//! Original iterative Ramer–Douglas–Peucker simplification of finite 2D curves.
//! Closed curves are split at a farthest-point pair before simplifying both arcs.
use crate::{
    imgproc_geometry::Point,
    imgproc_geometry_wasm::decode_browser_contour_query,
    mat::{Mat, MatDepth},
};

pub(crate) fn approx_poly_dp_into(
    source: &Mat,
    destination: &Mat,
    epsilon: f64,
    closed: bool,
) -> Result<(), String> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("epsilon must be finite and nonnegative".into());
    }
    let points = decode_browser_contour_query(source).map_err(|error| error.to_string())?;
    if points.is_empty() {
        return Err("contour must contain at least one point".into());
    }
    let indices = simplify(&points, epsilon, closed)?;
    // Preserve exact I32/F32 input coordinates, and snapshot before writing an alias.
    let source_bytes = source.compact_bytes();
    let mut bytes = Vec::with_capacity(indices.len() * 8);
    for &index in &indices {
        bytes.extend_from_slice(&source_bytes[index * 8..index * 8 + 8]);
    }
    if indices.is_empty() {
        destination.write_output(Vec::new(), 0, 0, 1, MatDepth::U8)
    } else {
        let count = u32::try_from(indices.len()).map_err(|_| "too many contour points")?;
        destination.write_output(bytes, count, 1, 2, source.depth())
    }
    .map_err(|error| error.to_string())
}

fn simplify(points: &[Point], epsilon: f64, closed: bool) -> Result<Vec<usize>, String> {
    if points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err("contour coordinates must be finite".into());
    }
    if points.len() <= 1 {
        return Ok((0..points.len()).collect());
    }
    let closed = closed || points.first() == points.last();
    let count = if closed && points.first() == points.last() {
        points.len() - 1
    } else {
        points.len()
    };
    if count <= 1 {
        return Ok(vec![0]);
    }
    let mut path: Vec<usize> = (0..count).collect();
    let mut spans = vec![(0, count - 1)];
    if closed {
        let farthest = |origin: usize| {
            (0..count).fold(origin, |best, index| {
                if distance_squared(points[origin], points[index])
                    > distance_squared(points[origin], points[best])
                {
                    index
                } else {
                    best
                }
            })
        };
        let first = farthest(0);
        let second = farthest(first);
        if distance_squared(points[first], points[second]) <= epsilon * epsilon {
            return Ok(vec![first]);
        }
        // Choose the earlier anchor to preserve input traversal order.
        let (start, end) = (first.min(second), first.max(second));
        path.rotate_left(start);
        path.push(start);
        spans = vec![(0, end - start), (end - start, count)];
    }
    let mut keep = vec![false; path.len()];
    while let Some((start, end)) = spans.pop() {
        keep[start] = true;
        keep[end] = true;
        let a = points[path[start]];
        let b = points[path[end]];
        let length = (b.x - a.x).hypot(b.y - a.y);
        let mut maximum = epsilon;
        let mut split = None;
        for position in start + 1..end {
            let p = points[path[position]];
            let projection = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);
            let distance = if length == 0.0 || projection <= 0.0 {
                (p.x - a.x).hypot(p.y - a.y)
            } else if projection >= length * length {
                (p.x - b.x).hypot(p.y - b.y)
            } else {
                ((p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)).abs() / length
            };
            if distance > maximum {
                maximum = distance;
                split = Some(position);
            }
        }
        if let Some(middle) = split {
            spans.push((start, middle));
            spans.push((middle, end));
        }
    }
    if closed {
        keep[path.len() - 1] = false;
    }
    Ok(path
        .into_iter()
        .zip(keep)
        .filter_map(|(index, keep)| keep.then_some(index))
        .collect())
}

fn distance_squared(a: Point, b: Point) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn points(values: &[(f64, f64)]) -> Vec<Point> {
        values.iter().map(|&(x, y)| Point { x, y }).collect()
    }
    fn matrix(values: &[i32], rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            rows,
            columns,
            channels,
            MatDepth::I32,
        )
        .unwrap()
    }
    #[test]
    fn matrix_alias_preserves_selected_i32_coordinates() {
        let source = matrix(&[0, 0, 0, 2, 0, 4, 4, 4, 4, 0], 5, 1, 2);
        approx_poly_dp_into(&source, &source, 0.5, true).unwrap();
        assert_eq!(source.rows(), 4);
        assert_eq!(
            source.compact_bytes(),
            matrix(&[0, 0, 0, 4, 4, 4, 4, 0], 4, 1, 2).compact_bytes()
        );
    }
    #[test]
    fn invalid_epsilon_and_layout_leave_destination_untouched() {
        let source = matrix(&[0, 0, 2, 2, 4, 0], 3, 1, 2);
        let destination = matrix(&[99, 99], 1, 1, 2);
        let before = destination.compact_bytes();
        for epsilon in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(approx_poly_dp_into(&source, &destination, epsilon, true).is_err());
            assert_eq!(destination.compact_bytes(), before);
        }
        let invalid = matrix(&[0, 1, 2, 3, 4, 5], 2, 3, 1);
        assert!(approx_poly_dp_into(&invalid, &destination, 1.0, false).is_err());
        assert_eq!(destination.compact_bytes(), before);
    }
    #[test]
    fn strided_source_rejected_and_destination_roi_writes_through() {
        let parent = matrix(&[0; 20], 5, 2, 2);
        let source = parent.roi(0, 0, 5, 1).unwrap();
        let destination = crate::mat::mat_empty();
        assert!(approx_poly_dp_into(&source, &destination, 1.0, true).is_err());
        let curve = matrix(&[0, 0, 0, 2, 0, 4, 4, 4, 4, 0], 5, 1, 2);
        let output_parent = matrix(&[99; 48], 8, 3, 2);
        let output_roi = output_parent.roi(1, 1, 4, 1).unwrap();
        approx_poly_dp_into(&curve, &output_roi, 0.5, true).unwrap();
        let bytes = output_parent.compact_bytes();
        assert_eq!(&bytes[32..40], &[0; 8]);
        assert_eq!(&bytes[..8], &matrix(&[99, 99], 1, 1, 2).compact_bytes());
    }
    #[test]
    fn open_curve_retains_endpoints_and_large_excursions() {
        let p = points(&[(0., 0.), (1., 0.1), (2., 0.), (3., 4.), (4., 0.)]);
        assert_eq!(simplify(&p, 0.2, false).unwrap(), [0, 2, 3, 4]);
        assert_eq!(simplify(&p, 5., false).unwrap(), [0, 4]);
    }
    #[test]
    fn closed_rectangle_removes_collinear_points_and_duplicate_endpoint() {
        let p = points(&[(0., 0.), (0., 2.), (0., 4.), (4., 4.), (4., 0.), (0., 0.)]);
        assert_eq!(simplify(&p, 0., true).unwrap(), [0, 2, 3, 4]);
        assert_eq!(simplify(&p, 0., false).unwrap(), [0, 2, 3, 4]);
    }
    #[test]
    fn segment_distance_preserves_collinear_backtracking() {
        let p = points(&[(0., 0.), (5., 0.), (1., 0.)]);
        assert_eq!(simplify(&p, 1., false).unwrap(), [0, 1, 2]);
    }
    #[test]
    fn degenerate_and_nonfinite_curves() {
        assert_eq!(simplify(&points(&[(2., 3.); 4]), 0., true).unwrap(), [0]);
        assert!(simplify(&points(&[(f64::NAN, 0.)]), 1., false).is_err());
        assert!(simplify(&[], 0., false).unwrap().is_empty());
    }
}
