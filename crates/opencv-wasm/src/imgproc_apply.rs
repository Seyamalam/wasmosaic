//! Adaptive thresholding, colormap application, and point-set geometry.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_border::border_index,
    imgproc_colormap::colormap_table,
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum ApplyError {
    UnsupportedDepth(MatDepth),
    InvalidKernelSize(i32),
    InvalidMethod(i32),
    InvalidThresholdType(i32),
    InvalidColormap(i32),
    InvalidPoints(&'static str),
    InvalidFlag(&'static str),
    OutputError(String),
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(depth) => {
                write!(f, "operation does not support matrix depth {depth:?}")
            }
            Self::InvalidKernelSize(size) => write!(f, "block size must be odd and >1: {size}"),
            Self::InvalidMethod(method) => write!(f, "unsupported adaptive method {method}"),
            Self::InvalidThresholdType(kind) => write!(f, "unsupported threshold type {kind}"),
            Self::InvalidColormap(index) => write!(f, "unsupported colormap {index}"),
            Self::InvalidPoints(reason) => write!(f, "invalid point set: {reason}"),
            Self::InvalidFlag(reason) => write!(f, "invalid flag: {reason}"),
            Self::OutputError(message) => write!(f, "{message}"),
        }
    }
}

impl Error for ApplyError {}

fn offset(row: u32, column: u32, columns: u32, channels: usize, channel: usize) -> usize {
    ((row * columns + column) as usize) * channels + channel
}

fn require_u8_single(source: &Mat) -> Result<(), ApplyError> {
    if source.channels() != 1 {
        return Err(ApplyError::InvalidPoints("single channel input required"));
    }
    if source.depth() != MatDepth::U8 {
        return Err(ApplyError::UnsupportedDepth(source.depth()));
    }
    Ok(())
}

const ADAPTIVE_MEAN: i32 = 0;
const ADAPTIVE_GAUSSIAN: i32 = 1;
const THRESH_BINARY: i32 = 0;
const THRESH_BINARY_INV: i32 = 1;

/// adaptiveThreshold: `dst = src > weightedMean - C ? maxValue : 0` (or inverse).
pub(crate) fn adaptive_threshold_into(
    source: &Mat,
    destination: &Mat,
    max_value: f64,
    adaptive_method: i32,
    threshold_type: i32,
    block_size: i32,
    delta: f64,
) -> Result<(), ApplyError> {
    require_u8_single(source)?;
    if !(ADAPTIVE_MEAN..=ADAPTIVE_GAUSSIAN).contains(&adaptive_method) {
        return Err(ApplyError::InvalidMethod(adaptive_method));
    }
    if !(THRESH_BINARY..=THRESH_BINARY_INV).contains(&threshold_type) {
        return Err(ApplyError::InvalidThresholdType(threshold_type));
    }
    if block_size <= 1 || block_size % 2 == 0 {
        return Err(ApplyError::InvalidKernelSize(block_size));
    }
    // Documented getGaussianKernel sigma for an odd kernel size.
    let sigma = if adaptive_method == ADAPTIVE_GAUSSIAN {
        0.3 * ((block_size - 1) as f64 * 0.5 - 1.0) + 0.8
    } else {
        0.0
    };
    let radius = block_size / 2;
    let weights: Vec<f64> = (0..block_size)
        .map(|i| {
            if adaptive_method == ADAPTIVE_MEAN {
                1.0
            } else {
                let x = f64::from(i - radius);
                (-x * x / (2.0 * sigma * sigma)).exp()
            }
        })
        .collect();
    let input = source.compact_bytes();
    let rows = source.rows();
    let columns = source.columns();
    let mut output = vec![0u8; (rows * columns) as usize];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for ky in 0..block_size {
                let sy = i64::from(row) + i64::from(ky) - i64::from(radius);
                let Some(y) = border_index(sy, rows, crate::imgproc_border::BORDER_REPLICATE) else {
                    continue;
                };
                for kx in 0..block_size {
                    let sx = i64::from(column) + i64::from(kx) - i64::from(radius);
                    let Some(x) = border_index(sx, columns, crate::imgproc_border::BORDER_REPLICATE) else {
                        continue;
                    };
                    let weight = weights[ky as usize] * weights[kx as usize];
                    sum += weight * f64::from(input[(y * columns + x) as usize]);
                    wsum += weight;
                }
            }
            let threshold = sum / wsum - delta;
            let value = f64::from(input[(row * columns + column) as usize]);
            let on = value > threshold;
            let out = if threshold_type == THRESH_BINARY {
                if on {
                    max_value
                } else {
                    0.0
                }
            } else if on {
                0.0
            } else {
                max_value
            };
            output[(row * columns + column) as usize] = write_u8(out);
        }
    }
    destination.write_output(output, rows, columns, 1, MatDepth::U8)
        .map_err(|e| ApplyError::OutputError(e.to_string()))?;
    Ok(())
}

fn write_u8(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// applyColorMap: U8 -> 8UC3 through a 256-entry BGR table.
pub(crate) fn apply_color_map_into(
    source: &Mat,
    destination: &Mat,
    colormap: i32,
) -> Result<(), ApplyError> {
    require_u8_single(source)?;
    let table = colormap_table(colormap).ok_or(ApplyError::InvalidColormap(colormap))?;
    let input = source.compact_bytes();
    let rows = source.rows();
    let columns = source.columns();
    let mut output = vec![0u8; input.len() * 3];
    for index in 0..input.len() {
        let value = input[index] as usize * 3;
        output[index * 3] = table[value];
        output[index * 3 + 1] = table[value + 1];
        output[index * 3 + 2] = table[value + 2];
    }
    destination.write_output(output, rows, columns, 3, MatDepth::U8)
        .map_err(|e| ApplyError::OutputError(e.to_string()))?;
    Ok(())
}

// ---- Point-set geometry ----

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

fn read_points(source: &Mat) -> Result<Vec<Point>, ApplyError> {
    let depth = scalar_depth(source.depth());
    if !matches!(depth, ScalarDepth::I32 | ScalarDepth::F32) || source.channels() != 2 {
        return Err(ApplyError::InvalidPoints(
            "expected a CV_32SC2 or CV_32FC2 point array",
        ));
    }
    let input = source.compact_bytes();
    let elements = input.len() / depth.width();
    if elements % 2 != 0 || source.rows() < 1 {
        return Err(ApplyError::InvalidPoints("expected an even element count"));
    }
    Ok((0..elements / 2)
        .map(|index| Point {
            x: read_scalar(&input, index * 2, depth),
            y: read_scalar(&input, index * 2 + 1, depth),
        })
        .collect())
}

/// Monotone-chain convex hull. Returns hull vertex indices into `points` in
/// counter-clockwise order starting at the lexicographically smallest point.
fn convex_hull_indices(points: &[Point]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[a]
            .x
            .partial_cmp(&points[b].x)
            .unwrap()
            .then(points[a].y.partial_cmp(&points[b].y).unwrap())
            .then(a.cmp(&b))
    });
    order.dedup_by(|a, b| points[*a].x == points[*b].x && points[*a].y == points[*b].y);
    if order.len() <= 1 {
        return order;
    }
    let cross = |o: &Point, a: &Point, b: &Point| {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    };
    let mut hull: Vec<usize> = Vec::with_capacity(order.len() * 2);
    for &index in &order {
        while hull.len() >= 2
            && cross(&points[hull[hull.len() - 2]], &points[hull[hull.len() - 1]], &points[index])
                <= 0.0
        {
            hull.pop();
        }
        hull.push(index);
    }
    let lower = hull.len() + 1;
    for &index in order.iter().rev().skip(1) {
        while hull.len() >= lower
            && cross(&points[hull[hull.len() - 2]], &points[hull[hull.len() - 1]], &points[index])
                <= 0.0
        {
            hull.pop();
        }
        hull.push(index);
    }
    hull.pop();
    // Rotate so the lexicographically smallest point is first.
    if let Some(start) = hull
        .iter()
        .enumerate()
        .min_by(|&(_, &a), &(_, &b)| {
            points[a]
                .x
                .partial_cmp(&points[b].x)
                .unwrap()
                .then(points[a].y.partial_cmp(&points[b].y).unwrap())
        })
        .map(|(index, _)| index)
    {
        hull.rotate_left(start);
    }
    hull
}

/// convexHull: monotone-chain hull; emits either vertex coords or indices.
pub(crate) fn convex_hull_into(
    source: &Mat,
    destination: &Mat,
    clockwise: bool,
    return_points: bool,
) -> Result<(), ApplyError> {
    let points = read_points(source)?;
    if points.is_empty() {
        return Err(ApplyError::InvalidPoints("empty point set"));
    }
    let mut hull = convex_hull_indices(&points);
    if clockwise {
        hull.reverse();
    }
    let (values, depth, channels): (Vec<f64>, MatDepth, u16) = if return_points {
        (
            hull.iter()
                .flat_map(|&index| [points[index].x, points[index].y])
                .collect(),
            source.depth(),
            2,
        )
    } else {
        (
            hull.iter().map(|&index| index as f64).collect(),
            MatDepth::I32,
            1,
        )
    };
    let scalar = scalar_depth(depth);
    let mut output = vec![0u8; values.len() * scalar.width()];
    for (index, value) in values.iter().enumerate() {
        write_scalar(&mut output, index, scalar, *value);
    }
    destination
        .write_output(output, hull.len() as u32, 1, channels, depth)
        .map_err(|e| ApplyError::OutputError(e.to_string()))?;
    Ok(())
}

/// Rotated rectangle: `(cx, cy, width, height, angle_degrees)`.
fn min_area_rect(hull: &[Point]) -> (f64, f64, f64, f64, f64) {
    let mut best = (0.0f64, 0.0f64, f64::MAX, f64::MAX, 0.0f64);
    for edge in 0..hull.len() {
        let a = hull[edge];
        let b = hull[(edge + 1) % hull.len()];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        if dx == 0.0 && dy == 0.0 {
            continue;
        }
        let angle = dy.atan2(dx);
        let (sin, cos) = angle.sin_cos();
        let mut min_u = f64::MAX;
        let mut max_u = f64::MIN;
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for point in hull {
            let u = point.x * cos + point.y * sin;
            let v = -point.x * sin + point.y * cos;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let width = max_u - min_u;
        let height = max_v - min_v;
        if width * height < best.2 * best.3 {
            let center_u = (min_u + max_u) / 2.0;
            let center_v = (min_v + max_v) / 2.0;
            let cx = center_u * cos - center_v * sin;
            let cy = center_u * sin + center_v * cos;
            // Pinned convention: `angle` is the width axis direction,
            // normalized into [-90, 0); width/height follow it.
            let mut angle_deg = (angle.to_degrees() + 90.0) % 180.0;
            if angle_deg >= 0.0 {
                angle_deg -= 180.0;
            }
            if angle_deg < -90.0 {
                angle_deg += 180.0;
            }
            best = (cx, cy, height, width, angle_deg);
        }
    }
    best
}

/// minAreaRect: rotating calipers over the hull; returns `[cx, cy, w, h, angle]`.
pub(crate) fn min_area_rect_points(source: &Mat) -> Result<[f64; 5], ApplyError> {
    let points = read_points(source)?;
    if points.len() < 2 {
        return Err(ApplyError::InvalidPoints("at least two points required"));
    }
    let hull: Vec<Point> = convex_hull_indices(&points)
        .iter()
        .map(|&index| points[index])
        .collect();
    let (cx, cy, w, h, angle) = min_area_rect(&hull);
    Ok([cx, cy, w, h, angle])
}

/// minEnclosingCircle: deterministic minimal enclosing circle.
pub(crate) fn min_enclosing_circle_points(source: &Mat) -> Result<[f64; 3], ApplyError> {
    let points = read_points(source)?;
    if points.is_empty() {
        return Err(ApplyError::InvalidPoints("empty point set"));
    }
    let (cx, cy, radius) = minimal_enclosing_circle(&points);
    Ok([cx, cy, radius])
}

fn circle_from_2(a: Point, b: Point) -> (f64, f64, f64) {
    ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, ((a.x - b.x).hypot(a.y - b.y)) / 2.0)
}

fn circle_from_3(a: Point, b: Point, c: Point) -> Option<(f64, f64, f64)> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-12 {
        return None;
    }
    let a2 = a.x * a.x + a.y * a.y;
    let b2 = b.x * b.x + b.y * b.y;
    let c2 = c.x * c.x + c.y * c.y;
    let ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d;
    let uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d;
    let radius = (ux - a.x).hypot(uy - a.y);
    Some((ux, uy, radius))
}

fn inside(circle: (f64, f64, f64), point: Point) -> bool {
    (point.x - circle.0).hypot(point.y - circle.1) <= circle.2 * (1.0 + 1e-9)
}

fn minimal_enclosing_circle(points: &[Point]) -> (f64, f64, f64) {
    // Deterministic incremental algorithm (Welzl order).
    let mut circle = (points[0].x, points[0].y, 0.0);
    for i in 1..points.len() {
        if inside(circle, points[i]) {
            continue;
        }
        circle = (points[i].x, points[i].y, 0.0);
        for j in 0..i {
            if inside(circle, points[j]) {
                continue;
            }
            circle = circle_from_2(points[i], points[j]);
            for k in 0..j {
                if inside(circle, points[k]) {
                    continue;
                }
                if let Some(c3) = circle_from_3(points[i], points[j], points[k]) {
                    circle = c3;
                }
            }
        }
    }
    circle
}

/// fitEllipse: Fitzgibbon least-squares conic fit; `[cx, cy, w, h, angle]`.
pub(crate) fn fit_ellipse_points(source: &Mat) -> Result<[f64; 5], ApplyError> {
    let points = read_points(source)?;
    if points.len() < 5 {
        return Err(ApplyError::InvalidPoints("at least five points required"));
    }
    // Direct least-squares conic fit (Fitzgibbon): solve for the conic
    // coefficients ax² + bxy + cy² + dx + ey + f = 0 under the constraint
    // a + c = 1.
    // Build the normal equations for [a, b, d, e, f] with c = 1 - a.
    let mut normal = [[0.0f64; 6]; 5];
    let mut rhs = [0.0f64; 5];
    for p in &points {
        let row = [p.x * p.x - p.y * p.y, p.x * p.y, p.x, p.y, 1.0];
        let target = -p.y * p.y;
        for i in 0..5 {
            rhs[i] += row[i] * target;
            for j in 0..5 {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    // Solve 5x5 by Gaussian elimination.
    for col in 0..5 {
        let mut pivot = col;
        for row in col + 1..5 {
            if normal[row][col].abs() > normal[pivot][col].abs() {
                pivot = row;
            }
        }
        normal.swap(col, pivot);
        rhs.swap(col, pivot);
        let divisor = normal[col][col];
        if divisor.abs() < 1e-12 {
            continue;
        }
        for row in col + 1..5 {
            let factor = normal[row][col] / divisor;
            for c in col..5 {
                normal[row][c] -= factor * normal[col][c];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut coeff = [0.0f64; 5];
    for row in (0..5).rev() {
        let mut value = rhs[row];
        for c in row + 1..5 {
            value -= normal[row][c] * coeff[c];
        }
        coeff[row] = if normal[row][row].abs() < 1e-12 {
            0.0
        } else {
            value / normal[row][row]
        };
    }
    let (a, b, d, e, f) = (coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]);
    let c = 1.0 - a;
    // Ellipse parameters from the conic.
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 || a.abs() < 1e-12 {
        // Degenerate: fall back to the enclosing-rectangle style result.
        let hull: Vec<Point> = convex_hull_indices(&points)
            .iter()
            .map(|&i| points[i])
            .collect();
        let (cx, cy, w, h, angle) = min_area_rect(&hull);
        return Ok([cx, cy, w, h, angle]);
    }
    let den = b * b - 4.0 * a * c;
    let cx = (2.0 * c * d - b * e) / den;
    let cy = (2.0 * a * e - b * d) / den;
    let common = a * cx * cx + b * cx * cy + c * cy * cy - f;
    let term = ((a - c) * (a - c) + b * b).sqrt();
    let major = (2.0 * common / ((a + c) - term)).sqrt().abs();
    let minor = (2.0 * common / ((a + c) + term)).sqrt().abs();
    let angle = if b.abs() < 1e-12 {
        if a < c {
            0.0
        } else {
            90.0
        }
    } else {
        (0.5 * (b / (a - c)).atan()).to_degrees()
    };
    Ok([cx, cy, 2.0 * major, 2.0 * minor, angle])
}

/// fitLine DIST_L2: direction = principal eigenvector; `[vx, vy, x0, y0]`.
pub(crate) fn fit_line_points(
    source: &Mat,
    destination: &Mat,
    dist_type: i32,
    _param: f64,
    _reps: f64,
    _aeps: f64,
) -> Result<(), ApplyError> {
    const DIST_L2: i32 = 2;
    if dist_type != DIST_L2 {
        return Err(ApplyError::InvalidFlag("only DIST_L2 is supported"));
    }
    let points = read_points(source)?;
    if points.len() < 2 {
        return Err(ApplyError::InvalidPoints("at least two points required"));
    }
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.x).sum::<f64>() / n;
    let my = points.iter().map(|p| p.y).sum::<f64>() / n;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut syy = 0.0f64;
    for p in &points {
        sxx += (p.x - mx) * (p.x - mx);
        sxy += (p.x - mx) * (p.y - my);
        syy += (p.y - my) * (p.y - my);
    }
    // Principal eigenvector of [[sxx, sxy], [sxy, syy]].
    let (mut vx, mut vy) = if sxy.abs() < 1e-12 {
        if sxx >= syy {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        }
    } else {
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        (theta.cos(), theta.sin())
    };
    let norm = vx.hypot(vy);
    if norm < 1e-12 {
        vx = 1.0;
        vy = 0.0;
    } else {
        vx /= norm;
        vy /= norm;
    }
    // Match the pinned sign convention (vx >= 0).
    if vx < 0.0 {
        vx = -vx;
        vy = -vy;
    }
    let values = [vx, vy, mx, my];
    let mut output = vec![0u8; 4 * ScalarDepth::F32.width()];
    for (index, value) in values.iter().enumerate() {
        write_scalar(&mut output, index, ScalarDepth::F32, *value);
    }
    destination
        .write_output(output, 4, 1, 1, MatDepth::F32)
        .map_err(|e| ApplyError::OutputError(e.to_string()))?;
    Ok(())
}
