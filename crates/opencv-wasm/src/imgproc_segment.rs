//! Connected components, distance transforms, and flood fill.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum SegmentError {
    UnsupportedDepth(MatDepth),
    InvalidConnectivity(i32),
    InvalidDistanceType(i32),
    InvalidMaskSize(i32),
    InvalidLabelType(i32),
    InvalidImage(&'static str),
    OutputError(String),
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(d) => write!(f, "unsupported matrix depth {d:?}"),
            Self::InvalidConnectivity(v) => write!(f, "connectivity must be 4 or 8: {v}"),
            Self::InvalidDistanceType(v) => write!(f, "unsupported distance type {v}"),
            Self::InvalidMaskSize(v) => write!(f, "mask size must be 3 or 5: {v}"),
            Self::InvalidLabelType(v) => write!(f, "label type must be CV_32S or CV_16U: {v}"),
            Self::InvalidImage(r) => write!(f, "invalid image: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}

impl Error for SegmentError {}

/// Two-pass union-find labeling in scan order (label 0 = background).
fn label_components(pixels: &[u8], rows: u32, cols: u32, connectivity: i32) -> (Vec<i32>, i32) {
    let n = (rows * cols) as usize;
    let mut labels = vec![0i32; n];
    let mut parent: Vec<i32> = vec![0];
    let find = |parent: &mut Vec<i32>, mut x: i32| -> i32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    };
    let mut next = 1i32;
    let w = cols as usize;
    for y in 0..rows as usize {
        for x in 0..cols as usize {
            if pixels[y * w + x] == 0 {
                continue;
            }
            let mut best = i32::MAX;
            let mut neighbors = Vec::with_capacity(4);
            if x > 0 && labels[y * w + x - 1] > 0 {
                neighbors.push(labels[y * w + x - 1]);
            }
            if y > 0 {
                if labels[(y - 1) * w + x] > 0 {
                    neighbors.push(labels[(y - 1) * w + x]);
                }
                if connectivity == 8 {
                    if x > 0 && labels[(y - 1) * w + x - 1] > 0 {
                        neighbors.push(labels[(y - 1) * w + x - 1]);
                    }
                    if x + 1 < cols as usize && labels[(y - 1) * w + x + 1] > 0 {
                        neighbors.push(labels[(y - 1) * w + x + 1]);
                    }
                }
            }
            for &l in &neighbors {
                let r = find(&mut parent, l);
                if r < best {
                    best = r;
                }
            }
            if neighbors.is_empty() {
                parent.push(next);
                labels[y * w + x] = next;
                next += 1;
            } else {
                labels[y * w + x] = best;
                for &l in &neighbors {
                    let r = find(&mut parent, l);
                    if r != best {
                        parent[r as usize] = best;
                    }
                }
            }
        }
    }
    // Compact labels to 1..count in first-seen order.
    let mut remap = vec![0i32; next as usize + 1];
    let mut count = 1i32;
    for label in labels.iter_mut() {
        if *label > 0 {
            let root = find(&mut parent, *label);
            if remap[root as usize] == 0 {
                remap[root as usize] = count;
                count += 1;
            }
            *label = remap[root as usize];
        }
    }
    (labels, count)
}

/// connectedComponents: emits the label map; returns the component count
/// including the background label.
pub(crate) fn connected_components_into(
    source: &Mat,
    labels_out: &Mat,
    connectivity: i32,
    label_type: i32,
) -> Result<i32, SegmentError> {
    if source.channels() != 1 || source.depth() != MatDepth::U8 {
        return Err(SegmentError::UnsupportedDepth(source.depth()));
    }
    if !(connectivity == 4 || connectivity == 8) {
        return Err(SegmentError::InvalidConnectivity(connectivity));
    }
    if !(label_type == 4 || label_type == 2) {
        return Err(SegmentError::InvalidLabelType(label_type));
    }
    let input = source.compact_bytes();
    let rows = source.rows();
    let cols = source.columns();
    let (labels, count) = label_components(&input, rows, cols, connectivity);
    let depth = if label_type == 2 {
        MatDepth::U16
    } else {
        MatDepth::I32
    };
    let scalar = scalar_depth(depth);
    let mut output = vec![0u8; labels.len() * scalar.width()];
    for (index, &label) in labels.iter().enumerate() {
        write_scalar(&mut output, index, scalar, f64::from(label));
    }
    labels_out
        .write_output(output, rows, cols, 1, depth)
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    Ok(count)
}

/// connectedComponentsWithStats: labels plus [left,top,w,h,area] and centroids.
#[allow(clippy::too_many_arguments)]
pub(crate) fn connected_components_with_stats_into(
    source: &Mat,
    labels_out: &Mat,
    stats_out: &Mat,
    centroids_out: &Mat,
    connectivity: i32,
    label_type: i32,
) -> Result<i32, SegmentError> {
    let count = connected_components_into(source, labels_out, connectivity, label_type)?;
    let labels: Vec<i32> = {
        let bytes = labels_out.compact_bytes();
        let depth = scalar_depth(labels_out.depth());
        (0..bytes.len() / depth.width())
            .map(|i| read_scalar(&bytes, i, depth) as i32)
            .collect()
    };
    let rows = source.rows();
    let cols = source.columns();
    let n = count as usize;
    let mut left = vec![u32::MAX; n];
    let mut top = vec![u32::MAX; n];
    let mut right = vec![0u32; n];
    let mut bottom = vec![0u32; n];
    let mut area = vec![0i32; n];
    let mut sx = vec![0.0f64; n];
    let mut sy = vec![0.0f64; n];
    for y in 0..rows {
        for x in 0..cols {
            let label = labels[(y * cols + x) as usize] as usize;
            area[label] += 1;
            left[label] = left[label].min(x);
            top[label] = top[label].min(y);
            right[label] = right[label].max(x);
            bottom[label] = bottom[label].max(y);
            sx[label] += f64::from(x);
            sy[label] += f64::from(y);
        }
    }
    let mut stats = vec![0i32; n * 5];
    let mut centroids = vec![0.0f64; n * 2];
    for l in 0..n {
        stats[l * 5] = if area[l] > 0 { left[l] as i32 } else { 0 };
        stats[l * 5 + 1] = if area[l] > 0 { top[l] as i32 } else { 0 };
        stats[l * 5 + 2] = if area[l] > 0 { (right[l] - left[l] + 1) as i32 } else { 0 };
        stats[l * 5 + 3] = if area[l] > 0 { (bottom[l] - top[l] + 1) as i32 } else { 0 };
        stats[l * 5 + 4] = area[l];
        centroids[l * 2] = if area[l] > 0 {
            sx[l] / f64::from(area[l])
        } else {
            0.0
        };
        centroids[l * 2 + 1] = if area[l] > 0 {
            sy[l] / f64::from(area[l])
        } else {
            0.0
        };
    }
    let mut stats_bytes = vec![0u8; stats.len() * 4];
    for (i, v) in stats.iter().enumerate() {
        write_scalar(&mut stats_bytes, i, ScalarDepth::I32, f64::from(*v));
    }
    stats_out
        .write_output(stats_bytes, count as u32, 5, 1, MatDepth::I32)
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    let mut cent_bytes = vec![0u8; centroids.len() * 8];
    for (i, v) in centroids.iter().enumerate() {
        write_scalar(&mut cent_bytes, i, ScalarDepth::F64, *v);
    }
    centroids_out
        .write_output(cent_bytes, count as u32, 2, 1, MatDepth::F64)
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    Ok(count)
}

/// distanceTransform: chamfer (mask 3) or exact (mask 5 for L2) distances.
pub(crate) fn distance_transform_into(
    source: &Mat,
    destination: &Mat,
    distance_type: i32,
    mask_size: i32,
) -> Result<(), SegmentError> {
    if source.channels() != 1 || source.depth() != MatDepth::U8 {
        return Err(SegmentError::UnsupportedDepth(source.depth()));
    }
    if !(mask_size == 3 || mask_size == 5) {
        return Err(SegmentError::InvalidMaskSize(mask_size));
    }
    const DIST_L1: i32 = 1;
    const DIST_L2: i32 = 2;
    const DIST_C: i32 = 3;
    const DIST_L12: i32 = 4;
    let input = source.compact_bytes();
    let rows = source.rows();
    let cols = source.columns();
    let w = cols as usize;
    let inf = 1e30f64;
    let mut dist = vec![inf; (rows * cols) as usize];
    for (i, &p) in input.iter().enumerate() {
        if p == 0 {
            dist[i] = 0.0;
        }
    }
    // Neighbor offsets and weights per distance type.
    let (fwd, bwd): (Vec<(i64, i64, f64)>, Vec<(i64, i64, f64)>) = match distance_type {
        DIST_L1 => (
            vec![(-1, 0, 1.0), (0, -1, 1.0)],
            vec![(0, 1, 1.0), (1, 0, 1.0)],
        ),
        DIST_L12 => (
            vec![(-1, 0, 2.0), (0, -1, 2.0)],
            vec![(0, 1, 2.0), (1, 0, 2.0)],
        ),
        DIST_C => (
            vec![(-1, -1, 1.0), (-1, 0, 1.0), (-1, 1, 1.0), (0, -1, 1.0)],
            vec![(0, 1, 1.0), (1, -1, 1.0), (1, 0, 1.0), (1, 1, 1.0)],
        ),
        DIST_L2 if mask_size == 3 => {
            let (d1, d2) = (0.955f64, 1.3693f64);
            (
                vec![(-1, -1, d2), (-1, 0, d1), (-1, 1, d2), (0, -1, d1)],
                vec![(0, 1, d1), (1, -1, d2), (1, 0, d1), (1, 1, d2)],
            )
        }
        DIST_L2 => {
            // 5x5 mask: adds knight-move neighbors with weight 1.4.
            let (d1, d2, d3) = (1.0f64, 1.4f64, 2.0f64);
            (
                vec![
                    (-2, -1, d3),
                    (-2, 0, d3),
                    (-2, 1, d3),
                    (-1, -2, d3),
                    (-1, -1, d2),
                    (-1, 0, d1),
                    (-1, 1, d2),
                    (-1, 2, d3),
                    (0, -2, d3),
                    (0, -1, d1),
                ],
                vec![
                    (0, 1, d1),
                    (0, 2, d3),
                    (1, -2, d3),
                    (1, -1, d2),
                    (1, 0, d1),
                    (1, 1, d2),
                    (1, 2, d3),
                    (2, -1, d3),
                    (2, 0, d3),
                    (2, 1, d3),
                ],
            )
        }
        other => return Err(SegmentError::InvalidDistanceType(other)),
    };
    // Forward pass.
    for y in 0..rows as i64 {
        for x in 0..cols as i64 {
            for &(dy, dx, weight) in &fwd {
                let (sy, sx) = (y + dy, x + dx);
                if sy >= 0 && sy < i64::from(rows) && sx >= 0 && sx < i64::from(cols) {
                    let candidate = dist[(sy * cols as i64 + sx) as usize] + weight;
                    if candidate < dist[(y * cols as i64 + x) as usize] {
                        dist[(y * cols as i64 + x) as usize] = candidate;
                    }
                }
            }
        }
    }
    // Backward pass.
    for y in (0..rows as i64).rev() {
        for x in (0..cols as i64).rev() {
            for &(dy, dx, weight) in &bwd {
                let (sy, sx) = (y + dy, x + dx);
                if sy >= 0 && sy < i64::from(rows) && sx >= 0 && sx < i64::from(cols) {
                    let candidate = dist[(sy * cols as i64 + sx) as usize] + weight;
                    if candidate < dist[(y * cols as i64 + x) as usize] {
                        dist[(y * cols as i64 + x) as usize] = candidate;
                    }
                }
            }
        }
    }
    let (out_depth, scale) = match distance_type {
        DIST_L2 => (MatDepth::F32, 1.0),
        DIST_L12 => (MatDepth::I32, 0.5),
        _ => (MatDepth::F32, 1.0),
    };
    let scalar = scalar_depth(out_depth);
    let mut output = vec![0u8; dist.len() * scalar.width()];
    for (i, &d) in dist.iter().enumerate() {
        write_scalar(&mut output, i, scalar, d * scale);
    }
    destination
        .write_output(output, rows, cols, 1, out_depth)
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    let _ = w;
    Ok(())
}

/// floodFill: region grow from the seed over in-range neighbors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flood_fill_into(
    image: &Mat,
    mask: &Mat,
    seed_x: i32,
    seed_y: i32,
    new_value: f64,
    lo_diff: f64,
    up_diff: f64,
    flags: i32,
) -> Result<i32, SegmentError> {
    if image.depth() != MatDepth::U8 && image.depth() != MatDepth::F32 {
        return Err(SegmentError::UnsupportedDepth(image.depth()));
    }
    if image.channels() != 1 && image.channels() != 3 {
        return Err(SegmentError::InvalidImage("expected a single- or 3-channel image"));
    }
    let rows = image.rows();
    let cols = image.columns();
    if mask.rows() != rows + 2 || mask.columns() != cols + 2 || mask.depth() != MatDepth::U8 {
        return Err(SegmentError::InvalidImage(
            "mask must be (rows+2)x(cols+2) CV_8U",
        ));
    }
    if seed_x < 0 || seed_x >= cols as i32 || seed_y < 0 || seed_y >= rows as i32 {
        return Err(SegmentError::InvalidImage("seed point out of range"));
    }
    let connectivity = flags & 0xFF;
    let fixed_range = (flags & 0x0001_0000) != 0;
    let use_mask_only = (flags & 0x0002_0000) != 0;
    if !(connectivity == 4 || connectivity == 8) {
        return Err(SegmentError::InvalidConnectivity(connectivity));
    }
    let depth = scalar_depth(image.depth());
    let channels = usize::from(image.channels());
    let mut img = image.compact_bytes();
    let mut msk = mask.compact_bytes();
    let w = cols as usize;
    let mw = (cols + 2) as usize;
    let at = |buf: &[u8], y: i64, x: i64, c: usize| -> f64 {
        read_scalar(buf, ((y as usize) * w + x as usize) * channels + c, depth)
    };
    let seed_vals: Vec<f64> = (0..channels)
        .map(|c| at(&img, i64::from(seed_y), i64::from(seed_x), c))
        .collect();
    let in_range = |value: f64, seed: f64| -> bool {
        let (lo, hi) = if fixed_range {
            (seed - lo_diff, seed + up_diff)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        value >= lo && value <= hi
    };
    let mut count = 0i32;
    let mut stack = vec![(i64::from(seed_y), i64::from(seed_x))];
    while let Some((y, x)) = stack.pop() {
        if y < 0 || y >= i64::from(rows) || x < 0 || x >= i64::from(cols) {
            continue;
        }
        let mask_index = ((y + 1) as usize) * mw + (x + 1) as usize;
        if msk[mask_index] != 0 {
            continue;
        }
        let matches = (0..channels).all(|c| {
            let v = at(&img, y, x, c);
            if fixed_range {
                in_range(v, seed_vals[c])
            } else {
                let (lo, hi) = (seed_vals[c] - lo_diff, seed_vals[c] + up_diff);
                v >= lo && v <= hi
            }
        });
        if !matches {
            continue;
        }
        msk[mask_index] = 1;
        count += 1;
        if !use_mask_only {
            for c in 0..channels {
                let index = ((y as usize) * w + x as usize) * channels + c;
                write_scalar(&mut img, index, depth, new_value);
            }
        }
        let offsets: &[(i64, i64)] = if connectivity == 4 {
            &[(-1, 0), (1, 0), (0, -1), (0, 1)]
        } else {
            &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        };
        for &(dy, dx) in offsets {
            stack.push((y + dy, x + dx));
        }
    }
    // Pinned behavior: the mask border frame is written as nonzero.
    for x in 0..mw {
        msk[x] = 1;
        msk[(mask.rows() as usize - 1) * mw + x] = 1;
    }
    for y in 0..mask.rows() as usize {
        msk[y * mw] = 1;
        msk[y * mw + mw - 1] = 1;
    }
    image
        .write_output(img, rows, cols, image.channels(), image.depth())
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    mask.write_output(msk, mask.rows(), mask.columns(), 1, MatDepth::U8)
        .map_err(|e| SegmentError::OutputError(e.to_string()))?;
    Ok(count)
}
