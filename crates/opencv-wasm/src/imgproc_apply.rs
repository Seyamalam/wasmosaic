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
        let length = (dx * dx + dy * dy).sqrt();
        let cos = (dx / length) as f32;
        let sin = (dy / length) as f32;
        let len2 = length * length;
        let (dx32, dy32) = (dx as f32, dy as f32);
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        let mut min_uu = f32::MAX;
        let mut max_uu = f32::MIN;
        let mut min_vu = f32::MAX;
        let mut max_vu = f32::MIN;
        for point in hull {
            let (px, py) = (point.x as f32, point.y as f32);
            let u = px * cos + py * sin;
            let v = -px * sin + py * cos;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            let uu = px * dx32 + py * dy32;
            let vu = -px * dy32 + py * dx32;
            min_uu = min_uu.min(uu);
            max_uu = max_uu.max(uu);
            min_vu = min_vu.min(vu);
            max_vu = max_vu.max(vu);
        }
        let width = max_v - min_v;
        let height = max_u - min_u;
        if f64::from(width) * f64::from(height) < best.2 * best.3 {
            let center_u = f64::from((min_uu + max_uu) * 0.5);
            let center_v = f64::from((min_vu + max_vu) * 0.5);
            let cx = (center_u * dx - center_v * dy) / len2;
            let cy = (center_u * dy + center_v * dx) / len2;
            // Pinned convention: `angle` is the width axis direction,
            // normalized into [-90, 0); width/height follow it.
            let mut angle_deg =
                f64::from(cos).atan2(-f64::from(sin)).to_degrees();
            while angle_deg >= 0.0 {
                angle_deg -= 180.0;
            }
            if angle_deg < -90.0 {
                angle_deg += 180.0;
            }
            best = (
                cx,
                cy,
                f64::from(width),
                f64::from(height),
                f64::from(angle_deg as f32),
            );
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
    Ok([
        f64::from(cx as f32),
        f64::from(cy as f32),
        f64::from(w as f32),
        f64::from(h as f32),
        f64::from(angle as f32),
    ])
}

/// minEnclosingCircle: deterministic minimal enclosing circle.
pub(crate) fn min_enclosing_circle_points(source: &Mat) -> Result<[f64; 3], ApplyError> {
    let points = read_points(source)?;
    if points.is_empty() {
        return Err(ApplyError::InvalidPoints("empty point set"));
    }
    let (cx, cy, radius) = minimal_enclosing_circle(&points);
    Ok([
        f64::from(cx as f32),
        f64::from(cy as f32),
        f64::from(radius as f32),
    ])
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
    conic_to_rotated(a, b, c, d, e, f).ok_or(ApplyError::InvalidPoints("degenerate conic"))
}

/// Convert a conic `ax² + bxy + cy² + dx + ey + f = 0` into `[cx, cy, w, h, angle]`.
/// Returns `None` when the conic is not a non-degenerate ellipse.
fn conic_to_rotated(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Option<[f64; 5]> {
    let den = b * b - 4.0 * a * c;
    if den >= 0.0 {
        return None;
    }
    let cx = (2.0 * c * d - b * e) / den;
    let cy = (2.0 * a * e - b * d) / den;
    let common = a * cx * cx + b * cx * cy + c * cy * cy - f;
    let term = ((a - c) * (a - c) + b * b).sqrt();
    let major = (2.0 * common / ((a + c) - term)).sqrt().abs();
    let minor = (2.0 * common / ((a + c) + term)).sqrt().abs();
    let mut angle = if b.abs() < 1e-12 {
        if a < c {
            0.0
        } else {
            90.0
        }
    } else {
        (0.5 * (b / (a - c)).atan()).to_degrees()
    };
    let mut width = 2.0 * major;
    let mut height = 2.0 * minor;
    // Pinned convention: angle of the width axis in [0, 180).
    if width > height {
        std::mem::swap(&mut width, &mut height);
        angle += 90.0;
    }
    let mut angle = ((angle % 180.0) + 180.0) % 180.0;
    if angle < 0.0 {
        angle += 180.0;
    }
    Some([
        f64::from(cx as f32),
        f64::from(cy as f32),
        f64::from(width as f32),
        f64::from(height as f32),
        f64::from(angle as f32),
    ])
}

/// Per-point conic scatter blocks: S1 = Σ d1·d1ᵀ, S2 = Σ d1·d2ᵀ, S3 = Σ d2·d2ᵀ
/// with d1 = [x², xy, y²] and d2 = [x, y, 1].
fn conic_scatter(points: &[Point]) -> ([[f64; 3]; 3], [[f64; 3]; 3], [[f64; 3]; 3]) {
    let mut s1 = [[0.0; 3]; 3];
    let mut s2 = [[0.0; 3]; 3];
    let mut s3 = [[0.0; 3]; 3];
    for p in points {
        let d1 = [p.x * p.x, p.x * p.y, p.y * p.y];
        let d2 = [p.x, p.y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += d1[i] * d1[j];
                s2[i][j] += d1[i] * d2[j];
                s3[i][j] += d2[i] * d2[j];
            }
        }
    }
    (s1, s2, s3)
}

fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    Some(inv)
}

/// Real roots of the characteristic polynomial of a 3×3 matrix.
fn matrix_roots3(a: &[[f64; 3]; 3]) -> Vec<f64> {
    let tr = a[0][0] + a[1][1] + a[2][2];
    let minors = a[0][0] * a[1][1] - a[0][1] * a[1][0] + a[0][0] * a[2][2]
        - a[0][2] * a[2][0]
        + a[1][1] * a[2][2]
        - a[1][2] * a[2][1];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    // λ³ + c2·λ² + c1·λ + c0 = 0
    let c2 = -tr;
    let c1 = minors;
    let c0 = -det;
    // Depress: λ = t − c2/3.
    let p = c1 - c2 * c2 / 3.0;
    let q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0;
    let shift = -c2 / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = Vec::new();
    if disc <= 0.0 && p < 0.0 {
        let radius = 2.0 * (-p / 3.0).sqrt();
        let arg = (3.0 * q / (2.0 * p)) * (-3.0 / p).sqrt();
        let theta = arg.clamp(-1.0, 1.0).acos() / 3.0;
        for k in 0..3 {
            roots.push(radius * (theta - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + shift);
        }
    } else {
        let sqrt_disc = disc.max(0.0).sqrt();
        let t = (-q / 2.0 + sqrt_disc).cbrt() + (-q / 2.0 - sqrt_disc).cbrt();
        roots.push(t + shift);
    }
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

/// Approximate null vector of `(a − λ·I)` via the largest row-pair cross product.
fn null3(a: &[[f64; 3]; 3], lambda: f64) -> [f64; 3] {
    let mut r = *a;
    for i in 0..3 {
        r[i][i] -= lambda;
    }
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let pairs = [
        cross(r[0], r[1]),
        cross(r[0], r[2]),
        cross(r[1], r[2]),
    ];
    let mut best = pairs[0];
    for p in &pairs[1..] {
        if p.iter().map(|v| v * v).sum::<f64>() > best.iter().map(|v| v * v).sum::<f64>() {
            best = *p;
        }
    }
    best
}

/// fitEllipseDirect: direct least-squares conic fit under the
/// ellipse-specificity constraint `4ac − b² = 1` (Fitzgibbon generalized
/// eigenproblem, Schur-reduced to 3×3).
pub(crate) fn fit_ellipse_direct_points(source: &Mat) -> Result<[f64; 5], ApplyError> {
    let points = read_points(source)?;
    if points.len() < 5 {
        return Err(ApplyError::InvalidPoints("at least five points required"));
    }
    let (s1, s2, s3) = conic_scatter(&points);
    let inv3 = invert3(&s3).ok_or(ApplyError::InvalidPoints("singular scatter"))?;
    // T = −S3⁻¹·S2ᵀ; M = S1 + S2·T.
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = -(0..3).map(|k| inv3[i][k] * s2[j][k]).sum::<f64>();
        }
    }
    let mut m = s1;
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] += (0..3).map(|k| s2[i][k] * t[k][j]).sum::<f64>();
        }
    }
    // Constraint matrix C = [[0,0,2],[0,-1,0],[2,0,0]] has inverse
    // [[0,0,0.5],[0,-1,0],[0.5,0,0]]; solve A·v = λ·v with A = C⁻¹·M.
    let a3 = [
        [0.5 * m[2][0], 0.5 * m[2][1], 0.5 * m[2][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [0.5 * m[0][0], 0.5 * m[0][1], 0.5 * m[0][2]],
    ];
    let mut conic: Option<[f64; 6]> = None;
    for lambda in matrix_roots3(&a3) {
        let a1 = null3(&a3, lambda);
        if 4.0 * a1[0] * a1[2] - a1[1] * a1[1] <= 0.0 {
            continue;
        }
        let mut a2 = [0.0; 3];
        for i in 0..3 {
            a2[i] = (0..3).map(|k| t[i][k] * a1[k]).sum();
        }
        conic = Some([a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]]);
        break;
    }
    let c = conic.ok_or(ApplyError::InvalidPoints("no elliptic conic"))?;
    conic_to_rotated(c[0], c[1], c[2], c[3], c[4], c[5])
        .ok_or(ApplyError::InvalidPoints("degenerate conic"))
}

/// fitEllipseAMS: approximate mean-square fit — minimize `aᵀSa / aᵀNa` where
/// N is the per-point data-gradient scatter. Solved as a generalized
/// eigenproblem via Cholesky of N plus a symmetric Jacobi sweep.
pub(crate) fn fit_ellipse_ams_points(source: &Mat) -> Result<[f64; 5], ApplyError> {
    let points = read_points(source)?;
    if points.len() < 5 {
        return Err(ApplyError::InvalidPoints("at least five points required"));
    }
    let mut s = [[0.0f64; 6]; 6];
    let mut n = [[0.0f64; 6]; 6];
    for p in &points {
        let d = [p.x * p.x, p.x * p.y, p.y * p.y, p.x, p.y, 1.0];
        // Jacobian of the design row with respect to (x, y).
        let j0 = [2.0 * p.x, p.y, 0.0, 1.0, 0.0, 0.0];
        let j1 = [0.0, p.x, 2.0 * p.y, 0.0, 1.0, 0.0];
        for i in 0..6 {
            for j in 0..6 {
                s[i][j] += d[i] * d[j];
                n[i][j] += j0[i] * j0[j] + j1[i] * j1[j];
            }
        }
    }
    // Cholesky N = L·Lᵀ.
    let mut l = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = n[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-14 {
                    // The gradient scatter is rank-deficient (the constant term
                    // carries no gradient), so the generalized problem has no
                    // finite solution — fall back to the direct fit.
                    return fit_ellipse_direct_points(source);
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // B = L⁻¹·S·L⁻ᵀ by forward substitution.
    let mut x = [[0.0f64; 6]; 6];
    for col in 0..6 {
        for i in 0..6 {
            let mut sum = s[i][col];
            for k in 0..i {
                sum -= l[i][k] * x[k][col];
            }
            x[i][col] = sum / l[i][i];
        }
    }
    let mut b = [[0.0f64; 6]; 6];
    for row in 0..6 {
        for j in 0..6 {
            let mut sum = x[row][j];
            for k in 0..j {
                sum -= x[row][k] * 0.0 + l[j][k] * b[row][k];
            }
            b[row][j] = sum / l[j][j];
        }
    }
    for i in 0..6 {
        for j in i + 1..6 {
            let v = 0.5 * (b[i][j] + b[j][i]);
            b[i][j] = v;
            b[j][i] = v;
        }
    }
    // Cyclic Jacobi sweeps on B.
    let mut vec = [[0.0f64; 6]; 6];
    for i in 0..6 {
        vec[i][i] = 1.0;
    }
    for _ in 0..120 {
        let mut off = 0.0;
        for p in 0..6 {
            for q in p + 1..6 {
                off += b[p][q] * b[p][q];
            }
        }
        if off < 1e-18 {
            break;
        }
        for p in 0..6 {
            for q in p + 1..6 {
                if b[p][q].abs() < 1e-14 {
                    continue;
                }
                let theta = 0.5 * (b[q][q] - b[p][p]) / b[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cs = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * cs;
                for k in 0..6 {
                    let (bp, bq) = (b[p][k], b[q][k]);
                    b[p][k] = cs * bp - sn * bq;
                    b[q][k] = sn * bp + cs * bq;
                }
                for k in 0..6 {
                    let (bp, bq) = (b[k][p], b[k][q]);
                    b[k][p] = cs * bp - sn * bq;
                    b[k][q] = sn * bp + cs * bq;
                }
                for k in 0..6 {
                    let (vp, vq) = (vec[p][k], vec[q][k]);
                    vec[p][k] = cs * vp - sn * vq;
                    vec[q][k] = sn * vp + cs * vq;
                }
            }
        }
    }
    // Eigenpairs ascending; conic coefficient a = L⁻ᵀ·w.
    let mut pairs: Vec<(f64, usize)> = (0..6).map(|i| (b[i][i], i)).collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Option<[f64; 6]> = None;
    for &(_, idx) in &pairs {
        // w is a row of `vec` (Jacobi accumulates rotations on rows).
        let w = vec[idx];
        // a = L⁻ᵀ·w: solve Lᵀ·a = w by back substitution.
        let mut a = [0.0f64; 6];
        for i in (0..6).rev() {
            let mut sum = w[i];
            for k in i + 1..6 {
                sum -= l[k][i] * a[k];
            }
            a[i] = sum / l[i][i];
        }
        if 4.0 * a[0] * a[2] - a[1] * a[1] > 0.0 {
            chosen = Some(a);
            break;
        }
    }
    let c = chosen.ok_or(ApplyError::InvalidPoints("no elliptic conic"))?;
    conic_to_rotated(c[0], c[1], c[2], c[3], c[4], c[5])
        .ok_or(ApplyError::InvalidPoints("degenerate conic"))
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
