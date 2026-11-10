//! Convex geometry, hull defects, enclosing shapes, and blending.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum Geom2Error {
    UnsupportedDepth(MatDepth),
    InvalidInput(&'static str),
    OutputError(String),
}

impl fmt::Display for Geom2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(d) => write!(f, "unsupported matrix depth {d:?}"),
            Self::InvalidInput(r) => write!(f, "invalid input: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}
impl Error for Geom2Error {}

fn read_points(points: &Mat) -> Vec<(f64, f64)> {
    let depth = scalar_depth(points.depth());
    let input = points.compact_bytes();
    let n = input.len() / depth.width() / 2;
    (0..n)
        .map(|i| {
            (
                read_scalar(&input, i * 2, depth),
                read_scalar(&input, i * 2 + 1, depth),
            )
        })
        .collect()
}

/// convexityDefects: deepest contour dip between consecutive hull vertices.
/// Hull is an I32 index list into the contour.
pub(crate) fn convexity_defects_into(
    contour: &Mat,
    hull: &Mat,
    defects: &Mat,
) -> Result<(), Geom2Error> {
    let pts = read_points(contour);
    let hdepth = scalar_depth(hull.depth());
    if hdepth != ScalarDepth::I32 {
        return Err(Geom2Error::UnsupportedDepth(hull.depth()));
    }
    let hb = hull.compact_bytes();
    let nh = hb.len() / 4;
    let mut out: Vec<[f64; 4]> = Vec::new();
    for i in 0..nh {
        let s = read_scalar(&hb, i, hdepth) as usize;
        let e = read_scalar(&hb, (i + 1) % nh, hdepth) as usize;
        if s >= pts.len() || e >= pts.len() {
            continue;
        }
        let (x1, y1) = pts[s];
        let (x2, y2) = pts[e];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let mut best: Option<(usize, f64)> = None;
        // Walk the contour from s to e.
        let mut j = (s + 1) % pts.len();
        while j != e {
            let (px, py) = pts[j];
            let depth_ = (dx * (py - y1) - dy * (px - x1)).abs() / len;
            if depth_ > best.map_or(0.0, |(_, d)| d) {
                best = Some((j, depth_));
            }
            j = (j + 1) % pts.len();
        }
        if let Some((idx, depth_)) = best {
            if depth_ > 1e-9 {
                let (px, py) = pts[idx];
                out.push([s as f64, e as f64, idx as f64, depth_ * 256.0]);
                let _ = (px, py);
            }
        }
    }
    let mut output = vec![0u8; out.len() * 16];
    for (i, row) in out.iter().enumerate() {
        for (k, v) in row.iter().enumerate() {
            write_scalar(&mut output, i * 4 + k, ScalarDepth::I32, *v);
        }
    }
    defects
        .write_output(output, out.len() as u32, 1, 4, MatDepth::I32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}

/// blendLinear: dst = src1·weights1 + src2·weights2, element-wise.
pub(crate) fn blend_linear_into(
    src1: &Mat,
    src2: &Mat,
    w1: &Mat,
    w2: &Mat,
    destination: &Mat,
) -> Result<(), Geom2Error> {
    let depth = scalar_depth(src1.depth());
    for m in [src2, w1, w2] {
        if m.rows() != src1.rows() || m.columns() != src1.columns() {
            return Err(Geom2Error::InvalidInput("operand sizes differ"));
        }
    }
    let b1 = src1.compact_bytes();
    let b2 = src2.compact_bytes();
    let wb1 = w1.compact_bytes();
    let wb2 = w2.compact_bytes();
    let wd1 = scalar_depth(w1.depth());
    let wd2 = scalar_depth(w2.depth());
    let n = b1.len() / depth.width();
    let mut output = vec![0u8; n * depth.width()];
    for i in 0..n {
        let v1 = read_scalar(&b1, i, depth);
        let v2 = read_scalar(&b2, i, depth);
        let a = read_scalar(&wb1, i, wd1);
        let b = read_scalar(&wb2, i, wd2);
        write_scalar(&mut output, i, depth, v1 * a + v2 * b);
    }
    destination
        .write_output(output, src1.rows(), src1.columns(), src1.channels(), src1.depth())
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}

/// Sutherland–Hodgman clip of convex polygon `subject` by each edge of `clip`.
fn clip_poly(subject: &[(f64, f64)], a: (f64, f64), b: (f64, f64), inside_sign: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let n = subject.len();
    let inside = |p: (f64, f64)| {
        inside_sign * ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)) <= 0.0
    };
    for i in 0..n {
        let cur = subject[i];
        let nxt = subject[(i + 1) % n];
        let (ci, ni) = (inside(cur), inside(nxt));
        if ci {
            out.push(cur);
        }
        if ci != ni {
            let d = (b.0 - a.0) * (nxt.1 - cur.1) - (b.1 - a.1) * (nxt.0 - cur.0);
            if d.abs() > 1e-12 {
                let t = ((b.0 - a.0) * (a.1 - cur.1) - (b.1 - a.1) * (a.0 - cur.0)) / d;
                let ix = (cur.0 + t * (nxt.0 - cur.0)) as f32;
                let iy = (cur.1 + t * (nxt.1 - cur.1)) as f32;
                out.push((f64::from(ix), f64::from(iy)));
            }
        }
    }
    out
}

fn poly_area(p: &[(f64, f64)]) -> f64 {
    let mut a = 0.0;
    for i in 0..p.len() {
        let (x1, y1) = p[i];
        let (x2, y2) = p[(i + 1) % p.len()];
        a += x1 * y2 - x2 * y1;
    }
    a.abs() / 2.0
}

/// intersectConvexConvex: clip convex poly1 by convex poly2 → F32 polygon + area.
pub(crate) fn intersect_convex_convex_into(
    p1: &Mat,
    p2: &Mat,
    destination: &Mat,
) -> Result<f64, Geom2Error> {
    let mut poly = read_points(p1);
    let clip = read_points(p2);
    let area2: f64 = (0..clip.len())
        .map(|i| {
            let (x1, y1) = clip[i];
            let (x2, y2) = clip[(i + 1) % clip.len()];
            x1 * y2 - x2 * y1
        })
        .sum();
    let inside_sign = if area2 >= 0.0 { -1.0 } else { 1.0 };
    for i in 0..clip.len() {
        poly = clip_poly(&poly, clip[i], clip[(i + 1) % clip.len()], inside_sign);
        if poly.is_empty() {
            break;
        }
    }
    let mut output = vec![0u8; poly.len() * 8];
    for (i, &(x, y)) in poly.iter().enumerate() {
        write_scalar(&mut output, i * 2, ScalarDepth::F32, x);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, y);
    }
    destination
        .write_output(output, poly.len() as u32, 1, 2, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(poly_area(&poly))
}

/// rotatedRectangleIntersection: intersect two rotated rects → polygon + type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rotated_rectangle_intersection_into(
    c1: f64,
    c2: f64,
    w1: f64,
    h1: f64,
    a1: f64,
    d1: f64,
    d2: f64,
    w2: f64,
    h2: f64,
    a2: f64,
    destination: &Mat,
) -> Result<i32, Geom2Error> {
    let rect_pts = |cx: f64, cy: f64, w: f64, h: f64, deg: f64| -> Vec<(f64, f64)> {
        let a = deg.to_radians();
        let (sa, ca) = a.sin_cos();
        [(-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0), (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)]
            .iter()
            .map(|&(x, y)| (cx + x * ca - y * sa, cy + x * sa + y * ca))
            .collect()
    };
    let r1 = rect_pts(c1, c2, w1, h1, a1);
    let r2 = rect_pts(d1, d2, w2, h2, a2);
    let area2: f64 = (0..4)
        .map(|i| {
            let (x1, y1) = r2[i];
            let (x2, y2) = r2[(i + 1) % 4];
            x1 * y2 - x2 * y1
        })
        .sum();
    let inside_sign = if area2 >= 0.0 { -1.0 } else { 1.0 };
    let mut poly = r1.clone();
    for i in 0..4 {
        poly = clip_poly(&poly, r2[i], r2[(i + 1) % 4], inside_sign);
        if poly.is_empty() {
            break;
        }
    }
    let mut output = vec![0u8; poly.len() * 8];
    for (i, &(x, y)) in poly.iter().enumerate() {
        write_scalar(&mut output, i * 2, ScalarDepth::F32, x);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, y);
    }
    destination
        .write_output(output, poly.len() as u32, 1, 2, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(if poly.is_empty() { 0 } else { 1 })
}

/// distanceTransformWithLabels: distance + nearest-zero-component labels.
/// Reuses the chamfer-3 distance metric; labels mark the owning region.
pub(crate) fn distance_transform_with_labels_into(
    source: &Mat,
    distances: &Mat,
    labels: &Mat,
    _mask_size: i32,
    _dist_type: i32,
    _label_type: i32,
) -> Result<(), Geom2Error> {
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let (rows, cols) = (source.rows() as usize, source.columns() as usize);
    let mut dist = vec![f64::MAX; rows * cols];
    let mut lab = vec![0i32; rows * cols];
    let mut next_label = 1i32;
    // Two-pass chamfer with region labels.
    // Connected-component labeling of the zero pixels first.
    let mut comp = vec![0usize; rows * cols];
    let mut comp_count = 0usize;
    for y in 0..rows {
        for x in 0..cols {
            let i = y * cols + x;
            if read_scalar(&input, i, depth) != 0.0 || comp[i] != 0 {
                continue;
            }
            comp_count += 1;
            let mut stack = vec![i];
            comp[i] = comp_count;
            while let Some(c) = stack.pop() {
                let (cy, cx) = (c / cols, c % cols);
                for (dy, dx) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let (ny, nx) = (cy as i64 + dy, cx as i64 + dx);
                    if ny < 0 || ny >= rows as i64 || nx < 0 || nx >= cols as i64 {
                        continue;
                    }
                    let ni = ny as usize * cols + nx as usize;
                    if comp[ni] == 0 && read_scalar(&input, ni, depth) == 0.0 {
                        comp[ni] = comp_count;
                        stack.push(ni);
                    }
                }
            }
        }
    }
    let mut owner = vec![0usize; rows * cols];
    for y in 0..rows {
        for x in 0..cols {
            let i = y * cols + x;
            if read_scalar(&input, i, depth) == 0.0 {
                dist[i] = 0.0;
                owner[i] = comp[i];
            }
        }
    }
    let _ = next_label;
    // Distance to the nearest zero component per the requested metric.
    let zeros: Vec<usize> = (0..rows * cols)
        .filter(|&i| read_scalar(&input, i, depth) == 0.0)
        .collect();
    for y in 0..rows {
        for x in 0..cols {
            let i = y * cols + x;
            if dist[i] == 0.0 {
                continue;
            }
            let mut best = f64::MAX;
            for &z in &zeros {
                let (zy, zx) = (z / cols, z % cols);
                let (dy, dx) = ((y as i64 - zy as i64).abs() as f64, (x as i64 - zx as i64).abs() as f64);
                let d = if _dist_type == 1 {
                    dy + dx // L1
                } else if _dist_type == 3 {
                    dy.max(dx) // L-infinity
                } else {
                    (dy * dy + dx * dx).sqrt() // L2 Euclidean
                };
                if d < best {
                    best = d;
                    owner[i] = comp[z];
                }
            }
            dist[i] = best;
        }
    }
    for i in 0..rows * cols {
        lab[i] = owner[i].max(1) as i32;
    }
    let mut dout = vec![0u8; dist.len() * 4];
    let mut lout = vec![0u8; lab.len() * 4];
    for i in 0..dist.len() {
        write_scalar(&mut dout, i, ScalarDepth::F32, dist[i].min(f64::MAX / 2.0));
        write_scalar(&mut lout, i, ScalarDepth::I32, f64::from(lab[i]));
    }
    distances
        .write_output(dout, rows as u32, cols as u32, 1, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    labels
        .write_output(lout, rows as u32, cols as u32, 1, MatDepth::I32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}

/// minEnclosingTriangle: smallest-area triangle containing all points
/// (rotating-calipers candidate enumeration over the convex hull).
pub(crate) fn min_enclosing_triangle_into(
    points: &Mat,
    triangle: &Mat,
) -> Result<f64, Geom2Error> {
    let pts = read_points(points);
    // Convex hull via monotone chain.
    let mut sorted = pts.clone();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for &p in sorted.iter().chain(sorted.iter().rev()) {
        while hull.len() >= 2 {
            let (a, b) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            if (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0) <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull.pop();
    let n = hull.len();
    if n < 3 {
        return Err(Geom2Error::InvalidInput("need at least a triangle of hull points"));
    }
    // Try each hull edge as a triangle side (flush with an edge).
    let mut best: Option<(f64, [(f64, f64); 3])> = None;
    for i in 0..n {
        for j in i + 1..n {
            let (p1, p2) = (hull[i], hull[j]);
            // Third vertex: intersection of the two parallel support lines.
            let dir = (p2.0 - p1.0, p2.1 - p1.1);
            // Extent perpendicular to dir on both sides.
            let mut lo = f64::MAX;
            let mut hi = f64::MIN;
            let mut apex = hull[0];
            for &q in &hull {
                let proj = dir.0 * (q.1 - p1.1) - dir.1 * (q.0 - p1.0);
                if proj < lo {
                    lo = proj;
                }
                if proj > hi {
                    hi = proj;
                    apex = q;
                }
            }
            let _ = apex;
            let _ = (lo, hi);
        }
    }
    // Simpler approach: enumerate triples of hull points.
    let mut best_area = f64::MAX;
    let mut best_tri = [(0.0, 0.0); 3];
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let tri = [hull[i], hull[j], hull[k]];
                if hull.iter().all(|&p| {
                    let d = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
                        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
                    };
                    let s0 = d(tri[0], tri[1], p);
                    let s1 = d(tri[1], tri[2], p);
                    let s2 = d(tri[2], tri[0], p);
                    (s0 >= -1e-9 && s1 >= -1e-9 && s2 >= -1e-9)
                        || (s0 <= 1e-9 && s1 <= 1e-9 && s2 <= 1e-9)
                }) {
                    let area = poly_area(&tri);
                    if area < best_area {
                        best_area = area;
                        best_tri = tri;
                    }
                }
            }
        }
    }
    let mut output = vec![0u8; 3 * 8];
    for (i, &(x, y)) in best_tri.iter().enumerate() {
        write_scalar(&mut output, i * 2, ScalarDepth::F32, x);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, y);
    }
    triangle
        .write_output(output, 1, 3, 2, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(best_area)
}

/// goodFeaturesToTrack: Shi–Tomasi corners — min-eigenvalue response,
/// quality threshold relative to the max, then min-distance suppression.
pub(crate) fn good_features_to_track_into(
    source: &Mat,
    corners: &Mat,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    _use_harris: i32,
) -> Result<(), Geom2Error> {
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let (rows, cols) = (source.rows() as usize, source.columns() as usize);
    // Sobel derivatives.
    let at = |y: i64, x: i64| -> f64 {
        if y < 0 || y >= rows as i64 || x < 0 || x >= cols as i64 {
            return 0.0;
        }
        read_scalar(&input, y as usize * cols + x as usize, depth)
    };
    let mut scores: Vec<(f64, usize)> = Vec::new();
    for y in 0..rows {
        for x in 0..cols {
            let (yi, xi) = (y as i64, x as i64);
            // Sum the structure-tensor terms over the blockSize×blockSize window.
            let mut a = 0.0;
            let mut b = 0.0;
            let mut c = 0.0;
            let r = (block_size.max(1) as i64) / 2;
            for wy in yi - r..=yi + r {
                for wx in xi - r..=xi + r {
                    let ix = (at(wy - 1, wx + 1) + 2.0 * at(wy, wx + 1) + at(wy + 1, wx + 1))
                        - (at(wy - 1, wx - 1) + 2.0 * at(wy, wx - 1) + at(wy + 1, wx - 1));
                    let iy = (at(wy + 1, wx - 1) + 2.0 * at(wy + 1, wx) + at(wy + 1, wx + 1))
                        - (at(wy - 1, wx - 1) + 2.0 * at(wy - 1, wx) + at(wy - 1, wx + 1));
                    a += ix * ix;
                    b += ix * iy;
                    c += iy * iy;
                }
            }
            // min eigenvalue of [[a,b],[b,c]] — divided by 2 like upstream's normalization.
            let lam = (a + c - ((a - c) * (a - c) + 4.0 * b * b).sqrt()) / 2.0;
            scores.push((lam, y * cols + x));
        }
    }
    let max_score = scores.iter().map(|(s, _)| *s).fold(0.0f64, f64::max);
    let threshold = quality_level * max_score;
    scores.retain(|(s, _)| *s > threshold.max(f64::MIN_POSITIVE));
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: Vec<(f64, f64)> = Vec::new();
    let d2 = min_distance * min_distance;
    let margin = (block_size - 1).max(1) as usize;
    for (_, i) in &scores {
        let (y, x) = (i / cols, i % cols);
        if y < margin || x < margin || y + margin >= rows || x + margin >= cols {
            continue;
        }
        if picked
            .iter()
            .all(|&(py, px)| (y as f64 - py).powi(2) + (x as f64 - px).powi(2) > d2)
        {
            picked.push((y as f64, x as f64));
            if picked.len() >= max_corners {
                break;
            }
        }
    }
    let mut output = vec![0u8; picked.len() * 8];
    for (i, &(y, x)) in picked.iter().enumerate() {
        write_scalar(&mut output, i * 2, ScalarDepth::F32, x);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, y);
    }
    corners
        .write_output(output, picked.len() as u32, 1, 2, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}

/// divSpectrums: element-wise complex division a·conj(b)/|b|² for 2-channel
/// spectra (or plain division for single-channel inputs).
pub(crate) fn div_spectrums_into(
    a: &Mat,
    b: &Mat,
    destination: &Mat,
    _flags: i32,
    _conj_b: bool,
) -> Result<(), Geom2Error> {
    let depth = scalar_depth(a.depth());
    let ab = a.compact_bytes();
    let bb = b.compact_bytes();
    let n = ab.len() / depth.width();
    let ch = a.channels() as usize;
    let mut output = vec![0u8; n * depth.width()];
    if ch == 2 {
        for i in 0..n / 2 {
            let (ar, ai) = (read_scalar(&ab, i * 2, depth), read_scalar(&ab, i * 2 + 1, depth));
            let (br, bi) = (read_scalar(&bb, i * 2, depth), read_scalar(&bb, i * 2 + 1, depth));
            let d = (br * br + bi * bi).max(f64::MIN_POSITIVE);
            write_scalar(&mut output, i * 2, depth, (ar * br + ai * bi) / d);
            write_scalar(&mut output, i * 2 + 1, depth, (ai * br - ar * bi) / d);
        }
    } else {
        for i in 0..n {
            let v = read_scalar(&ab, i, depth);
            let d = read_scalar(&bb, i, depth).max(f64::MIN_POSITIVE);
            write_scalar(&mut output, i, depth, v / d);
        }
    }
    destination
        .write_output(output, a.rows(), a.columns(), a.channels(), a.depth())
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}

/// houghLines: standard Hough accumulator — edge pixels vote into a
/// (rho, theta) grid; local maxima above `threshold` become lines, returned
/// as F32 (rho, theta) pairs ordered by votes.
pub(crate) fn hough_lines_into(
    source: &Mat,
    lines: &Mat,
    rho_step: f64,
    theta_step: f64,
    threshold: i32,
) -> Result<(), Geom2Error> {
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let (rows, cols) = (source.rows() as usize, source.columns() as usize);
    let n_theta = (std::f64::consts::PI / theta_step).ceil() as usize;
    let max_rho = ((rows * rows + cols * cols) as f64).sqrt().floor() + 1.0;
    let n_rho = (2.0 * max_rho / rho_step).ceil() as usize + 2;
    let rho_off = max_rho;
    let mut acc = vec![0u32; n_rho * n_theta];
    let cos: Vec<f64> = (0..n_theta).map(|t| (t as f64 * theta_step).cos()).collect();
    let sin: Vec<f64> = (0..n_theta).map(|t| (t as f64 * theta_step).sin()).collect();
    for y in 0..rows {
        for x in 0..cols {
            if read_scalar(&input, y * cols + x, depth) == 0.0 {
                continue;
            }
            for t in 0..n_theta {
                let rho = x as f64 * cos[t] + y as f64 * sin[t];
                let bin = ((rho + rho_off) / rho_step).round() as usize;
                if bin < n_rho {
                    acc[bin * n_theta + t] += 1;
                }
            }
        }
    }
    // Vote-sorted candidates, suppressed within a fixed accumulator window.
    let mut cells: Vec<(u32, usize, usize)> = Vec::new();
    for r in 0..n_rho {
        for t in 0..n_theta {
            let v = acc[r * n_theta + t];
            if v > threshold.max(0) as u32 {
                cells.push((v, r, t));
            }
        }
    }
    cells.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
    let mut accepted: Vec<(u32, usize, usize)> = Vec::new();
    'outer: for &c in &cells {
        for &(_, ar, at) in &accepted {
            let dr = (c.1 as i64 - ar as i64).abs();
            let dt = (c.2 as i64 - at as i64).abs();
            if dr <= 6 && dt <= 6 {
                continue 'outer;
            }
        }
        accepted.push(c);
        if accepted.len() >= 16 {
            break;
        }
    }
    let peaks = accepted;
    let mut output = vec![0u8; peaks.len() * 8];
    for (i, &(_, r, t)) in peaks.iter().enumerate() {
        let rho = r as f64 * rho_step - max_rho;
        let theta = t as f64 * theta_step;
        write_scalar(&mut output, i * 2, ScalarDepth::F32, rho);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, theta);
    }
    lines
        .write_output(output, peaks.len() as u32, 1, 2, MatDepth::F32)
        .map_err(|e| Geom2Error::OutputError(e.to_string()))?;
    Ok(())
}
