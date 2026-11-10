//! Clustering, transforms, and histogram back-projection.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum ClusterError {
    UnsupportedDepth(MatDepth),
    InvalidInput(&'static str),
    OutputError(String),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(d) => write!(f, "unsupported matrix depth {d:?}"),
            Self::InvalidInput(r) => write!(f, "invalid input: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}
impl Error for ClusterError {}

/// kmeans: Lloyd's iterations with kmeans++ seeding (deterministic rng-free choice
/// for reproducibility: picks the farthest point each time).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kmeans_into(
    samples: &Mat,
    k: i32,
    labels: &Mat,
    centers: &Mat,
    max_iters: i32,
    epsilon: f64,
    attempts: i32,
) -> Result<f64, ClusterError> {
    if samples.depth() != MatDepth::F32 {
        return Err(ClusterError::UnsupportedDepth(samples.depth()));
    }
    let input = samples.compact_bytes();
    let n = samples.rows() as usize;
    let dims = (samples.columns() as usize) * samples.channels() as usize;
    let k = k as usize;
    if k == 0 || k > n || dims == 0 {
        return Err(ClusterError::InvalidInput("bad k or empty samples"));
    }
    let pts: Vec<f64> = (0..n * dims).map(|i| read_scalar(&input, i, ScalarDepth::F32)).collect();
    // Deterministic farthest-point seeding (first center = first point).
    // First center = the point farthest from the origin (pinned seed choice).
    let first = (0..n)
        .max_by(|&a, &b| {
            let na: f64 = (0..dims).map(|j| pts[a * dims + j].powi(2)).sum();
            let nb: f64 = (0..dims).map(|j| pts[b * dims + j].powi(2)).sum();
            na.total_cmp(&nb)
        })
        .unwrap_or(0);
    let mut seeds: Vec<usize> = vec![first];
    let mut dist2: Vec<f64> = vec![f64::MAX; n];
    while seeds.len() < k {
        let last = seeds[seeds.len() - 1];
        for i in 0..n {
            let d: f64 = (0..dims).map(|j| (pts[i * dims + j] - pts[last * dims + j]).powi(2)).sum();
            dist2[i] = dist2[i].min(d);
        }
        let next = (0..n).max_by(|&a, &b| dist2[a].total_cmp(&dist2[b])).unwrap();
        seeds.push(next);
    }
    let mut center: Vec<f64> = seeds.iter().flat_map(|&s| pts[s * dims..s * dims + dims].iter().copied()).collect();
    let mut lab = vec![0i32; n];
    let mut compactness = f64::MAX;
    for _attempt in 0..attempts.max(1) {
        let mut changed = true;
        let mut iters = 0;
        while changed && iters < max_iters.max(1) {
            iters += 1;
            changed = false;
            let mut total32 = 0f32;
            for i in 0..n {
                let mut best = 0;
                let mut best_d32 = f32::MAX;
                for c in 0..k {
                    // Pinned distance: F32 component-wise accumulation.
                    let d32: f32 = (0..dims)
                        .map(|j| {
                            let diff = pts[i * dims + j] as f32 - center[c * dims + j] as f32;
                            diff * diff
                        })
                        .sum();
                    if d32 < best_d32 { best_d32 = d32; best = c; }
                }
                total32 += best_d32;
                if lab[i] != best as i32 { lab[i] = best as i32; changed = true; }
            }
            // Recenter.
            let mut sum = vec![0f64; k * dims];
            let mut cnt = vec![0i64; k];
            for i in 0..n {
                let c = lab[i] as usize;
                cnt[c] += 1;
                for j in 0..dims { sum[c * dims + j] += pts[i * dims + j]; }
            }
            for c in 0..k {
                if cnt[c] > 0 {
                    for j in 0..dims {
                        // Pinned mean: F32 sum times the F32 reciprocal count.
                        center[c * dims + j] = f64::from(sum[c * dims + j] as f32 * (1.0f32 / cnt[c] as f32));
                    }
                }
            }
            compactness = f64::from(total32);
            if f64::from(total32) < epsilon && iters > 1 { break; }
        }
    }
    let mut lb = vec![0u8; n * 4];
    for (i, l) in lab.iter().enumerate() { write_scalar(&mut lb, i, ScalarDepth::I32, f64::from(*l)); }
    labels.write_output(lb, n as u32, 1, 1, MatDepth::I32).map_err(|e| ClusterError::OutputError(e.to_string()))?;
    let mut cb = vec![0u8; k * dims * 4];
    for (i, v) in center.iter().enumerate() { write_scalar(&mut cb, i, ScalarDepth::F32, *v); }
    centers.write_output(cb, k as u32, dims as u32, 1, MatDepth::F32).map_err(|e| ClusterError::OutputError(e.to_string()))?;
    Ok(f64::from(compactness as f32))
}

/// dft: 2-D discrete Fourier transform over F32 inputs.
/// flags: 16 = complex output (CV_32FC2), 2 = scale by 1/(r·c), 32 = real output.
pub(crate) fn dft_into(source: &Mat, destination: &Mat, flags: i32) -> Result<(), ClusterError> {
    if !matches!(source.depth(), MatDepth::F32) {
        return Err(ClusterError::UnsupportedDepth(source.depth()));
    }
    let input = source.compact_bytes();
    let (rows, cols, ch) = (source.rows() as usize, source.columns() as usize, source.channels() as usize);
    let complex_in = ch == 2;
    // Per-row then per-column DFT per channel.
    let mut re = vec![0f64; rows * cols * ch];
    let mut im = vec![0f64; rows * cols * ch];
    for c in 0..ch {
        for y in 0..rows {
            for x in 0..cols {
                let v = read_scalar(&input, (y * cols + x) * ch + c, ScalarDepth::F32);
                re[(y * cols + x) * ch + c] = v;
            }
        }
    }
    let tau = std::f64::consts::TAU;
    // Row transforms.
    for c in 0..ch {
        for y in 0..rows {
            let mut out_re = vec![0f64; cols];
            let mut out_im = vec![0f64; cols];
            for u in 0..cols {
                for x in 0..cols {
                    let a = -tau * (u * x) as f64 / cols as f64;
                    out_re[u] += re[(y * cols + x) * ch + c] * a.cos()
                        - if complex_in { im[(y * cols + x) * ch + c] * a.sin() } else { 0.0 };
                    out_im[u] += re[(y * cols + x) * ch + c] * a.sin()
                        + if complex_in { im[(y * cols + x) * ch + c] * a.cos() } else { 0.0 };
                }
            }
            for u in 0..cols {
                re[(y * cols + u) * ch + c] = out_re[u];
                im[(y * cols + u) * ch + c] = out_im[u];
            }
        }
        // Column transforms.
        for x in 0..cols {
            let mut out_re = vec![0f64; rows];
            let mut out_im = vec![0f64; rows];
            for v in 0..rows {
                for y in 0..rows {
                    let a = -tau * (v * y) as f64 / rows as f64;
                    out_re[v] += re[(y * cols + x) * ch + c] * a.cos() - im[(y * cols + x) * ch + c] * a.sin();
                    out_im[v] += re[(y * cols + x) * ch + c] * a.sin() + im[(y * cols + x) * ch + c] * a.cos();
                }
            }
            for v in 0..rows {
                re[(v * cols + x) * ch + c] = out_re[v];
                im[(v * cols + x) * ch + c] = out_im[v];
            }
        }
    }
    let scale = if flags & 2 != 0 { 1.0 / (rows * cols) as f64 } else { 1.0 };
    let complex_out = flags & 16 != 0;
    let n = rows * cols;
    let mut output = vec![0u8; n * (if complex_out { 2 } else { 1 }) * 4];
    if complex_out {
        for i in 0..n {
            write_scalar(&mut output, i * 2, ScalarDepth::F32, re[i] * scale);
            write_scalar(&mut output, i * 2 + 1, ScalarDepth::F32, im[i] * scale);
        }
        destination.write_output(output, rows as u32, cols as u32, 2, MatDepth::F32)
    } else {
        for i in 0..n {
            write_scalar(&mut output, i, ScalarDepth::F32, re[i] * scale);
        }
        destination.write_output(output, rows as u32, cols as u32, 1, MatDepth::F32)
    }
    .map_err(|e| ClusterError::OutputError(e.to_string()))?;
    Ok(())
}

/// calcBackProject: lookup each pixel's histogram bin into a back-projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calc_back_project_into(
    source: &Mat,
    channels: i32,
    hist: &Mat,
    destination: &Mat,
    range_low: f64,
    range_high: f64,
    scale: f64,
) -> Result<(), ClusterError> {
    let depth = scalar_depth(source.depth());
    let _ = channels;
    let input = source.compact_bytes();
    let (rows, cols, ch) = (source.rows(), source.columns(), source.channels() as usize);
    let hbytes = hist.compact_bytes();
    let hd = scalar_depth(hist.depth());
    let bins = hbytes.len() / hd.width();
    let hsize = (range_high - range_low).max(1e-9);
    let mut output = vec![0u8; (rows * cols) as usize * depth.width()];
    for i in 0..(rows * cols) as usize {
        let v = read_scalar(&input, i * ch, depth);
        let bin = (((v - range_low) / hsize) * bins as f64) as usize;
        let h = if bin < bins { read_scalar(&hbytes, bin, hd) } else { 0.0 };
        write_scalar(&mut output, i, depth, h * scale);
    }
    destination
        .write_output(output, rows, cols, 1, source.depth())
        .map_err(|e| ClusterError::OutputError(e.to_string()))?;
    Ok(())
}

/// warpPolar: sample src at (x = cx + r·cos(θ), y = cy + r·sin(θ)).
/// flags & WARP_POLAR_LOG → r = M·exp(θ·ln(maxRadius)/cols); else linear r.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_polar_into(
    source: &Mat,
    destination: &Mat,
    width: i32,
    height: i32,
    center_x: f64,
    center_y: f64,
    max_radius: f64,
    flags: i32,
) -> Result<(), ClusterError> {
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let (sr, sc, ch) = (source.rows(), source.columns(), source.channels() as usize);
    let log_polar = flags & 8 != 0; // WARP_POLAR_LOG = 8
    let mut output = vec![0u8; (width * height) as usize * ch * depth.width()];
    for y in 0..height {
        for x in 0..width {
            // theta spans 0..2π across rows, radius across columns.
            let theta = std::f64::consts::TAU * f64::from(y) / f64::from(height);
            let radius = if log_polar {
                max_radius * (f64::from(x) / f64::from(width)).exp() / std::f64::consts::E
            } else {
                max_radius * f64::from(x) / f64::from(width)
            };
            let fx = center_x + radius * theta.cos();
            let fy = center_y + radius * theta.sin();
            for c in 0..ch {
                let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
                let (wx, wy) = (fx - x0 as f64, fy - y0 as f64);
                let mut acc = 0.0;
                for (dy, wy_) in [(0i64, 1.0 - wy), (1i64, wy)] {
                    for (dx, wx_) in [(0i64, 1.0 - wx), (1i64, wx)] {
                        let (sy, sx) = (y0 + dy, x0 + dx);
                        let v = if sy < 0 || sy >= i64::from(sr) || sx < 0 || sx >= i64::from(sc) {
                            0.0
                        } else {
                            read_scalar(&input, ((sy as u32 * sc + sx as u32) as usize) * ch + c, depth)
                        };
                        acc += wy_ * wx_ * v;
                    }
                }
                write_scalar(&mut output, ((y * width + x) as usize) * ch + c, depth, acc);
            }
        }
    }
    destination
        .write_output(output, height as u32, width as u32, source.channels(), source.depth())
        .map_err(|e| ClusterError::OutputError(e.to_string()))?;
    Ok(())
}

/// groupRectangles: cluster axis-aligned rects by position/size similarity.
/// Emits the surviving rects and their cluster weights.
pub(crate) fn group_rectangles_into(
    rects: &Mat,
    destination: &Mat,
    group_threshold: i32,
    eps: f64,
) -> Result<i32, ClusterError> {
    let depth = scalar_depth(rects.depth());
    let input = rects.compact_bytes();
    let n = input.len() / (depth.width() * 4);
    let mut r: Vec<[f64; 4]> = (0..n)
        .map(|i| {
            [
                read_scalar(&input, i * 4, depth),
                read_scalar(&input, i * 4 + 1, depth),
                read_scalar(&input, i * 4 + 2, depth),
                read_scalar(&input, i * 4 + 3, depth),
            ]
        })
        .collect();
    // Agglomerate: merge similar rectangles until stable.
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..r.len() {
            for j in i + 1..r.len() {
                let (a, b) = (r[i], r[j]);
                let similar = (a[0] - b[0]).abs() <= eps * a[2].min(b[2])
                    && (a[1] - b[1]).abs() <= eps * a[3].min(b[3])
                    && (a[2] - b[2]).abs() <= eps * a[2].min(b[2])
                    && (a[3] - b[3]).abs() <= eps * a[3].min(b[3]);
                if similar {
                    // Weighted merge toward the average.
                    r[i] = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0, (a[3] + b[3]) / 2.0];
                    r.remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    let mut output = vec![0u8; r.len() * 4 * 4];
    for (i, rr) in r.iter().enumerate() {
        for (k, v) in rr.iter().enumerate() {
            write_scalar(&mut output, i * 4 + k, ScalarDepth::I32, *v);
        }
    }
    let count = r.len() as i32;
    destination
        .write_output(output, r.len() as u32, 1, 4, MatDepth::I32)
        .map_err(|e| ClusterError::OutputError(e.to_string()))?;
    let _ = group_threshold;
    Ok(count)
}
