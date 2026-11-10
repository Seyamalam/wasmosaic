//! Dense linear algebra and polar conversions.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum LinalgError {
    UnsupportedDepth(MatDepth),
    ShapeMismatch(&'static str),
    InvalidInput(&'static str),
    OutputError(String),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(d) => write!(f, "unsupported matrix depth {d:?}"),
            Self::ShapeMismatch(r) => write!(f, "shape mismatch: {r}"),
            Self::InvalidInput(r) => write!(f, "invalid input: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}

impl Error for LinalgError {}

fn read_matrix_f64(source: &Mat) -> Result<Vec<f64>, LinalgError> {
    let depth = scalar_depth(source.depth());
    if !matches!(depth, ScalarDepth::F32 | ScalarDepth::F64) || source.channels() != 1 {
        return Err(LinalgError::UnsupportedDepth(source.depth()));
    }
    let input = source.compact_bytes();
    Ok((0..input.len() / depth.width())
        .map(|i| read_scalar(&input, i, depth))
        .collect())
}

/// gemm: `dst = alpha * op(src1) * op(src2) + beta * op(src3)`.
/// flags: 1 = transpose src1, 2 = transpose src2, 4 = transpose src3.
pub(crate) fn gemm_into(
    src1: &Mat,
    src2: &Mat,
    alpha: f64,
    src3: &Mat,
    beta: f64,
    destination: &Mat,
    flags: i32,
) -> Result<(), LinalgError> {
    let a = read_matrix_f64(src1)?;
    let b = read_matrix_f64(src2)?;
    let c = if beta != 0.0 || !src3.compact_bytes().is_empty() {
        Some(read_matrix_f64(src3)?)
    } else {
        None
    };
    let (r1, c1) = (src1.rows() as usize, src1.columns() as usize);
    let (r2, c2) = (src2.rows() as usize, src2.columns() as usize);
    let (am, an) = if flags & 1 != 0 { (c1, r1) } else { (r1, c1) };
    let (bm, bn) = if flags & 2 != 0 { (c2, r2) } else { (r2, c2) };
    if an != bm {
        return Err(LinalgError::ShapeMismatch("gemm inner dimensions differ"));
    }
    let at = |i: usize, j: usize| {
        if flags & 1 != 0 {
            a[j * c1 + i]
        } else {
            a[i * c1 + j]
        }
    };
    let bt = |i: usize, j: usize| {
        if flags & 2 != 0 {
            b[j * c2 + i]
        } else {
            b[i * c2 + j]
        }
    };
    let mut out = vec![0f64; am * bn];
    for i in 0..am {
        for j in 0..bn {
            let mut sum = 0f64;
            for k in 0..an {
                sum += at(i, k) * bt(k, j);
            }
            out[i * bn + j] = alpha * sum;
        }
    }
    if let Some(c_vals) = c {
        let (r3, c3) = (src3.rows() as usize, src3.columns() as usize);
        let (cm, cn) = if flags & 4 != 0 { (c3, r3) } else { (r3, c3) };
        if cm != am || cn != bn {
            return Err(LinalgError::ShapeMismatch("gemm C dimensions differ"));
        }
        for i in 0..am {
            for j in 0..bn {
                let cv = if flags & 4 != 0 {
                    c_vals[j * c3 + i]
                } else {
                    c_vals[i * c3 + j]
                };
                out[i * bn + j] += beta * cv;
            }
        }
    }
    let mut output = vec![0u8; out.len() * 8];
    for (i, v) in out.iter().enumerate() {
        write_scalar(&mut output, i, ScalarDepth::F64, *v);
    }
    destination
        .write_output(output, am as u32, bn as u32, 1, MatDepth::F64)
        .map_err(|e| LinalgError::OutputError(e.to_string()))?;
    Ok(())
}

/// eigen: symmetric-matrix eigensolver via cyclic Jacobi rotations.
/// Emits eigenvalues sorted descending and row-major eigenvectors.
pub(crate) fn eigen_into(
    source: &Mat,
    eigenvalues: &Mat,
    eigenvectors: &Mat,
) -> Result<bool, LinalgError> {
    let a = read_matrix_f64(source)?;
    let n = source.rows() as usize;
    if source.rows() != source.columns() {
        return Err(LinalgError::InvalidInput("eigen requires a square matrix"));
    }
    let mut m = a;
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off < 1e-15 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if m[p * n + q].abs() < 1e-12 {
                    continue;
                }
                let theta = 0.5 * (m[q * n + q] - m[p * n + p]) / m[p * n + q];
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mp, mq) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mp - s * mq;
                    m[q * n + k] = s * mp + c * mq;
                }
                for k in 0..n {
                    let (mp, mq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mp - s * mq;
                    m[k * n + q] = s * mp + c * mq;
                }
                for k in 0..n {
                    let (vp, vq) = (v[p * n + k], v[q * n + k]);
                    v[p * n + k] = c * vp - s * vq;
                    v[q * n + k] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[j * n + j].total_cmp(&m[i * n + i]));
    let mut vals = vec![0u8; n * 8];
    let mut vecs = vec![0u8; n * n * 8];
    for (out_i, &src_i) in order.iter().enumerate() {
        write_scalar(&mut vals, out_i, ScalarDepth::F64, m[src_i * n + src_i]);
        for k in 0..n {
            write_scalar(&mut vecs, out_i * n + k, ScalarDepth::F64, v[src_i * n + k]);
        }
    }
    eigenvalues
        .write_output(vals, n as u32, 1, 1, MatDepth::F64)
        .map_err(|e| LinalgError::OutputError(e.to_string()))?;
    eigenvectors
        .write_output(vecs, n as u32, n as u32, 1, MatDepth::F64)
        .map_err(|e| LinalgError::OutputError(e.to_string()))?;
    Ok(true)
}

/// solvePoly: roots of `c[0] + c[1]x + ... + c[n]x^n` via the companion
/// matrix and the unshifted QR iteration. Emits CV_64FC2 roots.
pub(crate) fn solve_poly_into(
    coeffs: &Mat,
    roots: &Mat,
    _max_iters: i32,
) -> Result<i32, LinalgError> {
    let c = read_matrix_f64(coeffs)?;
    // Strip leading zeros; coefficients are in ascending-degree order.
    let mut coef = c;
    while coef.len() > 1 && coef[coef.len() - 1] == 0.0 {
        coef.pop();
    }
    let n = coef.len() - 1;
    if n < 1 {
        return Err(LinalgError::InvalidInput("expected at least a linear polynomial"));
    }
    // Companion matrix (Frobenius form), roots = its eigenvalues.
    let an = coef[n];
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        m[i * n] = -coef[n - 1 - i] / an;
        if i + 1 < n {
            m[i * n + i + 1] = 1.0;
        }
    }
    // Unshifted QR iteration with Wilkinson-ish shift for convergence.
    let mut work = m;
    for _iter in 0..4000 {
        let mut residual = 0f64;
        for i in 0..n {
            for j in 0..i.saturating_sub(1) {
                residual += work[i * n + j].abs();
            }
        }
        if residual < 1e-12 {
            break;
        }
        // Wilkinson double shift via the trailing 2x2 eigenvalue.
        let shift = if n >= 2 {
            let (a, b, c, d) = (
                work[(n - 2) * n + n - 2],
                work[(n - 2) * n + n - 1],
                work[(n - 1) * n + n - 2],
                work[(n - 1) * n + n - 1],
            );
            let tr = a + d;
            let det = a * d - b * c;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let l1 = (tr + sq) / 2.0;
                let l2 = (tr - sq) / 2.0;
                if (l1 - d).abs() < (l2 - d).abs() { l1 } else { l2 }
            } else {
                tr / 2.0
            }
        } else {
            work[0]
        };
        for i in 0..n {
            work[i * n + i] -= shift;
        }
        // QR decomposition via Gram–Schmidt.
        let mut q = vec![0f64; n * n];
        let mut r = vec![0f64; n * n];
        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|i| work[i * n + j]).collect();
            for k in 0..j {
                let mut dot = 0f64;
                for i in 0..n {
                    dot += v[i] * q[i * n + k];
                }
                r[k * n + j] = dot;
                for i in 0..n {
                    v[i] -= dot * q[i * n + k];
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r[j * n + j] = norm;
            if norm > 1e-14 {
                for i in 0..n {
                    q[i * n + j] = v[i] / norm;
                }
            }
        }
        // work = R * Q + shift·I
        let mut next = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0f64;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
            next[i * n + i] += shift;
        }
        work = next;
    }
    // Extract eigenvalues from the quasi-triangular matrix.
    let mut roots_out: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        if i + 1 < n && work[(i + 1) * n + i].abs() > 1e-9 {
            // 2x2 block: complex-conjugate pair.
            let (a, b, cc, d) = (
                work[i * n + i],
                work[i * n + i + 1],
                work[(i + 1) * n + i],
                work[(i + 1) * n + i + 1],
            );
            let tr = a + d;
            let det = a * d - b * cc;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                let im = (-disc).sqrt() / 2.0;
                roots_out.push((tr / 2.0, im));
                roots_out.push((tr / 2.0, -im));
            } else {
                let sq = disc.sqrt();
                roots_out.push(((tr + sq) / 2.0, 0.0));
                roots_out.push(((tr - sq) / 2.0, 0.0));
            }
            i += 2;
        } else {
            roots_out.push((work[i * n + i], 0.0));
            i += 1;
        }
    }
    roots_out.truncate(n);
    let mut output = vec![0u8; n * 16];
    for (i, &(re, im)) in roots_out.iter().enumerate() {
        write_scalar(&mut output, i * 2, ScalarDepth::F64, re);
        write_scalar(&mut output, i * 2 + 1, ScalarDepth::F64, im);
    }
    roots
        .write_output(output, n as u32, 1, 2, MatDepth::F64)
        .map_err(|e| LinalgError::OutputError(e.to_string()))?;
    Ok(0)
}
