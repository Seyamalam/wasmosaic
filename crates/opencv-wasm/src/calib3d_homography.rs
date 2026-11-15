//! `findHomography`: robust homography estimation over point correspondences.
//!
//! Pinned contract (OpenCV.js 4.13.0 differential probing):
//! - Point inputs may be `N x 1` two-channel or `N x 2` single-channel
//!   matrices at any scalar depth; the two point sets must match. Fewer than
//!   four correspondences raise an error rather than returning an empty
//!   matrix.
//! - `method` selects the direct fit (0), LMeDS (4), RANSAC (8), or RHO (16);
//!   any other value raises an error.
//! - For the robust methods the reported inlier mask is the winning
//!   hypothesis's consensus set (Euclidean reprojection error strictly below
//!   the threshold); it is not recomputed under the refined model. Method 0
//!   instead reports the consensus under the refined all-point fit. The
//!   returned 3x3 CV_64F matrix is the geometric reprojection refit over the
//!   winning set for method 0 and RANSAC, and over the median-scale bound for
//!   LMeDS.
//! - Sampled subsets containing a collinear source triple are skipped.
//! - RANSAC subsets are drawn from a private per-call multiply-with-carry
//!   generator seeded at `u64::MAX`, while LMeDS scans subsets in index
//!   order.

use std::{error::Error, fmt};

use crate::calib3d_affine::{ransac_iteration_budget, rng_uniform};

/// Failures reported while estimating a homography.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum HomographyError {
    /// A method value other than 0, LMeDS (4), RANSAC (8), or RHO (16) was
    /// requested.
    UnsupportedMethod(i32),
    /// Fewer than four correspondences were supplied.
    TooFewPoints,
}

impl fmt::Display for HomographyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMethod(method) => write!(
                formatter,
                "findHomography method {method} is not implemented; use 0, LMeDS (4), RANSAC (8), or RHO (16)"
            ),
            Self::TooFewPoints => {
                write!(formatter, "findHomography requires at least 4 correspondences")
            }
        }
    }
}

impl Error for HomographyError {}

/// Outcome of a robust homography estimate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HomographyEstimate {
    /// The refined 3x3 model, or `None` when no valid hypothesis exists.
    pub(crate) model: Option<[f64; 9]>,
    /// The winning hypothesis's consensus set, one byte per correspondence.
    pub(crate) inliers: Vec<u8>,
}

/// Signed double-area of the triangle formed by three source points.
fn double_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Rejects four-point subsets containing a degenerate source triple.
fn subset_is_valid(from: &[[f64; 2]], subset: &[usize; 4]) -> bool {
    for a in 0..4 {
        for b in a + 1..4 {
            for c in b + 1..4 {
                if double_area(from[subset[a]], from[subset[b]], from[subset[c]]).abs()
                    <= f64::EPSILON
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Solves the normalized DLT for `from -> to` over the given indices.
///
/// Applies Hartley normalization to both point sets, solves the homogeneous
/// system via the smallest eigenvector of the 9x9 normal matrix, and maps the
/// result back to the input frame.
fn fit_least_squares(from: &[[f64; 2]], to: &[[f64; 2]], indices: &[usize]) -> Option<[f64; 9]> {
    if indices.len() < 4 {
        return None;
    }
    let (src_mean, src_scale) = hartley_transform(from, indices);
    let (dst_mean, dst_scale) = hartley_transform(to, indices);
    let mut normal = [[0f64; 9]; 9];
    for &index in indices {
        let x = (from[index][0] - src_mean[0]) * src_scale;
        let y = (from[index][1] - src_mean[1]) * src_scale;
        let u = (to[index][0] - dst_mean[0]) * dst_scale;
        let v = (to[index][1] - dst_mean[1]) * dst_scale;
        let row_a = [-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u];
        let row_b = [0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v];
        for i in 0..9 {
            for j in i..9 {
                normal[i][j] += row_a[i] * row_a[j] + row_b[i] * row_b[j];
            }
        }
    }
    for i in 0..9 {
        for j in 0..i {
            normal[i][j] = normal[j][i];
        }
    }
    let h = smallest_eigenvector(&normal)?;
    // Denormalize: H = inv(T_dst) * H_n * T_src
    let t_src = [
        [src_scale, 0.0, -src_scale * src_mean[0]],
        [0.0, src_scale, -src_scale * src_mean[1]],
        [0.0, 0.0, 1.0],
    ];
    let t_dst_inv = [
        [1.0 / dst_scale, 0.0, dst_mean[0]],
        [0.0, 1.0 / dst_scale, dst_mean[1]],
        [0.0, 0.0, 1.0],
    ];
    let hn = [h[0..3].try_into().ok()?, h[3..6].try_into().ok()?, h[6..9].try_into().ok()?];
    let tmp = mat3_mul(&t_dst_inv, &hn);
    let model3 = mat3_mul(&tmp, &t_src);
    let mut model = [0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            model[i * 3 + j] = model3[i][j];
        }
    }
    normalize_h(&mut model);
    Some(model)
}

/// Scales the last component to one when it is nonzero.
fn normalize_h(model: &mut [f64; 9]) {
    if model[8].abs() > 1e-15 {
        let inv = 1.0 / model[8];
        for value in model.iter_mut() {
            *value *= inv;
        }
    }
}

/// Centroid and mean-distance scale for Hartley normalization.
fn hartley_transform(points: &[[f64; 2]], indices: &[usize]) -> ([f64; 2], f64) {
    let n = indices.len() as f64;
    let mut mean = [0.0; 2];
    for &index in indices {
        mean[0] += points[index][0];
        mean[1] += points[index][1];
    }
    mean[0] /= n;
    mean[1] /= n;
    let mut distance = 0.0;
    for &index in indices {
        distance += (points[index][0] - mean[0]).hypot(points[index][1] - mean[1]);
    }
    let scale = if distance > f64::MIN_POSITIVE {
        2f64.sqrt() * n / distance
    } else {
        1.0
    };
    (mean, scale)
}

/// 3x3 matrix product.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Jacobi sweep returning the eigenvector of the smallest eigenvalue.
fn smallest_eigenvector(matrix: &[[f64; 9]; 9]) -> Option<[f64; 9]> {
    let mut m = *matrix;
    let mut v = [[0f64; 9]; 9];
    for i in 0..9 {
        v[i][i] = 1.0;
    }
    for _sweep in 0..200 {
        let mut off = 0f64;
        for i in 0..9 {
            for j in i + 1..9 {
                off += m[i][j] * m[i][j];
            }
        }
        if off < 1e-20 {
            break;
        }
        for p in 0..9 {
            for q in p + 1..9 {
                if m[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (m[q][q] - m[p][p]) / m[p][q];
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..9 {
                    let (mp, mq) = (m[p][k], m[q][k]);
                    m[p][k] = c * mp - s * mq;
                    m[q][k] = s * mp + c * mq;
                }
                for k in 0..9 {
                    let (mp, mq) = (m[k][p], m[k][q]);
                    m[k][p] = c * mp - s * mq;
                    m[k][q] = s * mp + c * mq;
                }
                for k in 0..9 {
                    let (vp, vq) = (v[p][k], v[q][k]);
                    v[p][k] = c * vp - s * vq;
                    v[q][k] = s * vp + c * vq;
                }
            }
        }
    }
    let mut smallest = 0usize;
    for i in 1..9 {
        if m[i][i] < m[smallest][smallest] {
            smallest = i;
        }
    }
    Some([
        v[smallest][0], v[smallest][1], v[smallest][2], v[smallest][3], v[smallest][4],
        v[smallest][5], v[smallest][6], v[smallest][7], v[smallest][8],
    ])
}

/// Euclidean reprojection error of one correspondence under `model`.
fn reprojection_error(model: &[f64; 9], from: [f64; 2], to: [f64; 2]) -> f64 {
    let w = model[6].mul_add(from[0], model[7].mul_add(from[1], model[8]));
    if w.abs() < 1e-15 {
        return f64::INFINITY;
    }
    let u = model[0].mul_add(from[0], model[1].mul_add(from[1], model[2])) / w - to[0];
    let v = model[3].mul_add(from[0], model[4].mul_add(from[1], model[5])) / w - to[1];
    u.hypot(v)
}

/// Consensus mask of `model` at `threshold`.
fn consensus_mask(
    model: &[f64; 9],
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    threshold: f64,
) -> Vec<bool> {
    (0..from.len())
        .map(|index| reprojection_error(model, from[index], to[index]) < threshold)
        .collect()
}

/// Reprojection residual of one correspondence under `model` (h33 fixed to 1).
fn projection_residual(model: &[f64; 8], from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
    let w = model[6].mul_add(from[0], model[7].mul_add(from[1], 1.0));
    [
        model[0].mul_add(from[0], model[1].mul_add(from[1], model[2])) / w - to[0],
        model[3].mul_add(from[0], model[4].mul_add(from[1], model[5])) / w - to[1],
    ]
}

/// Levenberg-Marquardt refinement of a DLT seed on geometric reprojection
/// error, matching the pinned build's converged refit on well-conditioned
/// data. Returns the seed unchanged when the linear system degenerates.
fn refine_geometric(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    indices: &[usize],
    seed: [f64; 9],
) -> [f64; 9] {
    if seed[8].abs() < 1e-15 {
        return seed;
    }
    let mut h = [0f64; 8];
    for (i, value) in h.iter_mut().enumerate() {
        *value = seed[i] / seed[8];
    }
    let error_at = |model: &[f64; 8]| -> f64 {
        indices
            .iter()
            .map(|&index| {
                let [du, dv] = projection_residual(model, from[index], to[index]);
                du * du + dv * dv
            })
            .sum()
    };
    let mut damping = 1e-8f64;
    for _iteration in 0..200 {
        let previous_error = error_at(&h);
        let mut normal = [[0f64; 8]; 8];
        let mut gradient = [0f64; 8];
        for &index in indices {
            let (x, y) = (from[index][0], from[index][1]);
            let w = h[6].mul_add(x, h[7].mul_add(y, 1.0));
            if w.abs() < 1e-15 {
                return normalize_and_flatten(h);
            }
            let nu = h[0].mul_add(x, h[1].mul_add(y, h[2]));
            let nv = h[3].mul_add(x, h[4].mul_add(y, h[5]));
            let residual = [nu / w - to[index][0], nv / w - to[index][1]];
            let w2 = w * w;
            let j_u = [x / w, y / w, 1.0 / w, 0.0, 0.0, 0.0, -nu * x / w2, -nu * y / w2];
            let j_v = [0.0, 0.0, 0.0, x / w, y / w, 1.0 / w, -nv * x / w2, -nv * y / w2];
            for p in 0..8 {
                for q in p..8 {
                    normal[p][q] += j_u[p] * j_u[q] + j_v[p] * j_v[q];
                }
                gradient[p] += j_u[p] * residual[0] + j_v[p] * residual[1];
            }
        }
        for p in 0..8 {
            for q in 0..p {
                normal[p][q] = normal[q][p];
            }
        }
        let mut accepted = false;
        for _attempt in 0..80 {
            let mut system = normal;
            for p in 0..8 {
                system[p][p] += damping;
            }
            let Some(step) = solve_8x8(&system, &gradient) else {
                damping *= 10.0;
                continue;
            };
            let mut candidate = h;
            for p in 0..8 {
                candidate[p] -= step[p];
            }
            if error_at(&candidate) < previous_error {
                h = candidate;
                damping = (damping * 0.3).max(1e-50);
                accepted = true;
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            break;
        }
    }
    normalize_and_flatten(h)
}

/// Solves `system * x = gradient` by Gaussian elimination with partial pivoting.
fn solve_8x8(system: &[[f64; 8]; 8], gradient: &[f64; 8]) -> Option<[f64; 8]> {
    let mut aug = [[0f64; 9]; 8];
    for i in 0..8 {
        aug[i][..8].copy_from_slice(&system[i]);
        aug[i][8] = gradient[i];
    }
    for column in 0..8 {
        let mut pivot = column;
        for row in column + 1..8 {
            if aug[row][column].abs() > aug[pivot][column].abs() {
                pivot = row;
            }
        }
        aug.swap(column, pivot);
        if aug[column][column].abs() < 1e-18 {
            return None;
        }
        for row in 0..8 {
            if row == column {
                continue;
            }
            let factor = aug[row][column] / aug[column][column];
            for k in 0..9 {
                aug[row][k] -= factor * aug[column][k];
            }
        }
    }
    let mut step = [0f64; 8];
    for i in 0..8 {
        step[i] = aug[i][8] / aug[i][i];
    }
    Some(step)
}

/// Appends the fixed unit last row element.
fn normalize_and_flatten(h: [f64; 8]) -> [f64; 9] {
    [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
}

/// DLT seed plus geometric refinement — the refit model over an index set.
fn refit_least_squares(from: &[[f64; 2]], to: &[[f64; 2]], indices: &[usize]) -> Option<[f64; 9]> {
    let seed = fit_least_squares(from, to, indices)?;
    Some(refine_geometric(from, to, indices, seed))
}

/// Draws the next four-index subset from the private generator.
fn draw_subset(state: &mut u64, count: usize) -> [usize; 4] {
    let count_i32 = count as i32;
    let mut subset = [0usize; 4];
    for position in 0..4 {
        let mut candidate = rng_uniform(state, 0, count_i32) as usize;
        while subset[..position].contains(&candidate) {
            candidate = rng_uniform(state, 0, count_i32) as usize;
        }
        subset[position] = candidate;
    }
    subset
}

/// Fits the model for one drawn subset via normalized DLT.
fn subset_model(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    subset: &[usize; 4],
) -> Option<[f64; 9]> {
    if !subset_is_valid(from, subset) {
        return None;
    }
    fit_least_squares(from, to, subset)
}

/// RANSAC: maximizes the consensus count, with an adaptive iteration cap.
fn estimate_ransac(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    threshold: f64,
    max_iters: i32,
    confidence: f64,
) -> Option<Vec<bool>> {
    let count = from.len();
    if count == 4 {
        let indices = [0usize, 1, 2, 3];
        let model = fit_least_squares(from, to, &indices)?;
        return Some(
            consensus_mask(&model, from, to, threshold),
        );
    }
    let mut state = u64::MAX;
    let mut best_mask: Option<Vec<bool>> = None;
    let mut best_count = 0usize;
    let mut iteration_budget = max_iters;
    let mut iteration = 0;
    while iteration < iteration_budget {
        // Degenerate subsets are redrawn inside the same iteration; only a
        // usable hypothesis consumes the iteration budget.
        let mut model = None;
        for _attempt in 0..4096 {
            let subset = draw_subset(&mut state, count);
            if let Some(candidate) = subset_model(from, to, &subset) {
                model = Some(candidate);
                break;
            }
        }
        let Some(model) = model else {
            break;
        };
        iteration += 1;
        let mask = consensus_mask(&model, from, to, threshold);
        let inlier_count = mask.iter().filter(|&&flag| flag).count();
        if inlier_count > best_count {
            best_count = inlier_count;
            best_mask = Some(mask);
            let fraction = inlier_count as f64 / count as f64;
            let budget = ransac_iteration_budget(confidence, fraction, 4).max(1);
            iteration_budget = iteration_budget.min(budget.max(1));
        }
    }
    best_mask
}

/// LMeDS: minimizes the median reprojection error over an exhaustive scan of
/// all four-point subsets in index order. The returned model is refit over the
/// points within a median-derived scale bound of the winning hypothesis, while
/// the reported mask still uses the caller's threshold.
fn estimate_lmeds(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    threshold: f64,
) -> Option<(Vec<bool>, [f64; 9])> {
    let count = from.len();
    let mut best_subset: Option<[usize; 4]> = None;
    let mut best_model: Option<[f64; 9]> = None;
    let mut best_median = f64::INFINITY;
    for a in 0..count {
        for b in a + 1..count {
            for c in b + 1..count {
                for d in c + 1..count {
                    let subset = [a, b, c, d];
                    let Some(model) = subset_model(from, to, &subset) else {
                        continue;
                    };
                    let mut errors: Vec<f64> = (0..count)
                        .map(|index| reprojection_error(&model, from[index], to[index]))
                        .collect();
                    errors.sort_by(f64::total_cmp);
                    let median = errors[count / 2];
                    if median < best_median {
                        best_median = median;
                        best_subset = Some(subset);
                        best_model = Some(model);
                    }
                }
            }
        }
    }
    let subset = best_subset?;
    let model = best_model?;
    let sigma = 1.4826 * (1.0 + 5.0 / (count.saturating_sub(4)).max(1) as f64) * best_median;
    let bound = threshold.min(3.0 * sigma);
    let refit_indices: Vec<usize> = (0..count)
        .filter(|&index| reprojection_error(&model, from[index], to[index]) < bound)
        .collect();
    let refined = if refit_indices.len() >= 4 {
        refit_least_squares(from, to, &refit_indices).unwrap_or(model)
    } else {
        refine_geometric(from, to, &subset, model)
    };
    Some((consensus_mask(&model, from, to, threshold), refined))
}

/// Rounds coordinates to single precision, matching the pinned build's
/// internal point storage.
fn quantize_points(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    points
        .iter()
        .map(|point| [point[0] as f32 as f64, point[1] as f32 as f64])
        .collect()
}

/// Estimates the 3x3 homography between matching point sets.
pub(crate) fn estimate_homography(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    method: i32,
    threshold: f64,
    max_iters: i32,
    confidence: f64,
) -> Result<HomographyEstimate, HomographyError> {
    if from.len() < 4 {
        return Err(HomographyError::TooFewPoints);
    }
    let from = quantize_points(from);
    let to = quantize_points(to);
    let (from, to) = (from.as_slice(), to.as_slice());
    let all: Vec<usize> = (0..from.len()).collect();
    let mask = match method {
        0 => {
            let model = refit_least_squares(from, to, &all);
            let inliers = match &model {
                Some(model) => consensus_mask(model, from, to, threshold)
                    .into_iter()
                    .map(u8::from)
                    .collect(),
                None => vec![0; from.len()],
            };
            return Ok(HomographyEstimate { model, inliers });
        }
        4 => {
            let Some((mask, model)) = estimate_lmeds(from, to, threshold) else {
                return Ok(HomographyEstimate {
                    model: None,
                    inliers: vec![0; from.len()],
                });
            };
            return Ok(HomographyEstimate {
                model: Some(model),
                inliers: mask.into_iter().map(u8::from).collect(),
            });
        }
        8 => estimate_ransac(from, to, threshold, max_iters, confidence),
        16 => estimate_ransac(from, to, threshold, max_iters, confidence),
        other => return Err(HomographyError::UnsupportedMethod(other)),
    };
    let Some(mask) = mask else {
        return Ok(HomographyEstimate {
            model: None,
            inliers: vec![0; from.len()],
        });
    };
    let indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &flag)| flag.then_some(index))
        .collect();
    let model = refit_least_squares(from, to, &indices);
    Ok(HomographyEstimate {
        model,
        inliers: mask.into_iter().map(u8::from).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(values: &[[f64; 2]]) -> Vec<[f64; 2]> {
        values.to_vec()
    }

    #[test]
    fn method_zero_fits_all_points() {
        let from = points(&[
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [5.0, 3.0],
        ]);
        let to = points(&[
            [2.0, 3.0], [4.0, 3.0], [2.0, 5.0], [4.0, 5.0], [6.0, 4.0], [12.0, 9.0],
        ]);
        let estimate = estimate_homography(&from, &to, 0, 3.0, 2000, 0.99).expect("estimate");
        let model = estimate.model.expect("model");
        for (actual, expected) in model.iter().zip([2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9, "model {model:?}");
        }
    }

    #[test]
    fn ransac_rejects_outliers() {
        let from = points(&[
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [5.0, 3.0], [3.0, 2.0], [4.0, 1.0],
        ]);
        let to = points(&[
            [2.0, 3.0], [4.0, 3.0], [2.0, 5.0], [4.0, 5.0], [6.0, 4.0], [12.0, 9.0], [8.9, 9.9], [10.0, 7.0],
        ]);
        let estimate = estimate_homography(&from, &to, 8, 3.0, 2000, 0.99).expect("estimate");
        assert_eq!(estimate.inliers, vec![1, 1, 1, 1, 1, 0, 1, 1]);
    }

    #[test]
    fn lmeds_finds_consensus() {
        let from = points(&[
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5], [5.0, 3.0], [3.0, 2.0], [4.0, 1.0],
        ]);
        let to = points(&[
            [2.0, 3.0], [4.0, 3.0], [2.0, 5.0], [4.0, 5.0], [6.0, 4.0], [12.0, 9.0], [8.9, 9.9], [10.0, 7.0],
        ]);
        let estimate = estimate_homography(&from, &to, 4, 3.0, 2000, 0.99).expect("estimate");
        assert_eq!(estimate.inliers, vec![1, 1, 1, 1, 1, 1, 0, 1]);
    }

    #[test]
    fn too_few_points_error() {
        let from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let to = points(&[[2.0, 3.0], [4.0, 3.0], [2.0, 5.0]]);
        assert_eq!(
            estimate_homography(&from, &to, 0, 3.0, 2000, 0.99),
            Err(HomographyError::TooFewPoints)
        );
    }

    #[test]
    fn unsupported_method_error() {
        let from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let to = points(&[[2.0, 3.0], [4.0, 3.0], [2.0, 5.0], [4.0, 5.0]]);
        assert_eq!(
            estimate_homography(&from, &to, 2, 3.0, 2000, 0.99),
            Err(HomographyError::UnsupportedMethod(2))
        );
    }
}
