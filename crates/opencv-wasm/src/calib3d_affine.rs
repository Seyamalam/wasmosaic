//! `estimateAffine2D`: robust affine estimation over point correspondences.
//!
//! Pinned contract (OpenCV.js 4.13.0 differential probing):
//! - Point inputs may be `N x 1` two-channel or `N x 2` single-channel
//!   matrices at any scalar depth; the two point sets must match.
//! - `method` selects RANSAC (8) or LMeDS (4); any other value is an error.
//! - Fewer than three correspondences, or no non-degenerate hypothesis, yield
//!   an empty result and an all-zero mask rather than an exception. Empty
//!   input itself raises an error.
//! - The reported inlier mask is the winning hypothesis's consensus set
//!   (Euclidean reprojection error strictly below the threshold); it is not
//!   recomputed under the refined model. The returned 2x3 CV_64F matrix is
//!   the least-squares refit over that set.
//! - RANSAC subsets are drawn from a private per-call multiply-with-carry
//!   generator seeded at `u64::MAX`, while LMeDS scans subsets in index
//!   order, matching the pinned call-sequence behavior.

use std::{error::Error, fmt};

/// Robust estimation method for [`estimate_affine_2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AffineMethod {
    /// Least-median-of-squares (`cv::LMEDS`, value 4).
    Lmeds,
    /// Random-sample consensus (`cv::RANSAC`, value 8).
    Ransac,
}

/// Failures reported while estimating an affine transform.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AffineError {
    /// A method value other than LMeDS (4) or RANSAC (8) was requested.
    UnsupportedMethod(i32),
}

impl fmt::Display for AffineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMethod(method) => write!(
                formatter,
                "estimateAffine2D method {method} is not implemented; use LMeDS (4) or RANSAC (8)"
            ),
        }
    }
}

impl Error for AffineError {}

/// Outcome of a robust affine estimate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AffineEstimate {
    /// The refined 2x3 model, or `None` when no valid hypothesis exists.
    pub(crate) model: Option<[f64; 6]>,
    /// The winning hypothesis's consensus set, one byte per correspondence.
    pub(crate) inliers: Vec<u8>,
}

/// Advances the multiply-with-carry generator and returns its output word.
///
/// The pinned robust estimator seeds a private generator per call and reads
/// the low half of the updated state.
pub(crate) fn rng_next(state: &mut u64) -> u32 {
    *state = (*state & 0xFFFF_FFFF)
        .wrapping_mul(4_164_903_690)
        .wrapping_add(*state >> 32);
    (*state & 0xFFFF_FFFF) as u32
}

/// Draws an integer uniformly in `[low, high)`.
pub(crate) fn rng_uniform(state: &mut u64, low: i32, high: i32) -> i32 {
    if low == high {
        return low;
    }
    low + (rng_next(state) % (high - low) as u32) as i32
}

/// Fits the exact affine map through three correspondences.
///
/// Returns `None` when the source triple is degenerate (a zero pivot during
/// elimination), mirroring the pinned behavior of rejecting singular
/// hypotheses.
fn fit_three_point(from: &[[f64; 2]], to: &[[f64; 2]]) -> Option<[f64; 6]> {
    let mut coefficients = [[0.0f64; 3]; 3];
    let mut right_hand_sides = [[0.0f64; 2]; 3];
    for index in 0..3 {
        coefficients[index] = [from[index][0], from[index][1], 1.0];
        right_hand_sides[index] = to[index];
    }
    for column in 0..3 {
        let pivot_row = (column..3)
            .max_by(|&left, &right| {
                coefficients[left][column]
                    .abs()
                    .total_cmp(&coefficients[right][column].abs())
            })
            .expect("each column has pivot candidates");
        if coefficients[pivot_row][column] == 0.0 {
            return None;
        }
        if pivot_row != column {
            coefficients.swap(column, pivot_row);
            right_hand_sides.swap(column, pivot_row);
        }
        let negative_inverse_pivot = -1.0 / coefficients[column][column];
        for row in column + 1..3 {
            let factor = coefficients[row][column] * negative_inverse_pivot;
            for target in column + 1..3 {
                coefficients[row][target] += factor * coefficients[column][target];
            }
            for target in 0..2 {
                right_hand_sides[row][target] += factor * right_hand_sides[column][target];
            }
        }
    }
    for row in (0..3).rev() {
        for target in 0..2 {
            let mut value = right_hand_sides[row][target];
            for column in row + 1..3 {
                value -= coefficients[row][column] * right_hand_sides[column][target];
            }
            right_hand_sides[row][target] = value / coefficients[row][row];
        }
    }
    Some([
        right_hand_sides[0][0],
        right_hand_sides[1][0],
        right_hand_sides[2][0],
        right_hand_sides[0][1],
        right_hand_sides[1][1],
        right_hand_sides[2][1],
    ])
}

/// Least-squares affine refit over an arbitrary correspondence set.
///
/// The two parameter rows share the same normal-equation system; a singular
/// system returns `None`.
fn fit_least_squares(from: &[[f64; 2]], to: &[[f64; 2]]) -> Option<[f64; 6]> {
    let mut columns = [[0.0f64; 3]; 3];
    let mut targets = [[0.0f64; 2]; 3];
    for (&[x, y], &[u, v]) in from.iter().zip(to.iter()) {
        let row = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                columns[i][j] += row[i] * row[j];
            }
            targets[i][0] += row[i] * u;
            targets[i][1] += row[i] * v;
        }
    }
    for column in 0..3 {
        let pivot_row = (column..3)
            .max_by(|&left, &right| {
                columns[left][column]
                    .abs()
                    .total_cmp(&columns[right][column].abs())
            })
            .expect("each column has pivot candidates");
        if columns[pivot_row][column] == 0.0 {
            return None;
        }
        if pivot_row != column {
            columns.swap(column, pivot_row);
            targets.swap(column, pivot_row);
        }
        let negative_inverse_pivot = -1.0 / columns[column][column];
        for row in column + 1..3 {
            let factor = columns[row][column] * negative_inverse_pivot;
            for target in column + 1..3 {
                columns[row][target] += factor * columns[column][target];
            }
            for side in 0..2 {
                targets[row][side] += factor * targets[column][side];
            }
        }
    }
    for row in (0..3).rev() {
        for side in 0..2 {
            let mut value = targets[row][side];
            for column in row + 1..3 {
                value -= columns[row][column] * targets[column][side];
            }
            targets[row][side] = value / columns[row][row];
        }
    }
    Some([
        targets[0][0],
        targets[1][0],
        targets[2][0],
        targets[0][1],
        targets[1][1],
        targets[2][1],
    ])
}

/// Euclidean reprojection error of one correspondence under `model`.
fn reprojection_error(model: &[f64; 6], from: [f64; 2], to: [f64; 2]) -> f64 {
    let dx = model[0].mul_add(from[0], model[1].mul_add(from[1], model[2])) - to[0];
    let dy = model[3].mul_add(from[0], model[4].mul_add(from[1], model[5])) - to[1];
    dx.hypot(dy)
}

/// Draws the next three-index subset.
///
/// Each position draws once from the generator and redraws while the index is
/// already present in the subset. Returns `None` when the sorted combination
/// was already used.
fn draw_subset(state: &mut u64, count: usize, used: &mut Vec<[usize; 3]>) -> Option<[usize; 3]> {
    let count_i32 = count as i32;
    let mut subset = [0usize; 3];
    for position in 0..3 {
        let mut candidate = rng_uniform(state, 0, count_i32) as usize;
        while subset[..position].contains(&candidate) {
            candidate = rng_uniform(state, 0, count_i32) as usize;
        }
        subset[position] = candidate;
    }
    let mut sorted = subset;
    sorted.sort_unstable();
    if used.contains(&sorted) {
        return None;
    }
    used.push(sorted);
    Some(subset)
}

/// Adaptive RANSAC iteration budget: rounds the log-ratio needed to reach
/// `confidence` given the current best inlier fraction.
pub(crate) fn ransac_iteration_budget(confidence: f64, inlier_fraction: f64, model_points: i32) -> i32 {
    let numerator = (1.0 - confidence).max(f64::MIN_POSITIVE);
    let denominator = 1.0 - inlier_fraction.powi(model_points);
    if numerator >= 1.0 - 1e-10 || denominator <= f64::MIN_POSITIVE {
        return 0;
    }
    (numerator.ln() / denominator.ln()).round() as i32
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
    let mut state = u64::MAX;
    let mut used: Vec<[usize; 3]> = Vec::new();
    let mut best_mask: Option<Vec<bool>> = None;
    let mut best_count = 0usize;
    let total_subsets = count * (count - 1) * (count - 2) / 6;
    let mut iteration_budget = max_iters;
    let mut iteration = 0;
    while iteration < iteration_budget {
        if used.len() >= total_subsets {
            break;
        }
        let Some(subset) = draw_subset(&mut state, count, &mut used) else {
            continue;
        };
        iteration += 1;
        let subset_from: Vec<[f64; 2]> = subset.iter().map(|&i| from[i]).collect();
        let subset_to: Vec<[f64; 2]> = subset.iter().map(|&i| to[i]).collect();
        let Some(model) = fit_three_point(&subset_from, &subset_to) else {
            continue;
        };
        let mut mask = vec![false; count];
        let mut inlier_count = 0usize;
        for index in 0..count {
            if reprojection_error(&model, from[index], to[index]) < threshold {
                mask[index] = true;
                inlier_count += 1;
            }
        }
        if inlier_count > best_count {
            best_count = inlier_count;
            best_mask = Some(mask);
            let fraction = inlier_count as f64 / count as f64;
            let budget = ransac_iteration_budget(confidence, fraction, 3).max(1);
            iteration_budget = iteration_budget.min(budget.max(1));
        }
    }
    best_mask
}

/// LMeDS: minimizes the median reprojection error over an exhaustive scan of
/// all three-point subsets in index order, then thresholds the best model at
/// the shared threshold.
fn estimate_lmeds(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    threshold: f64,
    _max_iters: i32,
) -> Option<Vec<bool>> {
    let count = from.len();
    let mut best_model: Option<[f64; 6]> = None;
    let mut best_median = f64::INFINITY;
    for a in 0..count {
        for b in a + 1..count {
            for c in b + 1..count {
                let subset_from = [from[a], from[b], from[c]];
                let subset_to = [to[a], to[b], to[c]];
                let Some(model) = fit_three_point(&subset_from, &subset_to) else {
                    continue;
                };
                let mut errors: Vec<f64> = (0..count)
                    .map(|index| reprojection_error(&model, from[index], to[index]))
                    .collect();
                errors.sort_by(f64::total_cmp);
                let median = errors[count / 2];
                if median < best_median {
                    best_median = median;
                    best_model = Some(model);
                }
            }
        }
    }
    let model = best_model?;
    let mask: Vec<bool> = (0..count)
        .map(|index| reprojection_error(&model, from[index], to[index]) < threshold)
        .collect();
    Some(mask)
}

/// Estimates the 2x3 affine map between matching point sets.
///
/// Returns the refined model and the winning hypothesis's consensus mask, or
/// an empty model with an all-zero mask when fewer than three correspondences
/// exist or every sampled hypothesis is degenerate.
pub(crate) fn estimate_affine_2d(
    from: &[[f64; 2]],
    to: &[[f64; 2]],
    method: i32,
    threshold: f64,
    max_iters: i32,
    confidence: f64,
) -> Result<AffineEstimate, AffineError> {
    let method = match method {
        4 => AffineMethod::Lmeds,
        8 => AffineMethod::Ransac,
        other => return Err(AffineError::UnsupportedMethod(other)),
    };
    let count = from.len();
    if count < 3 {
        return Ok(AffineEstimate {
            model: None,
            inliers: vec![0; count],
        });
    }
    let mask = match method {
        AffineMethod::Ransac => estimate_ransac(from, to, threshold, max_iters, confidence),
        AffineMethod::Lmeds => estimate_lmeds(from, to, threshold, max_iters),
    };
    let Some(mask) = mask else {
        return Ok(AffineEstimate {
            model: None,
            inliers: vec![0; count],
        });
    };
    let selected: Vec<usize> = (0..count).filter(|&index| mask[index]).collect();
    if selected.len() < 3 {
        return Ok(AffineEstimate {
            model: None,
            inliers: mask.iter().map(|&flag| u8::from(flag)).collect(),
        });
    }
    let fit_from: Vec<[f64; 2]> = selected.iter().map(|&index| from[index]).collect();
    let fit_to: Vec<[f64; 2]> = selected.iter().map(|&index| to[index]).collect();
    let model = fit_least_squares(&fit_from, &fit_to);
    Ok(AffineEstimate {
        model,
        inliers: mask.iter().map(|&flag| u8::from(flag)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(values: &[[f64; 2]]) -> Vec<[f64; 2]> {
        values.to_vec()
    }

    #[test]
    fn identity_is_recovered() {
        let from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let to = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let result = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac accepted");
        assert_eq!(result.inliers, vec![1, 1, 1]);
        let model = result.model.expect("model returned");
        assert!(
            model
                .iter()
                .zip([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn scale_and_translation_are_recovered() {
        let from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let to = points(&[[3.0, 1.0], [5.0, 1.0], [3.0, 3.0], [5.0, 3.0]]);
        let result = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac accepted");
        assert_eq!(result.inliers, vec![1; 4]);
        let model = result.model.expect("model returned");
        assert!(
            model
                .iter()
                .zip([2.0, 0.0, 3.0, 0.0, 2.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn outliers_are_rejected() {
        let mut from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let mut to = points(&[[3.0, 1.0], [5.0, 1.0], [3.0, 3.0], [5.0, 3.0]]);
        from.push([9.0, 9.0]);
        to.push([100.0, 100.0]);
        let result = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac accepted");
        assert_eq!(result.inliers.last(), Some(&0));
        let model = result.model.expect("model returned");
        assert!(
            model
                .iter()
                .zip([2.0, 0.0, 3.0, 0.0, 2.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn fewer_than_three_points_return_empty() {
        let from = points(&[[1.0, 2.0], [3.0, 4.0]]);
        let to = points(&[[5.0, 6.0], [7.0, 8.0]]);
        let result = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac accepted");
        assert!(result.model.is_none());
        assert_eq!(result.inliers, vec![0, 0]);
    }

    #[test]
    fn collinear_sources_return_empty() {
        let from = points(&[[0.0, 0.0], [1.0, 3.0], [2.0, 6.0], [3.0, 9.0]]);
        let to = points(&[[0.0, 0.0], [5.0, 1.0], [10.0, 2.0], [15.0, 3.0]]);
        let result = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac accepted");
        assert!(result.model.is_none());
    }

    #[test]
    fn unknown_method_errors() {
        let from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let to = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(estimate_affine_2d(&from, &to, 0, 3.0, 2000, 0.99).is_err());
        assert!(estimate_affine_2d(&from, &to, 1, 3.0, 2000, 0.99).is_err());
    }

    #[test]
    fn lmeds_finds_the_consensus_model() {
        let mut from = points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]);
        let mut to = points(&[[3.0, 1.0], [5.0, 1.0], [3.0, 3.0], [5.0, 3.0], [7.0, 1.0]]);
        for index in 0..4 {
            from.push([20.0 + index as f64, 30.0 + index as f64]);
            to.push([index as f64 * 7.0 % 23.0, index as f64 * 11.0 % 29.0]);
        }
        let result = estimate_affine_2d(&from, &to, 4, 3.0, 2000, 0.99).expect("lmeds accepted");
        let model = result.model.expect("model returned");
        assert!(
            model
                .iter()
                .zip([2.0, 0.0, 3.0, 0.0, 2.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }
}

#[cfg(test)]
mod sequence_tests {
    use super::*;

    fn tie_data() -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let from = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [5.0, 1.0],
            [6.0, 1.0],
        ];
        let to = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [15.0, 0.0],
            [16.0, 0.0],
            [15.0, 1.0],
            [16.0, 1.0],
        ];
        (from, to)
    }

    #[test]
    fn ransac_after_lmeds_matches_pinned() {
        let (from, to) = tie_data();
        let first = estimate_affine_2d(&from, &to, 4, 3.0, 2000, 0.99).expect("lmeds");
        eprintln!("lmeds model={:?} inl={:?}", first.model, first.inliers);
        let second = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac");
        eprintln!("ransac model={:?} inl={:?}", second.model, second.inliers);
        assert_eq!(first.inliers, vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(second.inliers, vec![1, 1, 1, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn ransac_first_call_matches_pinned() {
        let (from, to) = tie_data();
        let second = estimate_affine_2d(&from, &to, 8, 3.0, 2000, 0.99).expect("ransac");
        eprintln!(
            "fresh ransac model={:?} inl={:?}",
            second.model, second.inliers
        );
    }
}
