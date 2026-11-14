//! `meanShift`: iterative mean-shift tracking of a rectangular window over a
//! single-channel probability image.
//!
//! Pinned contract (OpenCV.js 4.13.0 differential probing):
//! - The window is intersected with the image rect before iterating; the
//!   returned rectangle is that clamped window, and each new position is
//!   clamped to `[0, cols - w] x [0, rows - h]` so the size is preserved.
//! - `maxIters` is `criteria.maxCount` when the COUNT flag is set, else 100.
//! - When the EPS flag is set, the only early exit is the initial centroid
//!   displacement being within `epsilon` on the first iteration; afterward
//!   the loop runs all iterations even when the window stops moving.
//! - Without the EPS flag, the loop exits when the new position equals the
//!   current one. `m00 == 0` (empty region mass) exits in both modes.
//! - The window center is `x + w / 2.0`, `y + h / 2.0`; the shift applies
//!   round-half-even to the displacement, and the return value counts the
//!   completed iterations.

use std::{error::Error, fmt};

use crate::{
    core_norm::read_scalar,
    imgproc_window::scalar_depth,
    mat::{Mat, MatError},
};

#[derive(Debug)]
pub(crate) enum MeanShiftError {
    Matrix(MatError),
    UnsupportedChannels,
}

impl fmt::Display for MeanShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedChannels => {
                formatter.write_str("probImage requires a single channel")
            }
        }
    }
}

impl Error for MeanShiftError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for MeanShiftError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

fn round_half_even(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// Runs the mean-shift loop on a single-channel image and returns the
/// completed iteration count plus the final clamped window
/// `[x, y, width, height]`.
pub(crate) fn mean_shift(
    prob_image: &Mat,
    mut x: i64,
    mut y: i64,
    mut width: i64,
    mut height: i64,
    criteria_type: i32,
    max_count: i32,
    epsilon: f64,
) -> Result<(i32, [i64; 4]), MeanShiftError> {
    if prob_image.channels() != 1 {
        return Err(MeanShiftError::UnsupportedChannels);
    }
    let cols = i64::from(prob_image.columns());
    let rows = i64::from(prob_image.rows());

    // Intersect the window with the image rect.
    let x_end = (x + width).clamp(0, cols);
    let y_end = (y + height).clamp(0, rows);
    x = x.clamp(0, cols);
    y = y.clamp(0, rows);
    width = x_end - x;
    height = y_end - y;

    let depth = scalar_depth(prob_image.depth());
    let data = prob_image.compact_bytes();
    let stride = cols as usize;
    let max_iters = if criteria_type & 1 != 0 { max_count } else { 100 };
    let use_eps = criteria_type & 2 != 0;

    let mut iterations = 0;
    for i in 0..max_iters {
        if width <= 0 || height <= 0 {
            break;
        }
        let (wx0, wy0) = (x as usize, y as usize);
        let (wx1, wy1) = (wx0 + width as usize, wy0 + height as usize);
        let (mut m00, mut m10, mut m01) = (0.0f64, 0.0f64, 0.0f64);
        for row in wy0..wy1 {
            for column in wx0..wx1 {
                let value = read_scalar(&data, row * stride + column, depth);
                m00 += value;
                m10 += column as f64 * value;
                m01 += row as f64 * value;
            }
        }
        if m00 == 0.0 {
            break;
        }
        let dx = m10 / m00 - (x as f64 + width as f64 / 2.0);
        let dy = m01 / m00 - (y as f64 + height as f64 / 2.0);
        // Pinned behavior: the epsilon test only decides the first
        // iteration; afterward every pass counts even with zero shift.
        if use_eps && i == 0 && (dx * dx + dy * dy).sqrt() <= epsilon {
            break;
        }
        let nx = (x + round_half_even(dx)).clamp(0, (cols - width).max(0));
        let ny = (y + round_half_even(dy)).clamp(0, (rows - height).max(0));
        if !use_eps && nx == x && ny == y {
            break;
        }
        x = nx;
        y = ny;
        iterations += 1;
    }
    Ok((iterations, [x, y, width, height]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::mat_from_u8;

    fn image(fill: &[(u32, u32, u8)], rows: u32, cols: u32) -> Mat {
        let mut bytes = vec![0u8; (rows * cols) as usize];
        for (x, y, v) in fill {
            bytes[(*y as usize) * (cols as usize) + (*x as usize)] = *v;
        }
        mat_from_u8(&bytes, rows, cols, 1).expect("image")
    }

    #[test]
    fn shifts_window_onto_blob() {
        let img = image(&[(10, 12, 255)], 20, 20);
        let (iters, rect) = mean_shift(&img, 5, 5, 10, 10, 1, 10, 0.0).expect("shift");
        assert_eq!((iters, rect), (1, [5, 7, 10, 10]));
    }

    #[test]
    fn count_only_breaks_on_no_move() {
        let img = image(&[(9, 5, 153), (10, 5, 102)], 20, 20);
        let (iters, rect) = mean_shift(&img, 5, 0, 10, 10, 1, 10, 0.0).expect("shift");
        assert_eq!((iters, rect), (1, [4, 0, 10, 10]));
    }

    #[test]
    fn eps_mode_runs_max_iters_after_convergence() {
        let img = image(&[(10, 12, 255)], 20, 20);
        let (iters, rect) = mean_shift(&img, 5, 5, 10, 10, 3, 5, 0.001).expect("shift");
        assert_eq!((iters, rect), (5, [5, 7, 10, 10]));
    }

    #[test]
    fn eps_mode_zero_iters_when_initially_converged() {
        let img = image(&[(6, 6, 200), (7, 6, 200), (6, 7, 200), (7, 7, 255)], 20, 20);
        let (iters, _) = mean_shift(&img, 3, 3, 8, 8, 3, 5, 10.0).expect("shift");
        assert_eq!(iters, 0);
    }

    #[test]
    fn no_count_flag_defaults_to_100() {
        let img = image(&[(10, 12, 255)], 20, 20);
        let (iters, _) = mean_shift(&img, 5, 5, 10, 10, 2, 5, 0.001).expect("shift");
        assert_eq!(iters, 100);
    }

    #[test]
    fn clamps_window_to_image() {
        let img = image(&[(19, 5, 255)], 20, 20);
        let (iters, rect) = mean_shift(&img, 12, 0, 10, 10, 1, 10, 0.0).expect("shift");
        assert_eq!((iters, rect), (0, [12, 0, 8, 10]));
    }

    #[test]
    fn empty_window_returns_zero() {
        let img = image(&[(3, 3, 255)], 20, 20);
        let (iters, rect) = mean_shift(&img, 10, 10, 5, 5, 1, 10, 0.0).expect("shift");
        assert_eq!((iters, rect), (0, [10, 10, 5, 5]));
    }

    #[test]
    fn multi_channel_rejected() {
        let img = mat_from_u8(&vec![0u8; 20 * 20 * 3], 20, 20, 3).expect("mat");
        assert!(mean_shift(&img, 0, 0, 10, 10, 1, 10, 0.0).is_err());
    }
}
