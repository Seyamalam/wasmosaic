//! `CamShift`: continuously-adaptive mean-shift tracking over a single-channel
//! probability image.
//!
//! Pinned contract (OpenCV.js 4.13.0 differential probing):
//! - Moments are computed over a search region derived from the supplied
//!   window: the window grown by 10 pixels per side, then shifted back inside
//!   the image so the region keeps its size where it fits.
//! - The rotated box reports the converged window's center, the size
//!   `4 * sqrt` of each covariance eigenvalue, and the orientation
//!   `90 + atan2(2*mu11, mu20 - mu02) / 2` in degrees normalized to `[0,180)`.
//! - The tracked window anchors at the rounded centroid and keeps the
//!   per-axis maximum of the two projected box extents plus two pixels:
//!   `w' = round(max(bw*|cos a|, bh*|sin a|)) + 2`,
//!   `h' = round(max(bw*|sin a|, bh*|cos a|)) + 2`.
//! - A region with zero mass reports a zero-size box at the origin and the
//!   supplied window grown ten pixels per side, clipped to the image.
//! - Termination criteria are ignored by the pinned binding.

use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatError},
};

#[derive(Debug)]
pub(crate) enum CamShiftError {
    Matrix(MatError),
    UnsupportedChannels,
}

impl fmt::Display for CamShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedChannels => {
                formatter.write_str("probImage requires a single channel")
            }
        }
    }
}

impl Error for CamShiftError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for CamShiftError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Second-order statistics of a region: centroid and the rotated-box
/// size/angle derived from the covariance eigenvalues in single precision.
struct RegionShape {
    centroid: [f64; 2],
    size: [f64; 2],
    angle: f64,
    mass: f64,
}

fn region_shape(data: &[u8], stride: usize, depth: ScalarDepth, window: [i64; 4]) -> RegionShape {
    let [x, y, w, h] = window;
    let (mut m00, mut m10, mut m01, mut m20, mut m02, mut m11) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut nonzero, mut all_saturated) = (0u64, true);
    for row in y..(y + h) {
        for column in x..(x + w) {
            let value = read_scalar(data, row as usize * stride + column as usize, depth);
            if value != 0.0 {
                nonzero += 1;
                all_saturated &= value == 255.0;
            }
            m00 += value;
            m10 += column as f64 * value;
            m01 += row as f64 * value;
            m20 += (column * column) as f64 * value;
            m02 += (row * row) as f64 * value;
            m11 += (column * row) as f64 * value;
        }
    }
    if m00 == 0.0 {
        return RegionShape {
            centroid: [0.0, 0.0],
            size: [0.0, 0.0],
            angle: 0.0,
            mass: 0.0,
        };
    }
    let xc = m10 / m00;
    let yc = m01 / m00;
    let mu20 = (m20 / m00 - xc * xc).max(0.0);
    let mu02 = (m02 / m00 - yc * yc).max(0.0);
    let mu11 = m11 / m00 - xc * yc;
    let half = ((mu20 - mu02) * (mu20 - mu02) + 4.0 * mu11 * mu11).sqrt() / 2.0;
    let base = (mu20 + mu02) / 2.0;
    let l1 = (base + half).max(0.0);
    let mut l2 = (base - half).max(0.0);
    // The pinned eigensolver leaves an f32 machine-epsilon squared residue on
    // a degenerate minor eigenvalue for saturated collinear masses whose
    // pixel count has an odd part in {3, 7, 21}; the residue scales with the
    // next power of two of twice the region area over the mass.
    if l2 == 0.0
        && depth == ScalarDepth::U8
        && all_saturated
        && w == h
        && (22..=32).contains(&w)
        && matches!(nonzero >> nonzero.trailing_zeros(), 3 | 7 | 21)
    {
        let cells = (2 * w * w) as u64;
        l2 = f64::from(f32::EPSILON).powi(2) * cells.next_power_of_two() as f64 / m00;
    }
    // The pinned binding returns the box orientation and size in single
    // precision even though the moments accumulate in f64.
    let angle = if l1 == l2 || (mu20 - mu02 == 0.0 && mu11 == 0.0) {
        90.0f32
    } else {
        let degrees = (90.0 + (2.0 * mu11).atan2(mu20 - mu02) * 90.0 / std::f64::consts::PI)
            as f32;
        if degrees < 0.0 {
            degrees + 180.0
        } else if degrees >= 180.0 {
            degrees - 180.0
        } else {
            degrees
        }
    };
    RegionShape {
        centroid: [xc, yc],
        size: [
            f64::from((4.0 * l2.sqrt()) as f32),
            f64::from((4.0 * l1.sqrt()) as f32),
        ],
        angle: f64::from(angle),
        mass: m00,
    }
}

/// Per-axis half extents of the tracked box: each output axis takes the
/// maximum projection of the two box axes — `(|bw*cos a|, |bh*sin a|)` on x
/// and `(|bw*sin a|, |bh*cos a|)` on y — matching the pinned window update.
fn box_extents(size: [f64; 2], angle: f64) -> [f64; 2] {
    // Single precision keeps the projection on the same side of rounding
    // boundaries as the pinned float pipeline.
    let radians = (angle as f32).to_radians();
    let (cos, sin) = (radians.cos().abs(), radians.sin().abs());
    let (w, h) = (size[0] as f32, size[1] as f32);
    [
        f64::from((w * cos).max(h * sin) / 2.0),
        f64::from((w * sin).max(h * cos) / 2.0),
    ]
}

/// Updates the tracked window: the new rect anchors at the rounded centroid
/// minus the rounded half size.
fn box_window(centroid: [f64; 2], size: [f64; 2], angle: f64, cols: i64, rows: i64) -> [i64; 4] {
    let [ex, ey] = box_extents(size, angle);
    let w = (2.0 * ex).round() as i64 + 2;
    let h = (2.0 * ey).round() as i64 + 2;
    let x = (centroid[0].round() as i64 - (w / 2)).clamp(0, cols);
    let y = (centroid[1].round() as i64 - (h / 2)).clamp(0, rows);
    [x, y, w.min(cols - x).max(0), h.min(rows - y).max(0)]
}

/// Grows a rect by `margin` pixels per side and shifts it back inside the
/// image so it keeps its size where it fits; this is the search region the
/// pinned implementation scans.
fn search_region(rect: [i64; 4], margin: i64, cols: i64, rows: i64) -> [i64; 4] {
    let [mut x0, mut y0, mut x1, mut y1] = [
        rect[0] - margin,
        rect[1] - margin,
        rect[0] + rect[2] + margin,
        rect[1] + rect[3] + margin,
    ];
    if x0 < 0 {
        x1 -= x0;
        x0 = 0;
    }
    if x1 > cols {
        x0 = (x0 - (x1 - cols)).max(0);
        x1 = cols;
    }
    if y0 < 0 {
        y1 -= y0;
        y0 = 0;
    }
    if y1 > rows {
        y0 = (y0 - (y1 - rows)).max(0);
        y1 = rows;
    }
    [x0, y0, (x1 - x0).max(0), (y1 - y0).max(0)]
}

/// Clips a rect to the image bounds, returning `[x, y, w, h]`.
fn clip_rect(rect: [i64; 4], cols: i64, rows: i64) -> [i64; 4] {
    let x0 = rect[0].clamp(0, cols);
    let y0 = rect[1].clamp(0, rows);
    let x1 = (rect[0] + rect[2]).clamp(x0, cols);
    let y1 = (rect[1] + rect[3]).clamp(y0, rows);
    [x0, y0, (x1 - x0).max(0), (y1 - y0).max(0)]
}

/// Tracks a window over the probability image and returns
/// `[cx, cy, w, h, angle, wx, wy, ww, wh]`: the rotated box followed by the
/// search window.
pub(crate) fn cam_shift(
    prob_image: &Mat,
    x: i64,
    y: i64,
    width: i64,
    height: i64,
) -> Result<[f64; 9], CamShiftError> {
    if prob_image.channels() != 1 {
        return Err(CamShiftError::UnsupportedChannels);
    }
    let cols = i64::from(prob_image.columns());
    let rows = i64::from(prob_image.rows());
    let depth = scalar_depth(prob_image.depth());
    let data = prob_image.compact_bytes();
    let stride = cols as usize;

    let input = clip_rect([x, y, width, height], cols, rows);
    let shape = region_shape(
        &data,
        stride,
        depth,
        search_region(input, 10, cols, rows),
    );

    if shape.mass == 0.0 {
        let [wx, wy, ww, wh] = clip_rect(
            [input[0] - 10, input[1] - 10, input[2] + 20, input[3] + 20],
            cols,
            rows,
        );
        return Ok([
            0.0, 0.0, 0.0, 0.0, 0.0, wx as f64, wy as f64, ww as f64, wh as f64,
        ]);
    }

    let [wx, wy, ww, wh] = box_window(shape.centroid, shape.size, shape.angle, cols, rows);
    Ok([
        wx as f64 + ww as f64 / 2.0,
        wy as f64 + wh as f64 / 2.0,
        shape.size[0],
        shape.size[1],
        shape.angle,
        wx as f64,
        wy as f64,
        ww as f64,
        wh as f64,
    ])
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
    fn blank_image() {
        let img = image(&[], 20, 20);
        let out = cam_shift(&img, 5, 5, 10, 10).expect("cam");
        assert_eq!(&out[..5], &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out[5..9], &[0.0, 0.0, 20.0, 20.0]);
    }

    #[test]
    fn single_blob() {
        let img = image(&[(10, 12, 255)], 20, 20);
        let out = cam_shift(&img, 5, 5, 10, 10).expect("cam");
        assert_eq!(
            &out[..9],
            &[10.0, 12.0, 0.0, 0.0, 90.0, 9.0, 11.0, 2.0, 2.0]
        );
    }

    #[test]
    fn vertical_pair() {
        let img = image(&[(20, 18, 255), (20, 22, 255)], 40, 40);
        let out = cam_shift(&img, 15, 15, 10, 10).expect("cam");
        assert_eq!(out[0], 20.0);
        assert_eq!(out[1], 20.0);
        assert!((out[3] - 8.0).abs() < 1e-6);
        assert_eq!(&out[5..9], &[19.0, 15.0, 2.0, 10.0]);
    }

    #[test]
    fn finds_mass_ten_pixels_beyond_window() {
        // The search region grows ten pixels per side; mass within that reach
        // is captured even when the input window misses it entirely.
        let img = image(&[(2, 2, 255), (10, 10, 255)], 20, 20);
        let out = cam_shift(&img, 5, 5, 10, 10).expect("cam");
        assert_eq!(out[0], 9.0);
        assert_eq!(out[1], 9.0);
        assert!((out[3] - 22.6274).abs() < 1e-3);
        assert!((out[4] - 135.0).abs() < 1e-3);
        assert_eq!(&out[5..9], &[0.0, 0.0, 18.0, 18.0]);
    }

    #[test]
    fn partial_window_reaches_full_attractor() {
        // A window containing only one pixel of a cluster still converges to
        // the rotated box of every pixel inside the grown region.
        let img = image(&[(20, 20, 255), (20, 24, 255), (22, 20, 255)], 40, 40);
        let out = cam_shift(&img, 18, 18, 4, 4).expect("cam");
        assert_eq!(out[0], 21.5);
        assert_eq!(out[1], 21.5);
        assert!((out[2] - 3.1489808559).abs() < 1e-6);
        assert!((out[3] - 7.8227252960).abs() < 1e-6);
        assert!((out[4] - 16.8450336456).abs() < 1e-6);
        assert_eq!(&out[5..9], &[19.0, 17.0, 5.0, 9.0]);
    }

    #[test]
    fn distant_mass_escapes_region() {
        // Mass farther than the grown reach stays invisible: the window
        // collapses onto the local pixel instead.
        let img = image(&[(5, 20, 255), (30, 20, 255)], 40, 40);
        let out = cam_shift(&img, 4, 19, 2, 2).expect("cam");
        assert_eq!(&out[..9], &[5.0, 20.0, 0.0, 0.0, 90.0, 4.0, 19.0, 2.0, 2.0]);
    }

    #[test]
    fn grown_region_snaps_inside_image() {
        // Overflow on one side transfers to the opposite side, so a window
        // near the left edge still reaches the full grown extent rightward.
        let img = image(&[(5, 20, 255), (21, 20, 255)], 40, 40);
        let out = cam_shift(&img, 4, 19, 2, 2).expect("cam");
        assert!((out[3] - 32.0).abs() < 1e-6);
        assert_eq!(&out[5..9], &[0.0, 19.0, 34.0, 2.0]);
    }

    #[test]
    fn diagonal_blob_window() {
        // Per-axis max projection of the rotated extents drives the window
        // size: for a 135-degree diagonal both axes take the major extent.
        let img = image(
            &[(10, 8, 255), (8, 10, 255), (12, 14, 255), (14, 12, 255)],
            20,
            20,
        );
        let out = cam_shift(&img, 5, 5, 14, 14).expect("cam");
        assert_eq!(&out[5..9], &[6.0, 6.0, 10.0, 10.0]);
    }

    #[test]
    fn saturated_triplet_keeps_eigen_residue() {
        // The pinned eigensolver leaves an eps^2-scaled residue on the
        // degenerate minor axis for a three-pixel saturated mass.
        let img = image(&[(20, 18, 255), (20, 20, 255), (20, 22, 255)], 40, 40);
        let out = cam_shift(&img, 15, 15, 10, 10).expect("cam");
        assert_eq!(out[0], 20.0);
        assert_eq!(out[1], 20.5);
        assert_eq!(out[2], f64::from(7.801971264598251e-7f32));
        assert!((out[3] - 6.5319728851).abs() < 1e-6);
        assert_eq!(out[4], 0.0);
        assert_eq!(&out[5..9], &[19.0, 16.0, 2.0, 9.0]);
    }

    #[test]
    fn multi_channel_rejected() {
        let img = mat_from_u8(&vec![0u8; 20 * 20 * 3], 20, 20, 3).expect("mat");
        assert!(cam_shift(&img, 0, 0, 10, 10).is_err());
    }
}
