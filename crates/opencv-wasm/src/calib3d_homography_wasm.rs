//! Browser WebAssembly export for `findHomography`.

use wasm_bindgen::prelude::*;

use crate::{
    calib3d_homography,
    core_norm::read_scalar,
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

/// Extracts correspondences from an `N x 1` two-channel or `N x 2`
/// single-channel point matrix at any scalar depth.
fn decode_points(matrix: &Mat) -> Result<Vec<[f64; 2]>, JsError> {
    let depth = scalar_depth(matrix.depth());
    let (rows, columns, channels) = (matrix.rows(), matrix.columns(), matrix.channels());
    let count = if channels == 2 && columns == 1 {
        rows as usize
    } else if channels == 1 && columns == 2 {
        rows as usize
    } else {
        return Err(JsError::new(&format!(
            "point input requires an Nx1 two-channel or Nx2 single-channel matrix; received {rows}x{columns} with {channels} channels"
        )));
    };
    if count == 0 {
        return Err(JsError::new("point input must not be empty"));
    }
    let data = matrix.compact_bytes();
    let mut points = Vec::with_capacity(count);
    for index in 0..count {
        points.push([
            read_scalar(&data, index * 2, depth),
            read_scalar(&data, index * 2 + 1, depth),
        ]);
    }
    Ok(points)
}

/// Estimates the 3x3 homography between two point sets.
///
/// Returns the estimated 3x3 CV_64F matrix, or an empty matrix when no valid
/// hypothesis exists. When `mask` is supplied it is rebound to an `N x 1`
/// CV_8U mask of the winning hypothesis's consensus set.
///
/// # Errors
/// Returns an error for empty or malformed point input, fewer than four
/// correspondences, a method value other than 0, LMeDS (4), RANSAC (8), or
/// RHO (16), or a failed mask write.
#[wasm_bindgen(js_name = matFindHomography)]
pub fn mat_find_homography(
    from: &Mat,
    to: &Mat,
    method: i32,
    ransac_reprojection_threshold: f64,
    mask: &Mat,
    max_iters: i32,
    confidence: f64,
) -> Result<Mat, JsError> {
    let from_points = decode_points(from)?;
    let to_points = decode_points(to)?;
    if to_points.len() != from_points.len() {
        return Err(JsError::new(&format!(
            "point sets must contain the same number of correspondences; received {} and {}",
            from_points.len(),
            to_points.len()
        )));
    }
    let estimate = calib3d_homography::estimate_homography(
        &from_points,
        &to_points,
        method,
        ransac_reprojection_threshold,
        max_iters,
        confidence,
    )
    .map_err(|error| JsError::new(&error.to_string()))?;
    let mask_length = estimate.inliers.len() as u32;
    mask
        .write_output(estimate.inliers, mask_length, 1, 1, MatDepth::U8)
        .map_err(JsError::from)?;
    let Some(model) = estimate.model else {
        return Ok(crate::mat::mat_empty());
    };
    let bytes = model
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<u8>>();
    Mat::from_owned_bytes(bytes, 3, 3, 1, MatDepth::F64).map_err(JsError::from)
}
