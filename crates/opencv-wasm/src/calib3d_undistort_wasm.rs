//! Browser WebAssembly exports for undistortion.

use wasm_bindgen::prelude::*;

use crate::{calib3d_undistort, mat::Mat};

/// Computes dense backward maps for image undistortion/rectification.
#[wasm_bindgen(js_name = matInitUndistortRectifyMap)]
pub fn mat_init_undistort_rectify_map(
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rotation: &Mat,
    new_camera_matrix: &Mat,
    width: i32,
    height: i32,
    m1type: i32,
    map1: &Mat,
    map2: &Mat,
) -> Result<(), JsError> {
    if width < 0 || height < 0 {
        return Err(JsError::new("map size must be positive"));
    }
    calib3d_undistort::init_undistort_rectify_map_into(
        camera_matrix,
        dist_coeffs,
        rotation,
        new_camera_matrix,
        width as u32,
        height as u32,
        m1type,
        map1,
        map2,
    )
    .map_err(JsError::from)
}

/// Removes lens distortion from an image.
#[wasm_bindgen(js_name = matUndistortInto)]
pub fn mat_undistort_into(
    source: &Mat,
    destination: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    new_camera_matrix: &Mat,
) -> Result<(), JsError> {
    calib3d_undistort::undistort_into(
        source,
        destination,
        camera_matrix,
        dist_coeffs,
        new_camera_matrix,
    )
    .map_err(JsError::from)
}
