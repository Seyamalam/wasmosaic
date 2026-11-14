//! Browser WebAssembly exports for fisheye calibration functions.

use wasm_bindgen::prelude::*;

use crate::{calib3d_fisheye, mat::Mat};

/// Projects 3D points through the equidistant fisheye model, optionally
/// writing the 2N x 15 Jacobian.
#[wasm_bindgen(js_name = matFisheyeProjectPointsInto)]
pub fn mat_fisheye_project_points_into(
    object_points: &Mat,
    image_points: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    alpha: f64,
    jacobian: &Mat,
    with_jacobian: i32,
) -> Result<(), JsError> {
    calib3d_fisheye::fisheye_project_points_into(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        alpha,
        jacobian,
        with_jacobian != 0,
    )
    .map_err(JsError::from)
}

/// Computes dense backward maps for fisheye undistortion/rectification.
#[wasm_bindgen(js_name = matFisheyeInitUndistortRectifyMap)]
pub fn mat_fisheye_init_undistort_rectify_map(
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
    calib3d_fisheye::fisheye_init_undistort_rectify_map_into(
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
