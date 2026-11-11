//! Browser WebAssembly exports for camera-matrix helpers.

use wasm_bindgen::prelude::*;

use crate::{calib3d_camera, mat::Mat};

/// Copies a camera matrix to F64, optionally re-centering the principal point.
#[wasm_bindgen(js_name = matDefaultNewCameraMatrix)]
pub fn mat_default_new_camera_matrix(
    camera_matrix: &Mat,
    destination: &Mat,
    image_width: i32,
    image_height: i32,
    center_principal_point: i32,
) -> Result<(), JsError> {
    calib3d_camera::default_new_camera_matrix(
        camera_matrix,
        destination,
        image_width,
        image_height,
        center_principal_point != 0,
    )
    .map_err(JsError::from)
}
