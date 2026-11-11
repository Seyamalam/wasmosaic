//! Browser WebAssembly exports for point projection.

use wasm_bindgen::prelude::*;

use crate::{calib3d_project, mat::Mat};

/// Projects 3D points to the image plane, optionally writing the Jacobian.
#[wasm_bindgen(js_name = matProjectPointsInto)]
pub fn mat_project_points_into(
    object_points: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    image_points: &Mat,
    jacobian: &Mat,
    with_jacobian: i32,
    aspect_ratio: f64,
) -> Result<(), JsError> {
    calib3d_project::project_points(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        image_points,
        jacobian,
        with_jacobian != 0,
        aspect_ratio,
    )
    .map_err(JsError::from)
}
