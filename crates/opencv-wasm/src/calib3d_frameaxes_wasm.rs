//! Browser WebAssembly export for `drawFrameAxes`.

use wasm_bindgen::prelude::*;

use crate::{calib3d_frameaxes, mat::Mat};

/// Draws the projected world axes on `image`: x in red, y in green, z in
/// blue (BGR), each as an 8-connected line of the given thickness.
#[wasm_bindgen(js_name = matDrawFrameAxesInto)]
pub fn mat_draw_frame_axes_into(
    image: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    length: f64,
    thickness: i32,
) -> Result<(), JsError> {
    calib3d_frameaxes::draw_frame_axes_into(
        image,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        length,
        thickness,
    )
    .map_err(JsError::from)
}
