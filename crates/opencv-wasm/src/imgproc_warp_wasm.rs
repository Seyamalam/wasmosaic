use wasm_bindgen::prelude::*;

use crate::{imgproc_warp, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matWarpAffineInto)]
pub fn mat_warp_affine_into(
    source: &Mat,
    destination: &Mat,
    transform: &Mat,
    width: i32,
    height: i32,
    flags: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
    #[wasm_bindgen(js_name = borderValue)] border_value: &[f64],
) -> Result<(), JsError> {
    imgproc_warp::warp_affine_into(
        source,
        destination,
        transform,
        width,
        height,
        flags,
        border_type,
        border_value,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matWarpPerspectiveInto)]
pub fn mat_warp_perspective_into(
    source: &Mat,
    destination: &Mat,
    transform: &Mat,
    width: i32,
    height: i32,
    flags: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
    #[wasm_bindgen(js_name = borderValue)] border_value: &[f64],
) -> Result<(), JsError> {
    crate::imgproc_perspective::warp_perspective_into(
        source,
        destination,
        transform,
        width,
        height,
        flags,
        border_type,
        border_value,
    )
    .map_err(JsError::from)
}
