use wasm_bindgen::prelude::*;

use crate::{imgproc_contours, mat::Mat, mat_vector::MatVector};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matFindContoursInto)]
pub fn mat_find_contours_into(
    source: &Mat,
    contours: &MatVector,
    hierarchy: &Mat,
    mode: i32,
    method: i32,
    #[wasm_bindgen(js_name = offsetX)] offset_x: i32,
    #[wasm_bindgen(js_name = offsetY)] offset_y: i32,
) -> Result<(), JsError> {
    imgproc_contours::find_contours_into(
        source, contours, hierarchy, mode, method, offset_x, offset_y,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matApproxPolyDPInto)]
pub fn mat_approx_poly_dp_into(
    source: &Mat,
    destination: &Mat,
    epsilon: f64,
    closed: bool,
) -> Result<(), JsError> {
    crate::imgproc_approx::approx_poly_dp_into(source, destination, epsilon, closed)
        .map_err(|error| JsError::new(&error))
}
