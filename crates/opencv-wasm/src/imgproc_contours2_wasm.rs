use wasm_bindgen::prelude::*;

use crate::{
    imgproc_contours,
    mat::Mat,
    mat_vector::MatVector,
};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matDrawContoursInto)]
pub fn mat_draw_contours_into(
    image: &Mat,
    contours: &MatVector,
    contour_index: i32,
    color: &[f64],
    thickness: i32,
    line_type: i32,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), JsError> {
    imgproc_contours::draw_contours_into(image, contours, contour_index, color, thickness, line_type, offset_x, offset_y)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matApproxPolyNInto)]
pub fn mat_approx_poly_n_into(
    curve: &Mat,
    approximation: &Mat,
    nsides: i32,
    closed: bool,
) -> Result<(), JsError> {
    imgproc_contours::approx_poly_n_into(curve, approximation, nsides.max(0) as usize, closed)
        .map_err(JsError::from)
}
