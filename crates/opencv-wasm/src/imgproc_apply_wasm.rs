use wasm_bindgen::prelude::*;

use crate::{imgproc_apply, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matAdaptiveThresholdInto)]
pub fn mat_adaptive_threshold_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = maxValue)] max_value: f64,
    #[wasm_bindgen(js_name = adaptiveMethod)] adaptive_method: i32,
    #[wasm_bindgen(js_name = thresholdType)] threshold_type: i32,
    #[wasm_bindgen(js_name = blockSize)] block_size: i32,
    c: f64,
) -> Result<(), JsError> {
    imgproc_apply::adaptive_threshold_into(
        source,
        destination,
        max_value,
        adaptive_method,
        threshold_type,
        block_size,
        c,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matApplyColorMapInto)]
pub fn mat_apply_color_map_into(
    source: &Mat,
    destination: &Mat,
    colormap: i32,
) -> Result<(), JsError> {
    imgproc_apply::apply_color_map_into(source, destination, colormap).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matConvexHullInto)]
pub fn mat_convex_hull_into(
    points: &Mat,
    destination: &Mat,
    clockwise: bool,
    #[wasm_bindgen(js_name = returnPoints)] return_points: bool,
) -> Result<(), JsError> {
    imgproc_apply::convex_hull_into(points, destination, clockwise, return_points)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMinAreaRect)]
pub fn mat_min_area_rect(points: &Mat) -> Result<Vec<f64>, JsError> {
    imgproc_apply::min_area_rect_points(points)
        .map(|rect| rect.to_vec())
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMinEnclosingCircle)]
pub fn mat_min_enclosing_circle(points: &Mat) -> Result<Vec<f64>, JsError> {
    imgproc_apply::min_enclosing_circle_points(points)
        .map(|circle| circle.to_vec())
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matFitEllipse)]
pub fn mat_fit_ellipse(points: &Mat) -> Result<Vec<f64>, JsError> {
    imgproc_apply::fit_ellipse_points(points)
        .map(|ellipse| ellipse.to_vec())
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matFitLineInto)]
pub fn mat_fit_line_into(
    points: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = distType)] dist_type: i32,
    param: f64,
    reps: f64,
    aeps: f64,
) -> Result<(), JsError> {
    imgproc_apply::fit_line_points(points, destination, dist_type, param, reps, aeps)
        .map_err(JsError::from)
}
