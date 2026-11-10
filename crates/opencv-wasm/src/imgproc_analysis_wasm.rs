use wasm_bindgen::prelude::*;

use crate::{imgproc_analysis, mat::Mat};

#[wasm_bindgen(js_name = matIntegralInto)]
pub fn mat_integral_into(
    source: &Mat,
    sum: &Mat,
    #[wasm_bindgen(js_name = sumDepth)] sum_depth: i32,
    #[wasm_bindgen(js_name = squaredSumDepth)] _squared_sum_depth: i32,
) -> Result<(), JsError> {
    imgproc_analysis::integral_into(source, sum, sum_depth).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matIntegral2Into)]
pub fn mat_integral2_into(
    source: &Mat,
    sum: &Mat,
    #[wasm_bindgen(js_name = squaredSum)] squared_sum: &Mat,
    #[wasm_bindgen(js_name = sumDepth)] sum_depth: i32,
    #[wasm_bindgen(js_name = squaredSumDepth)] squared_sum_depth: i32,
) -> Result<(), JsError> {
    imgproc_analysis::integral2_into(source, sum, squared_sum, sum_depth, squared_sum_depth)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMoments)]
pub fn mat_moments(source: &Mat, binary: bool) -> Result<Box<[f64]>, JsError> {
    let moments = imgproc_analysis::raster_moments(source, binary).map_err(JsError::from)?;
    Ok(moments.values.into())
}

#[wasm_bindgen(js_name = matHuMomentsInto)]
pub fn mat_hu_moments_into(moments: &[f64], destination: &Mat) -> Result<(), JsError> {
    imgproc_analysis::hu_moments_into(moments, destination).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMatchShapes)]
pub fn mat_match_shapes(
    first: &Mat,
    second: &Mat,
    method: i32,
    parameter: f64,
) -> Result<f64, JsError> {
    imgproc_analysis::match_shapes(first, second, method, parameter).map_err(JsError::from)
}

#[wasm_bindgen(js_name = fontScaleFromHeight)]
pub fn font_scale_from_height(
    #[wasm_bindgen(js_name = fontFace)] font_face: i32,
    #[wasm_bindgen(js_name = pixelHeight)] pixel_height: i32,
    thickness: i32,
) -> Result<f64, JsError> {
    imgproc_analysis::font_scale_from_height(font_face, pixel_height, thickness)
        .map_err(JsError::from)
}
