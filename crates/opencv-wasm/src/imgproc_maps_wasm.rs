use wasm_bindgen::prelude::*;

use crate::{imgproc_maps, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matRemapInto)]
pub fn mat_remap_into(
    source: &Mat,
    destination: &Mat,
    map1: &Mat,
    map2: &Mat,
    interpolation: i32,
    #[wasm_bindgen(js_name = borderValue)] border_value: f64,
) -> Result<(), JsError> {
    imgproc_maps::remap_into(source, destination, map1, map2, interpolation, border_value)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matConvertMapsInto)]
pub fn mat_convert_maps_into(
    map1: &Mat,
    map2: &Mat,
    dst1: &Mat,
    dst2: &Mat,
) -> Result<(), JsError> {
    imgproc_maps::convert_maps_into(map1, map2, dst1, dst2).map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matGetRectSubPixInto)]
pub fn mat_get_rect_sub_pix_into(
    source: &Mat,
    destination: &Mat,
    width: i32,
    height: i32,
    #[wasm_bindgen(js_name = centerX)] center_x: f64,
    #[wasm_bindgen(js_name = centerY)] center_y: f64,
) -> Result<(), JsError> {
    imgproc_maps::get_rect_sub_pix_into(source, destination, width, height, center_x, center_y)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matDemosaicingInto)]
pub fn mat_demosaicing_into(
    source: &Mat,
    destination: &Mat,
    code: i32,
) -> Result<(), JsError> {
    imgproc_maps::demosaicing_into(source, destination, code).map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matCalcHistInto)]
pub fn mat_calc_hist_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = histSize)] hist_size: i32,
    #[wasm_bindgen(js_name = rangeLow)] range_low: f64,
    #[wasm_bindgen(js_name = rangeHigh)] range_high: f64,
    accumulate: bool,
) -> Result<(), JsError> {
    imgproc_maps::calc_hist_into(source, destination, hist_size, range_low, range_high, accumulate)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matCompareHist)]
pub fn mat_compare_hist(h1: &Mat, h2: &Mat, method: i32) -> Result<f64, JsError> {
    imgproc_maps::compare_hist_into(h1, h2, method).map_err(JsError::from)
}
