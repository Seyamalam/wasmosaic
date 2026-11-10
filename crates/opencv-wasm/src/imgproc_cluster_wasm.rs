use wasm_bindgen::prelude::*;

use crate::{imgproc_cluster, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matKmeansInto)]
pub fn mat_kmeans_into(
    samples: &Mat,
    k: i32,
    labels: &Mat,
    centers: &Mat,
    #[wasm_bindgen(js_name = maxIters)] max_iters: i32,
    epsilon: f64,
    attempts: i32,
) -> Result<f64, JsError> {
    imgproc_cluster::kmeans_into(samples, k, labels, centers, max_iters, epsilon, attempts)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matDftInto)]
pub fn mat_dft_into(source: &Mat, destination: &Mat, flags: i32) -> Result<(), JsError> {
    imgproc_cluster::dft_into(source, destination, flags).map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matCalcBackProjectInto)]
pub fn mat_calc_back_project_into(
    source: &Mat,
    channels: i32,
    hist: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = rangeLow)] range_low: f64,
    #[wasm_bindgen(js_name = rangeHigh)] range_high: f64,
    scale: f64,
) -> Result<(), JsError> {
    imgproc_cluster::calc_back_project_into(source, channels, hist, destination, range_low, range_high, scale)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matWarpPolarInto)]
pub fn mat_warp_polar_into(
    source: &Mat,
    destination: &Mat,
    width: i32,
    height: i32,
    #[wasm_bindgen(js_name = centerX)] center_x: f64,
    #[wasm_bindgen(js_name = centerY)] center_y: f64,
    #[wasm_bindgen(js_name = maxRadius)] max_radius: f64,
    flags: i32,
) -> Result<(), JsError> {
    imgproc_cluster::warp_polar_into(source, destination, width, height, center_x, center_y, max_radius, flags)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matGroupRectanglesInto)]
pub fn mat_group_rectangles_into(
    rects: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = groupThreshold)] group_threshold: i32,
    eps: f64,
) -> Result<i32, JsError> {
    imgproc_cluster::group_rectangles_into(rects, destination, group_threshold, eps)
        .map_err(JsError::from)
}
