use wasm_bindgen::prelude::*;

use crate::{imgproc_window, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matErodeInto)]
pub fn mat_erode_into(
    source: &Mat,
    destination: &Mat,
    kernel: &Mat,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    iterations: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
    #[wasm_bindgen(js_name = borderValue)] border_value: &[f64],
    #[wasm_bindgen(js_name = defaultBorderValue)] default_border_value: bool,
) -> Result<(), JsError> {
    imgproc_window::morphology_into(
        source,
        destination,
        kernel,
        anchor_x,
        anchor_y,
        iterations,
        border_type,
        border_value,
        default_border_value,
        false,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matDilateInto)]
pub fn mat_dilate_into(
    source: &Mat,
    destination: &Mat,
    kernel: &Mat,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    iterations: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
    #[wasm_bindgen(js_name = borderValue)] border_value: &[f64],
    #[wasm_bindgen(js_name = defaultBorderValue)] default_border_value: bool,
) -> Result<(), JsError> {
    imgproc_window::morphology_into(
        source,
        destination,
        kernel,
        anchor_x,
        anchor_y,
        iterations,
        border_type,
        border_value,
        default_border_value,
        true,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matBlurInto)]
pub fn mat_blur_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = kernelWidth)] kernel_width: i32,
    #[wasm_bindgen(js_name = kernelHeight)] kernel_height: i32,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_window::blur_into(
        source,
        destination,
        kernel_width,
        kernel_height,
        anchor_x,
        anchor_y,
        border_type,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matBoxFilterInto)]
pub fn mat_box_filter_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = destinationDepth)] destination_depth: i32,
    #[wasm_bindgen(js_name = kernelWidth)] kernel_width: i32,
    #[wasm_bindgen(js_name = kernelHeight)] kernel_height: i32,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    normalize: bool,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_window::box_filter_into(
        source,
        destination,
        destination_depth,
        kernel_width,
        kernel_height,
        anchor_x,
        anchor_y,
        normalize,
        border_type,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMedianBlurInto)]
pub fn mat_median_blur_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = kernelSize)] kernel_size: i32,
) -> Result<(), JsError> {
    imgproc_window::median_blur_into(source, destination, kernel_size).map_err(JsError::from)
}
