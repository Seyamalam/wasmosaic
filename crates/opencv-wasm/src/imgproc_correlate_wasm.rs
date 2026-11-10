use wasm_bindgen::prelude::*;

use crate::{imgproc_correlate, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matFilter2DInto)]
pub fn mat_filter_2d_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = destinationDepth)] destination_depth: i32,
    kernel: &Mat,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    delta: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::filter_2d_into(
        source,
        destination,
        destination_depth,
        kernel,
        anchor_x,
        anchor_y,
        delta,
        border_type,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matSepFilter2DInto)]
pub fn mat_sep_filter_2d_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = destinationDepth)] destination_depth: i32,
    #[wasm_bindgen(js_name = kernelX)] kernel_x: &Mat,
    #[wasm_bindgen(js_name = kernelY)] kernel_y: &Mat,
    #[wasm_bindgen(js_name = anchorX)] anchor_x: i32,
    #[wasm_bindgen(js_name = anchorY)] anchor_y: i32,
    delta: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::sep_filter_2d_into(
        source,
        destination,
        destination_depth,
        kernel_x,
        kernel_y,
        anchor_x,
        anchor_y,
        delta,
        border_type,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matSqrBoxFilterInto)]
pub fn mat_sqr_box_filter_into(
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
    imgproc_correlate::sqr_box_filter_into(
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

#[wasm_bindgen(js_name = matStackBlurInto)]
pub fn mat_stack_blur_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = kernelWidth)] kernel_width: i32,
    #[wasm_bindgen(js_name = kernelHeight)] kernel_height: i32,
) -> Result<(), JsError> {
    imgproc_correlate::stack_blur_into(source, destination, kernel_width, kernel_height)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matBilateralFilterInto)]
pub fn mat_bilateral_filter_into(
    source: &Mat,
    destination: &Mat,
    diameter: i32,
    #[wasm_bindgen(js_name = sigmaColor)] sigma_color: f64,
    #[wasm_bindgen(js_name = sigmaSpace)] sigma_space: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::bilateral_filter_into(
        source,
        destination,
        diameter,
        sigma_color,
        sigma_space,
        border_type,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matPyrDownInto)]
pub fn mat_pyr_down_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = dstRows)] dst_rows: i32,
    #[wasm_bindgen(js_name = dstCols)] dst_cols: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::pyr_down_into(source, destination, dst_rows, dst_cols, border_type)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matPyrUpInto)]
pub fn mat_pyr_up_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = dstRows)] dst_rows: i32,
    #[wasm_bindgen(js_name = dstCols)] dst_cols: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::pyr_up_into(source, destination, dst_rows, dst_cols, border_type)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matLaplacianInto)]
pub fn mat_laplacian_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = destinationDepth)] destination_depth: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::laplacian_into(
        source,
        destination,
        destination_depth,
        ksize,
        scale,
        delta,
        border_type,
    )
    .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matScharrInto)]
pub fn mat_scharr_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = destinationDepth)] destination_depth: i32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::scharr_into(
        source,
        destination,
        destination_depth,
        dx,
        dy,
        scale,
        delta,
        border_type,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matSpatialGradientInto)]
pub fn mat_spatial_gradient_into(
    source: &Mat,
    #[wasm_bindgen(js_name = destinationX)] destination_x: &Mat,
    #[wasm_bindgen(js_name = destinationY)] destination_y: &Mat,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::spatial_gradient_into(source, destination_x, destination_y, border_type)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matPreCornerDetectInto)]
pub fn mat_pre_corner_detect_into(
    source: &Mat,
    destination: &Mat,
    ksize: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::pre_corner_detect_into(source, destination, ksize, border_type)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matCornerHarrisInto)]
pub fn mat_corner_harris_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = blockSize)] block_size: i32,
    ksize: i32,
    k: f64,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::corner_harris_into(source, destination, block_size, ksize, k, border_type)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matCornerMinEigenValInto)]
pub fn mat_corner_min_eigen_val_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = blockSize)] block_size: i32,
    ksize: i32,
    #[wasm_bindgen(js_name = borderType)] border_type: i32,
) -> Result<(), JsError> {
    imgproc_correlate::corner_min_eigen_val_into(
        source,
        destination,
        block_size,
        ksize,
        border_type,
    )
    .map_err(JsError::from)
}
