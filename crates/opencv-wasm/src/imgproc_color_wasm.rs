use wasm_bindgen::prelude::*;

use crate::{imgproc_color, mat::Mat};

/// Converts a matrix between the browser-critical grayscale and RGB/BGR channel layouts.
///
/// Codes 0 through 11 match the pinned OpenCV.js color conversion constants on U8, U16, and F32
/// matrices, including mutable destination replacement and exact in-place use. Codes 32 through
/// 39 and 82 through 85 cover the XYZ, YCrCb, and YUV families on U8 matrices.
///
/// # Errors
///
/// Returns an error for unsupported codes, depths, source channel counts, or destination counts.
#[wasm_bindgen(js_name = matCvtColorInto)]
pub fn mat_cvt_color_into(
    source: &Mat,
    destination: &Mat,
    code: i32,
    #[wasm_bindgen(js_name = destinationChannels)] destination_channels: i32,
) -> Result<(), JsError> {
    imgproc_color::cvt_color_into(source, destination, code, destination_channels)
        .map_err(JsError::from)
}
