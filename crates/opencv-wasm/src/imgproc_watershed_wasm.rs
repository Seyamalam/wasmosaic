//! Browser WebAssembly exports for marker-controlled watershed segmentation.

use wasm_bindgen::prelude::*;

use crate::{imgproc_watershed, mat::Mat};

/// Floods the marker matrix in place from seed labels using image intensity.
#[wasm_bindgen(js_name = matWatershedInto)]
pub fn mat_watershed_into(image: &Mat, markers: &Mat) -> Result<(), JsError> {
    imgproc_watershed::watershed_into(image, markers).map_err(JsError::from)
}
