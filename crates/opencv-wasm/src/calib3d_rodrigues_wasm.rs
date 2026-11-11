//! Browser WebAssembly exports for rotation vector and matrix conversion.

use wasm_bindgen::prelude::*;

use crate::{calib3d_rodrigues, mat::Mat};

/// Converts a rotation vector to a rotation matrix, or the reverse.
#[wasm_bindgen(js_name = matRodriguesInto)]
pub fn mat_rodrigues_into(
    source: &Mat,
    destination: &Mat,
    jacobian: &Mat,
    with_jacobian: i32,
) -> Result<(), JsError> {
    calib3d_rodrigues::rodrigues_into(source, destination, jacobian, with_jacobian != 0)
        .map_err(JsError::from)
}
