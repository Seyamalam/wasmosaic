//! Browser WebAssembly export for `CamShift`.

use wasm_bindgen::prelude::*;

use crate::{mat::Mat, video_cam};

/// Tracks the window over the single-channel probability image and returns
/// `[cx, cy, w, h, angle, x, y, width, height]`: the rotated box followed by
/// the converged search window.
#[wasm_bindgen(js_name = matCamShift)]
pub fn mat_cam_shift(
    prob_image: &Mat,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    _criteria_type: i32,
    _max_count: i32,
    _epsilon: f64,
) -> Result<Vec<f64>, JsError> {
    Ok(video_cam::cam_shift(
        prob_image,
        i64::from(x),
        i64::from(y),
        i64::from(width),
        i64::from(height),
    )
    .map_err(JsError::from)?
    .to_vec())
}
