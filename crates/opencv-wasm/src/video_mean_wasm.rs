//! Browser WebAssembly export for `meanShift`.

use wasm_bindgen::prelude::*;

use crate::{mat::Mat, video_mean};

/// Tracks the window over the single-channel probability image and returns
/// `[iterations, x, y, width, height]` for the final clamped window.
#[wasm_bindgen(js_name = matMeanShift)]
pub fn mat_mean_shift(
    prob_image: &Mat,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    criteria_type: i32,
    max_count: i32,
    epsilon: f64,
) -> Result<Vec<f64>, JsError> {
    let (iterations, rect) = video_mean::mean_shift(
        prob_image,
        i64::from(x),
        i64::from(y),
        i64::from(width),
        i64::from(height),
        criteria_type,
        max_count,
        epsilon,
    )
    .map_err(JsError::from)?;
    Ok(vec![
        f64::from(iterations),
        rect[0] as f64,
        rect[1] as f64,
        rect[2] as f64,
        rect[3] as f64,
    ])
}
