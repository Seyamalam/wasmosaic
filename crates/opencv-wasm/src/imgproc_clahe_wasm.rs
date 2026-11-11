//! Browser WebAssembly handle for CLAHE configuration and application.

use wasm_bindgen::prelude::*;

use crate::{imgproc_clahe, mat::Mat};

/// Owned CLAHE processor mirroring the upstream algorithm-object contract.
#[wasm_bindgen(js_name = CLAHEConfig)]
pub struct Clahe {
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
}

#[wasm_bindgen(js_class = CLAHEConfig)]
impl Clahe {
    #[wasm_bindgen(js_name = create)]
    pub fn create(clip_limit: Option<f64>, tiles_x: Option<u32>, tiles_y: Option<u32>) -> Clahe {
        Self {
            clip_limit: clip_limit.unwrap_or(40.0),
            tiles_x: tiles_x.unwrap_or(8).max(1),
            tiles_y: tiles_y.unwrap_or(8).max(1),
        }
    }

    #[wasm_bindgen(js_name = apply)]
    pub fn apply(&self, source: &Mat, destination: &Mat) -> Result<(), JsError> {
        imgproc_clahe::clahe_apply_into(
            source,
            destination,
            self.clip_limit,
            self.tiles_x,
            self.tiles_y,
        )
        .map_err(JsError::from)
    }

    #[wasm_bindgen(js_name = getClipLimit)]
    pub fn get_clip_limit(&self) -> f64 {
        self.clip_limit
    }

    #[wasm_bindgen(js_name = getTilesGridSize)]
    pub fn get_tiles_grid_size(&self) -> Vec<u32> {
        vec![self.tiles_x, self.tiles_y]
    }

    #[wasm_bindgen(js_name = setClipLimit)]
    pub fn set_clip_limit(&mut self, clip_limit: f64) {
        self.clip_limit = clip_limit;
    }

    #[wasm_bindgen(js_name = setTilesGridSize)]
    pub fn set_tiles_grid_size(&mut self, width: u32, height: u32) {
        self.tiles_x = width.max(1);
        self.tiles_y = height.max(1);
    }

    #[wasm_bindgen(js_name = collectGarbage)]
    #[allow(clippy::unused_self)]
    pub fn collect_garbage(&self) {}
}
