use wasm_bindgen::prelude::*;

use crate::{imgproc_segment, mat::Mat};

#[wasm_bindgen(js_name = matConnectedComponentsInto)]
pub fn mat_connected_components_into(
    source: &Mat,
    labels: &Mat,
    connectivity: i32,
    #[wasm_bindgen(js_name = labelType)] label_type: i32,
) -> Result<i32, JsError> {
    imgproc_segment::connected_components_into(source, labels, connectivity, label_type)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matConnectedComponentsWithStatsInto)]
pub fn mat_connected_components_with_stats_into(
    source: &Mat,
    labels: &Mat,
    stats: &Mat,
    centroids: &Mat,
    connectivity: i32,
    #[wasm_bindgen(js_name = labelType)] label_type: i32,
) -> Result<i32, JsError> {
    imgproc_segment::connected_components_with_stats_into(
        source,
        labels,
        stats,
        centroids,
        connectivity,
        label_type,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matDistanceTransformInto)]
pub fn mat_distance_transform_into(
    source: &Mat,
    destination: &Mat,
    #[wasm_bindgen(js_name = distanceType)] distance_type: i32,
    #[wasm_bindgen(js_name = maskSize)] mask_size: i32,
) -> Result<(), JsError> {
    imgproc_segment::distance_transform_into(source, destination, distance_type, mask_size)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matFloodFillInto)]
pub fn mat_flood_fill_into(
    image: &Mat,
    mask: &Mat,
    #[wasm_bindgen(js_name = seedX)] seed_x: i32,
    #[wasm_bindgen(js_name = seedY)] seed_y: i32,
    #[wasm_bindgen(js_name = newValue)] new_value: f64,
    #[wasm_bindgen(js_name = loDiff)] lo_diff: f64,
    #[wasm_bindgen(js_name = upDiff)] up_diff: f64,
    flags: i32,
) -> Result<i32, JsError> {
    imgproc_segment::flood_fill_into(
        image, mask, seed_x, seed_y, new_value, lo_diff, up_diff, flags,
    )
    .map_err(JsError::from)
}
