use wasm_bindgen::prelude::*;

use crate::{imgproc_geom2, mat::Mat};

#[wasm_bindgen(js_name = matConvexityDefectsInto)]
pub fn mat_convexity_defects_into(
    contour: &Mat,
    hull: &Mat,
    defects: &Mat,
) -> Result<(), JsError> {
    imgproc_geom2::convexity_defects_into(contour, hull, defects).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matBlendLinearInto)]
pub fn mat_blend_linear_into(
    src1: &Mat,
    src2: &Mat,
    w1: &Mat,
    w2: &Mat,
    destination: &Mat,
) -> Result<(), JsError> {
    imgproc_geom2::blend_linear_into(src1, src2, w1, w2, destination).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matIntersectConvexConvexInto)]
pub fn mat_intersect_convex_convex_into(
    p1: &Mat,
    p2: &Mat,
    destination: &Mat,
) -> Result<f64, JsError> {
    imgproc_geom2::intersect_convex_convex_into(p1, p2, destination).map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matRotatedRectangleIntersectionInto)]
pub fn mat_rotated_rectangle_intersection_into(
    c1: f64,
    c2: f64,
    w1: f64,
    h1: f64,
    a1: f64,
    d1: f64,
    d2: f64,
    w2: f64,
    h2: f64,
    a2: f64,
    destination: &Mat,
) -> Result<i32, JsError> {
    imgproc_geom2::rotated_rectangle_intersection_into(c1, c2, w1, h1, a1, d1, d2, w2, h2, a2, destination)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matDistanceTransformWithLabelsInto)]
pub fn mat_distance_transform_with_labels_into(
    source: &Mat,
    distances: &Mat,
    labels: &Mat,
    mask_size: i32,
    dist_type: i32,
    label_type: i32,
) -> Result<(), JsError> {
    imgproc_geom2::distance_transform_with_labels_into(source, distances, labels, mask_size, dist_type, label_type)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMinEnclosingTriangleInto)]
pub fn mat_min_enclosing_triangle_into(
    points: &Mat,
    triangle: &Mat,
) -> Result<f64, JsError> {
    imgproc_geom2::min_enclosing_triangle_into(points, triangle).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matGoodFeaturesToTrackInto)]
pub fn mat_good_features_to_track_into(
    source: &Mat,
    corners: &Mat,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    use_harris: i32,
) -> Result<(), JsError> {
    imgproc_geom2::good_features_to_track_into(
        source,
        corners,
        max_corners.max(0) as usize,
        quality_level,
        min_distance,
        block_size,
        use_harris,
    )
    .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matDivSpectrumsInto)]
pub fn mat_div_spectrums_into(
    a: &Mat,
    b: &Mat,
    destination: &Mat,
    flags: i32,
    conj_b: bool,
) -> Result<(), JsError> {
    imgproc_geom2::div_spectrums_into(a, b, destination, flags, conj_b).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matHoughLinesInto)]
pub fn mat_hough_lines_into(
    source: &Mat,
    lines: &Mat,
    rho_step: f64,
    theta_step: f64,
    threshold: i32,
) -> Result<(), JsError> {
    imgproc_geom2::hough_lines_into(source, lines, rho_step, theta_step, threshold)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matHoughLinesPInto)]
pub fn mat_hough_lines_p_into(
    source: &Mat,
    lines: &Mat,
    rho_step: f64,
    theta_step: f64,
    threshold: i32,
    min_length: i32,
    max_gap: i32,
) -> Result<(), JsError> {
    imgproc_geom2::hough_lines_p_into(source, lines, rho_step, theta_step, threshold, max_gap, min_length)
        .map_err(JsError::from)
}
