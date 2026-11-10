use wasm_bindgen::prelude::*;

use crate::{imgproc_draw, mat::Mat};

fn color_from(c: &[f64]) -> Vec<f64> {
    let mut v = c.to_vec();
    v.resize(4, 0.0);
    v
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matLineInto)]
pub fn mat_line_into(
    image: &Mat,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: &[f64],
    thickness: i32,
    line_type: i32,
    shift: i32,
) -> Result<(), JsError> {
    imgproc_draw::line_into(image, i64::from(x0), i64::from(y0), i64::from(x1), i64::from(y1), &color_from(color), thickness, line_type, shift)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matRectangleInto)]
pub fn mat_rectangle_into(
    image: &Mat,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: &[f64],
    thickness: i32,
) -> Result<(), JsError> {
    imgproc_draw::rectangle_into(image, i64::from(x0), i64::from(y0), i64::from(x1), i64::from(y1), &color_from(color), thickness)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matCircleInto)]
pub fn mat_circle_into(
    image: &Mat,
    cx: i32,
    cy: i32,
    radius: i32,
    color: &[f64],
    thickness: i32,
) -> Result<(), JsError> {
    imgproc_draw::circle_into(image, i64::from(cx), i64::from(cy), i64::from(radius), &color_from(color), thickness)
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matEllipseInto)]
pub fn mat_ellipse_into(
    image: &Mat,
    cx: f64,
    cy: f64,
    ax: f64,
    ay: f64,
    angle_deg: f64,
    start_deg: f64,
    end_deg: f64,
    color: &[f64],
    thickness: i32,
) -> Result<(), JsError> {
    imgproc_draw::ellipse_into(image, cx, cy, ax, ay, angle_deg, start_deg, end_deg, &color_from(color), thickness)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matFillPolyInto)]
pub fn mat_fill_poly_into(
    image: &Mat,
    points: &[i32],
    closed: bool,
    color: &[f64],
) -> Result<(), JsError> {
    let pts: Vec<i64> = points.iter().map(|&v| i64::from(v)).collect();
    imgproc_draw::fill_poly_into(image, &pts, closed, &color_from(color)).map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matDrawMarkerInto)]
pub fn mat_draw_marker_into(
    image: &Mat,
    x: i32,
    y: i32,
    color: &[f64],
    marker_type: i32,
    marker_size: i32,
    thickness: i32,
) -> Result<(), JsError> {
    imgproc_draw::draw_marker_into(image, i64::from(x), i64::from(y), &color_from(color), marker_type, marker_size, thickness)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matPolylinesInto)]
pub fn mat_polylines_into(
    image: &Mat,
    points: &[i32],
    closed: bool,
    color: &[f64],
    thickness: i32,
) -> Result<(), JsError> {
    let pts: Vec<i64> = points.iter().map(|&v| i64::from(v)).collect();
    imgproc_draw::polylines_into(image, &pts, closed, &color_from(color), thickness)
        .map_err(JsError::from)
}
