//! Browser WebAssembly bindings for small image-processing helpers.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    imgproc_helpers::{
        HelperError, Point, Rect, clip_line, ellipse_poly, hanning_window, structuring_element,
    },
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
enum HelperWasmError {
    UnsupportedWindowDepth(MatDepth),
    UnsupportedWindowType(i32),
    Kernel(HelperError),
    Matrix(MatError),
}

impl fmt::Display for HelperWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedWindowDepth(depth) => write!(
                formatter,
                "Hanning window depth must be F32 or F64; received {depth:?}"
            ),
            Self::UnsupportedWindowType(code) => write!(
                formatter,
                "Hanning window type must be CV_32F (5) or CV_64F (6); received {code}"
            ),
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for HelperWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::UnsupportedWindowDepth(_) | Self::UnsupportedWindowType(_) => None,
        }
    }
}

impl From<HelperError> for HelperWasmError {
    fn from(error: HelperError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for HelperWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Creates a single-channel U8 structuring kernel.
///
/// Supported shape codes are 0 for rectangle, 1 for cross, 2 for ellipse, and 3 for diamond. An anchor
/// coordinate of -1 selects that dimension's center. The anchor changes the cross intersection;
/// rectangle and ellipse geometry remain centered. This partial binding accepts positive kernel
/// dimensions whose compact U8 result fits the WASM matrix limit.
///
/// # Errors
///
/// Returns an error for an unsupported shape, empty size, invalid anchor, or size overflow.
#[wasm_bindgen(js_name = getStructuringElement)]
pub fn get_structuring_element(
    shape: i32,
    columns: u32,
    rows: u32,
    anchor_x: i32,
    anchor_y: i32,
) -> Result<Mat, JsError> {
    let element = structuring_element(shape, columns, rows, anchor_x, anchor_y)
        .map_err(HelperWasmError::from)?;
    Mat::from_owned_bytes(
        element.bytes,
        element.rows,
        element.columns,
        1,
        MatDepth::U8,
    )
    .map_err(HelperWasmError::from)
    .map_err(JsError::from)
}

/// Creates a two-dimensional Hanning window with F32 or F64 elements.
///
/// The result is the outer product of the non-negative sine weights that are the square roots of
/// the one-dimensional Hann coefficients. Both dimensions must be at least two. This partial
/// binding supports only F32 and F64 output and caps allocation at the conservative F64 WASM
/// matrix limit.
///
/// # Errors
///
/// Returns an error for dimensions below two, another depth, or size overflow.
#[wasm_bindgen(js_name = createHanningWindow)]
pub fn create_hanning_window(columns: u32, rows: u32, depth: MatDepth) -> Result<Mat, JsError> {
    create_hanning_window_mat(columns, rows, depth).map_err(JsError::from)
}

/// Writes a two-dimensional Hanning window into a mutable destination.
///
/// Dimensions use signed 32-bit binding values and must each be at least two. Type code 5 selects
/// F32 and type code 6 selects F64. A destination with another layout is rebound.
///
/// # Errors
/// Returns an error for dimensions below two, another type code, size overflow, or an invalid
/// destination write.
#[wasm_bindgen(js_name = createHanningWindowInto)]
pub fn create_hanning_window_into(
    destination: &Mat,
    columns: i32,
    rows: i32,
    depth: i32,
) -> Result<(), JsError> {
    create_hanning_window_into_adapter(destination, columns, rows, depth).map_err(JsError::from)
}

fn create_hanning_window_into_adapter(
    destination: &Mat,
    columns: i32,
    rows: i32,
    depth: i32,
) -> Result<(), HelperWasmError> {
    let columns = u32::try_from(columns).map_err(|_| HelperError::WindowTooSmall)?;
    let rows = u32::try_from(rows).map_err(|_| HelperError::WindowTooSmall)?;
    let depth = match depth {
        5 => MatDepth::F32,
        6 => MatDepth::F64,
        code => return Err(HelperWasmError::UnsupportedWindowType(code)),
    };
    let output = create_hanning_window_mat(columns, rows, depth)?;
    destination.write_output(
        output.compact_bytes(),
        output.rows(),
        output.columns(),
        output.channels(),
        output.depth(),
    )?;
    Ok(())
}

fn create_hanning_window_mat(
    columns: u32,
    rows: u32,
    depth: MatDepth,
) -> Result<Mat, HelperWasmError> {
    if !matches!(depth, MatDepth::F32 | MatDepth::F64) {
        return Err(HelperWasmError::UnsupportedWindowDepth(depth));
    }
    let values = hanning_window(columns, rows)?;
    let mut bytes = Vec::with_capacity(values.len() * depth.byte_width());
    match depth {
        MatDepth::F32 => {
            for value in values {
                bytes.extend_from_slice(&narrow_window_value(value).to_ne_bytes());
            }
        }
        MatDepth::F64 => {
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        _ => unreachable!("window depth was validated"),
    }
    Mat::from_owned_bytes(bytes, rows, columns, 1, depth).map_err(HelperWasmError::from)
}

#[allow(clippy::cast_possible_truncation)]
fn narrow_window_value(value: f64) -> f32 {
    value as f32
}

/// Samples an integer polyline along a rotated ellipse arc.
///
/// The returned `Int32Array` stores interleaved x and y coordinates. Axes must be non-negative,
/// arc bounds must satisfy `0 <= start <= end <= 360`, and the angular step must be in 1 through
/// 180. The exact end angle is always sampled. Consecutive points that round to the same integer
/// coordinate are collapsed. This partial binding does not normalize or swap arc bounds.
///
/// # Errors
///
/// Returns an error for invalid axes, arc bounds, angular step, or generated coordinate overflow.
#[wasm_bindgen(js_name = ellipse2Poly)]
#[allow(clippy::too_many_arguments)]
pub fn ellipse_2_poly(
    center_x: i32,
    center_y: i32,
    axis_x: i32,
    axis_y: i32,
    rotation_degrees: i32,
    arc_start: i32,
    arc_end: i32,
    delta: i32,
) -> Result<Vec<i32>, JsError> {
    let points = ellipse_poly(
        Point {
            x: center_x,
            y: center_y,
        },
        axis_x,
        axis_y,
        rotation_degrees,
        arc_start,
        arc_end,
        delta,
    )
    .map_err(HelperWasmError::from)?;
    Ok(flatten_points(&points))
}

/// Reports whether a line segment intersects the inclusive pixel bounds of a rectangle.
///
/// Matches the pinned `cv.clipLine` binding: only the boolean result is observable because the
/// caller's point objects cannot be mutated through Embind. Degenerate or empty rectangles return
/// `false` instead of raising.
#[must_use]
#[wasm_bindgen(js_name = clipLineBool)]
#[allow(clippy::too_many_arguments)]
pub fn clip_line_bool(
    rectangle_x: i32,
    rectangle_y: i32,
    rectangle_width: i32,
    rectangle_height: i32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) -> bool {
    if rectangle_width <= 0 || rectangle_height <= 0 {
        return false;
    }
    let Ok(width) = u32::try_from(rectangle_width) else {
        return false;
    };
    let Ok(height) = u32::try_from(rectangle_height) else {
        return false;
    };
    clip_line(
        Rect {
            x: rectangle_x,
            y: rectangle_y,
            width,
            height,
        },
        Point {
            x: start_x,
            y: start_y,
        },
        Point { x: end_x, y: end_y },
    )
    .is_ok_and(|clipped| clipped.is_some())
}

/// Clips a line segment to the inclusive pixel bounds of an integer rectangle.
///
/// A visible segment returns `[x1, y1, x2, y2]`; a disjoint segment returns an empty
/// `Int32Array`. Width and height must be positive, and the inclusive right and bottom edges must
/// fit signed 32-bit coordinates. This coordinate-returning extension complements the
/// bool-only upstream form.
///
/// # Errors
///
/// Returns an error for an empty rectangle or a rectangle edge outside the signed 32-bit range.
#[wasm_bindgen(js_name = clipLine)]
#[allow(clippy::too_many_arguments)]
pub fn clip_line_rect(
    rectangle_x: i32,
    rectangle_y: i32,
    rectangle_width: u32,
    rectangle_height: u32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) -> Result<Vec<i32>, JsError> {
    let clipped = clip_line(
        Rect {
            x: rectangle_x,
            y: rectangle_y,
            width: rectangle_width,
            height: rectangle_height,
        },
        Point {
            x: start_x,
            y: start_y,
        },
        Point { x: end_x, y: end_y },
    )
    .map_err(HelperWasmError::from)?;
    Ok(clipped.map_or_else(Vec::new, |(start, end)| {
        vec![start.x, start.y, end.x, end.y]
    }))
}

fn flatten_points(points: &[Point]) -> Vec<i32> {
    let mut output = Vec::with_capacity(points.len() * 2);
    for point in points {
        output.extend_from_slice(&[point.x, point.y]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_adapters_return_typed_matrices_and_flat_point_arrays() {
        let kernel = get_structuring_element(1, 3, 3, 1, 1).expect("valid kernel");
        assert_eq!(kernel.depth(), MatDepth::U8);
        assert_eq!(kernel.to_u8_array(), [0, 1, 0, 1, 1, 1, 0, 1, 0]);

        let window = create_hanning_window(3, 3, MatDepth::F32).expect("valid window");
        assert_eq!(
            window.to_f32_array().expect("F32 output"),
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        let window_f64 = create_hanning_window(3, 3, MatDepth::F64).expect("valid window");
        assert_eq!(
            window_f64.to_f64_array().expect("F64 output"),
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            ellipse_2_poly(0, 0, 10, 5, 0, 0, 90, 90).expect("valid arc"),
            [10, 0, 0, 5]
        );
        assert_eq!(
            clip_line_rect(10, 20, 5, 4, 8, 21, 16, 21).expect("valid clip"),
            [10, 21, 14, 21]
        );
    }

    #[test]
    fn hanning_destination_adapter_rebinds_and_writes_compatible_regions() {
        let destination = Mat::empty_output();
        create_hanning_window_into_adapter(&destination, 4, 3, 6)
            .expect("empty destination rebinds");
        assert_eq!(
            (
                destination.rows(),
                destination.columns(),
                destination.channels(),
                destination.depth()
            ),
            (3, 4, 1, MatDepth::F64)
        );
        assert_eq!(
            destination
                .to_f64_array()
                .expect("F64 window")
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            [
                0,
                0,
                0,
                0,
                0,
                0x3FEB_B67A_E858_4CAA,
                0x3FEB_B67A_E858_4CAC,
                0,
                0,
                0,
                0,
                0,
            ]
        );

        let parent = Mat::from_owned_bytes(
            vec![99.0_f32; 20]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
            4,
            5,
            1,
            MatDepth::F32,
        )
        .expect("destination parent");
        let region = parent
            .roi(1, 1, 2, 3)
            .expect("compatible destination region");
        create_hanning_window_into_adapter(&region, 3, 2, 5).expect("compatible region write");
        assert_eq!(region.to_f32_array().expect("F32 region"), [0.0; 6]);
        assert_eq!(
            parent.to_f32_array().expect("F32 parent"),
            [
                99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.0, 0.0, 0.0, 99.0, 99.0, 0.0, 0.0, 0.0, 99.0,
                99.0, 99.0, 99.0, 99.0, 99.0,
            ]
        );

        assert!(create_hanning_window_into_adapter(&destination, 1, 3, 5).is_err());
        assert!(create_hanning_window_into_adapter(&destination, 3, 3, 4).is_err());
    }
}
