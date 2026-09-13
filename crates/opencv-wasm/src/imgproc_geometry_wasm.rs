//! WebAssembly matrix adapters for two-dimensional contour geometry.
//!
//! Browser-matched contour operations accept I32 and F32 contours stored as `Nx1C2`, `1xNC2`, or
//! `Nx2C1`. Other curve containers and higher-dimensional points are not supported.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    imgproc_geometry::{
        BoundingRect, GeometryError, Point, arc_length, bounding_rect, contour_area,
        is_contour_convex, point_polygon_test,
    },
    mat::{Mat, MatDepth},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GeometryWasmError {
    Kernel(GeometryError),
    UnsupportedDepth(MatDepth),
    NonContinuousContour,
    InvalidContourLayout {
        rows: u32,
        columns: u32,
        channels: u16,
    },
}

impl fmt::Display for GeometryWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => error.fmt(formatter),
            Self::UnsupportedDepth(depth) => {
                write!(
                    formatter,
                    "contour depth {depth:?} is unsupported by this operation"
                )
            }
            Self::NonContinuousContour => {
                formatter.write_str("contour must use continuous matrix storage")
            }
            Self::InvalidContourLayout {
                rows,
                columns,
                channels,
            } => write!(
                formatter,
                "contour shape {rows}x{columns}C{channels} is unsupported; expected Nx1C2, 1xNC2, or Nx2C1"
            ),
        }
    }
}

impl Error for GeometryWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::UnsupportedDepth(_)
            | Self::NonContinuousContour
            | Self::InvalidContourLayout { .. } => None,
        }
    }
}

impl From<GeometryError> for GeometryWasmError {
    fn from(error: GeometryError) -> Self {
        Self::Kernel(error)
    }
}

/// Computes an open or closed contour perimeter.
///
/// Accepted contour layouts are `Nx1C2`, `1xNC2`, and `Nx2C1` at I32 or F32 depth. Strided regions
/// are read through their logical bytes.
///
/// # Errors
/// Returns an error for unsupported layout or depth, non-finite coordinates, or numeric overflow.
#[wasm_bindgen(js_name = matArcLength)]
pub fn mat_arc_length(contour: &Mat, closed: bool) -> Result<f64, JsError> {
    arc_length_adapter(contour, closed).map_err(JsError::from)
}

/// Computes unsigned or oriented polygon area.
///
/// Fewer than three points have zero area. Positive oriented area denotes counter-clockwise order
/// in Cartesian coordinates. Accepted contours use I32 or F32 depth.
///
/// # Errors
/// Returns an error for unsupported layout or depth, non-finite coordinates, or numeric overflow.
#[wasm_bindgen(js_name = matContourArea)]
pub fn mat_contour_area(contour: &Mat, oriented: bool) -> Result<f64, JsError> {
    contour_area_adapter(contour, oriented).map_err(JsError::from)
}

/// Returns `[x, y, width, height]` for the inclusive integer contour bounds.
///
/// Fractional coordinates are floored. A single integer point produces a 1-by-1 rectangle. Accepted
/// nonempty contours use I32 or F32 depth. The canonical empty matrix returns a zero rectangle.
///
/// # Errors
/// Returns an error for unsupported input, non-finite points, or an I32-unrepresentable rectangle.
#[wasm_bindgen(js_name = matBoundingRect)]
pub fn mat_bounding_rect(contour: &Mat) -> Result<Vec<i32>, JsError> {
    bounding_rect_adapter(contour)
        .map(|rect| vec![rect.x, rect.y, rect.width, rect.height])
        .map_err(JsError::from)
}

/// Reports whether a contour has one consistent nonzero turn direction.
///
/// Every turn must be nonzero and have the same direction. Fewer than three points, collinear edge
/// points, duplicate vertices, concavity, and self-intersection return false.
///
/// # Errors
/// Returns an error for unsupported layout, depth, storage continuity, non-finite contour
/// coordinates, or numeric overflow.
#[wasm_bindgen(js_name = matIsContourConvex)]
pub fn mat_is_contour_convex(contour: &Mat) -> Result<bool, JsError> {
    is_contour_convex_adapter(contour).map_err(JsError::from)
}

/// Classifies a point against a polygon or returns signed nearest-boundary distance.
///
/// Positive results are inside, negative results are outside, and zero lies on the boundary. When
/// `measureDistance` is false, nonzero results are exactly 1 or -1.
/// One-point and two-point contours are valid. Non-finite query coordinates follow the pinned
/// browser sentinel behavior.
///
/// # Errors
/// Returns an error for empty contours, unsupported layout, depth, storage continuity, non-finite
/// contour coordinates, or numeric overflow.
#[wasm_bindgen(js_name = matPointPolygonTest)]
pub fn mat_point_polygon_test(
    contour: &Mat,
    x: f64,
    y: f64,
    #[wasm_bindgen(js_name = measureDistance)] measure_distance: bool,
) -> Result<f64, JsError> {
    point_polygon_test_adapter(contour, Point { x, y }, measure_distance).map_err(JsError::from)
}

fn arc_length_adapter(contour: &Mat, closed: bool) -> Result<f64, GeometryWasmError> {
    let points = decode_browser_contour(contour)?;
    arc_length(&points, closed).map_err(GeometryWasmError::from)
}

fn contour_area_adapter(contour: &Mat, oriented: bool) -> Result<f64, GeometryWasmError> {
    let points = decode_browser_contour(contour)?;
    contour_area(&points, oriented).map_err(GeometryWasmError::from)
}

fn bounding_rect_adapter(contour: &Mat) -> Result<BoundingRect, GeometryWasmError> {
    if contour.rows() == 0
        && contour.columns() == 0
        && contour.channels() == 1
        && contour.depth() == MatDepth::U8
        && !contour.is_continuous()
    {
        return Ok(BoundingRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        });
    }
    let points = decode_browser_contour(contour)?;
    bounding_rect(&points).map_err(GeometryWasmError::from)
}

fn is_contour_convex_adapter(contour: &Mat) -> Result<bool, GeometryWasmError> {
    let points = decode_browser_contour_query(contour)?;
    is_contour_convex(&points).map_err(GeometryWasmError::from)
}

fn point_polygon_test_adapter(
    contour: &Mat,
    query: Point,
    measure_distance: bool,
) -> Result<f64, GeometryWasmError> {
    let points = decode_browser_contour_query(contour)?;
    point_polygon_test(&points, query, measure_distance).map_err(GeometryWasmError::from)
}

pub(crate) fn decode_browser_contour_query(contour: &Mat) -> Result<Vec<Point>, GeometryWasmError> {
    if !contour.is_continuous() {
        return Err(GeometryWasmError::NonContinuousContour);
    }
    decode_browser_contour(contour)
}

fn decode_browser_contour(contour: &Mat) -> Result<Vec<Point>, GeometryWasmError> {
    match contour.depth() {
        MatDepth::I32 | MatDepth::F32 => decode_contour(contour),
        depth => Err(GeometryWasmError::UnsupportedDepth(depth)),
    }
}

fn decode_contour(contour: &Mat) -> Result<Vec<Point>, GeometryWasmError> {
    validate_layout(contour)?;
    let bytes = contour.compact_bytes();
    match contour.depth() {
        MatDepth::I32 => Ok(bytes
            .chunks_exact(8)
            .map(|pair| Point {
                x: f64::from(i32::from_ne_bytes(
                    pair[..4].try_into().expect("I32 scalar width"),
                )),
                y: f64::from(i32::from_ne_bytes(
                    pair[4..].try_into().expect("I32 scalar width"),
                )),
            })
            .collect()),
        MatDepth::F32 => Ok(bytes
            .chunks_exact(8)
            .map(|pair| Point {
                x: f64::from(f32::from_ne_bytes(
                    pair[..4].try_into().expect("F32 scalar width"),
                )),
                y: f64::from(f32::from_ne_bytes(
                    pair[4..].try_into().expect("F32 scalar width"),
                )),
            })
            .collect()),
        MatDepth::F64 => Ok(bytes
            .chunks_exact(16)
            .map(|pair| Point {
                x: f64::from_ne_bytes(pair[..8].try_into().expect("F64 scalar width")),
                y: f64::from_ne_bytes(pair[8..].try_into().expect("F64 scalar width")),
            })
            .collect()),
        depth => Err(GeometryWasmError::UnsupportedDepth(depth)),
    }
}

fn validate_layout(contour: &Mat) -> Result<(), GeometryWasmError> {
    let two_channel_vector =
        contour.channels() == 2 && (contour.rows() == 1 || contour.columns() == 1);
    let two_column_matrix = contour.channels() == 1 && contour.columns() == 2;
    if two_channel_vector || two_column_matrix {
        Ok(())
    } else {
        Err(GeometryWasmError::InvalidContourLayout {
            rows: contour.rows(),
            columns: contour.columns(),
            channels: contour.channels(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode<T: Copy>(values: &[T], append: impl Fn(T, &mut Vec<u8>)) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in values {
            append(value, &mut bytes);
        }
        bytes
    }

    fn matrix<T: Copy>(
        values: &[T],
        rows: u32,
        columns: u32,
        channels: u16,
        depth: MatDepth,
        append: impl Fn(T, &mut Vec<u8>),
    ) -> Mat {
        Mat::from_owned_bytes(encode(values, append), rows, columns, channels, depth)
            .expect("valid test contour")
    }

    fn i32_contour(values: &[i32], rows: u32, columns: u32, channels: u16) -> Mat {
        matrix(
            values,
            rows,
            columns,
            channels,
            MatDepth::I32,
            |value, out| {
                out.extend_from_slice(&value.to_ne_bytes());
            },
        )
    }

    fn f32_contour(values: &[f32], rows: u32, columns: u32, channels: u16) -> Mat {
        matrix(
            values,
            rows,
            columns,
            channels,
            MatDepth::F32,
            |value, out| {
                out.extend_from_slice(&value.to_ne_bytes());
            },
        )
    }

    fn f64_contour(values: &[f64], rows: u32, columns: u32, channels: u16) -> Mat {
        matrix(
            values,
            rows,
            columns,
            channels,
            MatDepth::F64,
            |value, out| {
                out.extend_from_slice(&value.to_ne_bytes());
            },
        )
    }

    #[test]
    fn adapters_accept_documented_depths_and_layouts() {
        let i32_points = i32_contour(&[0, 0, 4, 0, 4, 3, 0, 3], 4, 1, 2);
        let f32_points = f32_contour(&[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0], 1, 4, 2);
        assert_eq!(arc_length_adapter(&i32_points, true), Ok(14.0));
        assert_eq!(contour_area_adapter(&f32_points, false), Ok(12.0));
        assert_eq!(is_contour_convex_adapter(&f32_points), Ok(true));
    }

    #[test]
    fn browser_contour_operations_reject_f64() {
        let contour = f64_contour(&[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0], 4, 1, 2);
        assert_eq!(
            arc_length_adapter(&contour, true),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::F64))
        );
        assert_eq!(
            contour_area_adapter(&contour, false),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::F64))
        );
        assert_eq!(
            bounding_rect_adapter(&contour),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::F64))
        );
        assert_eq!(
            is_contour_convex_adapter(&contour),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::F64))
        );
        assert_eq!(
            point_polygon_test_adapter(&contour, Point { x: 2.0, y: 1.0 }, false),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::F64))
        );
    }

    #[test]
    fn bounding_rect_accepts_only_the_canonical_empty_matrix() {
        assert_eq!(
            bounding_rect_adapter(&crate::mat::mat_empty()),
            Ok(BoundingRect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            })
        );

        for (rows, columns) in [(0, 0), (0, 3), (3, 0)] {
            let typed_empty = Mat::empty_with_layout(rows, columns, 2, MatDepth::I32, true)
                .expect("valid typed empty contour");
            assert!(arc_length_adapter(&typed_empty, false).is_err());
            assert!(contour_area_adapter(&typed_empty, false).is_err());
            assert!(bounding_rect_adapter(&typed_empty).is_err());
        }
    }

    #[test]
    fn adapters_read_strided_roi_logical_bytes() {
        let parent = f32_contour(
            &[
                99.0, 99.0, 0.0, 0.0, 99.0, 99.0, 4.0, 0.0, 99.0, 99.0, 4.0, 3.0, 99.0, 99.0, 0.0,
                3.0,
            ],
            4,
            2,
            2,
        );
        let contour = parent.roi(0, 1, 4, 1).expect("valid strided ROI");
        assert!(!contour.is_continuous());
        assert_eq!(contour_area_adapter(&contour, false), Ok(12.0));
        assert_eq!(arc_length_adapter(&contour, false), Ok(11.0));
        assert!(is_contour_convex_adapter(&contour).is_err());
        assert!(point_polygon_test_adapter(&contour, Point { x: 2.0, y: 1.0 }, false).is_err());
    }

    #[test]
    fn bounding_rect_and_polygon_distance_have_stable_results() {
        let contour = f32_contour(&[-0.2, 1.9, 2.8, -1.1, 0.0, 3.0], 3, 1, 2);
        assert_eq!(
            bounding_rect_adapter(&contour),
            Ok(BoundingRect {
                x: -1,
                y: -2,
                width: 4,
                height: 6,
            })
        );
        let rectangle = i32_contour(&[0, 0, 4, 0, 4, 3, 0, 3], 4, 1, 2);
        assert_eq!(
            point_polygon_test_adapter(&rectangle, Point { x: 2.0, y: 1.0 }, true),
            Ok(1.0)
        );
        assert_eq!(
            point_polygon_test_adapter(&rectangle, Point { x: 5.0, y: 1.0 }, false),
            Ok(-1.0)
        );
    }

    #[test]
    fn invalid_layout_depth_and_values_fail_before_geometry() {
        let bad_layout = f32_contour(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 2, 3, 1);
        assert_eq!(
            arc_length_adapter(&bad_layout, false),
            Err(GeometryWasmError::InvalidContourLayout {
                rows: 2,
                columns: 3,
                channels: 1,
            })
        );
        let unsupported = Mat::from_owned_bytes(vec![0, 0], 1, 1, 2, MatDepth::U8)
            .expect("valid unsupported-depth matrix");
        assert_eq!(
            contour_area_adapter(&unsupported, false),
            Err(GeometryWasmError::UnsupportedDepth(MatDepth::U8))
        );
        let non_finite = f32_contour(&[0.0, 0.0, f32::NAN, 1.0], 2, 1, 2);
        assert_eq!(
            arc_length_adapter(&non_finite, false),
            Err(GeometryWasmError::Kernel(GeometryError::NonFinitePoint {
                index: 1,
            }))
        );
    }
}
