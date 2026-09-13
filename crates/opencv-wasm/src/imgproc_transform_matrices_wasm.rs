//! WebAssembly adapters for small image-transform matrix constructors.

use crate::{
    imgproc_transform_matrices::{self, TransformMatrixError},
    mat::{Mat, MatDepth, MatError},
};
use std::{error::Error, fmt};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum TransformMatrixWasmError {
    Kernel(TransformMatrixError),
    Matrix(MatError),
    FloatingPointInputRequired(MatDepth),
    F32PointInputRequired(MatDepth),
    NonContinuousPointInput,
    PointMatrixShape {
        point_count: usize,
        rows: u32,
        columns: u32,
        channels: u16,
    },
    AffineMatrixShape {
        rows: u32,
        columns: u32,
        channels: u16,
    },
}

impl fmt::Display for TransformMatrixWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
            Self::FloatingPointInputRequired(depth) => write!(
                formatter,
                "transform matrix inputs require F32 or F64 depth; received {depth:?}"
            ),
            Self::F32PointInputRequired(depth) => write!(
                formatter,
                "affine point matrices require F32 depth; received {depth:?}"
            ),
            Self::NonContinuousPointInput => {
                formatter.write_str("affine point matrices require continuous storage")
            }
            Self::PointMatrixShape {
                point_count,
                rows,
                columns,
                channels,
            } => write!(
                formatter,
                "expected {point_count} two-dimensional points; received {rows}x{columns} with {channels} channels"
            ),
            Self::AffineMatrixShape {
                rows,
                columns,
                channels,
            } => write!(
                formatter,
                "affine transform input must be 2x3 with one channel; received {rows}x{columns} with {channels} channels"
            ),
        }
    }
}

impl Error for TransformMatrixWasmError {}

impl From<TransformMatrixError> for TransformMatrixWasmError {
    fn from(error: TransformMatrixError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for TransformMatrixWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Allocates a 2x3 F64 matrix for rotation in degrees around a center point.
///
/// Every IEEE-754 scalar value is accepted, including non-finite values and signed zero. The
/// result always has F64 depth.
///
/// # Errors
/// Returns an error when the output matrix cannot be allocated.
#[wasm_bindgen(js_name = matGetRotationMatrix2D)]
pub fn mat_get_rotation_matrix_2d(
    center_x: f64,
    center_y: f64,
    angle_degrees: f64,
    scale: f64,
) -> Result<Mat, JsError> {
    rotation_adapter(center_x, center_y, angle_degrees, scale).map_err(JsError::from)
}

/// Allocates the 2x3 F64 affine map between three source and destination points.
///
/// Point inputs may be 3x2 single-channel matrices or 3x1 and 1x3 two-channel vectors. F32 and
/// F64 inputs are supported, including strided regions of interest.
///
/// # Errors
/// Returns an error for unsupported depth or shape, non-finite coordinates, or collinear source
/// points.
#[wasm_bindgen(js_name = matGetAffineTransform)]
pub fn mat_get_affine_transform(source: &Mat, destination: &Mat) -> Result<Mat, JsError> {
    affine_adapter(source, destination).map_err(JsError::from)
}

/// Allocates the inverse of a 2x3 F32 or F64 affine matrix at the source depth.
///
/// Strided single-channel regions are supported.
///
/// # Errors
/// Returns an error for unsupported depth or shape.
#[wasm_bindgen(js_name = matInvertAffineTransform)]
pub fn mat_invert_affine_transform(transform: &Mat) -> Result<Mat, JsError> {
    invert_affine_adapter(transform).map_err(JsError::from)
}

/// Writes the inverse of a 2x3 F32 or F64 affine matrix into a mutable destination.
///
/// The destination is rebound when its shape or depth differs. Compatible regions are written
/// through, and exact in-place operation is supported.
///
/// # Errors
/// Returns an error for unsupported source depth or shape, or an invalid destination write.
#[wasm_bindgen(js_name = matInvertAffineTransformInto)]
pub fn mat_invert_affine_transform_into(transform: &Mat, destination: &Mat) -> Result<(), JsError> {
    invert_affine_into_adapter(transform, destination).map_err(JsError::from)
}

/// Allocates the 3x3 F64 projective map between four source and destination points.
///
/// Point inputs may be 4x2 single-channel matrices or 4x1 and 1x4 two-channel vectors. F32
/// inputs must be continuous. LU (0) and QR (4), optionally combined with NORMAL (16),
/// normalize the lower-right output coefficient to one.
///
/// # Errors
/// Returns an error for unsupported depth, shape or solve method, degenerate point
/// configurations, or a transform that cannot use the selected normalization.
#[wasm_bindgen(js_name = matGetPerspectiveTransform)]
pub fn mat_get_perspective_transform(
    source: &Mat,
    destination: &Mat,
    method: i32,
) -> Result<Mat, JsError> {
    perspective_adapter(source, destination, method).map_err(JsError::from)
}

fn rotation_adapter(
    center_x: f64,
    center_y: f64,
    angle_degrees: f64,
    scale: f64,
) -> Result<Mat, TransformMatrixWasmError> {
    let result =
        imgproc_transform_matrices::rotation_matrix_2d([center_x, center_y], angle_degrees, scale);
    f64_matrix(&result, 2, 3)
}

fn affine_adapter(source: &Mat, destination: &Mat) -> Result<Mat, TransformMatrixWasmError> {
    let source = decode_affine_points(source)?;
    let destination = decode_affine_points(destination)?;
    let result = imgproc_transform_matrices::affine_transform(&source, &destination)?;
    f64_matrix(&result, 2, 3)
}

fn decode_affine_points(matrix: &Mat) -> Result<[[f64; 2]; 3], TransformMatrixWasmError> {
    if matrix.depth() != MatDepth::F32 {
        return Err(TransformMatrixWasmError::F32PointInputRequired(
            matrix.depth(),
        ));
    }
    if !matrix.is_continuous() {
        return Err(TransformMatrixWasmError::NonContinuousPointInput);
    }
    decode_points::<3>(matrix)
}

fn invert_affine_adapter(transform: &Mat) -> Result<Mat, TransformMatrixWasmError> {
    let (bytes, depth) = invert_affine_bytes(transform)?;
    Ok(Mat::from_owned_bytes(bytes, 2, 3, 1, depth)?)
}

fn invert_affine_into_adapter(
    transform: &Mat,
    destination: &Mat,
) -> Result<(), TransformMatrixWasmError> {
    let (bytes, depth) = invert_affine_bytes(transform)?;
    destination.write_output(bytes, 2, 3, 1, depth)?;
    Ok(())
}

fn invert_affine_bytes(transform: &Mat) -> Result<(Vec<u8>, MatDepth), TransformMatrixWasmError> {
    validate_floating_depth(transform)?;
    if transform.rows() != 2 || transform.columns() != 3 || transform.channels() != 1 {
        return Err(TransformMatrixWasmError::AffineMatrixShape {
            rows: transform.rows(),
            columns: transform.columns(),
            channels: transform.channels(),
        });
    }
    let bytes = transform.compact_bytes();
    match transform.depth() {
        MatDepth::F32 => {
            let mut coefficients = [0.0_f32; 6];
            for (coefficient, chunk) in coefficients.iter_mut().zip(bytes.chunks_exact(4)) {
                *coefficient = f32::from_ne_bytes(
                    chunk
                        .try_into()
                        .expect("validated F32 affine chunks contain four bytes"),
                );
            }
            let result = imgproc_transform_matrices::invert_affine_transform_f32(&coefficients);
            Ok((
                result
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>(),
                MatDepth::F32,
            ))
        }
        MatDepth::F64 => {
            let values = decode_floating(transform)?;
            let coefficients: [f64; 6] = values
                .try_into()
                .expect("validated affine matrix contains six coefficients");
            let result = imgproc_transform_matrices::invert_affine_transform(&coefficients)?;
            Ok((
                result
                    .into_iter()
                    .flat_map(f64::to_ne_bytes)
                    .collect::<Vec<_>>(),
                MatDepth::F64,
            ))
        }
        depth => Err(TransformMatrixWasmError::FloatingPointInputRequired(depth)),
    }
}

fn perspective_adapter(
    source: &Mat,
    destination: &Mat,
    method: i32,
) -> Result<Mat, TransformMatrixWasmError> {
    for matrix in [source, destination] {
        if matrix.depth() != MatDepth::F32 {
            return Err(TransformMatrixWasmError::F32PointInputRequired(
                matrix.depth(),
            ));
        }
        if !matrix.is_continuous() {
            return Err(TransformMatrixWasmError::NonContinuousPointInput);
        }
    }
    let source = decode_points::<4>(source)?;
    let destination = decode_points::<4>(destination)?;
    let result = imgproc_transform_matrices::perspective_transform(&source, &destination, method)?;
    f64_matrix(&result, 3, 3)
}

fn decode_points<const COUNT: usize>(
    matrix: &Mat,
) -> Result<[[f64; 2]; COUNT], TransformMatrixWasmError> {
    validate_floating_depth(matrix)?;
    let count = u32::try_from(COUNT).map_err(|_| TransformMatrixError::NumericalFailure)?;
    let valid_shape = (matrix.channels() == 1 && matrix.rows() == count && matrix.columns() == 2)
        || (matrix.channels() == 2
            && ((matrix.rows() == count && matrix.columns() == 1)
                || (matrix.rows() == 1 && matrix.columns() == count)));
    if !valid_shape {
        return Err(TransformMatrixWasmError::PointMatrixShape {
            point_count: COUNT,
            rows: matrix.rows(),
            columns: matrix.columns(),
            channels: matrix.channels(),
        });
    }
    let values = decode_floating(matrix)?;
    let mut points = [[0.0; 2]; COUNT];
    for (point, pair) in points.iter_mut().zip(values.chunks_exact(2)) {
        *point = [pair[0], pair[1]];
    }
    Ok(points)
}

fn validate_floating_depth(matrix: &Mat) -> Result<(), TransformMatrixWasmError> {
    match matrix.depth() {
        MatDepth::F32 | MatDepth::F64 => Ok(()),
        depth => Err(TransformMatrixWasmError::FloatingPointInputRequired(depth)),
    }
}

fn decode_floating(matrix: &Mat) -> Result<Vec<f64>, TransformMatrixWasmError> {
    let bytes = matrix.compact_bytes();
    match matrix.depth() {
        MatDepth::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                let encoded: [u8; 4] = chunk.try_into().expect("F32 matrix chunks have four bytes");
                f64::from(f32::from_ne_bytes(encoded))
            })
            .collect()),
        MatDepth::F64 => Ok(bytes
            .chunks_exact(8)
            .map(|chunk| {
                let encoded: [u8; 8] = chunk
                    .try_into()
                    .expect("F64 matrix chunks have eight bytes");
                f64::from_ne_bytes(encoded)
            })
            .collect()),
        depth => Err(TransformMatrixWasmError::FloatingPointInputRequired(depth)),
    }
}

fn f64_matrix(values: &[f64], rows: u32, columns: u32) -> Result<Mat, TransformMatrixWasmError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    Ok(Mat::from_owned_bytes(
        bytes,
        rows,
        columns,
        1,
        MatDepth::F64,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_mat(values: &[f64], rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            rows,
            columns,
            channels,
            MatDepth::F64,
        )
        .expect("valid F64 matrix")
    }

    fn f32_mat(values: &[f32], rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            rows,
            columns,
            channels,
            MatDepth::F32,
        )
        .expect("valid F32 matrix")
    }

    fn assert_close(actual: &Mat, expected: &[f64], tolerance: f64) {
        assert_eq!(actual.depth(), MatDepth::F64);
        let actual = decode_floating(actual).expect("F64 output");
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index} was {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn rotation_adapter_allocates_a_f64_matrix() {
        let result = rotation_adapter(10.0, 20.0, 90.0, 2.0).expect("finite rotation");

        assert_eq!(
            (result.rows(), result.columns(), result.channels()),
            (2, 3, 1)
        );
        assert_close(&result, &[0.0, 2.0, -30.0, -2.0, 0.0, 40.0], 1.0e-12);
    }

    #[test]
    fn rotation_adapter_allocates_non_finite_f64_results() {
        let result = rotation_adapter(1.0, 2.0, f64::NAN, 2.0)
            .expect("the pinned binding propagates non-finite coefficients");

        assert_eq!(
            (
                result.rows(),
                result.columns(),
                result.channels(),
                result.depth()
            ),
            (2, 3, 1, MatDepth::F64)
        );
        assert!(
            decode_floating(&result)
                .expect("F64 output")
                .into_iter()
                .all(f64::is_nan)
        );
    }

    #[test]
    fn affine_adapter_rejects_f64_and_strided_point_matrices() {
        let source_parent = f64_mat(&[99.0, 0.0, 0.0, 99.0, 1.0, 0.0, 99.0, 0.0, 1.0], 3, 3, 1);
        let source = source_parent
            .roi(0, 1, 3, 2)
            .expect("strided source points");
        let destination = f32_mat(&[2.0, 3.0, 4.0, 4.0, 1.0, 6.0], 3, 1, 2);

        assert!(affine_adapter(&source, &destination).is_err());

        let strided_parent = f32_mat(
            &[
                99.0, 99.0, 0.0, 0.0, 99.0, 99.0, 1.0, 0.0, 99.0, 99.0, 0.0, 1.0,
            ],
            3,
            2,
            2,
        );
        let strided = strided_parent.roi(0, 1, 3, 1).expect("strided F32 points");
        assert!(affine_adapter(&strided, &destination).is_err());
    }

    #[test]
    fn inverse_adapter_reads_a_strided_affine_matrix() {
        let parent = f32_mat(&[99.0, 2.0, 0.0, 4.0, 99.0, 0.0, 3.0, -6.0], 2, 4, 1);
        let transform = parent.roi(0, 1, 2, 3).expect("strided affine matrix");

        let result = invert_affine_adapter(&transform).expect("invertible matrix");

        assert_eq!(result.depth(), MatDepth::F32);
        assert_eq!(
            decode_floating(&result)
                .expect("F32 output")
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            [0.5_f32, -0.0, -2.0, -0.0, 1.0 / 3.0, 2.0].map(|value| f64::from(value).to_bits())
        );
    }

    #[test]
    fn inverse_adapter_mutates_regions_rebinds_and_supports_exact_aliasing() {
        let source = f64_mat(&[2.0, 0.0, 4.0, 0.0, 3.0, -6.0], 2, 3, 1);
        let parent = f64_mat(&[99.0; 15], 3, 5, 1);
        let region = parent.roi(0, 1, 2, 3).expect("destination region");
        invert_affine_into_adapter(&source, &region).expect("compatible region write");
        assert_close(&region, &[0.5, -0.0, -2.0, -0.0, 1.0 / 3.0, 2.0], 0.0);
        assert_eq!(
            decode_floating(&parent).expect("parent remains attached"),
            vec![
                99.0,
                0.5,
                -0.0,
                -2.0,
                99.0,
                99.0,
                -0.0,
                1.0 / 3.0,
                2.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ]
        );

        invert_affine_into_adapter(&source, &source).expect("exact in-place write");
        assert_close(&source, &[0.5, -0.0, -2.0, -0.0, 1.0 / 3.0, 2.0], 0.0);

        let replacement = f32_mat(&[9.0], 1, 1, 1);
        invert_affine_into_adapter(&source, &replacement).expect("destination replacement");
        assert_eq!(
            (
                replacement.rows(),
                replacement.columns(),
                replacement.channels(),
                replacement.depth()
            ),
            (2, 3, 1, MatDepth::F64)
        );
    }

    #[test]
    fn perspective_adapter_builds_a_non_affine_projective_map() {
        let source = f32_mat(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1, 4, 2);
        let destination = f32_mat(
            &[1.0, 2.0, 2.0, 7.0 / 6.0, 2.0, 19.0 / 7.0, 1.2, 4.0],
            4,
            2,
            1,
        );

        let result = perspective_adapter(&source, &destination, 0).expect("valid correspondences");

        assert_eq!((result.rows(), result.columns()), (3, 3));
        assert_close(
            &result,
            &[2.0, 0.5, 1.0, -0.25, 3.0, 2.0, 0.5, 0.25, 1.0],
            1.0e-6,
        );
    }

    #[test]
    fn adapters_reject_bad_shape_depth_and_degenerate_geometry() {
        let integer_points =
            Mat::from_owned_bytes(vec![0; 6], 3, 2, 1, MatDepth::U8).expect("valid integer matrix");
        assert_eq!(
            affine_adapter(&integer_points, &integer_points)
                .expect_err("integer point depth must fail"),
            TransformMatrixWasmError::F32PointInputRequired(MatDepth::U8)
        );

        let wrong_shape = f64_mat(&[0.0; 6], 1, 6, 1);
        assert_eq!(
            decode_points::<3>(&wrong_shape),
            Err(TransformMatrixWasmError::PointMatrixShape {
                point_count: 3,
                rows: 1,
                columns: 6,
                channels: 1,
            })
        );

        let collinear = f32_mat(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 1, 2);
        let destination = f32_mat(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 1, 2);
        let zero = affine_adapter(&collinear, &destination).expect("singular maps return zeros");
        assert_eq!(decode_floating(&zero).expect("F64 result"), vec![0.0; 6]);

        let non_finite = f32_mat(&[f32::NAN, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 1, 2);
        let nan = affine_adapter(&non_finite, &destination).expect("NaN coordinates propagate");
        assert!(
            decode_floating(&nan)
                .expect("F64 result")
                .into_iter()
                .all(f64::is_nan)
        );
    }
}
