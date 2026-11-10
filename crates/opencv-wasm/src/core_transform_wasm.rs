//! Matrix-aware WebAssembly adapters for per-element transforms.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    core_transform::{ScalarDepth, TransformError, perspective_transform_bytes, transform_bytes},
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
enum TransformWasmError {
    CoefficientsMustBeSingleChannel(u16),
    UnsupportedCoefficientDepth(MatDepth),
    CoefficientDimensionsOverflow,
    DestinationMetadata,
    Kernel(TransformError),
    Matrix(MatError),
}

impl fmt::Display for TransformWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoefficientsMustBeSingleChannel(channels) => write!(
                formatter,
                "transform coefficients must be single-channel; found {channels} channels"
            ),
            Self::UnsupportedCoefficientDepth(depth) => write!(
                formatter,
                "transform coefficients must have F32 or F64 depth; found {depth:?}"
            ),
            Self::CoefficientDimensionsOverflow => formatter
                .write_str("transform coefficient dimensions exceed the matrix channel limit"),
            Self::DestinationMetadata => formatter.write_str(
                "transform destination must match source rows, columns, and depth and coefficient rows",
            ),
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for TransformWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TransformError> for TransformWasmError {
    fn from(error: TransformError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for TransformWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Applies a per-element linear or affine channel transform and allocates its result.
///
/// # Errors
///
/// Returns an error when coefficient metadata does not match the source channel count.
#[wasm_bindgen(js_name = matTransform)]
pub fn mat_transform(source: &Mat, coefficients: &Mat) -> Result<Mat, JsError> {
    transform_mat(source, coefficients).map_err(JsError::from)
}

/// Applies a per-element linear or affine channel transform into an existing matrix.
///
/// Source and coefficient bytes are copied before the destination changes, so overlapping regions
/// produce deterministic results.
///
/// # Errors
///
/// Returns an error when input metadata is invalid or the destination metadata differs from the
/// required output.
#[wasm_bindgen(js_name = matTransformInto)]
pub fn mat_transform_into(
    source: &Mat,
    coefficients: &Mat,
    destination: &Mat,
) -> Result<(), JsError> {
    transform_into_mat(source, coefficients, destination).map_err(JsError::from)
}

/// Applies a homogeneous perspective transform to 2D or 3D floating-point vectors.
///
/// # Errors
///
/// Returns an error unless the source has F32 or F64 depth and two or three channels, with a
/// matching square F32 or F64 coefficient matrix.
#[wasm_bindgen(js_name = matPerspectiveTransform)]
pub fn mat_perspective_transform(source: &Mat, coefficients: &Mat) -> Result<Mat, JsError> {
    perspective_transform_mat(source, coefficients).map_err(JsError::from)
}

/// Applies a homogeneous perspective transform into an existing floating-point matrix.
///
/// The adapter snapshots source and coefficients before writing, including when regions overlap.
///
/// # Errors
///
/// Returns an error for invalid input metadata or a destination that does not have the same shape,
/// channels, and depth as the source.
#[wasm_bindgen(js_name = matPerspectiveTransformInto)]
pub fn mat_perspective_transform_into(
    source: &Mat,
    coefficients: &Mat,
    destination: &Mat,
) -> Result<(), JsError> {
    perspective_transform_into_mat(source, coefficients, destination).map_err(JsError::from)
}

fn transform_mat(source: &Mat, coefficients: &Mat) -> Result<Mat, TransformWasmError> {
    let coefficient_values = decode_coefficients(coefficients)?;
    let output_channels = u16::try_from(coefficients.rows())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let coefficient_columns = u16::try_from(coefficients.columns())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let bytes = transform_bytes(
        &source.compact_bytes(),
        scalar_depth(source.depth()),
        source.channels(),
        &coefficient_values,
        output_channels,
        coefficient_columns,
    )?;
    Mat::from_owned_bytes(
        bytes,
        source.rows(),
        source.columns(),
        output_channels,
        source.depth(),
    )
    .map_err(TransformWasmError::from)
}

fn transform_into_mat(
    source: &Mat,
    coefficients: &Mat,
    destination: &Mat,
) -> Result<(), TransformWasmError> {
    let coefficient_values = decode_coefficients(coefficients)?;
    let output_channels = u16::try_from(coefficients.rows())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let coefficient_columns = u16::try_from(coefficients.columns())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let bytes = transform_bytes(
        &source.compact_bytes(),
        scalar_depth(source.depth()),
        source.channels(),
        &coefficient_values,
        output_channels,
        coefficient_columns,
    )?;
    destination.write_output(
        bytes,
        source.rows(),
        source.columns(),
        output_channels,
        source.depth(),
    )?;
    Ok(())
}

fn perspective_transform_mat(source: &Mat, coefficients: &Mat) -> Result<Mat, TransformWasmError> {
    let coefficient_values = decode_coefficients(coefficients)?;
    let coefficient_rows = u16::try_from(coefficients.rows())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let coefficient_columns = u16::try_from(coefficients.columns())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let bytes = perspective_transform_bytes(
        &source.compact_bytes(),
        scalar_depth(source.depth()),
        source.channels(),
        &coefficient_values,
        coefficient_rows,
        coefficient_columns,
    )?;
    Mat::from_owned_bytes(
        bytes,
        source.rows(),
        source.columns(),
        source.channels(),
        source.depth(),
    )
    .map_err(TransformWasmError::from)
}

fn perspective_transform_into_mat(
    source: &Mat,
    coefficients: &Mat,
    destination: &Mat,
) -> Result<(), TransformWasmError> {
    let coefficient_values = decode_coefficients(coefficients)?;
    let coefficient_rows = u16::try_from(coefficients.rows())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let coefficient_columns = u16::try_from(coefficients.columns())
        .map_err(|_| TransformWasmError::CoefficientDimensionsOverflow)?;
    let bytes = perspective_transform_bytes(
        &source.compact_bytes(),
        scalar_depth(source.depth()),
        source.channels(),
        &coefficient_values,
        coefficient_rows,
        coefficient_columns,
    )?;
    destination.write_output(
        bytes,
        source.rows(),
        source.columns(),
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

fn decode_coefficients(coefficients: &Mat) -> Result<Vec<f64>, TransformWasmError> {
    if coefficients.channels() != 1 {
        return Err(TransformWasmError::CoefficientsMustBeSingleChannel(
            coefficients.channels(),
        ));
    }
    let bytes = coefficients.compact_bytes();
    match coefficients.depth() {
        MatDepth::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                f64::from(f32::from_ne_bytes(
                    chunk.try_into().expect("F32 coefficient chunk width"),
                ))
            })
            .collect()),
        MatDepth::F64 => Ok(bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("F64 coefficient chunk width")))
            .collect()),
        depth => Err(TransformWasmError::UnsupportedCoefficientDepth(depth)),
    }
}

const fn scalar_depth(depth: MatDepth) -> ScalarDepth {
    match depth {
        MatDepth::U8 => ScalarDepth::U8,
        MatDepth::I8 => ScalarDepth::I8,
        MatDepth::U16 => ScalarDepth::U16,
        MatDepth::I16 => ScalarDepth::I16,
        MatDepth::I32 => ScalarDepth::I32,
        MatDepth::F32 => ScalarDepth::F32,
        MatDepth::F64 => ScalarDepth::F64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(bytes: Vec<u8>, rows: u32, columns: u32, channels: u16, depth: MatDepth) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, channels, depth).expect("valid test matrix")
    }

    #[test]
    fn transform_reads_strided_source_and_f32_coefficients() {
        let source_parent = matrix(
            vec![0, 0, 1, 10, 2, 20, 0, 0, 0, 0, 3, 30, 4, 40, 0, 0],
            2,
            4,
            2,
            MatDepth::U8,
        );
        let source = source_parent.roi(0, 1, 2, 2).expect("source ROI");
        let coefficients = matrix(
            [1.0_f32, 0.5, 2.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
            1,
            3,
            1,
            MatDepth::F32,
        );

        let output = transform_mat(&source, &coefficients).expect("valid transform");

        assert_eq!(output.rows(), 2);
        assert_eq!(output.columns(), 2);
        assert_eq!(output.channels(), 1);
        assert_eq!(output.depth(), MatDepth::U8);
        assert_eq!(output.compact_bytes(), [8, 14, 20, 26]);
    }

    #[test]
    fn transform_into_snapshots_overlapping_rois_before_writing() {
        let parent = matrix(vec![1, 2, 3, 4, 5], 1, 5, 1, MatDepth::U8);
        let source = parent.roi(0, 0, 1, 3).expect("source ROI");
        let destination = parent.roi(0, 1, 1, 3).expect("destination ROI");
        let coefficients = matrix(
            [1.0_f64, 10.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            1,
            2,
            1,
            MatDepth::F64,
        );

        transform_into_mat(&source, &coefficients, &destination).expect("valid aliased transform");

        assert_eq!(parent.compact_bytes(), [1, 11, 12, 13, 5]);
    }

    #[test]
    fn perspective_transform_supports_three_dimensional_f32_points() {
        let source = matrix(
            [1.0_f32, 2.0, 3.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
            1,
            1,
            3,
            MatDepth::F32,
        );
        let coefficients = matrix(
            [
                1.0_f64, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.5,
                0.5,
            ]
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect(),
            4,
            4,
            1,
            MatDepth::F64,
        );

        let output =
            perspective_transform_mat(&source, &coefficients).expect("valid projective transform");

        assert_eq!(output.to_f32_array().expect("F32 output"), [5.5, 2.0, 1.0],);
    }

    #[test]
    fn perspective_transform_into_handles_overlapping_f64_rois() {
        let parent = matrix(
            [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            1,
            4,
            2,
            MatDepth::F64,
        );
        let source = parent.roi(0, 0, 1, 3).expect("source ROI");
        let destination = parent.roi(0, 1, 1, 3).expect("destination ROI");
        let coefficients = matrix(
            [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            3,
            3,
            1,
            MatDepth::F64,
        );

        perspective_transform_into_mat(&source, &coefficients, &destination)
            .expect("valid aliased perspective transform");

        assert_eq!(
            parent.to_f64_array().expect("F64 parent"),
            [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
    }

    #[test]
    fn adapters_reject_invalid_coefficients_and_destinations_without_writing() {
        let source = matrix(vec![7], 1, 1, 1, MatDepth::U8);
        let integer_coefficients = matrix(vec![1, 0], 1, 2, 1, MatDepth::U8);
        assert!(matches!(
            transform_mat(&source, &integer_coefficients),
            Err(TransformWasmError::UnsupportedCoefficientDepth(
                MatDepth::U8
            )),
        ));

        let multi_channel_coefficients = matrix(
            [1.0_f32, 0.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
            1,
            1,
            2,
            MatDepth::F32,
        );
        assert!(matches!(
            transform_mat(&source, &multi_channel_coefficients),
            Err(TransformWasmError::CoefficientsMustBeSingleChannel(2)),
        ));

        let wrong_shape = matrix(
            [1.0_f64, 0.0, 0.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            1,
            3,
            1,
            MatDepth::F64,
        );
        let destination = matrix(vec![99], 1, 1, 1, MatDepth::U8);
        assert!(matches!(
            transform_into_mat(&source, &wrong_shape, &destination),
            Err(TransformWasmError::Kernel(
                TransformError::InvalidCoefficientShape
            )),
        ));
        assert_eq!(destination.compact_bytes(), [99]);

        let valid_coefficients = matrix(
            [1.0_f64, 0.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            1,
            2,
            1,
            MatDepth::F64,
        );
        let wrong_destination = matrix(vec![77], 1, 1, 1, MatDepth::I8);
        transform_into_mat(&source, &valid_coefficients, &wrong_destination)
            .expect("incompatible destinations reallocate to source depth");
        assert_eq!(wrong_destination.depth(), MatDepth::U8);
        assert_eq!(wrong_destination.compact_bytes(), [7]);

        let perspective_coefficients = matrix(
            [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect(),
            3,
            3,
            1,
            MatDepth::F64,
        );
        assert!(matches!(
            perspective_transform_mat(&source, &perspective_coefficients),
            Err(TransformWasmError::Kernel(
                TransformError::UnsupportedPerspectiveDepth(ScalarDepth::U8)
            )),
        ));
    }
}
