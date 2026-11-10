//! WebAssembly adapters for dense single-channel matrix algebra.

use crate::{
    core_algebra::{self, AlgebraError},
    mat::{Mat, MatDepth, MatError},
};
use std::{error::Error, fmt};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum AlgebraWasmError {
    Kernel(AlgebraError),
    Matrix(MatError),
    SingleChannelRequired,
    SquareMatrixRequired { rows: u32, columns: u32 },
    FloatingSourceRequired(MatDepth),
    FloatingDestinationRequired(MatDepth),
    InvertDestinationMismatch,
    SolveShapeMismatch,
    DecompositionRequiresSquare,
    UnsupportedMethod(u32),
    DestinationOverflow,
}

impl fmt::Display for AlgebraWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
            Self::SingleChannelRequired => {
                formatter.write_str("matrix algebra requires single-channel matrices")
            }
            Self::SquareMatrixRequired { rows, columns } => write!(
                formatter,
                "operation requires a square matrix; received {rows}x{columns}"
            ),
            Self::FloatingSourceRequired(depth) => write!(
                formatter,
                "determinant source requires F32 or F64 depth; received {depth:?}"
            ),
            Self::FloatingDestinationRequired(depth) => write!(
                formatter,
                "matrix algebra destinations require F32 or F64 depth; received {depth:?}"
            ),
            Self::InvertDestinationMismatch => formatter.write_str(
                "inverse destination must match the source rows and columns and have one channel",
            ),
            Self::SolveShapeMismatch => formatter
                .write_str("solve requires B.rows = A.rows and X shape A.columns-by-B.columns"),
            Self::DecompositionRequiresSquare => {
                formatter.write_str("selected decomposition requires a square coefficient matrix")
            }
            Self::UnsupportedMethod(method) => write!(
                formatter,
                "unsupported decomposition method {method}; expected 0 (LU), 3 (Cholesky), or 4 (QR)"
            ),
            Self::DestinationOverflow => formatter.write_str(
                "algebra result cannot be represented by the destination's scalar depth",
            ),
        }
    }
}

impl Error for AlgebraWasmError {}
impl From<AlgebraError> for AlgebraWasmError {
    fn from(error: AlgebraError) -> Self {
        Self::Kernel(error)
    }
}
impl From<MatError> for AlgebraWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Computes the determinant of a square, single-channel matrix.
///
/// F32 sources retain F32 elimination arithmetic; F64 sources retain F64 arithmetic.
///
/// # Errors
/// Returns an error for an empty, multi-channel, nonsquare, or non-floating source.
#[wasm_bindgen(js_name = matDeterminant)]
pub fn mat_determinant(source: &Mat) -> Result<f64, JsError> {
    determinant_adapter(source).map_err(JsError::from)
}

/// Inverts a square matrix into a caller-owned F32 or F64 destination.
///
/// Method 0 uses partial-pivoted LU, method 3 uses Cholesky for symmetric positive-definite input,
/// and method 4 uses Householder QR. The function returns 1 on success and 0 for a singular or
/// numerically rank-deficient source. A zero result leaves the destination unchanged.
///
/// # Errors
/// Returns an error for invalid shapes, channels, destination depth, method, or non-finite input.
#[wasm_bindgen(js_name = matInvertInto)]
pub fn mat_invert_into(source: &Mat, destination: &Mat, method: u32) -> Result<f64, JsError> {
    invert_adapter(source, destination, method).map_err(JsError::from)
}

/// Solves `A * X = B` into a caller-owned F32 or F64 destination.
///
/// Method 0 solves square systems with partial-pivoted LU. Method 3 uses Cholesky for symmetric
/// positive-definite systems. Method 4 uses Householder QR and also handles overdetermined
/// least-squares systems. The function returns false on rank loss without changing the destination.
///
/// # Errors
/// Returns an error for invalid shapes, channels, destination depth, method, or non-finite input.
#[wasm_bindgen(js_name = matSolveInto)]
pub fn mat_solve_into(
    coefficients: &Mat,
    right_hand_sides: &Mat,
    destination: &Mat,
    method: u32,
) -> Result<bool, JsError> {
    solve_adapter(coefficients, right_hand_sides, destination, method).map_err(JsError::from)
}

fn determinant_adapter(source: &Mat) -> Result<f64, AlgebraWasmError> {
    validate_single_channel(source)?;
    validate_square(source)?;
    let order = usize::try_from(source.rows()).map_err(|_| AlgebraError::SizeOverflow)?;
    match source.depth() {
        MatDepth::F32 => Ok(core_algebra::determinant_f32(&decode_f32(source), order)?),
        MatDepth::F64 => Ok(core_algebra::determinant(&decode_f64(source), order)?),
        depth => Err(AlgebraWasmError::FloatingSourceRequired(depth)),
    }
}

fn invert_adapter(source: &Mat, destination: &Mat, method: u32) -> Result<f64, AlgebraWasmError> {
    validate_single_channel(source)?;
    validate_square(source)?;
    match source.depth() {
        MatDepth::F32 | MatDepth::F64 => {}
        depth => return Err(AlgebraWasmError::FloatingSourceRequired(depth)),
    }
    let values = decode(source);
    let order = usize::try_from(source.rows()).map_err(|_| AlgebraError::SizeOverflow)?;
    let result = match method {
        0 => core_algebra::invert(&values, order)?,
        3 => {
            let mut identity = vec![0.0; values.len()];
            for diagonal in 0..order {
                identity[diagonal * order + diagonal] = 1.0;
            }
            core_algebra::solve_cholesky(&values, order, &identity, order)?
        }
        4 => {
            let mut identity = vec![0.0; values.len()];
            for diagonal in 0..order {
                identity[diagonal * order + diagonal] = 1.0;
            }
            core_algebra::solve_qr(&values, order, order, &identity, order)?
        }
        value => return Err(AlgebraWasmError::UnsupportedMethod(value)),
    };
    let Some(result) = result else {
        return Ok(0.0);
    };
    write_result(
        destination,
        &result,
        source.depth(),
        source.rows(),
        source.columns(),
    )?;
    Ok(1.0)
}

fn solve_adapter(
    coefficients: &Mat,
    right_hand_sides: &Mat,
    destination: &Mat,
    method: u32,
) -> Result<bool, AlgebraWasmError> {
    validate_single_channel(coefficients)?;
    validate_single_channel(right_hand_sides)?;
    for matrix in [coefficients, right_hand_sides] {
        match matrix.depth() {
            MatDepth::F32 | MatDepth::F64 => {}
            depth => return Err(AlgebraWasmError::FloatingSourceRequired(depth)),
        }
    }
    if right_hand_sides.rows() != coefficients.rows() {
        return Err(AlgebraWasmError::SolveShapeMismatch);
    }
    let coefficients_values = decode(coefficients);
    let right_hand_side_values = decode(right_hand_sides);
    let rows = usize::try_from(coefficients.rows()).map_err(|_| AlgebraError::SizeOverflow)?;
    let columns =
        usize::try_from(coefficients.columns()).map_err(|_| AlgebraError::SizeOverflow)?;
    let right_columns =
        usize::try_from(right_hand_sides.columns()).map_err(|_| AlgebraError::SizeOverflow)?;
    let result = match method {
        0 => {
            if rows != columns {
                return Err(AlgebraWasmError::DecompositionRequiresSquare);
            }
            core_algebra::solve_lu(
                &coefficients_values,
                rows,
                &right_hand_side_values,
                right_columns,
            )?
        }
        3 => {
            if rows != columns {
                return Err(AlgebraWasmError::DecompositionRequiresSquare);
            }
            core_algebra::solve_cholesky(
                &coefficients_values,
                rows,
                &right_hand_side_values,
                right_columns,
            )?
        }
        4 => core_algebra::solve_qr(
            &coefficients_values,
            rows,
            columns,
            &right_hand_side_values,
            right_columns,
        )?,
        value => return Err(AlgebraWasmError::UnsupportedMethod(value)),
    };
    let Some(result) = result else {
        return Ok(false);
    };
    write_result(
        destination,
        &result,
        right_hand_sides.depth(),
        coefficients.columns(),
        right_hand_sides.columns(),
    )?;
    Ok(true)
}

fn validate_single_channel(matrix: &Mat) -> Result<(), AlgebraWasmError> {
    if matrix.channels() == 1 {
        Ok(())
    } else {
        Err(AlgebraWasmError::SingleChannelRequired)
    }
}

fn validate_square(matrix: &Mat) -> Result<(), AlgebraWasmError> {
    if matrix.rows() == matrix.columns() {
        Ok(())
    } else {
        Err(AlgebraWasmError::SquareMatrixRequired {
            rows: matrix.rows(),
            columns: matrix.columns(),
        })
    }
}

fn validate_floating_destination(matrix: &Mat) -> Result<(), AlgebraWasmError> {
    match matrix.depth() {
        MatDepth::F32 | MatDepth::F64 => Ok(()),
        depth => Err(AlgebraWasmError::FloatingDestinationRequired(depth)),
    }
}

fn write_result(
    destination: &Mat,
    result: &[f64],
    depth: MatDepth,
    rows: u32,
    columns: u32,
) -> Result<(), AlgebraWasmError> {
    let bytes = match depth {
        MatDepth::F64 => result
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        MatDepth::F32 => encode_f32(result)?,
        depth => return Err(AlgebraWasmError::FloatingDestinationRequired(depth)),
    };
    destination.write_output(bytes, rows, columns, 1, depth)?;
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn encode_f32(values: &[f64]) -> Result<Vec<u8>, AlgebraWasmError> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for &value in values {
        if value.abs() > f64::from(f32::MAX) {
            return Err(AlgebraWasmError::DestinationOverflow);
        }
        bytes.extend_from_slice(&(value as f32).to_ne_bytes());
    }
    Ok(bytes)
}

fn decode(matrix: &Mat) -> Vec<f64> {
    let bytes = matrix.compact_bytes();
    match matrix.depth() {
        MatDepth::U8 => bytes.into_iter().map(f64::from).collect(),
        MatDepth::I8 => bytes
            .into_iter()
            .map(|value| f64::from(i8::from_ne_bytes([value])))
            .collect(),
        MatDepth::U16 => decode_chunks::<2, u16>(&bytes, u16::from_ne_bytes),
        MatDepth::I16 => decode_chunks::<2, i16>(&bytes, i16::from_ne_bytes),
        MatDepth::I32 => decode_chunks::<4, i32>(&bytes, i32::from_ne_bytes),
        MatDepth::F32 => decode_chunks::<4, f32>(&bytes, f32::from_ne_bytes),
        MatDepth::F64 => decode_chunks::<8, f64>(&bytes, f64::from_ne_bytes),
    }
}

fn decode_f32(matrix: &Mat) -> Vec<f32> {
    matrix
        .compact_bytes()
        .chunks_exact(4)
        .map(|chunk| {
            f32::from_ne_bytes(
                chunk
                    .try_into()
                    .expect("F32 matrix chunks always contain four bytes"),
            )
        })
        .collect()
}

fn decode_f64(matrix: &Mat) -> Vec<f64> {
    decode_chunks::<8, f64>(&matrix.compact_bytes(), f64::from_ne_bytes)
}

fn decode_chunks<const WIDTH: usize, T>(
    bytes: &[u8],
    decode_value: impl Fn([u8; WIDTH]) -> T,
) -> Vec<f64>
where
    f64: From<T>,
{
    bytes
        .chunks_exact(WIDTH)
        .map(|chunk| {
            let encoded: [u8; WIDTH] = chunk
                .try_into()
                .expect("matrix chunks always match the scalar byte width");
            f64::from(decode_value(encoded))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(bytes: Vec<u8>, rows: u32, columns: u32, depth: MatDepth) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, 1, depth).expect("valid matrix")
    }

    fn scalar_bytes<T: Copy>(values: &[T], encode: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
        values.iter().copied().flat_map(encode).collect()
    }

    #[test]
    fn determinant_accepts_only_nonempty_floating_square_matrices() {
        let f32_source = matrix(
            scalar_bytes(&[1_f32, 2.0, 3.0, 4.0], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F32,
        );
        let f64_source = matrix(
            scalar_bytes(&[1_f64, 2.0, 3.0, 4.0], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F64,
        );
        assert_eq!(determinant_adapter(&f32_source), Ok(-2.0));
        assert_eq!(determinant_adapter(&f64_source), Ok(-2.0));

        let typed_empty = crate::mat::mat_from_f64(&[], 0, 0, 1).expect("typed empty matrix");
        assert_eq!(
            determinant_adapter(&typed_empty),
            Err(AlgebraWasmError::Kernel(AlgebraError::EmptyMatrix))
        );
        let canonical_empty = crate::mat::mat_empty();
        assert!(determinant_adapter(&canonical_empty).is_err());

        for (depth, byte_width) in [
            (MatDepth::U8, 1),
            (MatDepth::I8, 1),
            (MatDepth::U16, 2),
            (MatDepth::I16, 2),
            (MatDepth::I32, 4),
        ] {
            let source = matrix(vec![0; 4 * byte_width], 2, 2, depth);
            assert_eq!(
                determinant_adapter(&source),
                Err(AlgebraWasmError::FloatingSourceRequired(depth))
            );
        }

        let parent_f32 = matrix(
            scalar_bytes(&[99.0_f32, 1.0, 2.0, 99.0, 3.0, 4.0], |v| {
                v.to_ne_bytes().to_vec()
            }),
            2,
            3,
            MatDepth::F32,
        );
        let roi_f32 = parent_f32.roi(0, 1, 2, 2).expect("strided F32 source");
        assert!(!roi_f32.is_continuous());
        let before = parent_f32.compact_bytes();
        assert_eq!(determinant_adapter(&roi_f32), Ok(-2.0));
        assert_eq!(parent_f32.compact_bytes(), before);
    }

    #[test]
    fn invert_writes_a_strided_f64_destination_and_leaves_it_on_singularity() {
        let source = matrix(
            scalar_bytes(&[4.0_f64, 7.0, 2.0, 6.0], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F64,
        );
        let parent = matrix(
            scalar_bytes(&[99.0_f64; 6], |v| v.to_ne_bytes().to_vec()),
            2,
            3,
            MatDepth::F64,
        );
        let destination = parent.roi(0, 1, 2, 2).expect("strided destination");
        assert_eq!(invert_adapter(&source, &destination, 0), Ok(1.0));
        let values = decode(&destination);
        for (actual, expected) in values.iter().zip([0.6, -0.7, -0.2, 0.4]) {
            assert!((actual - expected).abs() <= 1.0e-14);
        }
        assert_eq!(decode(&parent)[0].to_bits(), 99.0_f64.to_bits());
        assert_eq!(decode(&parent)[3].to_bits(), 99.0_f64.to_bits());

        let singular = matrix(
            scalar_bytes(&[1.0_f64, 2.0, 2.0, 4.0], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F64,
        );
        let before = destination.compact_bytes();
        assert_eq!(invert_adapter(&singular, &destination, 0), Ok(0.0));
        assert_eq!(destination.compact_bytes(), before);
    }

    #[test]
    fn solve_qr_writes_a_strided_f32_least_squares_destination() {
        let coefficients = matrix(
            scalar_bytes(&[1.0_f32, 0.0, 1.0, 1.0, 1.0, 2.0], |v| {
                v.to_ne_bytes().to_vec()
            }),
            3,
            2,
            MatDepth::F32,
        );
        let right_hand_side = matrix(
            scalar_bytes(&[1.0_f32, 2.0, 2.0], |v| v.to_ne_bytes().to_vec()),
            3,
            1,
            MatDepth::F32,
        );
        let parent = matrix(
            scalar_bytes(&[99.0_f32; 4], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F32,
        );
        let destination = parent.roi(0, 1, 2, 1).expect("strided destination");

        assert_eq!(
            solve_adapter(&coefficients, &right_hand_side, &destination, 4),
            Ok(true)
        );
        let values = decode(&destination);
        assert!((values[0] - 7.0 / 6.0).abs() <= 1.0e-6);
        assert!((values[1] - 0.5).abs() <= 1.0e-6);
        assert_eq!(decode(&parent)[0].to_bits(), 99.0_f64.to_bits());
        assert_eq!(decode(&parent)[2].to_bits(), 99.0_f64.to_bits());
    }

    #[test]
    fn strided_sources_and_in_place_inverse_use_compact_snapshots() {
        let parent = matrix(
            scalar_bytes(&[99.0_f64, 4.0, 7.0, 99.0, 99.0, 2.0, 6.0, 99.0], |v| {
                v.to_ne_bytes().to_vec()
            }),
            2,
            4,
            MatDepth::F64,
        );
        let source = parent.roi(0, 1, 2, 2).expect("strided source");
        assert!(!source.is_continuous());
        let before_determinant = parent.compact_bytes();
        assert!((determinant_adapter(&source).expect("determinant") - 10.0).abs() <= 1.0e-14);
        assert_eq!(parent.compact_bytes(), before_determinant);

        assert_eq!(invert_adapter(&source, &source, 0), Ok(1.0));
        let values = decode(&source);
        for (actual, expected) in values.iter().zip([0.6, -0.7, -0.2, 0.4]) {
            assert!((actual - expected).abs() <= 1.0e-14);
        }
        let parent_values = decode(&parent);
        assert_eq!(parent_values[0].to_bits(), 99.0_f64.to_bits());
        assert_eq!(parent_values[3].to_bits(), 99.0_f64.to_bits());
        assert_eq!(parent_values[4].to_bits(), 99.0_f64.to_bits());
        assert_eq!(parent_values[7].to_bits(), 99.0_f64.to_bits());
    }

    #[test]
    fn adapters_reject_invalid_metadata_before_mutating_destinations() {
        let two_channel = Mat::from_owned_bytes(
            scalar_bytes(&[1.0_f32, 2.0, 3.0, 4.0], |v| v.to_ne_bytes().to_vec()),
            1,
            2,
            2,
            MatDepth::F32,
        )
        .expect("valid matrix storage");
        assert_eq!(
            determinant_adapter(&two_channel),
            Err(AlgebraWasmError::SingleChannelRequired)
        );
        let nonsquare = matrix(
            scalar_bytes(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], |v| {
                v.to_ne_bytes().to_vec()
            }),
            2,
            3,
            MatDepth::F64,
        );
        assert_eq!(
            determinant_adapter(&nonsquare),
            Err(AlgebraWasmError::SquareMatrixRequired {
                rows: 2,
                columns: 3,
            })
        );

        let source = matrix(
            scalar_bytes(&[1.0_f64, 0.0, 0.0, 1.0], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F64,
        );
        let integer_destination = matrix(vec![17; 4], 2, 2, MatDepth::U8);
        assert_eq!(invert_adapter(&source, &integer_destination, 0), Ok(1.0));
        assert_eq!(integer_destination.depth(), MatDepth::F64);

        let destination = matrix(
            scalar_bytes(&[17.0_f64; 4], |v| v.to_ne_bytes().to_vec()),
            2,
            2,
            MatDepth::F64,
        );
        let before = destination.compact_bytes();
        assert_eq!(
            invert_adapter(&source, &destination, 99),
            Err(AlgebraWasmError::UnsupportedMethod(99))
        );
        assert_eq!(destination.compact_bytes(), before);

        let underdetermined = matrix(
            scalar_bytes(&[1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0], |v| {
                v.to_ne_bytes().to_vec()
            }),
            2,
            3,
            MatDepth::F64,
        );
        let right_hand_side = matrix(
            scalar_bytes(&[1.0_f64, 2.0], |v| v.to_ne_bytes().to_vec()),
            2,
            1,
            MatDepth::F64,
        );
        let solution = matrix(
            scalar_bytes(&[0.0_f64; 3], |v| v.to_ne_bytes().to_vec()),
            3,
            1,
            MatDepth::F64,
        );
        assert_eq!(
            solve_adapter(&underdetermined, &right_hand_side, &solution, 4),
            Err(AlgebraWasmError::Kernel(
                AlgebraError::UnderdeterminedSystem {
                    rows: 2,
                    columns: 3,
                }
            ))
        );
    }

    #[test]
    fn f32_overflow_is_atomic() {
        let source = matrix(
            scalar_bytes(&[1.0e-39_f32], |v| v.to_ne_bytes().to_vec()),
            1,
            1,
            MatDepth::F32,
        );
        let destination = matrix(
            scalar_bytes(&[7.0_f32], |v| v.to_ne_bytes().to_vec()),
            1,
            1,
            MatDepth::F32,
        );
        let before = destination.compact_bytes();
        assert_eq!(
            invert_adapter(&source, &destination, 0),
            Err(AlgebraWasmError::DestinationOverflow)
        );
        assert_eq!(destination.compact_bytes(), before);
    }
}
