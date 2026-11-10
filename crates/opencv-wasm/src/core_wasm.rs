//! Matrix-aware WebAssembly adapters for unsigned 8-bit core kernels.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    core_ops::{
        CoreOpError, absdiff_u8, add_u8, bitwise_and_u8, bitwise_not_u8, bitwise_or_u8,
        bitwise_xor_u8, compare_eq_u8, count_non_zero_u8, in_range_u8, max_u8, min_u8, subtract_u8,
    },
    mat::{Mat, MatDepth, MatError},
};

type BinaryKernel = fn(&[u8], &[u8]) -> Result<Vec<u8>, CoreOpError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MatShape {
    rows: u32,
    columns: u32,
    channels: u16,
}

impl MatShape {
    fn of(matrix: &Mat) -> Self {
        Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            channels: matrix.channels(),
        }
    }
}

impl fmt::Display for MatShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} row(s) by {} column(s) with {} channel(s)",
            self.rows, self.columns, self.channels
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CoreWasmError {
    IncorrectDepth {
        operand: &'static str,
        actual: MatDepth,
    },
    ShapeMismatch {
        operand: &'static str,
        expected: MatShape,
        actual: MatShape,
    },
    IncorrectChannelCount {
        expected: u16,
        actual: u16,
    },
    InvalidMaskDepth {
        actual: MatDepth,
    },
    InvalidMaskChannels {
        actual: u16,
    },
    CountOverflow,
    Kernel(CoreOpError),
    Matrix(MatError),
}

impl fmt::Display for CoreWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncorrectDepth { operand, actual } => {
                write!(
                    formatter,
                    "{operand} matrix depth is {actual:?}; expected U8"
                )
            }
            Self::ShapeMismatch {
                operand,
                expected,
                actual,
            } => write!(
                formatter,
                "{operand} matrix shape is {actual}; expected {expected}"
            ),
            Self::IncorrectChannelCount { expected, actual } => write!(
                formatter,
                "matrix has {actual} channel(s); expected {expected} channel(s)"
            ),
            Self::InvalidMaskDepth { actual } => write!(
                formatter,
                "bitwise masks require U8 or I8 depth; received {actual:?}"
            ),
            Self::InvalidMaskChannels { actual } => write!(
                formatter,
                "bitwise masks require one channel; received {actual}"
            ),
            Self::CountOverflow => {
                formatter.write_str("non-zero element count exceeds the WASM integer limit")
            }
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for CoreWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::IncorrectDepth { .. }
            | Self::ShapeMismatch { .. }
            | Self::IncorrectChannelCount { .. }
            | Self::InvalidMaskDepth { .. }
            | Self::InvalidMaskChannels { .. }
            | Self::CountOverflow => None,
        }
    }
}

impl From<CoreOpError> for CoreWasmError {
    fn from(error: CoreOpError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for CoreWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Adds corresponding U8 matrix elements with saturation at 255.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matAddU8)]
pub fn mat_add_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, add_u8).map_err(JsError::from)
}

/// Subtracts corresponding U8 matrix elements with saturation at zero.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matSubtractU8)]
pub fn mat_subtract_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, subtract_u8).map_err(JsError::from)
}

/// Computes the absolute difference between corresponding U8 matrix elements.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matAbsdiffU8)]
pub fn mat_absdiff_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, absdiff_u8).map_err(JsError::from)
}

/// Computes a bitwise AND between corresponding U8 matrix elements.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matBitwiseAndU8)]
pub fn mat_bitwise_and_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, bitwise_and_u8).map_err(JsError::from)
}

/// Computes a bitwise OR between corresponding U8 matrix elements.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matBitwiseOrU8)]
pub fn mat_bitwise_or_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, bitwise_or_u8).map_err(JsError::from)
}

/// Computes a bitwise XOR between corresponding U8 matrix elements.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matBitwiseXorU8)]
pub fn mat_bitwise_xor_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, bitwise_xor_u8).map_err(JsError::from)
}

/// Inverts every bit in a U8 matrix.
///
/// # Errors
///
/// Returns an error unless the matrix has U8 depth.
#[wasm_bindgen(js_name = matBitwiseNotU8)]
pub fn mat_bitwise_not_u8(source: &Mat) -> Result<Mat, JsError> {
    apply_unary_u8(source, bitwise_not_u8).map_err(JsError::from)
}

/// Inverts every stored bit while preserving the source matrix metadata.
///
/// # Errors
/// Returns an error when matrix metadata and storage are inconsistent.
#[wasm_bindgen(js_name = matBitwiseNot)]
pub fn mat_bitwise_not(source: &Mat) -> Result<Mat, JsError> {
    apply_bitwise_not(source).map_err(JsError::from)
}

/// Writes an all-depth bitwise inversion into a caller-owned destination.
///
/// # Errors
/// Returns an error when matrix metadata, storage, or destination writes are invalid.
#[wasm_bindgen(js_name = matBitwiseNotInto)]
pub fn mat_bitwise_not_into(source: &Mat, destination: &Mat) -> Result<(), JsError> {
    apply_bitwise_not_into(source, destination, None).map_err(JsError::from)
}

/// Writes an all-depth bitwise inversion for pixels selected by an 8-bit mask.
///
/// # Errors
/// Returns an error for an invalid mask or destination write.
#[wasm_bindgen(js_name = matBitwiseNotMaskedInto)]
pub fn mat_bitwise_not_masked_into(
    source: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    let mask = bitwise_mask(source, mask).map_err(JsError::from)?;
    apply_bitwise_not_into(source, destination, mask.as_deref()).map_err(JsError::from)
}

/// Selects the smaller corresponding U8 matrix element.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matMinU8)]
pub fn mat_min_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, min_u8).map_err(JsError::from)
}

/// Selects the larger corresponding U8 matrix element.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matMaxU8)]
pub fn mat_max_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, max_u8).map_err(JsError::from)
}

/// Compares corresponding U8 elements for equality and returns a 255/0 mask.
///
/// # Errors
///
/// Returns an error unless both matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matCompareEqU8)]
pub fn mat_compare_eq_u8(left: &Mat, right: &Mat) -> Result<Mat, JsError> {
    apply_binary_u8(left, right, compare_eq_u8).map_err(JsError::from)
}

/// Tests U8 elements against corresponding inclusive lower and upper bounds.
///
/// The result has one U8 channel. A pixel is 255 only when every source channel lies within its
/// corresponding bounds.
///
/// # Errors
///
/// Returns an error unless all matrices have U8 depth and identical shapes.
#[wasm_bindgen(js_name = matInRangeU8)]
pub fn mat_in_range_u8(source: &Mat, lower_bound: &Mat, upper_bound: &Mat) -> Result<Mat, JsError> {
    apply_in_range_u8(source, lower_bound, upper_bound).map_err(JsError::from)
}

/// Counts non-zero elements in a single-channel U8 matrix.
///
/// # Errors
///
/// Returns an error unless the matrix has U8 depth and exactly one channel.
#[wasm_bindgen(js_name = matCountNonZeroU8)]
pub fn mat_count_non_zero_u8(source: &Mat) -> Result<u32, JsError> {
    count_mat_non_zero_u8(source).map_err(JsError::from)
}

fn apply_binary_u8(left: &Mat, right: &Mat, kernel: BinaryKernel) -> Result<Mat, CoreWasmError> {
    validate_u8(left, "left")?;
    validate_u8(right, "right")?;
    validate_shape(MatShape::of(left), right, "right")?;

    let output = kernel(&left.compact_bytes(), &right.compact_bytes())?;
    matrix_from_u8(output, MatShape::of(left))
}

fn apply_unary_u8(source: &Mat, kernel: fn(&[u8]) -> Vec<u8>) -> Result<Mat, CoreWasmError> {
    validate_u8(source, "source")?;
    matrix_from_u8(kernel(&source.compact_bytes()), MatShape::of(source))
}

fn apply_bitwise_not(source: &Mat) -> Result<Mat, CoreWasmError> {
    let output = bitwise_not_u8(&source.compact_bytes());
    Mat::from_owned_bytes(
        output,
        source.rows(),
        source.columns(),
        source.channels(),
        source.depth(),
    )
    .map_err(CoreWasmError::from)
}

fn apply_bitwise_not_into(
    source: &Mat,
    destination: &Mat,
    mask: Option<&[u8]>,
) -> Result<(), CoreWasmError> {
    if source.rows() == 0 || source.columns() == 0 {
        if !source.is_continuous()
            && source.rows() == 0
            && source.columns() == 0
            && source.channels() == 1
            && source.depth() == MatDepth::U8
        {
            destination.write_output(Vec::new(), 0, 0, 1, MatDepth::U8)?;
        } else {
            destination.write_empty_layout(
                source.rows(),
                source.columns(),
                source.channels(),
                source.depth(),
                true,
            )?;
        }
        return Ok(());
    }

    if mask.is_none() && destination.try_write_shared_bitwise_not(source)? {
        return Ok(());
    }

    let source_bytes = source.compact_bytes();
    let compatible = destination.rows() == source.rows()
        && destination.columns() == source.columns()
        && destination.channels() == source.channels()
        && destination.depth() == source.depth();
    let mut output = if compatible {
        destination.compact_bytes()
    } else {
        vec![0; source_bytes.len()]
    };
    let pixel_bytes = usize::from(source.channels()) * source.depth().byte_width();
    for (pixel, source_pixel) in source_bytes.chunks_exact(pixel_bytes).enumerate() {
        if mask.is_some_and(|values| values[pixel] == 0) {
            continue;
        }
        let first = pixel * pixel_bytes;
        for (target, value) in output[first..first + pixel_bytes]
            .iter_mut()
            .zip(source_pixel)
        {
            *target = !value;
        }
    }
    destination.write_output(
        output,
        source.rows(),
        source.columns(),
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

pub(crate) fn bitwise_mask(source: &Mat, mask: &Mat) -> Result<Option<Vec<u8>>, CoreWasmError> {
    if mask.byte_length() == 0 {
        return Ok(None);
    }
    if !matches!(mask.depth(), MatDepth::U8 | MatDepth::I8) {
        return Err(CoreWasmError::InvalidMaskDepth {
            actual: mask.depth(),
        });
    }
    if mask.channels() != 1 {
        return Err(CoreWasmError::InvalidMaskChannels {
            actual: mask.channels(),
        });
    }
    validate_shape(
        MatShape {
            rows: source.rows(),
            columns: source.columns(),
            channels: 1,
        },
        mask,
        "mask",
    )?;
    Ok(Some(mask.compact_bytes()))
}

fn apply_in_range_u8(
    source: &Mat,
    lower_bound: &Mat,
    upper_bound: &Mat,
) -> Result<Mat, CoreWasmError> {
    validate_u8(source, "source")?;
    validate_u8(lower_bound, "lower bound")?;
    validate_u8(upper_bound, "upper bound")?;
    let source_shape = MatShape::of(source);
    validate_shape(source_shape, lower_bound, "lower bound")?;
    validate_shape(source_shape, upper_bound, "upper bound")?;

    let component_mask = in_range_u8(
        &source.compact_bytes(),
        &lower_bound.compact_bytes(),
        &upper_bound.compact_bytes(),
    )?;
    let pixel_mask = component_mask
        .chunks_exact(usize::from(source_shape.channels))
        .map(|channels| u8::from(channels.iter().all(|&value| value == u8::MAX)) * u8::MAX)
        .collect();

    matrix_from_u8(
        pixel_mask,
        MatShape {
            channels: 1,
            ..source_shape
        },
    )
}

fn count_mat_non_zero_u8(source: &Mat) -> Result<u32, CoreWasmError> {
    validate_u8(source, "source")?;
    if source.channels() != 1 {
        return Err(CoreWasmError::IncorrectChannelCount {
            expected: 1,
            actual: source.channels(),
        });
    }

    u32::try_from(count_non_zero_u8(&source.compact_bytes()))
        .map_err(|_| CoreWasmError::CountOverflow)
}

fn validate_u8(matrix: &Mat, operand: &'static str) -> Result<(), CoreWasmError> {
    if matrix.depth() != MatDepth::U8 {
        return Err(CoreWasmError::IncorrectDepth {
            operand,
            actual: matrix.depth(),
        });
    }
    Ok(())
}

fn validate_shape(
    expected: MatShape,
    actual: &Mat,
    operand: &'static str,
) -> Result<(), CoreWasmError> {
    let actual = MatShape::of(actual);
    if actual != expected {
        return Err(CoreWasmError::ShapeMismatch {
            operand,
            expected,
            actual,
        });
    }
    Ok(())
}

fn matrix_from_u8(data: Vec<u8>, shape: MatShape) -> Result<Mat, CoreWasmError> {
    Mat::from_owned_bytes(
        data,
        shape.rows,
        shape.columns,
        shape.channels,
        MatDepth::U8,
    )
    .map_err(CoreWasmError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u8_matrix(data: Vec<u8>, rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(data, rows, columns, channels, MatDepth::U8)
            .expect("valid U8 test matrix")
    }

    #[test]
    fn binary_adapter_rejects_shape_mismatches_before_running_a_kernel() {
        let left = u8_matrix(vec![1, 2], 1, 2, 1);
        let right = u8_matrix(vec![1, 2], 2, 1, 1);

        let error =
            apply_binary_u8(&left, &right, add_u8).expect_err("different matrix shapes must fail");
        assert_eq!(
            error,
            CoreWasmError::ShapeMismatch {
                operand: "right",
                expected: MatShape {
                    rows: 1,
                    columns: 2,
                    channels: 1,
                },
                actual: MatShape {
                    rows: 2,
                    columns: 1,
                    channels: 1,
                },
            }
        );
    }

    #[test]
    fn adapter_rejects_non_u8_depths() {
        let left = u8_matrix(vec![1], 1, 1, 1);
        let right = Mat::from_owned_bytes(vec![1, 0], 1, 1, 1, MatDepth::I16)
            .expect("valid I16 test matrix");

        let error = apply_binary_u8(&left, &right, add_u8).expect_err("non-U8 matrix must fail");
        assert_eq!(
            error,
            CoreWasmError::IncorrectDepth {
                operand: "right",
                actual: MatDepth::I16,
            }
        );
    }

    #[test]
    fn binary_adapter_compacts_strided_regions_before_processing() {
        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6, 7, 8], 2, 4, 1);
        let region = source.roi(0, 1, 2, 2).expect("valid strided region");
        let right = u8_matrix(vec![10, 20, 30, 40], 2, 2, 1);

        let output = apply_binary_u8(&region, &right, add_u8).expect("matching U8 matrices");
        assert_eq!(output.compact_bytes(), [12, 23, 36, 47]);
    }

    #[test]
    fn element_wise_output_preserves_source_metadata() {
        let source = u8_matrix(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2, 2, 3);
        let output = apply_unary_u8(&source, bitwise_not_u8).expect("U8 matrix");

        assert_eq!(output.rows(), 2);
        assert_eq!(output.columns(), 2);
        assert_eq!(output.channels(), 3);
        assert_eq!(output.depth(), MatDepth::U8);
        assert!(output.is_continuous());
        assert_eq!(output.compact_bytes()[0], 255);
        assert_eq!(output.compact_bytes()[11], 244);
    }

    #[test]
    fn bitwise_not_preserves_all_depth_bits_and_masked_destinations() {
        let source = Mat::from_owned_bytes(
            vec![0x00, 0x80, 0x34, 0x12, 0xFF, 0x7F, 0xAA, 0x55],
            1,
            1,
            1,
            MatDepth::F64,
        )
        .expect("valid F64 raw bits");
        let inverted = apply_bitwise_not(&source).expect("all-depth inversion");
        assert_eq!(inverted.depth(), MatDepth::F64);
        assert_eq!(
            inverted.compact_bytes(),
            [0xFF, 0x7F, 0xCB, 0xED, 0x00, 0x80, 0x55, 0xAA]
        );

        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        let mask = Mat::from_owned_bytes(vec![1, 0, 2, 0, 3, 0], 2, 3, 1, MatDepth::I8)
            .expect("valid signed mask");
        let mask_bytes = bitwise_mask(&source, &mask).expect("valid mask");
        let populated = u8_matrix(vec![99, 98, 97, 96, 95, 94], 2, 3, 1);
        apply_bitwise_not_into(&source, &populated, mask_bytes.as_deref())
            .expect("compatible masked destination");
        assert_eq!(populated.compact_bytes(), [254, 98, 252, 96, 250, 94]);

        let fresh = crate::mat::mat_empty();
        apply_bitwise_not_into(&source, &fresh, mask_bytes.as_deref())
            .expect("fresh masked destination");
        assert_eq!(fresh.compact_bytes(), [254, 0, 252, 0, 250, 0]);

        let alias_mask = u8_matrix(vec![1, 0, 3, 0, 5, 0], 2, 3, 1);
        let mask_bytes = bitwise_mask(&alias_mask, &alias_mask).expect("aliased mask");
        apply_bitwise_not_into(&alias_mask, &alias_mask, mask_bytes.as_deref())
            .expect("fully aliased operation");
        assert_eq!(alias_mask.compact_bytes(), [254, 0, 252, 0, 250, 0]);
    }

    #[test]
    fn bitwise_not_propagates_typed_empties_and_validates_nonempty_masks() {
        let typed = Mat::empty_with_layout(0, 3, 2, MatDepth::I16, true).expect("typed empty");
        let destination = crate::mat::mat_empty();
        apply_bitwise_not_into(&typed, &destination, None).expect("typed empty inversion");
        assert_eq!(
            (
                destination.rows(),
                destination.columns(),
                destination.channels(),
                destination.depth(),
                destination.is_continuous(),
            ),
            (0, 3, 2, MatDepth::I16, true)
        );

        let source = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);
        let invalid_depth =
            Mat::from_owned_bytes(vec![0; 8], 2, 2, 1, MatDepth::U16).expect("U16 mask");
        assert_eq!(
            bitwise_mask(&source, &invalid_depth),
            Err(CoreWasmError::InvalidMaskDepth {
                actual: MatDepth::U16,
            })
        );
        let empty_wrong_metadata =
            Mat::empty_with_layout(0, 2, 3, MatDepth::F64, true).expect("typed empty mask");
        assert_eq!(bitwise_mask(&source, &empty_wrong_metadata), Ok(None));
    }

    #[test]
    fn bitwise_not_matches_two_byte_live_traversal_for_overlapping_regions() {
        let parent = u8_matrix(
            vec![1, 38, 75, 112, 149, 186, 223, 4, 41, 78, 115, 152],
            2,
            6,
            1,
        );
        let source = parent.roi(0, 0, 2, 4).expect("valid source region");
        let destination = parent.roi(0, 1, 2, 4).expect("valid destination region");

        apply_bitwise_not_into(&source, &destination, None)
            .expect("valid overlapping bitwise inversion");

        assert_eq!(
            parent.compact_bytes(),
            [1, 254, 217, 38, 143, 186, 223, 32, 251, 4, 177, 152]
        );
    }

    #[test]
    fn in_range_combines_channels_into_a_single_channel_mask() {
        let source = u8_matrix(vec![10, 20, 30, 10, 99, 30], 1, 2, 3);
        let lower = u8_matrix(vec![10, 20, 30, 10, 20, 30], 1, 2, 3);
        let upper = u8_matrix(vec![10, 20, 30, 10, 50, 30], 1, 2, 3);

        let output = apply_in_range_u8(&source, &lower, &upper).expect("matching U8 matrices");
        assert_eq!(output.rows(), 1);
        assert_eq!(output.columns(), 2);
        assert_eq!(output.channels(), 1);
        assert_eq!(output.depth(), MatDepth::U8);
        assert_eq!(output.compact_bytes(), [255, 0]);
    }

    #[test]
    fn count_handles_strided_single_channel_regions() {
        let source = u8_matrix(vec![0, 1, 9, 0, 2, 3, 9, 9], 2, 4, 1);
        let region = source.roi(0, 0, 2, 2).expect("valid strided region");
        assert_eq!(count_mat_non_zero_u8(&region), Ok(3));
    }

    #[test]
    fn count_rejects_multi_channel_input() {
        let source = u8_matrix(vec![1, 2], 1, 1, 2);
        assert_eq!(
            count_mat_non_zero_u8(&source),
            Err(CoreWasmError::IncorrectChannelCount {
                expected: 1,
                actual: 2,
            })
        );
    }
}
