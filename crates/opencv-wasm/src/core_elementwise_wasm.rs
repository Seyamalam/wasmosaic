use crate::core_elementwise::{
    BinaryElementwise, ElementwiseError, binary_elementwise, bitwise_elementwise,
    compare_elementwise, in_range,
};
use crate::core_norm::ScalarDepth;
use crate::core_wasm::bitwise_mask;
use crate::mat::{Mat, MatDepth, MatError};
use std::{error::Error, fmt};
use wasm_bindgen::prelude::*;

#[derive(Debug, PartialEq)]
pub(crate) enum ElementwiseWasmError {
    DepthMismatch { left: MatDepth, right: MatDepth },
    ShapeMismatch,
    BoundChannels { source: u16, bound: u16 },
    InvalidDtype(i32),
    Mask(crate::core_wasm::CoreWasmError),
    Kernel(ElementwiseError),
    Matrix(MatError),
}

impl fmt::Display for ElementwiseWasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthMismatch { left, right } => {
                write!(f, "operand depths differ: {left:?} versus {right:?}")
            }
            Self::ShapeMismatch => f.write_str("operand shapes differ"),
            Self::BoundChannels { source, bound } => write!(
                f,
                "in-range bound channel count {bound} does not match source channels {source}"
            ),
            Self::InvalidDtype(value) => {
                write!(
                    f,
                    "dtype must be -1 or a scalar depth 0..=6; received {value}"
                )
            }
            Self::Mask(error) => error.fmt(f),
            Self::Kernel(error) => error.fmt(f),
            Self::Matrix(error) => error.fmt(f),
        }
    }
}

impl Error for ElementwiseWasmError {}

impl From<ElementwiseError> for ElementwiseWasmError {
    fn from(value: ElementwiseError) -> Self {
        Self::Kernel(value)
    }
}

impl From<MatError> for ElementwiseWasmError {
    fn from(value: MatError) -> Self {
        Self::Matrix(value)
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

fn depth_from_dtype(dtype: i32) -> Result<MatDepth, ElementwiseWasmError> {
    match dtype & 7 {
        0 => Ok(MatDepth::U8),
        1 => Ok(MatDepth::I8),
        2 => Ok(MatDepth::U16),
        3 => Ok(MatDepth::I16),
        4 => Ok(MatDepth::I32),
        5 => Ok(MatDepth::F32),
        6 => Ok(MatDepth::F64),
        value => Err(ElementwiseWasmError::InvalidDtype(value)),
    }
}

fn validate_operands(left: &Mat, right: &Mat) -> Result<(), ElementwiseWasmError> {
    if left.depth() != right.depth() {
        return Err(ElementwiseWasmError::DepthMismatch {
            left: left.depth(),
            right: right.depth(),
        });
    }
    if left.rows() != right.rows()
        || left.columns() != right.columns()
        || left.channels() != right.channels()
    {
        return Err(ElementwiseWasmError::ShapeMismatch);
    }
    Ok(())
}

fn binary_dispatch(
    operation: BinaryElementwise,
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: Option<&Mat>,
    dtype: i32,
) -> Result<(), ElementwiseWasmError> {
    validate_operands(left, right)?;
    let mask_bytes = match mask {
        Some(mask) => bitwise_mask(left, mask).map_err(ElementwiseWasmError::Mask)?,
        None => None,
    };
    let output_depth = if dtype == -1 {
        left.depth()
    } else {
        depth_from_dtype(dtype)?
    };
    let seed = mask_bytes.as_ref().and_then(|_| {
        (destination.rows() == left.rows()
            && destination.columns() == left.columns()
            && destination.channels() == left.channels()
            && destination.depth() == output_depth)
            .then(|| destination.compact_bytes())
    });
    let bytes = binary_elementwise(
        &left.compact_bytes(),
        &right.compact_bytes(),
        left.rows(),
        left.columns(),
        left.channels(),
        scalar_depth(left.depth()),
        scalar_depth(output_depth),
        operation,
        mask_bytes.as_deref(),
        seed.as_deref(),
    )?;
    destination.write_output(
        bytes,
        left.rows(),
        left.columns(),
        left.channels(),
        output_depth,
    )?;
    Ok(())
}

fn bitwise_dispatch(
    kind: u8,
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: Option<&Mat>,
) -> Result<(), ElementwiseWasmError> {
    validate_operands(left, right)?;
    let mask_bytes = match mask {
        Some(mask) => bitwise_mask(left, mask).map_err(ElementwiseWasmError::Mask)?,
        None => None,
    };
    let seed = mask_bytes.as_ref().and_then(|_| {
        (destination.rows() == left.rows()
            && destination.columns() == left.columns()
            && destination.channels() == left.channels()
            && destination.depth() == left.depth())
        .then(|| destination.compact_bytes())
    });
    let bytes = bitwise_elementwise(
        &left.compact_bytes(),
        &right.compact_bytes(),
        left.rows(),
        left.columns(),
        left.channels(),
        scalar_depth(left.depth()),
        kind,
        mask_bytes.as_deref(),
        seed.as_deref(),
    )?;
    destination.write_output(
        bytes,
        left.rows(),
        left.columns(),
        left.channels(),
        left.depth(),
    )?;
    Ok(())
}

/// Writes the elementwise sum into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matAddInto)]
pub fn mat_add_into(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    binary_dispatch(BinaryElementwise::Add, left, right, destination, None, -1)
        .map_err(JsError::from)
}

/// Writes the masked elementwise sum into the destination.
///
/// # Errors
/// Returns an error when operands or the mask are inconsistent.
#[wasm_bindgen(js_name = matAddMaskedInto)]
pub fn mat_add_masked_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Add,
        left,
        right,
        destination,
        Some(mask),
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise sum into the destination at the requested dtype.
///
/// # Errors
/// Returns an error when operands differ or the dtype is unsupported.
#[wasm_bindgen(js_name = matAddDtypeInto)]
pub fn mat_add_dtype_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    dtype: i32,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Add,
        left,
        right,
        destination,
        None,
        dtype,
    )
    .map_err(JsError::from)
}

/// Writes the masked elementwise sum into the destination at the requested dtype.
///
/// # Errors
/// Returns an error when operands, mask, or dtype are inconsistent.
#[wasm_bindgen(js_name = matAddMaskedDtypeInto)]
pub fn mat_add_masked_dtype_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
    dtype: i32,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Add,
        left,
        right,
        destination,
        Some(mask),
        dtype,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise difference into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matSubtractInto)]
pub fn mat_subtract_into(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Subtract,
        left,
        right,
        destination,
        None,
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the masked elementwise difference into the destination.
///
/// # Errors
/// Returns an error when operands or the mask are inconsistent.
#[wasm_bindgen(js_name = matSubtractMaskedInto)]
pub fn mat_subtract_masked_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Subtract,
        left,
        right,
        destination,
        Some(mask),
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise difference into the destination at the requested dtype.
///
/// # Errors
/// Returns an error when operands differ or the dtype is unsupported.
#[wasm_bindgen(js_name = matSubtractDtypeInto)]
pub fn mat_subtract_dtype_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    dtype: i32,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Subtract,
        left,
        right,
        destination,
        None,
        dtype,
    )
    .map_err(JsError::from)
}

/// Writes the masked elementwise difference into the destination at the requested dtype.
///
/// # Errors
/// Returns an error when operands, mask, or dtype are inconsistent.
#[wasm_bindgen(js_name = matSubtractMaskedDtypeInto)]
pub fn mat_subtract_masked_dtype_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
    dtype: i32,
) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Subtract,
        left,
        right,
        destination,
        Some(mask),
        dtype,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise absolute difference into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matAbsdiffInto)]
pub fn mat_absdiff_into_fn(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::AbsoluteDifference,
        left,
        right,
        destination,
        None,
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise minimum into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matMinInto)]
pub fn mat_min_into(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Minimum,
        left,
        right,
        destination,
        None,
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise maximum into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matMaxInto)]
pub fn mat_max_into(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    binary_dispatch(
        BinaryElementwise::Maximum,
        left,
        right,
        destination,
        None,
        -1,
    )
    .map_err(JsError::from)
}

/// Writes the elementwise bitwise AND into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matBitwiseAndInto)]
pub fn mat_bitwise_and_into_fn(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    bitwise_dispatch(0, left, right, destination, None).map_err(JsError::from)
}

/// Writes the masked elementwise bitwise AND into the destination.
///
/// # Errors
/// Returns an error when operands or the mask are inconsistent.
#[wasm_bindgen(js_name = matBitwiseAndMaskedInto)]
pub fn mat_bitwise_and_masked_into_fn(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    bitwise_dispatch(0, left, right, destination, Some(mask)).map_err(JsError::from)
}

/// Writes the elementwise bitwise OR into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matBitwiseOrInto)]
pub fn mat_bitwise_or_into_fn(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    bitwise_dispatch(1, left, right, destination, None).map_err(JsError::from)
}

/// Writes the masked elementwise bitwise OR into the destination.
///
/// # Errors
/// Returns an error when operands or the mask are inconsistent.
#[wasm_bindgen(js_name = matBitwiseOrMaskedInto)]
pub fn mat_bitwise_or_masked_into_fn(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    bitwise_dispatch(1, left, right, destination, Some(mask)).map_err(JsError::from)
}

/// Writes the elementwise bitwise XOR into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ.
#[wasm_bindgen(js_name = matBitwiseXorInto)]
pub fn mat_bitwise_xor_into_fn(left: &Mat, right: &Mat, destination: &Mat) -> Result<(), JsError> {
    bitwise_dispatch(2, left, right, destination, None).map_err(JsError::from)
}

/// Writes the masked elementwise bitwise XOR into the destination.
///
/// # Errors
/// Returns an error when operands or the mask are inconsistent.
#[wasm_bindgen(js_name = matBitwiseXorMaskedInto)]
pub fn mat_bitwise_xor_masked_into_fn(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    bitwise_dispatch(2, left, right, destination, Some(mask)).map_err(JsError::from)
}

/// Writes the per-scalar comparison mask into the destination.
///
/// # Errors
/// Returns an error when operand depths or shapes differ or the comparison kind is invalid.
#[wasm_bindgen(js_name = matCompareInto)]
pub fn mat_compare_into(
    left: &Mat,
    right: &Mat,
    destination: &Mat,
    kind: u8,
) -> Result<(), JsError> {
    validate_operands(left, right)?;
    let bytes = compare_elementwise(
        &left.compact_bytes(),
        &right.compact_bytes(),
        left.rows(),
        left.columns(),
        left.channels(),
        scalar_depth(left.depth()),
        kind,
    )?;
    destination.write_output(
        bytes,
        left.rows(),
        left.columns(),
        left.channels(),
        MatDepth::U8,
    )?;
    Ok(())
}

/// Tests source scalars against per-channel inclusive bounds and writes a single-channel U8 map.
///
/// # Errors
/// Returns an error when bound depths differ from the source or shapes are inconsistent.
#[wasm_bindgen(js_name = matInRangeInto)]
pub fn mat_in_range_into(
    source: &Mat,
    lower: &Mat,
    upper: &Mat,
    destination: &Mat,
) -> Result<(), JsError> {
    validate_operands(source, lower)?;
    validate_operands(source, upper)?;
    if lower.channels() != 1 && lower.channels() != source.channels() {
        return Err(ElementwiseWasmError::BoundChannels {
            source: source.channels(),
            bound: lower.channels(),
        }
        .into());
    }
    let bytes = in_range(
        &source.compact_bytes(),
        &lower.compact_bytes(),
        &upper.compact_bytes(),
        source.rows(),
        source.columns(),
        source.channels(),
        scalar_depth(source.depth()),
        lower.channels(),
    )?;
    destination.write_output(bytes, source.rows(), source.columns(), 1, MatDepth::U8)?;
    Ok(())
}
