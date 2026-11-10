//! WebAssembly adapters for matrix norms and destination-mutating normalization.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    core_norm::{NormError, ScalarDepth, norm, normalize_into},
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug, PartialEq, Eq)]
enum NormWasmError {
    ShapeMismatch,
    InvalidMask,
    Kernel(NormError),
    Destination(MatError),
}

impl fmt::Display for NormWasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch => f.write_str(
                "source and destination matrices must have identical rows, columns, and channels",
            ),
            Self::InvalidMask => {
                f.write_str("mask must be an equally sized single-channel U8 matrix")
            }
            Self::Kernel(error) => error.fmt(f),
            Self::Destination(error) => error.fmt(f),
        }
    }
}

impl Error for NormWasmError {}
impl From<NormError> for NormWasmError {
    fn from(value: NormError) -> Self {
        Self::Kernel(value)
    }
}
impl From<MatError> for NormWasmError {
    fn from(value: MatError) -> Self {
        Self::Destination(value)
    }
}

/// Computes an absolute matrix norm for every supported scalar depth.
#[wasm_bindgen(js_name = matNorm)]
pub fn mat_norm(source: &Mat, norm_type: u32) -> Result<f64, JsError> {
    norm_adapter(source, None, None, norm_type).map_err(JsError::from)
}

/// Computes an absolute matrix norm over pixels selected by an unsigned byte mask.
#[wasm_bindgen(js_name = matNormMasked)]
pub fn mat_norm_masked(source: &Mat, norm_type: u32, mask: &Mat) -> Result<f64, JsError> {
    norm_adapter(source, None, Some(mask), norm_type).map_err(JsError::from)
}

/// Computes the norm of the element-wise difference between equally typed matrices.
#[wasm_bindgen(js_name = matNormDiff)]
pub fn mat_norm_diff(first: &Mat, second: &Mat, norm_type: u32) -> Result<f64, JsError> {
    norm_adapter(first, Some(second), None, norm_type).map_err(JsError::from)
}

/// Computes a masked norm of the element-wise difference between equally typed matrices.
#[wasm_bindgen(js_name = matNormDiffMasked)]
pub fn mat_norm_diff_masked(
    first: &Mat,
    second: &Mat,
    norm_type: u32,
    mask: &Mat,
) -> Result<f64, JsError> {
    norm_adapter(first, Some(second), Some(mask), norm_type).map_err(JsError::from)
}

/// Normalizes a matrix into an equally shaped destination, optionally converting scalar depth.
#[wasm_bindgen(js_name = matNormalizeInto)]
pub fn mat_normalize_into(
    source: &Mat,
    destination: &Mat,
    alpha: f64,
    beta: f64,
    norm_type: u32,
    dtype: i32,
) -> Result<(), JsError> {
    normalize_adapter(source, destination, None, alpha, beta, norm_type, dtype)
        .map_err(JsError::from)
}

/// Normalizes selected pixels into a destination while preserving unselected pixels.
#[wasm_bindgen(js_name = matNormalizeMaskedInto)]
pub fn mat_normalize_masked_into(
    source: &Mat,
    destination: &Mat,
    alpha: f64,
    beta: f64,
    norm_type: u32,
    dtype: i32,
    mask: &Mat,
) -> Result<(), JsError> {
    normalize_adapter(
        source,
        destination,
        Some(mask),
        alpha,
        beta,
        norm_type,
        dtype,
    )
    .map_err(JsError::from)
}

fn norm_adapter(
    first: &Mat,
    second: Option<&Mat>,
    mask: Option<&Mat>,
    norm_type: u32,
) -> Result<f64, NormWasmError> {
    if second.is_some_and(|other| !same_layout_and_depth(first, other)) {
        return Err(NormWasmError::ShapeMismatch);
    }
    let mask_bytes = checked_mask(first, mask)?;
    let second_bytes = second.map(Mat::compact_bytes);
    let resolved_type = if second.is_none() {
        norm_type & !(crate::core_norm::NORM_RELATIVE | crate::core_norm::NORM_MINMAX)
    } else {
        norm_type
    };
    norm(
        &first.compact_bytes(),
        second_bytes.as_deref(),
        mask_bytes.as_deref(),
        first.rows(),
        first.columns(),
        first.channels(),
        scalar_depth(first.depth()),
        resolved_type,
    )
    .map_err(NormWasmError::from)
}

fn normalize_adapter(
    source: &Mat,
    destination: &Mat,
    mask: Option<&Mat>,
    alpha: f64,
    beta: f64,
    norm_type: u32,
    dtype: i32,
) -> Result<(), NormWasmError> {
    let output_depth = match dtype {
        -1 => source.depth(),
        0 => MatDepth::U8,
        1 => MatDepth::I8,
        2 => MatDepth::U16,
        3 => MatDepth::I16,
        4 => MatDepth::I32,
        5 => MatDepth::F32,
        6 => MatDepth::F64,
        _ => return Err(NormWasmError::ShapeMismatch),
    };
    let mask_bytes = checked_mask(source, mask)?;
    let byte_width = output_depth.byte_width();
    let bytes_length = source.compact_bytes().len() * byte_width / source.depth().byte_width();
    let mut destination_bytes =
        if same_layout(source, destination) && destination.depth() == output_depth {
            destination.compact_bytes()
        } else {
            vec![0u8; bytes_length]
        };
    normalize_into(
        &source.compact_bytes(),
        &mut destination_bytes,
        mask_bytes.as_deref(),
        source.rows(),
        source.columns(),
        source.channels(),
        scalar_depth(source.depth()),
        scalar_depth(output_depth),
        alpha,
        beta,
        norm_type,
    )?;
    destination.write_output(
        destination_bytes,
        source.rows(),
        source.columns(),
        source.channels(),
        output_depth,
    )?;
    Ok(())
}

fn checked_mask(source: &Mat, mask: Option<&Mat>) -> Result<Option<Vec<u8>>, NormWasmError> {
    match mask {
        None => Ok(None),
        Some(value)
            if value.rows() == source.rows()
                && value.columns() == source.columns()
                && value.channels() == 1
                && value.depth() == MatDepth::U8 =>
        {
            Ok(Some(value.compact_bytes()))
        }
        Some(_) => Err(NormWasmError::InvalidMask),
    }
}

fn same_layout(first: &Mat, second: &Mat) -> bool {
    first.rows() == second.rows()
        && first.columns() == second.columns()
        && first.channels() == second.channels()
}

fn same_layout_and_depth(first: &Mat, second: &Mat) -> bool {
    same_layout(first, second) && first.depth() == second.depth()
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
        Mat::from_owned_bytes(bytes, rows, columns, channels, depth).expect("valid matrix")
    }

    #[test]
    fn strided_rois_are_compacted_for_norm_and_written_back_for_normalize() {
        let parent = matrix(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, 1, MatDepth::U8);
        let source = parent.roi(0, 1, 2, 2).expect("source ROI");
        assert_eq!(
            norm_adapter(&source, None, None, crate::core_norm::NORM_L1),
            Ok(16.0)
        );

        let target_parent = matrix(vec![99; 9], 3, 3, 1, MatDepth::U8);
        let target = target_parent.roi(1, 1, 2, 2).expect("target ROI");
        normalize_adapter(
            &source,
            &target,
            None,
            10.0,
            0.0,
            crate::core_norm::NORM_L1,
            -1,
        )
        .unwrap();
        assert_eq!(
            target_parent.compact_bytes(),
            [99, 99, 99, 99, 1, 2, 99, 3, 4]
        );
    }

    #[test]
    fn masked_destination_preserves_unselected_pixels() {
        let source = matrix(vec![3, 4], 1, 2, 1, MatDepth::U8);
        let target = matrix(vec![77, 88], 1, 2, 1, MatDepth::U8);
        let mask = matrix(vec![1, 0], 1, 2, 1, MatDepth::U8);
        normalize_adapter(
            &source,
            &target,
            Some(&mask),
            9.0,
            0.0,
            crate::core_norm::NORM_L2,
            -1,
        )
        .unwrap();
        assert_eq!(target.compact_bytes(), [9, 88]);
    }
}
