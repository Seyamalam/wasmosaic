//! WebAssembly adapters for statistical and dimensional reductions.

use crate::{
    core_reductions::ScalarDepth,
    core_stats::{ReduceKind, StatsError, mean_std_dev, reduce},
    mat::{Mat, MatDepth, MatError},
};
use std::{error::Error, fmt};
use wasm_bindgen::prelude::*;

#[derive(Debug)]
enum StatsWasmError {
    Stats(StatsError),
    Matrix(MatError),
    InvalidMask,
    InvalidReduceKind(u32),
    InvalidReduceDepth(i32),
    UnsupportedReduce,
}

impl fmt::Display for StatsWasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stats(error) => error.fmt(f),
            Self::Matrix(error) => error.fmt(f),
            Self::InvalidMask => {
                f.write_str("mask must be an equally sized single-channel U8 matrix")
            }
            Self::InvalidReduceDepth(value) => {
                write!(
                    f,
                    "reduction dtype must be -1 or a scalar depth 0..=6; received {value}"
                )
            }
            Self::InvalidReduceKind(value) => write!(
                f,
                "reduction kind must be 0 (sum), 1 (average), 2 (max), or 3 (min); received {value}"
            ),
            Self::UnsupportedReduce => {
                f.write_str("reduction depth and kind combination is not supported")
            }
        }
    }
}
impl Error for StatsWasmError {}
impl From<StatsError> for StatsWasmError {
    fn from(value: StatsError) -> Self {
        Self::Stats(value)
    }
}
impl From<MatError> for StatsWasmError {
    fn from(value: MatError) -> Self {
        Self::Matrix(value)
    }
}

/// Writes per-channel population means and standard deviations to F64 destinations.
#[wasm_bindgen(js_name = matMeanStdDevInto)]
pub fn mat_mean_std_dev_into(
    source: &Mat,
    means: &Mat,
    standard_deviations: &Mat,
) -> Result<(), JsError> {
    mean_std_dev_adapter(source, means, standard_deviations, None).map_err(JsError::from)
}

/// Writes masked per-channel population means and standard deviations to F64 destinations.
#[wasm_bindgen(js_name = matMeanStdDevMaskedInto)]
pub fn mat_mean_std_dev_masked_into(
    source: &Mat,
    means: &Mat,
    standard_deviations: &Mat,
    mask: &Mat,
) -> Result<(), JsError> {
    mean_std_dev_adapter(source, means, standard_deviations, Some(mask)).map_err(JsError::from)
}

/// Reduces one matrix dimension into a destination, optionally converting scalar depth.
#[wasm_bindgen(js_name = matReduceInto)]
pub fn mat_reduce_into(
    source: &Mat,
    destination: &Mat,
    axis: i32,
    kind: u32,
    dtype: i32,
) -> Result<(), JsError> {
    reduce_adapter(source, destination, axis, kind, dtype).map_err(JsError::from)
}

fn mean_std_dev_adapter(
    source: &Mat,
    means: &Mat,
    deviations: &Mat,
    mask: Option<&Mat>,
) -> Result<(), StatsWasmError> {
    let mask_bytes = match mask {
        None => None,
        Some(mask)
            if mask.rows() == source.rows()
                && mask.columns() == source.columns()
                && mask.channels() == 1
                && mask.depth() == MatDepth::U8 =>
        {
            Some(mask.compact_bytes())
        }
        Some(_) => return Err(StatsWasmError::InvalidMask),
    };
    let result = mean_std_dev(
        &source.compact_bytes(),
        source.rows(),
        source.columns(),
        source.channels(),
        scalar_depth(source.depth()),
        mask_bytes.as_deref(),
    )?;
    means.write_output(
        f64_bytes(&result.means),
        u32::from(source.channels()),
        1,
        1,
        MatDepth::F64,
    )?;
    deviations.write_output(
        f64_bytes(&result.standard_deviations),
        u32::from(source.channels()),
        1,
        1,
        MatDepth::F64,
    )?;
    Ok(())
}

fn reduce_adapter(
    source: &Mat,
    destination: &Mat,
    axis: i32,
    kind: u32,
    dtype: i32,
) -> Result<(), StatsWasmError> {
    let (rows, columns) = match axis {
        0 => (1, source.columns()),
        1 => (source.rows(), 1),
        _ => return Err(StatsError::InvalidAxis { actual: axis }.into()),
    };
    let kind = match kind {
        0 => ReduceKind::Sum,
        1 => ReduceKind::Average,
        2 => ReduceKind::Maximum,
        3 => ReduceKind::Minimum,
        4 => ReduceKind::SumSquares,
        value => return Err(StatsWasmError::InvalidReduceKind(value)),
    };
    let output_depth = match dtype {
        -1 => source.depth(),
        0..=6 => depth_from_dtype(dtype)?,
        value => return Err(StatsWasmError::InvalidReduceDepth(value)),
    };
    match (kind, source.depth()) {
        (_, MatDepth::I8 | MatDepth::I32) => return Err(StatsWasmError::UnsupportedReduce),
        (ReduceKind::Maximum | ReduceKind::Minimum, _)
            if dtype != -1 && output_depth != source.depth() =>
        {
            return Err(StatsWasmError::UnsupportedReduce);
        }
        (ReduceKind::Sum | ReduceKind::SumSquares, MatDepth::F64)
            if output_depth == MatDepth::F32 =>
        {
            return Err(StatsWasmError::UnsupportedReduce);
        }
        (ReduceKind::Sum | ReduceKind::SumSquares, _)
            if !matches!(output_depth, MatDepth::F32 | MatDepth::F64) =>
        {
            return Err(StatsWasmError::UnsupportedReduce);
        }
        (ReduceKind::Average, MatDepth::U8)
            if matches!(output_depth, MatDepth::U8 | MatDepth::F32 | MatDepth::F64) => {}
        (ReduceKind::Average, _) if !matches!(output_depth, MatDepth::F32 | MatDepth::F64) => {
            return Err(StatsWasmError::UnsupportedReduce);
        }
        (ReduceKind::Average, MatDepth::F64) if output_depth == MatDepth::F32 => {
            return Err(StatsWasmError::UnsupportedReduce);
        }
        _ => {}
    }
    let bytes = reduce(
        &source.compact_bytes(),
        source.rows(),
        source.columns(),
        source.channels(),
        scalar_depth(source.depth()),
        scalar_depth(output_depth),
        axis,
        kind,
    )?;
    destination.write_output(bytes, rows, columns, source.channels(), output_depth)?;
    Ok(())
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn depth_from_dtype(dtype: i32) -> Result<MatDepth, StatsWasmError> {
    match dtype {
        0 => Ok(MatDepth::U8),
        1 => Ok(MatDepth::I8),
        2 => Ok(MatDepth::U16),
        3 => Ok(MatDepth::I16),
        4 => Ok(MatDepth::I32),
        5 => Ok(MatDepth::F32),
        6 => Ok(MatDepth::F64),
        value => Err(StatsWasmError::InvalidReduceDepth(value)),
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
    fn matrix(data: Vec<u8>, rows: u32, columns: u32, channels: u16, depth: MatDepth) -> Mat {
        Mat::from_owned_bytes(data, rows, columns, channels, depth).expect("valid matrix")
    }
    fn f64_values(matrix: &Mat) -> Vec<f64> {
        matrix
            .compact_bytes()
            .chunks_exact(8)
            .map(|bytes| f64::from_ne_bytes(bytes.try_into().expect("f64 width")))
            .collect()
    }

    #[test]
    fn masked_statistics_write_strided_destinations() {
        let source = matrix(vec![1, 10, 3, 30, 5, 50, 7, 70], 2, 2, 2, MatDepth::U8);
        let mask_parent = matrix(vec![9, 1, 0, 9, 9, 0, 1, 9], 2, 4, 1, MatDepth::U8);
        let mask = mask_parent.roi(0, 1, 2, 2).expect("mask ROI");
        let mean_parent = matrix(vec![0; 32], 2, 2, 1, MatDepth::F64);
        let means = mean_parent.roi(0, 1, 2, 1).expect("destination ROI");
        let deviations = matrix(vec![0; 16], 2, 1, 1, MatDepth::F64);
        mean_std_dev_adapter(&source, &means, &deviations, Some(&mask)).expect("masked statistics");
        assert_eq!(f64_values(&means), vec![4.0, 40.0]);
        assert_eq!(f64_values(&deviations), vec![3.0, 30.0]);
    }

    #[test]
    fn reduce_writes_all_depths_and_strided_roi() {
        let source = matrix(
            [1_i16, 2, 3, 4, 5, 6]
                .into_iter()
                .flat_map(i16::to_ne_bytes)
                .collect(),
            2,
            3,
            1,
            MatDepth::I16,
        );
        let destination = matrix(vec![0; 6], 1, 3, 1, MatDepth::I16);
        reduce_adapter(&source, &destination, 0, 0, 6).expect("sum converts to F64");
        assert_eq!(
            destination.compact_bytes(),
            [5.0_f64, 7.0, 9.0]
                .into_iter()
                .flat_map(f64::to_ne_bytes)
                .collect::<Vec<u8>>()
        );
        let u8_source = matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1, MatDepth::U8);
        let parent = matrix(vec![0; 5], 1, 5, 1, MatDepth::U8);
        let roi = parent.roi(0, 1, 1, 3).expect("strided destination");
        reduce_adapter(&u8_source, &roi, 0, 2, -1).expect("maximum into U8 ROI");
        assert_eq!(parent.compact_bytes(), vec![0, 4, 5, 6, 0]);
    }

    #[test]
    fn adapters_reject_wrong_output_and_mask_shapes() {
        let source = matrix(vec![1, 2], 1, 2, 1, MatDepth::U8);
        let wrong = matrix(vec![0], 1, 1, 1, MatDepth::U8);
        mean_std_dev_adapter(&source, &wrong, &wrong, None)
            .expect("destinations reallocate to F64 statistics");
        assert_eq!(wrong.depth(), MatDepth::F64);
        let mask = matrix(vec![1, 1], 1, 1, 2, MatDepth::U8);
        let output = matrix(vec![0; 8], 1, 1, 1, MatDepth::F64);
        assert!(matches!(
            mean_std_dev_adapter(&source, &output, &output, Some(&mask)),
            Err(StatsWasmError::InvalidMask)
        ));
    }
}
