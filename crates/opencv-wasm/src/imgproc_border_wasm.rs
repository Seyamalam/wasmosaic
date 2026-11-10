//! Matrix adapters for OpenCV-compatible border expansion.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    imgproc_border::{BorderError, BorderSpec, copy_make_border},
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug, Clone, PartialEq)]
enum BorderWasmError {
    ConstantChannels { expected: usize, actual: usize },
    DestinationMetadata,
    Kernel(BorderError),
    Matrix(MatError),
}

impl fmt::Display for BorderWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConstantChannels { expected, actual } => write!(
                formatter,
                "constant border has {actual} values; expected one or at least {expected}"
            ),
            Self::DestinationMetadata => formatter.write_str(
                "destination shape, channels, or depth do not match the bordered matrix",
            ),
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for BorderWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::ConstantChannels { .. } | Self::DestinationMetadata => None,
        }
    }
}

impl From<BorderError> for BorderWasmError {
    fn from(error: BorderError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for BorderWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Adds a border and returns a compact matrix.
///
/// Border type values match `OpenCV`: 0 constant, 1 replicate, 2 reflect, 3 wrap, and 4
/// reflect-101. Bit 16 requests isolated ROI behavior, which is also the default because Mat
/// adapters operate on logical ROI bytes. A one-value constant broadcasts to every channel.
/// Otherwise the adapter reads the first value for each channel, matching `OpenCV` Scalar use.
/// Integer constants use nearest-even rounding and saturation.
///
/// # Errors
///
/// Returns an error for unsupported border types, invalid constants, size overflow, or allocation
/// failure.
#[wasm_bindgen(js_name = matCopyMakeBorder)]
#[allow(clippy::too_many_arguments)]
pub fn mat_copy_make_border(
    source: &Mat,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    border_type: i32,
    constant: &[f64],
) -> Result<Mat, JsError> {
    make_border(source, top, bottom, left, right, border_type, constant).map_err(JsError::from)
}

/// Adds a border into an exactly matching caller-owned destination.
///
/// Source bytes are snapshotted before destination validation and mutation, so overlapping regions
/// behave deterministically. Failed validation leaves the destination unchanged.
///
/// # Errors
///
/// Returns an error for invalid border arguments or destination metadata.
#[wasm_bindgen(js_name = matCopyMakeBorderInto)]
#[allow(clippy::too_many_arguments)]
pub fn mat_copy_make_border_into(
    source: &Mat,
    destination: &Mat,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    border_type: i32,
    constant: &[f64],
) -> Result<(), JsError> {
    let output = make_border(source, top, bottom, left, right, border_type, constant)
        .map_err(JsError::from)?;
    destination
        .write_output(
            output.compact_bytes(),
            output.rows(),
            output.columns(),
            output.channels(),
            output.depth(),
        )
        .map_err(JsError::from)
}

#[allow(clippy::too_many_arguments)]
fn make_border(
    source: &Mat,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    border_type: i32,
    constant: &[f64],
) -> Result<Mat, BorderWasmError> {
    let constant_pixel = encode_constant(constant, source.channels(), source.depth())?;
    let pixel_width = usize::from(source.channels())
        .checked_mul(source.depth().byte_width())
        .ok_or(BorderError::SizeOverflow)?;
    let output = copy_make_border(
        &source.compact_bytes(),
        BorderSpec {
            rows: source.rows(),
            columns: source.columns(),
            pixel_width,
            top,
            bottom,
            left,
            right,
            border_type,
            constant_pixel: &constant_pixel,
        },
    )?;
    let channels = source.channels();
    let depth = source.depth();
    let (bytes, rows, columns) = output.into_parts();
    Mat::from_owned_bytes(bytes, rows, columns, channels, depth).map_err(BorderWasmError::from)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn encode_constant(
    values: &[f64],
    channels: u16,
    depth: MatDepth,
) -> Result<Vec<u8>, BorderWasmError> {
    let channels = usize::from(channels);
    if values.len() != 1 && values.len() < channels {
        return Err(BorderWasmError::ConstantChannels {
            expected: channels,
            actual: values.len(),
        });
    }
    let mut bytes = Vec::with_capacity(channels * depth.byte_width());
    for channel in 0..channels {
        let value = values[if values.len() == 1 { 0 } else { channel }];
        match depth {
            MatDepth::U8 => {
                bytes.push(saturating_integer(value, u8::MIN as i128, u8::MAX as i128) as u8);
            }
            MatDepth::I8 => bytes.extend_from_slice(
                &(saturating_integer(value, i8::MIN as i128, i8::MAX as i128) as i8).to_ne_bytes(),
            ),
            MatDepth::U16 => bytes.extend_from_slice(
                &(saturating_integer(value, u16::MIN as i128, u16::MAX as i128) as u16)
                    .to_ne_bytes(),
            ),
            MatDepth::I16 => bytes.extend_from_slice(
                &(saturating_integer(value, i16::MIN as i128, i16::MAX as i128) as i16)
                    .to_ne_bytes(),
            ),
            MatDepth::I32 => bytes.extend_from_slice(
                &(saturating_integer(value, i32::MIN as i128, i32::MAX as i128) as i32)
                    .to_ne_bytes(),
            ),
            MatDepth::F32 => bytes.extend_from_slice(&(value as f32).to_ne_bytes()),
            MatDepth::F64 => bytes.extend_from_slice(&value.to_ne_bytes()),
        }
    }
    Ok(bytes)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn saturating_integer(value: f64, minimum: i128, maximum: i128) -> i128 {
    if value.is_nan() {
        return 0;
    }
    let rounded = value.round_ties_even();
    if rounded <= minimum as f64 {
        minimum
    } else if rounded >= maximum as f64 {
        maximum
    } else {
        rounded as i128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imgproc_border::{BORDER_CONSTANT, BORDER_REFLECT_101};

    fn u8_matrix(bytes: Vec<u8>, rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, channels, MatDepth::U8)
            .expect("valid test matrix")
    }

    #[test]
    fn compacts_strided_rois_before_reflecting() {
        let parent = u8_matrix((1..=12).collect(), 3, 4, 1);
        let roi = parent.roi(0, 1, 3, 2).expect("valid strided ROI");
        let output =
            make_border(&roi, 1, 0, 1, 0, BORDER_REFLECT_101, &[0.0]).expect("valid border");
        assert_eq!(
            output.to_u8_array(),
            [7, 6, 7, 3, 2, 3, 7, 6, 7, 11, 10, 11]
        );
    }

    #[test]
    fn constant_values_round_to_even_and_saturate_per_channel() {
        let source = u8_matrix(vec![1, 2], 1, 1, 2);
        let output = make_border(&source, 0, 0, 1, 0, BORDER_CONSTANT, &[2.5, 999.0])
            .expect("valid constant border");
        assert_eq!(output.to_u8_array(), [2, 255, 1, 2]);
    }

    #[test]
    fn destination_roi_write_updates_parent() {
        let source = u8_matrix(vec![5], 1, 1, 1);
        let parent = u8_matrix(vec![0; 9], 3, 3, 1);
        let destination = parent.roi(0, 0, 2, 2).expect("valid destination ROI");
        let output =
            make_border(&source, 1, 0, 1, 0, BORDER_CONSTANT, &[9.0]).expect("valid output");
        destination
            .write_compact_bytes(&output.compact_bytes())
            .expect("write output");
        assert_eq!(parent.to_u8_array(), [9, 9, 0, 9, 5, 0, 0, 0, 0]);
    }
}
