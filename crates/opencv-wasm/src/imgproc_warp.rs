use std::{error::Error, fmt};

use crate::{
    imgproc_border::{BORDER_CONSTANT, BORDER_ISOLATED, BORDER_REPLICATE, border_index},
    mat::{Mat, MatDepth, MatError},
};

const INTER_NEAREST: i32 = 0;
const INTER_LINEAR: i32 = 1;
const INTER_MASK: i32 = 7;
const WARP_INVERSE_MAP: i32 = 16;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WarpError {
    EmptySource,
    InvalidBorderType(i32),
    InvalidInterpolation(i32),
    InvalidTarget { width: i32, height: i32 },
    InvalidTransform { rows: u32 },
    NonFiniteTransform,
    Matrix(MatError),
    SingularTransform,
    UnsupportedDepth(MatDepth),
}

impl fmt::Display for WarpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => formatter.write_str("image warp source must not be empty"),
            Self::InvalidBorderType(value) => write!(formatter, "unsupported border type {value}"),
            Self::InvalidInterpolation(value) => {
                write!(formatter, "unsupported image warp interpolation {value}")
            }
            Self::InvalidTarget { width, height } => write!(
                formatter,
                "warpAffine target dimensions must be positive; received {width} by {height}"
            ),
            Self::InvalidTransform { rows } => write!(
                formatter,
                "transform must be a {rows}x3 single-channel F32/F64 Mat"
            ),
            Self::NonFiniteTransform => {
                formatter.write_str("transform coefficients must be finite")
            }
            Self::Matrix(error) => error.fmt(formatter),
            Self::SingularTransform => formatter.write_str("warpAffine transform is singular"),
            Self::UnsupportedDepth(depth) => {
                write!(
                    formatter,
                    "image warp source depth {depth:?} is not implemented"
                )
            }
        }
    }
}

impl Error for WarpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for WarpError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_into(
    source: &Mat,
    destination: &Mat,
    transform: &Mat,
    width: i32,
    height: i32,
    flags: i32,
    border_type: i32,
    border_value: &[f64],
) -> Result<(), WarpError> {
    if source.rows() == 0 || source.columns() == 0 {
        return Err(WarpError::EmptySource);
    }
    if source.depth() != MatDepth::U8 {
        return Err(WarpError::UnsupportedDepth(source.depth()));
    }
    if width <= 0 || height <= 0 {
        return Err(WarpError::InvalidTarget { width, height });
    }
    let interpolation = flags & INTER_MASK;
    if !matches!(interpolation, INTER_NEAREST | INTER_LINEAR) {
        return Err(WarpError::InvalidInterpolation(interpolation));
    }
    if !matches!(
        border_type & !BORDER_ISOLATED,
        BORDER_CONSTANT | BORDER_REPLICATE
    ) {
        return Err(WarpError::InvalidBorderType(border_type));
    }
    let coefficients = read_transform::<6>(transform, 2)?;
    let inverse = if flags & WARP_INVERSE_MAP != 0 {
        coefficients
    } else {
        invert_transform(coefficients)?
    };
    let target_width = width.unsigned_abs();
    let target_height = height.unsigned_abs();
    let channels = usize::from(source.channels());
    let input = source.compact_bytes();
    let output_length = usize::try_from(target_width)
        .ok()
        .and_then(|width| {
            usize::try_from(target_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(MatError::BufferSizeOverflow)?;
    let mut output = vec![0; output_length];
    for target_y in 0..target_height {
        for target_x in 0..target_width {
            let source_x =
                inverse[0] * f64::from(target_x) + inverse[1] * f64::from(target_y) + inverse[2];
            let source_y =
                inverse[3] * f64::from(target_x) + inverse[4] * f64::from(target_y) + inverse[5];
            for channel in 0..channels {
                let value = if interpolation == INTER_NEAREST {
                    sample_nearest(
                        &input,
                        source.rows(),
                        source.columns(),
                        channels,
                        source_x,
                        source_y,
                        channel,
                        border_type,
                        border_value,
                    )
                } else {
                    sample_linear(
                        &input,
                        source.rows(),
                        source.columns(),
                        channels,
                        source_x,
                        source_y,
                        channel,
                        border_type,
                        border_value,
                    )
                };
                output[pixel_offset(target_y, target_x, target_width, channels, channel)] = value;
            }
        }
    }
    destination.write_output(
        output,
        target_height,
        target_width,
        source.channels(),
        MatDepth::U8,
    )?;
    Ok(())
}

pub(crate) fn read_transform<const N: usize>(
    transform: &Mat,
    rows: u32,
) -> Result<[f64; N], WarpError> {
    if transform.rows() != rows || transform.columns() != 3 || transform.channels() != 1 {
        return Err(WarpError::InvalidTransform { rows });
    }
    let bytes = transform.compact_bytes();
    let values = match transform.depth() {
        MatDepth::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f64::from(f32::from_ne_bytes(bytes.try_into().expect("f32 bytes"))))
            .collect::<Vec<_>>(),
        MatDepth::F64 => bytes
            .chunks_exact(8)
            .map(|bytes| f64::from_ne_bytes(bytes.try_into().expect("f64 bytes")))
            .collect::<Vec<_>>(),
        _ => return Err(WarpError::InvalidTransform { rows }),
    };
    values
        .try_into()
        .map_err(|_| WarpError::InvalidTransform { rows })
}

fn invert_transform(transform: [f64; 6]) -> Result<[f64; 6], WarpError> {
    let determinant = transform[0] * transform[4] - transform[1] * transform[3];
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(WarpError::SingularTransform);
    }
    let inverse = 1.0 / determinant;
    Ok([
        transform[4] * inverse,
        -transform[1] * inverse,
        (transform[1] * transform[5] - transform[4] * transform[2]) * inverse,
        -transform[3] * inverse,
        transform[0] * inverse,
        (transform[3] * transform[2] - transform[0] * transform[5]) * inverse,
    ])
}

#[allow(clippy::too_many_arguments)]
fn sample_nearest(
    input: &[u8],
    rows: u32,
    columns: u32,
    channels: usize,
    x: f64,
    y: f64,
    channel: usize,
    border_type: i32,
    border_value: &[f64],
) -> u8 {
    sample_at(
        input,
        rows,
        columns,
        channels,
        round_i64(x),
        round_i64(y),
        channel,
        border_type,
        border_value,
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_linear(
    input: &[u8],
    rows: u32,
    columns: u32,
    channels: usize,
    x: f64,
    y: f64,
    channel: usize,
    border_type: i32,
    border_value: &[f64],
) -> u8 {
    let left = floor_i64(x);
    let top = floor_i64(y);
    let horizontal = x - left as f64;
    let vertical = y - top as f64;
    let top_left = f64::from(sample_at(
        input,
        rows,
        columns,
        channels,
        left,
        top,
        channel,
        border_type,
        border_value,
    ));
    let top_right = f64::from(sample_at(
        input,
        rows,
        columns,
        channels,
        left + 1,
        top,
        channel,
        border_type,
        border_value,
    ));
    let bottom_left = f64::from(sample_at(
        input,
        rows,
        columns,
        channels,
        left,
        top + 1,
        channel,
        border_type,
        border_value,
    ));
    let bottom_right = f64::from(sample_at(
        input,
        rows,
        columns,
        channels,
        left + 1,
        top + 1,
        channel,
        border_type,
        border_value,
    ));
    round_u8(
        (top_left * (1.0 - horizontal) + top_right * horizontal) * (1.0 - vertical)
            + (bottom_left * (1.0 - horizontal) + bottom_right * horizontal) * vertical,
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_at(
    input: &[u8],
    rows: u32,
    columns: u32,
    channels: usize,
    x: i64,
    y: i64,
    channel: usize,
    border_type: i32,
    border_value: &[f64],
) -> u8 {
    let mapped_x = border_index(x, columns, border_type & !BORDER_ISOLATED);
    let mapped_y = border_index(y, rows, border_type & !BORDER_ISOLATED);
    match (mapped_y, mapped_x) {
        (Some(row), Some(column)) => input[pixel_offset(row, column, columns, channels, channel)],
        _ => round_u8(*border_value.get(channel).unwrap_or(&0.0)),
    }
}

fn floor_i64(value: f64) -> i64 {
    #[allow(clippy::cast_possible_truncation)]
    {
        value.floor().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }
}

fn round_i64(value: f64) -> i64 {
    #[allow(clippy::cast_possible_truncation)]
    {
        value
            .round_ties_even()
            .clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }
}

fn round_u8(value: f64) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        value.round_ties_even().clamp(0.0, 255.0) as u8
    }
}

fn pixel_offset(row: u32, column: u32, columns: u32, channels: usize, channel: usize) -> usize {
    (usize::try_from(row).expect("row fits usize")
        * usize::try_from(columns).expect("columns fit usize")
        + usize::try_from(column).expect("column fits usize"))
        * channels
        + channel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64, mat_from_u8};

    #[test]
    fn forward_translation_is_inverted_for_sampling() {
        let source = mat_from_u8(&[1, 2, 3, 4, 5, 6], 2, 3, 1).expect("source");
        let transform = mat_from_f64(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2, 3, 1).expect("transform");
        let destination = mat_empty();
        warp_affine_into(
            &source,
            &destination,
            &transform,
            3,
            2,
            INTER_NEAREST,
            BORDER_CONSTANT,
            &[9.0, 0.0, 0.0, 0.0],
        )
        .expect("warp");
        assert_eq!(destination.compact_bytes(), [9, 1, 2, 9, 4, 5]);
    }
}
