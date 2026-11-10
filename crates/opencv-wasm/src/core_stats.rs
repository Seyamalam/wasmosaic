//! Statistical and dimensional reductions over compact, native-endian matrix bytes.

use crate::core_reductions::ScalarDepth;
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceKind {
    Sum,
    Average,
    Maximum,
    Minimum,
    SumSquares,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeanStdDev {
    pub(crate) means: Vec<f64>,
    pub(crate) standard_deviations: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatsError {
    EmptyDimensions,
    EmptyChannels,
    BufferSizeOverflow,
    IncorrectBufferLength { expected: usize, actual: usize },
    InvalidAxis { actual: i32 },
    IncorrectMaskLength { expected: usize, actual: usize },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => f.write_str("matrix dimensions must be greater than zero"),
            Self::EmptyChannels => f.write_str("matrix channels must be greater than zero"),
            Self::BufferSizeOverflow => f.write_str("matrix byte length overflowed"),
            Self::IncorrectBufferLength { expected, actual } => write!(
                f,
                "matrix buffer has {actual} bytes; expected {expected} bytes"
            ),
            Self::InvalidAxis { actual } => {
                write!(f, "reduction axis must be 0 or 1; received {actual}")
            }
            Self::IncorrectMaskLength { expected, actual } => {
                write!(f, "mask has {actual} bytes; expected {expected} bytes")
            }
        }
    }
}
impl Error for StatsError {}

/// Calculates per-channel population mean and standard deviation.
#[allow(clippy::cast_precision_loss)] // WASM storage bounds keep the selected count exactly representable.
pub(crate) fn mean_std_dev(
    data: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    mask: Option<&[u8]>,
) -> Result<MeanStdDev, StatsError> {
    let scalars = validate(data, rows, columns, channels, depth)?;
    let channel_count = usize::from(channels);
    let pixels = scalars / channel_count;
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(StatsError::IncorrectMaskLength {
                expected: pixels,
                actual: mask.len(),
            });
        }
    }
    let integral = matches!(
        depth,
        ScalarDepth::U8 | ScalarDepth::I8 | ScalarDepth::U16 | ScalarDepth::I16 | ScalarDepth::I32
    );
    let mut integer_sums = vec![0_i128; channel_count];
    let mut integer_squares = vec![0_i128; channel_count];
    let mut sums = vec![0.0; channel_count];
    let mut squares = vec![0.0; channel_count];
    let mut selected = 0_u64;
    for pixel in 0..pixels {
        if mask.is_some_and(|values| values[pixel] == 0) {
            continue;
        }
        selected += 1;
        for channel in 0..channel_count {
            let value = decode(data, pixel * channel_count + channel, depth);
            if integral {
                let integer = value as i64 as i128;
                integer_sums[channel] += integer;
                integer_squares[channel] += integer * integer;
            } else {
                sums[channel] += value;
                squares[channel] += value * value;
            }
        }
    }
    let count = selected.max(1) as f64;
    let (resolved_sums, resolved_squares): (Vec<f64>, Vec<f64>) = if integral {
        (
            integer_sums.iter().map(|sum| *sum as f64).collect(),
            integer_squares.iter().map(|sum| *sum as f64).collect(),
        )
    } else {
        (sums, squares)
    };
    let means = resolved_sums
        .iter()
        .map(|sum| sum / count)
        .collect::<Vec<_>>();
    let standard_deviations = if selected == 0 {
        vec![0.0; channel_count]
    } else if integral {
        integer_squares
            .iter()
            .zip(&means)
            .map(|(square, mean)| {
                let high = *square as f64;
                let low = (*square - high as i128) as f64;
                (high / count + low / count - mean * mean).sqrt()
            })
            .collect()
    } else {
        resolved_squares
            .iter()
            .zip(&means)
            .map(|(square, mean)| (square / count - mean * mean).sqrt())
            .collect()
    };
    Ok(MeanStdDev {
        means,
        standard_deviations,
    })
}

/// Reduces rows (`axis == 0`) or columns (`axis == 1`) and retains channels.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)] // Mirrors OpenCV's reduction contract; matrix extents are WASM-bounded.
pub(crate) fn reduce(
    data: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    source_depth: ScalarDepth,
    destination_depth: ScalarDepth,
    axis: i32,
    kind: ReduceKind,
) -> Result<Vec<u8>, StatsError> {
    validate(data, rows, columns, channels, source_depth)?;
    if axis != 0 && axis != 1 {
        return Err(StatsError::InvalidAxis { actual: axis });
    }
    let rows = rows as usize;
    let columns = columns as usize;
    let channels = usize::from(channels);
    let major_count = if axis == 0 { columns } else { rows };
    let reduction_count = if axis == 0 { rows } else { columns };
    let capacity = major_count
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(byte_width(destination_depth)))
        .ok_or(StatsError::BufferSizeOverflow)?;
    let mut output = Vec::with_capacity(capacity);
    for major in 0..major_count {
        for channel in 0..channels {
            let mut accumulator = match kind {
                ReduceKind::Sum | ReduceKind::Average | ReduceKind::SumSquares => 0.0,
                ReduceKind::Maximum | ReduceKind::Minimum => f64::NAN,
            };
            for position in 0..reduction_count {
                let (row, column) = if axis == 0 {
                    (position, major)
                } else {
                    (major, position)
                };
                let value = decode(
                    data,
                    (row * columns + column) * channels + channel,
                    source_depth,
                );
                accumulator = match kind {
                    ReduceKind::Sum | ReduceKind::Average => accumulator + value,
                    ReduceKind::SumSquares => accumulator + value * value,
                    ReduceKind::Maximum if accumulator.is_nan() || value > accumulator => value,
                    ReduceKind::Minimum if accumulator.is_nan() || value < accumulator => value,
                    ReduceKind::Maximum | ReduceKind::Minimum => accumulator,
                };
            }
            if kind == ReduceKind::Average {
                accumulator /= reduction_count as f64;
            }
            encode(accumulator, destination_depth, &mut output);
        }
    }
    Ok(output)
}

fn validate(
    data: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
) -> Result<usize, StatsError> {
    if rows == 0 || columns == 0 {
        return Err(StatsError::EmptyDimensions);
    }
    if channels == 0 {
        return Err(StatsError::EmptyChannels);
    }
    let scalars = (rows as usize)
        .checked_mul(columns as usize)
        .and_then(|v| v.checked_mul(usize::from(channels)))
        .ok_or(StatsError::BufferSizeOverflow)?;
    let expected = scalars
        .checked_mul(byte_width(depth))
        .ok_or(StatsError::BufferSizeOverflow)?;
    if data.len() != expected {
        return Err(StatsError::IncorrectBufferLength {
            expected,
            actual: data.len(),
        });
    }
    Ok(scalars)
}

const fn byte_width(depth: ScalarDepth) -> usize {
    match depth {
        ScalarDepth::U8 | ScalarDepth::I8 => 1,
        ScalarDepth::U16 | ScalarDepth::I16 => 2,
        ScalarDepth::I32 | ScalarDepth::F32 => 4,
        ScalarDepth::F64 => 8,
    }
}

fn decode(data: &[u8], index: usize, depth: ScalarDepth) -> f64 {
    let start = index * byte_width(depth);
    match depth {
        ScalarDepth::U8 => f64::from(data[start]),
        ScalarDepth::I8 => f64::from(i8::from_ne_bytes([data[start]])),
        ScalarDepth::U16 => f64::from(u16::from_ne_bytes(
            data[start..start + 2].try_into().expect("validated width"),
        )),
        ScalarDepth::I16 => f64::from(i16::from_ne_bytes(
            data[start..start + 2].try_into().expect("validated width"),
        )),
        ScalarDepth::I32 => f64::from(i32::from_ne_bytes(
            data[start..start + 4].try_into().expect("validated width"),
        )),
        ScalarDepth::F32 => f64::from(f32::from_ne_bytes(
            data[start..start + 4].try_into().expect("validated width"),
        )),
        ScalarDepth::F64 => {
            f64::from_ne_bytes(data[start..start + 8].try_into().expect("validated width"))
        }
    }
}

#[allow(clippy::float_cmp)] // Exactly representable half-integers intentionally select ties-to-even.
fn nearest_even(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let floor = value.floor();
    if value - floor == 0.5 {
        if floor % 2.0 == 0.0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Values are rounded and saturated immediately before conversion.
fn encode(value: f64, depth: ScalarDepth, output: &mut Vec<u8>) {
    match depth {
        ScalarDepth::U8 => output.push(if value.is_nan() {
            0
        } else {
            nearest_even(value).clamp(0.0, f64::from(u8::MAX)) as u8
        }),
        ScalarDepth::I8 => output.extend_from_slice(
            &(if value.is_nan() {
                0
            } else {
                nearest_even(value).clamp(f64::from(i8::MIN), f64::from(i8::MAX)) as i8
            })
            .to_ne_bytes(),
        ),
        ScalarDepth::U16 => output.extend_from_slice(
            &(if value.is_nan() {
                0
            } else {
                nearest_even(value).clamp(0.0, f64::from(u16::MAX)) as u16
            })
            .to_ne_bytes(),
        ),
        ScalarDepth::I16 => output.extend_from_slice(
            &(if value.is_nan() {
                0
            } else {
                nearest_even(value).clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
            })
            .to_ne_bytes(),
        ),
        ScalarDepth::I32 => output.extend_from_slice(
            &(if value.is_nan() {
                0
            } else {
                nearest_even(value).clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
            })
            .to_ne_bytes(),
        ),
        ScalarDepth::F32 => output.extend_from_slice(&(value as f32).to_ne_bytes()),
        ScalarDepth::F64 => output.extend_from_slice(&value.to_ne_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes<T: Copy>(values: &[T], encode: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
        values.iter().copied().flat_map(encode).collect()
    }

    #[test]
    fn masked_mean_std_dev_tracks_each_interleaved_channel() {
        let result = mean_std_dev(
            &[1, 10, 3, 30, 5, 50, 7, 70],
            2,
            2,
            2,
            ScalarDepth::U8,
            Some(&[1, 0, 0, 1]),
        )
        .expect("valid matrix");
        assert_eq!(
            result,
            MeanStdDev {
                means: vec![4.0, 40.0],
                standard_deviations: vec![3.0, 30.0]
            }
        );
    }

    #[test]
    fn mean_std_dev_supports_every_depth_and_empty_selection() {
        let cases = [
            (ScalarDepth::U8, vec![2]),
            (ScalarDepth::I8, (-2_i8).to_ne_bytes().to_vec()),
            (ScalarDepth::U16, 2_u16.to_ne_bytes().to_vec()),
            (ScalarDepth::I16, (-2_i16).to_ne_bytes().to_vec()),
            (ScalarDepth::I32, (-2_i32).to_ne_bytes().to_vec()),
            (ScalarDepth::F32, (-2_f32).to_ne_bytes().to_vec()),
            (ScalarDepth::F64, (-2_f64).to_ne_bytes().to_vec()),
        ];
        for (depth, source) in cases {
            assert_eq!(
                mean_std_dev(&source, 1, 1, 1, depth, Some(&[0])).expect("supported depth"),
                MeanStdDev {
                    means: vec![0.0],
                    standard_deviations: vec![0.0]
                }
            );
        }
    }

    #[test]
    fn reduce_covers_axes_kinds_channels_and_conversion() {
        let source = bytes(&[1_i16, 2, 3, 4, 5, 6], |v| v.to_ne_bytes().to_vec());
        assert_eq!(
            reduce(
                &source,
                2,
                3,
                1,
                ScalarDepth::I16,
                ScalarDepth::I32,
                0,
                ReduceKind::Sum
            )
            .expect("sum"),
            bytes(&[5_i32, 7, 9], |v| v.to_ne_bytes().to_vec())
        );
        assert_eq!(
            reduce(
                &source,
                2,
                3,
                1,
                ScalarDepth::I16,
                ScalarDepth::F64,
                1,
                ReduceKind::Average
            )
            .expect("average"),
            bytes(&[2_f64, 5.0], |v| v.to_ne_bytes().to_vec())
        );
        assert_eq!(
            reduce(
                &source,
                2,
                3,
                1,
                ScalarDepth::I16,
                ScalarDepth::I16,
                0,
                ReduceKind::Maximum
            )
            .expect("max"),
            bytes(&[4_i16, 5, 6], |v| v.to_ne_bytes().to_vec())
        );
        assert_eq!(
            reduce(
                &source,
                2,
                3,
                1,
                ScalarDepth::I16,
                ScalarDepth::I16,
                0,
                ReduceKind::Minimum
            )
            .expect("min"),
            bytes(&[1_i16, 2, 3], |v| v.to_ne_bytes().to_vec())
        );
        let channels = bytes(&[200_u16, 10, 200, 20], |v| v.to_ne_bytes().to_vec());
        assert_eq!(
            reduce(
                &channels,
                2,
                1,
                2,
                ScalarDepth::U16,
                ScalarDepth::U8,
                0,
                ReduceKind::Sum
            )
            .expect("channels"),
            vec![255, 30]
        );
    }

    #[test]
    fn reduce_accepts_every_source_and_destination_depth() {
        let sources = [
            (ScalarDepth::U8, vec![2]),
            (ScalarDepth::I8, 2_i8.to_ne_bytes().to_vec()),
            (ScalarDepth::U16, 2_u16.to_ne_bytes().to_vec()),
            (ScalarDepth::I16, 2_i16.to_ne_bytes().to_vec()),
            (ScalarDepth::I32, 2_i32.to_ne_bytes().to_vec()),
            (ScalarDepth::F32, 2_f32.to_ne_bytes().to_vec()),
            (ScalarDepth::F64, 2_f64.to_ne_bytes().to_vec()),
        ];
        for (source_depth, source) in sources {
            for destination_depth in [
                ScalarDepth::U8,
                ScalarDepth::I8,
                ScalarDepth::U16,
                ScalarDepth::I16,
                ScalarDepth::I32,
                ScalarDepth::F32,
                ScalarDepth::F64,
            ] {
                let output = reduce(
                    &source,
                    1,
                    1,
                    1,
                    source_depth,
                    destination_depth,
                    0,
                    ReduceKind::Sum,
                )
                .expect("supported depth pair");
                assert_eq!(
                    decode(&output, 0, destination_depth).to_bits(),
                    2.0_f64.to_bits()
                );
            }
        }
    }

    #[test]
    fn invalid_mask_and_axis_are_rejected() {
        assert_eq!(
            mean_std_dev(&[1], 1, 1, 1, ScalarDepth::U8, Some(&[])),
            Err(StatsError::IncorrectMaskLength {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            reduce(
                &[1],
                1,
                1,
                1,
                ScalarDepth::U8,
                ScalarDepth::U8,
                2,
                ReduceKind::Sum
            ),
            Err(StatsError::InvalidAxis { actual: 2 })
        );
    }
}
