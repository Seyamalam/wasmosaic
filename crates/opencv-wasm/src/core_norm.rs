//! Type-generic norm and normalization kernels over compact matrix storage.

use std::{error::Error, fmt};

pub(crate) const NORM_INF: u32 = 1;
pub(crate) const NORM_L1: u32 = 2;
pub(crate) const NORM_L2: u32 = 4;
pub(crate) const NORM_L2SQR: u32 = 5;
pub(crate) const NORM_HAMMING: u32 = 6;
pub(crate) const NORM_HAMMING2: u32 = 7;
pub(crate) const NORM_RELATIVE: u32 = 8;
pub(crate) const NORM_MINMAX: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarDepth {
    U8,
    I8,
    U16,
    I16,
    I32,
    F32,
    F64,
}

impl ScalarDepth {
    pub(crate) const fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NormError {
    InvalidShape,
    BufferSizeOverflow,
    IncorrectBufferLength { expected: usize, actual: usize },
    ShapeMismatch,
    InvalidMask,
    UnsupportedNormType(u32),
    HammingRequiresU8,
    RelativeRequiresSecondInput,
    MinMaxIsNotANorm,
}

impl fmt::Display for NormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                f.write_str("matrix dimensions and channels must be greater than zero")
            }
            Self::BufferSizeOverflow => f.write_str("matrix byte length overflowed"),
            Self::IncorrectBufferLength { expected, actual } => write!(
                f,
                "matrix buffer has {actual} bytes; expected {expected} bytes"
            ),
            Self::ShapeMismatch => {
                f.write_str("source matrices must have identical shape and depth")
            }
            Self::InvalidMask => {
                f.write_str("mask must be an equally sized single-channel U8 matrix")
            }
            Self::UnsupportedNormType(value) => write!(f, "unsupported norm type {value}"),
            Self::HammingRequiresU8 => f.write_str("Hamming norms require unsigned 8-bit matrices"),
            Self::RelativeRequiresSecondInput => {
                f.write_str("relative norm requires a second source matrix")
            }
            Self::MinMaxIsNotANorm => f.write_str("NORM_MINMAX is valid only for normalization"),
        }
    }
}

impl Error for NormError {}

#[allow(clippy::too_many_arguments)]
pub(crate) fn norm(
    first: &[u8],
    second: Option<&[u8]>,
    mask: Option<&[u8]>,
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    norm_type: u32,
) -> Result<f64, NormError> {
    let elements = validate(first, rows, columns, channels, depth)?;
    if let Some(other) = second {
        validate(other, rows, columns, channels, depth).map_err(|_| NormError::ShapeMismatch)?;
    }
    validate_mask(mask, rows, columns)?;
    let relative = norm_type & NORM_RELATIVE != 0;
    let base = norm_type & !NORM_RELATIVE;
    if relative && second.is_none() {
        return Err(NormError::RelativeRequiresSecondInput);
    }
    if base == NORM_MINMAX {
        return Err(NormError::MinMaxIsNotANorm);
    }
    if matches!(base, NORM_HAMMING | NORM_HAMMING2) {
        if depth != ScalarDepth::U8 {
            return Err(NormError::HammingRequiresU8);
        }
        let value = hamming(
            first,
            second,
            mask,
            columns,
            channels,
            base == NORM_HAMMING2,
        );
        if relative {
            let denominator = hamming(
                second.expect("checked above"),
                None,
                mask,
                columns,
                channels,
                base == NORM_HAMMING2,
            );
            return Ok(value / denominator);
        }
        return Ok(value);
    }
    if !matches!(base, NORM_INF | NORM_L1 | NORM_L2 | NORM_L2SQR) {
        return Err(NormError::UnsupportedNormType(norm_type));
    }
    let value = numeric_norm(
        first, second, mask, columns, channels, depth, elements, base,
    );
    if relative {
        let denominator = numeric_norm(
            second.expect("checked above"),
            None,
            mask,
            columns,
            channels,
            depth,
            elements,
            base,
        );
        Ok(value / denominator)
    } else {
        Ok(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_into(
    source: &[u8],
    destination: &mut [u8],
    mask: Option<&[u8]>,
    rows: u32,
    columns: u32,
    channels: u16,
    source_depth: ScalarDepth,
    destination_depth: ScalarDepth,
    alpha: f64,
    beta: f64,
    norm_type: u32,
) -> Result<(), NormError> {
    let elements = validate(source, rows, columns, channels, source_depth)?;
    validate(destination, rows, columns, channels, destination_depth)?;
    validate_mask(mask, rows, columns)?;
    let (scale, shift, minmax) = if norm_type == NORM_MINMAX {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for index in 0..elements {
            if selected(mask, index / usize::from(channels), columns) {
                let value = read(source, index, source_depth);
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
        }
        if minimum == f64::INFINITY {
            return Ok(());
        }
        if maximum.total_cmp(&minimum).is_eq() {
            (0.0, alpha, None)
        } else {
            let low = alpha.min(beta);
            let high = alpha.max(beta);
            let scale = (high - low) / (maximum - minimum);
            (scale, low, Some((minimum, low)))
        }
    } else {
        if !matches!(norm_type, NORM_INF | NORM_L1 | NORM_L2) {
            return Err(NormError::UnsupportedNormType(norm_type));
        }
        let magnitude = numeric_norm(
            source,
            None,
            mask,
            columns,
            channels,
            source_depth,
            elements,
            norm_type,
        );
        (
            if magnitude == 0.0 {
                0.0
            } else {
                alpha / magnitude
            },
            0.0,
            None,
        )
    };
    for index in 0..elements {
        if selected(mask, index / usize::from(channels), columns) {
            let value = match minmax {
                Some((base, low)) => (read(source, index, source_depth) - base) * scale + low,
                None => read(source, index, source_depth).mul_add(scale, shift),
            };
            write(destination, index, destination_depth, value);
        }
    }
    Ok(())
}

fn validate(
    data: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
) -> Result<usize, NormError> {
    if rows == 0 || columns == 0 || channels == 0 {
        return Err(NormError::InvalidShape);
    }
    let elements = (rows as usize)
        .checked_mul(columns as usize)
        .and_then(|n| n.checked_mul(usize::from(channels)))
        .ok_or(NormError::BufferSizeOverflow)?;
    let expected = elements
        .checked_mul(depth.width())
        .ok_or(NormError::BufferSizeOverflow)?;
    if data.len() != expected {
        return Err(NormError::IncorrectBufferLength {
            expected,
            actual: data.len(),
        });
    }
    Ok(elements)
}

fn validate_mask(mask: Option<&[u8]>, rows: u32, columns: u32) -> Result<(), NormError> {
    let expected = (rows as usize)
        .checked_mul(columns as usize)
        .ok_or(NormError::BufferSizeOverflow)?;
    if mask.is_some_and(|bytes| bytes.len() != expected) {
        return Err(NormError::InvalidMask);
    }
    Ok(())
}

fn selected(mask: Option<&[u8]>, pixel: usize, _columns: u32) -> bool {
    mask.is_none_or(|bytes| bytes[pixel] != 0)
}

#[allow(clippy::too_many_arguments)]
fn numeric_norm(
    data: &[u8],
    second: Option<&[u8]>,
    mask: Option<&[u8]>,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    elements: usize,
    base: u32,
) -> f64 {
    let mut aggregate: f64 = 0.0;
    for index in 0..elements {
        if !selected(mask, index / usize::from(channels), columns) {
            continue;
        }
        let value = second
            .map_or(read(data, index, depth), |other| {
                read(data, index, depth) - read(other, index, depth)
            })
            .abs();
        aggregate = match base {
            NORM_INF => aggregate.max(value),
            NORM_L1 => aggregate + value,
            NORM_L2 | NORM_L2SQR => value.mul_add(value, aggregate),
            _ => unreachable!(),
        };
    }
    if base == NORM_L2 {
        aggregate.sqrt()
    } else {
        aggregate
    }
}

fn hamming(
    data: &[u8],
    second: Option<&[u8]>,
    mask: Option<&[u8]>,
    _columns: u32,
    channels: u16,
    cell2: bool,
) -> f64 {
    let mut total = 0.0_f64;
    for (index, &byte) in data.iter().enumerate() {
        if !selected(mask, index / usize::from(channels), 0) {
            continue;
        }
        let value = second.map_or(byte, |other| byte ^ other[index]);
        total += if cell2 {
            (0..4).fold(0.0, |count, shift| {
                count + f64::from(u8::from((value & (3 << (shift * 2))) != 0))
            })
        } else {
            f64::from(value.count_ones())
        };
    }
    total
}

fn read(data: &[u8], index: usize, depth: ScalarDepth) -> f64 {
    let offset = index * depth.width();
    match depth {
        ScalarDepth::U8 => f64::from(data[offset]),
        ScalarDepth::I8 => f64::from(i8::from_ne_bytes([data[offset]])),
        ScalarDepth::U16 => f64::from(u16::from_ne_bytes(
            data[offset..offset + 2].try_into().expect("width"),
        )),
        ScalarDepth::I16 => f64::from(i16::from_ne_bytes(
            data[offset..offset + 2].try_into().expect("width"),
        )),
        ScalarDepth::I32 => f64::from(i32::from_ne_bytes(
            data[offset..offset + 4].try_into().expect("width"),
        )),
        ScalarDepth::F32 => f64::from(f32::from_ne_bytes(
            data[offset..offset + 4].try_into().expect("width"),
        )),
        ScalarDepth::F64 => f64::from_ne_bytes(data[offset..offset + 8].try_into().expect("width")),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn write(data: &mut [u8], index: usize, depth: ScalarDepth, value: f64) {
    let offset = index * depth.width();
    let rounded = if value.is_finite() && value.abs() < 2_147_483_648.0 {
        value.round_ties_even()
    } else {
        f64::from(i32::MIN)
    };
    let bytes: Vec<u8> = match depth {
        ScalarDepth::U8 => vec![(rounded.clamp(0.0, f64::from(u8::MAX))) as u8],
        ScalarDepth::I8 => (rounded.clamp(f64::from(i8::MIN), f64::from(i8::MAX)) as i8)
            .to_ne_bytes()
            .to_vec(),
        ScalarDepth::U16 => (rounded.clamp(0.0, f64::from(u16::MAX)) as u16)
            .to_ne_bytes()
            .to_vec(),
        ScalarDepth::I16 => (rounded.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16)
            .to_ne_bytes()
            .to_vec(),
        ScalarDepth::I32 => (rounded.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32)
            .to_ne_bytes()
            .to_vec(),
        ScalarDepth::F32 => (value as f32).to_ne_bytes().to_vec(),
        ScalarDepth::F64 => value.to_ne_bytes().to_vec(),
    };
    data[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i16_bytes(values: &[i16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn computes_numeric_difference_norms_and_relative_norm() {
        let first = i16_bytes(&[3, -4]);
        let second = i16_bytes(&[0, 0]);
        assert_eq!(
            norm(&first, None, None, 1, 2, 1, ScalarDepth::I16, NORM_L2),
            Ok(5.0)
        );
        assert_eq!(
            norm(
                &first,
                Some(&second),
                None,
                1,
                2,
                1,
                ScalarDepth::I16,
                NORM_L1
            ),
            Ok(7.0)
        );
        let relative_source = i16_bytes(&[3, 4]);
        let denominator = i16_bytes(&[6, 8]);
        assert_eq!(
            norm(
                &relative_source,
                Some(&denominator),
                None,
                1,
                2,
                1,
                ScalarDepth::I16,
                NORM_L2 | NORM_RELATIVE
            ),
            Ok(0.5)
        );
    }

    #[test]
    fn mask_selects_whole_pixels_across_channels() {
        assert_eq!(
            norm(
                &[3, 4, 50, 50],
                None,
                Some(&[1, 0]),
                1,
                2,
                2,
                ScalarDepth::U8,
                NORM_L2
            ),
            Ok(5.0)
        );
    }

    #[test]
    fn hamming_and_hamming2_count_bits_and_two_bit_cells() {
        let value = [0b1110_0100];
        assert_eq!(
            norm(&value, None, None, 1, 1, 1, ScalarDepth::U8, NORM_HAMMING),
            Ok(4.0)
        );
        assert_eq!(
            norm(&value, None, None, 1, 1, 1, ScalarDepth::U8, NORM_HAMMING2),
            Ok(3.0)
        );
    }

    #[test]
    fn normalize_mutates_only_masked_pixels_and_converts_depth() {
        let source = i16_bytes(&[3, 4, 12]);
        let mut destination = vec![99_u8; 3];
        normalize_into(
            &source,
            &mut destination,
            Some(&[1, 1, 0]),
            1,
            3,
            1,
            ScalarDepth::I16,
            ScalarDepth::U8,
            10.0,
            0.0,
            NORM_L2,
        )
        .unwrap();
        assert_eq!(destination, [6, 8, 99]);
    }

    #[test]
    fn minmax_normalization_maps_range_and_constant_input() {
        let mut output = vec![0_u8; 3];
        normalize_into(
            &[10, 20, 30],
            &mut output,
            None,
            1,
            3,
            1,
            ScalarDepth::U8,
            ScalarDepth::U8,
            0.0,
            100.0,
            NORM_MINMAX,
        )
        .unwrap();
        assert_eq!(output, [0, 50, 100]);
        normalize_into(
            &[7, 7, 7],
            &mut output,
            None,
            1,
            3,
            1,
            ScalarDepth::U8,
            ScalarDepth::U8,
            9.0,
            20.0,
            NORM_MINMAX,
        )
        .unwrap();
        assert_eq!(output, [9, 9, 9]);
    }
}
