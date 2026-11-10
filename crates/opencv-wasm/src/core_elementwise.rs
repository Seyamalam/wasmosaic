use crate::core_norm::ScalarDepth;
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryElementwise {
    Add,
    Subtract,
    AbsoluteDifference,
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementwiseError {
    BufferSizeOverflow,
    MaskLengthMismatch { expected: usize, actual: usize },
    InvalidCompareKind(u8),
}

impl fmt::Display for ElementwiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferSizeOverflow => formatter.write_str("matrix byte length overflowed"),
            Self::MaskLengthMismatch { expected, actual } => write!(
                formatter,
                "mask length {actual} does not match the expected {expected} pixels"
            ),
            Self::InvalidCompareKind(value) => {
                write!(formatter, "unsupported comparison kind {value}")
            }
        }
    }
}

impl Error for ElementwiseError {}

fn scalar_count(rows: u32, columns: u32, channels: u16) -> Result<usize, ElementwiseError> {
    let pixels = u64::from(rows) * u64::from(columns);
    let scalars = pixels * u64::from(channels);
    usize::try_from(scalars).map_err(|_| ElementwiseError::BufferSizeOverflow)
}

pub(crate) fn validate_mask(mask: &[u8], rows: u32, columns: u32) -> Result<(), ElementwiseError> {
    let expected = (u64::from(rows) * u64::from(columns)) as usize;
    if mask.len() != expected {
        return Err(ElementwiseError::MaskLengthMismatch {
            expected,
            actual: mask.len(),
        });
    }
    Ok(())
}

fn read_scalar(data: &[u8], index: usize, depth: ScalarDepth) -> f64 {
    let offset = index * depth.width();
    match depth {
        ScalarDepth::U8 => f64::from(data[offset]),
        ScalarDepth::I8 => f64::from(i8::from_ne_bytes([data[offset]])),
        ScalarDepth::U16 => f64::from(u16::from_ne_bytes([data[offset], data[offset + 1]])),
        ScalarDepth::I16 => f64::from(i16::from_ne_bytes([data[offset], data[offset + 1]])),
        ScalarDepth::I32 => f64::from(i32::from_ne_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])),
        ScalarDepth::F32 => f64::from(f32::from_ne_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])),
        ScalarDepth::F64 => f64::from_ne_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn write_scalar(data: &mut [u8], index: usize, depth: ScalarDepth, value: f64) {
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
    data[offset..offset + depth.width()].copy_from_slice(&bytes);
}

/// Evaluates a per-scalar binary operation between two same-shaped matrices.
///
/// When `mask` is supplied, only selected pixels are written; the destination buffer is seeded
/// from `seed` when present so callers can preserve existing destination bytes.
pub(crate) fn binary_elementwise(
    left: &[u8],
    right: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    source_depth: ScalarDepth,
    output_depth: ScalarDepth,
    operation: BinaryElementwise,
    mask: Option<&[u8]>,
    seed: Option<&[u8]>,
) -> Result<Vec<u8>, ElementwiseError> {
    let scalars = scalar_count(rows, columns, channels)?;
    if left.len() != scalars * source_depth.width() || right.len() != scalars * source_depth.width()
    {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    if let Some(mask) = mask {
        validate_mask(mask, rows, columns)?;
    }
    let mut output = match seed {
        Some(seed) if seed.len() == scalars * output_depth.width() => seed.to_vec(),
        _ => vec![0_u8; scalars * output_depth.width()],
    };
    let wrap_i32 = source_depth == ScalarDepth::I32 && output_depth == ScalarDepth::I32;
    for scalar in 0..scalars {
        let pixel = scalar / usize::from(channels);
        if let Some(values) = mask {
            if values[pixel] == 0 {
                continue;
            }
        }
        let a = read_scalar(left, scalar, source_depth);
        let b = read_scalar(right, scalar, source_depth);
        let value = match (operation, wrap_i32) {
            (BinaryElementwise::Minimum, _) => a.min(b),
            (BinaryElementwise::Maximum, _) => a.max(b),
            (BinaryElementwise::Add, true) => {
                f64::from((a as i64 as i32).wrapping_add(b as i64 as i32))
            }
            (BinaryElementwise::Subtract, true) => {
                f64::from((a as i64 as i32).wrapping_sub(b as i64 as i32))
            }
            (BinaryElementwise::AbsoluteDifference, true) => f64::from(
                (a as i64 as i32)
                    .wrapping_sub(b as i64 as i32)
                    .wrapping_abs(),
            ),
            (BinaryElementwise::Add, false) => a + b,
            (BinaryElementwise::Subtract, false) => a - b,
            (BinaryElementwise::AbsoluteDifference, false) => (a - b).abs(),
        };
        write_scalar(&mut output, scalar, output_depth, value);
    }
    Ok(output)
}

/// Applies a raw bitwise byte operation between two identical-layout matrices.
pub(crate) fn bitwise_elementwise(
    left: &[u8],
    right: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    kind: u8,
    mask: Option<&[u8]>,
    seed: Option<&[u8]>,
) -> Result<Vec<u8>, ElementwiseError> {
    let scalars = scalar_count(rows, columns, channels)?;
    let length = scalars * depth.width();
    if left.len() != length || right.len() != length {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    if let Some(mask) = mask {
        validate_mask(mask, rows, columns)?;
    }
    let channel_count = usize::from(channels);
    let scalars_per_pixel = channel_count * depth.width();
    let mut output = match seed {
        Some(seed) if seed.len() == length => seed.to_vec(),
        _ => vec![0_u8; length],
    };
    for (offset, (a, b)) in left.iter().zip(right).enumerate() {
        if let Some(values) = mask {
            if values[offset / scalars_per_pixel] == 0 {
                continue;
            }
        }
        output[offset] = match kind {
            0 => a & b,
            1 => a | b,
            _ => a ^ b,
        };
    }
    Ok(output)
}

/// Compares corresponding scalars and writes a single-channel U8 result per pixel.
pub(crate) fn compare_elementwise(
    left: &[u8],
    right: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    compare: u8,
) -> Result<Vec<u8>, ElementwiseError> {
    let scalars = scalar_count(rows, columns, channels)?;
    if left.len() != scalars * depth.width() || right.len() != scalars * depth.width() {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    if compare > 5 {
        return Err(ElementwiseError::InvalidCompareKind(compare));
    }
    let channel_count = usize::from(channels);
    let pixels = scalars / channel_count;
    let mut output = vec![0_u8; pixels * channel_count];
    for scalar in 0..scalars {
        let a = read_scalar(left, scalar, depth);
        let b = read_scalar(right, scalar, depth);
        let matched = match compare {
            0 => a == b,
            1 => a > b,
            2 => a >= b,
            3 => a < b,
            4 => a <= b,
            _ => a != b,
        };
        if matched {
            output[scalar] = 255;
        }
    }
    Ok(output)
}

/// Tests each source scalar against per-channel inclusive bounds and writes a single-channel
/// U8 result that is 255 only when every channel of the pixel lies within bounds.
///
/// Bounds matrices must be single channel or match the source channel count.
pub(crate) fn in_range(
    source: &[u8],
    lower: &[u8],
    upper: &[u8],
    rows: u32,
    columns: u32,
    channels: u16,
    depth: ScalarDepth,
    bound_channels: u16,
) -> Result<Vec<u8>, ElementwiseError> {
    let scalars = scalar_count(rows, columns, channels)?;
    if source.len() != scalars * depth.width() {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    let bound_count = usize::from(bound_channels);
    if bound_count != 1 && bound_count != usize::from(channels) {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    let bound_scalars = scalars / usize::from(channels) * bound_count;
    if lower.len() != bound_scalars * depth.width() || upper.len() != bound_scalars * depth.width()
    {
        return Err(ElementwiseError::BufferSizeOverflow);
    }
    let channel_count = usize::from(channels);
    let pixels = scalars / channel_count;
    let mut output = vec![0_u8; pixels];
    for pixel in 0..pixels {
        let mut inside = true;
        for channel in 0..channel_count {
            let source_scalar = pixel * channel_count + channel;
            let bound_scalar = pixel * bound_count + channel.min(bound_count - 1);
            let value = read_scalar(source, source_scalar, depth);
            let lo = read_scalar(lower, bound_scalar, depth);
            let hi = read_scalar(upper, bound_scalar, depth);
            if !(value >= lo && value <= hi) {
                inside = false;
                break;
            }
        }
        if inside {
            output[pixel] = 255;
        }
    }
    Ok(output)
}
