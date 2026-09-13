//! Original separable image resampling from pixel-centre and pixel-area geometry.
use crate::{
    imgproc_sample::{decode, encode},
    mat::{Mat, MatDepth, MatError},
};
use std::{error::Error, fmt};

pub(crate) const INTER_NEAREST: i32 = 0;
pub(crate) const INTER_LINEAR: i32 = 1;
pub(crate) const INTER_AREA: i32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResizeError {
    EmptySource,
    InvalidScale { x: f64, y: f64 },
    Matrix(MatError),
    SizeOverflow,
    UnsupportedInterpolation(i32),
    UnsupportedDepth(MatDepth),
    UnsupportedChannels,
}
impl fmt::Display for ResizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => f.write_str("resize source must not be empty"),
            Self::InvalidScale { x, y } => write!(
                f,
                "resize scales must be positive and finite; received {x}, {y}"
            ),
            Self::Matrix(error) => error.fmt(f),
            Self::SizeOverflow => f.write_str("resize dimensions exceed the matrix limit"),
            Self::UnsupportedInterpolation(mode) => {
                write!(f, "unsupported resize interpolation {mode}")
            }
            Self::UnsupportedChannels => {
                f.write_str("fractional area resize supports at most four channels")
            }
            Self::UnsupportedDepth(depth) => {
                write!(f, "resize interpolation does not support {depth:?}")
            }
        }
    }
}
impl Error for ResizeError {}
impl From<MatError> for ResizeError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

struct Geometry {
    width: u32,
    height: u32,
    x: f64,
    y: f64,
}
impl Geometry {
    fn resolve(
        source: &Mat,
        width: i32,
        height: i32,
        fx: f64,
        fy: f64,
    ) -> Result<Self, ResizeError> {
        if width > 0 && height > 0 {
            let width = width.unsigned_abs();
            let height = height.unsigned_abs();
            return Ok(Self {
                width,
                height,
                x: 1.0 / (f64::from(width) / f64::from(source.columns())),
                y: 1.0 / (f64::from(height) / f64::from(source.rows())),
            });
        }
        if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
            return Err(ResizeError::InvalidScale { x: fx, y: fy });
        }
        Ok(Self {
            width: dimension(f64::from(source.columns()) * fx)?,
            height: dimension(f64::from(source.rows()) * fy)?,
            x: 1.0 / fx,
            y: 1.0 / fy,
        })
    }
}
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dimension(value: f64) -> Result<u32, ResizeError> {
    let rounded = value.round_ties_even();
    if !rounded.is_finite() || rounded < 1.0 || rounded > f64::from(i32::MAX) {
        return Err(ResizeError::SizeOverflow);
    }
    Ok(rounded as u32)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resize_into(
    source: &Mat,
    destination: &Mat,
    width: i32,
    height: i32,
    fx: f64,
    fy: f64,
    mode: i32,
) -> Result<(), ResizeError> {
    if source.rows() == 0 || source.columns() == 0 {
        return Err(ResizeError::EmptySource);
    }
    let geometry = Geometry::resolve(source, width, height, fx, fy)?;
    if geometry.width == source.columns() && geometry.height == source.rows() {
        destination.write_output(
            source.compact_bytes(),
            source.rows(),
            source.columns(),
            source.channels(),
            source.depth(),
        )?;
        return Ok(());
    }
    let mode = match resolve_mode(source, &geometry, mode) {
        Ok(mode) => mode,
        Err(error) => {
            prepare_error_output(source, destination, &geometry)?;
            return Err(error);
        }
    };
    let pixel_width = usize::from(source.channels()) * source.depth().byte_width();
    let length = (geometry.width as usize)
        .checked_mul(geometry.height as usize)
        .and_then(|n| n.checked_mul(pixel_width))
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or(ResizeError::SizeOverflow)?;
    if mode == INTER_NEAREST || mode == 6 {
        let columns = (0..geometry.width)
            .map(|x| nearest_index(x, geometry.width, source.columns(), geometry.x, mode == 6))
            .collect::<Vec<_>>();
        let rows = (0..geometry.height)
            .map(|y| nearest_index(y, geometry.height, source.rows(), geometry.y, mode == 6))
            .collect::<Vec<_>>();
        if destination.try_write_shared_nearest(source, &columns, &rows)? {
            return Ok(());
        }
    }
    let input = source.compact_bytes();
    let mut output = vec![0; length];
    if mode == INTER_NEAREST || mode == 6 {
        nearest(source, &input, &mut output, &geometry, mode == 6);
    } else {
        filtered(source, &input, &mut output, &geometry, mode);
    }
    destination.write_output(
        output,
        geometry.height,
        geometry.width,
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

// Exact scale identities select distinct interpolation rounding contracts.
#[allow(clippy::float_cmp)]
fn resolve_mode(source: &Mat, geometry: &Geometry, mode: i32) -> Result<i32, ResizeError> {
    if !(0..=6).contains(&mode) {
        return Err(ResizeError::UnsupportedInterpolation(mode));
    }
    let mode = if mode == 5 && matches!(source.depth(), MatDepth::F32 | MatDepth::F64) {
        1
    } else {
        mode
    };
    let mode = if (mode == INTER_LINEAR || (mode == 5 && source.channels() != 2))
        && geometry.x == 2.0
        && geometry.y == 2.0
    {
        3
    } else {
        mode
    };
    if matches!(source.depth(), MatDepth::I8 | MatDepth::I32) && matches!(mode, 1..=4) {
        return Err(ResizeError::UnsupportedDepth(source.depth()));
    }
    if mode == INTER_AREA
        && geometry.x >= 1.0
        && geometry.y >= 1.0
        && source.channels() > 4
        && (geometry.x.fract() != 0.0 || geometry.y.fract() != 0.0)
    {
        return Err(ResizeError::UnsupportedChannels);
    }
    Ok(mode)
}

fn prepare_error_output(source: &Mat, destination: &Mat, g: &Geometry) -> Result<(), ResizeError> {
    if destination.rows() == g.height
        && destination.columns() == g.width
        && destination.channels() == source.channels()
        && destination.depth() == source.depth()
    {
        return Ok(());
    }
    let bytes = (g.width as usize)
        .checked_mul(g.height as usize)
        .and_then(|n| n.checked_mul(usize::from(source.channels())))
        .and_then(|n| n.checked_mul(source.depth().byte_width()))
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or(ResizeError::SizeOverflow)?;
    destination.write_output(
        vec![0; bytes],
        g.height,
        g.width,
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn nearest_index(position: u32, length: u32, source_length: u32, scale: f64, exact: bool) -> u32 {
    if exact {
        // The pinned exact mode rounds its source/destination step to 16 fractional bits.
        let numerator = u64::from(source_length) * 65536;
        let step = (numerator + u64::from(length) / 2) / u64::from(length);
        (((u64::from(position) * 2 + 1) * step / 2 / 65536) as u32).min(source_length - 1)
    } else {
        ((f64::from(position) * scale).floor() as u32).min(source_length - 1)
    }
}

fn nearest(source: &Mat, input: &[u8], output: &mut [u8], g: &Geometry, exact: bool) {
    let bytes = usize::from(source.channels()) * source.depth().byte_width();
    for y in 0..g.height {
        let sy = nearest_index(y, g.height, source.rows(), g.y, exact);
        for x in 0..g.width {
            let sx = nearest_index(x, g.width, source.columns(), g.x, exact);
            let src = (sy as usize * source.columns() as usize + sx as usize) * bytes;
            let dst = (y as usize * g.width as usize + x as usize) * bytes;
            output[dst..dst + bytes].copy_from_slice(&input[src..src + bytes]);
        }
    }
}

struct Tap {
    index: u32,
    weight: f64,
}
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn axis(
    position: u32,
    length: u32,
    scale: f64,
    mode: i32,
    area: bool,
    depth: MatDepth,
) -> Vec<Tap> {
    let clamp = |i: i64| i.clamp(0, i64::from(length) - 1) as u32;
    if area {
        let start = f64::from(position) * scale;
        let end = (f64::from(position + 1) * scale).min(f64::from(length));
        let denominator = (end - start).max(f64::MIN_POSITIVE);
        return (start.floor() as u32..end.ceil() as u32)
            .map(|i| Tap {
                index: i.min(length - 1),
                weight: f64::from(
                    ((end.min(f64::from(i + 1)) - start.max(f64::from(i))) / denominator) as f32,
                ),
            })
            .collect();
    }
    let location = if mode == INTER_AREA {
        f64::from(position) * scale
    } else {
        (f64::from(position) + 0.5) * scale - 0.5
    };
    let location = if mode == 5 {
        location
    } else {
        f64::from(location as f32)
    };
    let lower = location.floor() as i32;
    let fraction = if mode == INTER_AREA {
        ((f64::from(position + 1) - (f64::from(lower) + 1.0) / scale).max(0.0)).fract()
    } else {
        location - f64::from(lower)
    };
    let fraction = if mode == 5 {
        fraction
    } else {
        f64::from(fraction as f32)
    };
    let (start, weights) = match mode {
        2 => {
            let x = fraction as f32;
            let first = cubic(x + 1.0);
            let second = cubic(x);
            let third = cubic(1.0 - x);
            (
                lower - 1,
                vec![
                    f64::from(first),
                    f64::from(second),
                    f64::from(third),
                    f64::from(1.0 - first - second - third),
                ],
            )
        }
        4 => {
            let mut weights = (-3..=4)
                .map(|offset| lanczos(fraction, offset) as f32)
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f32>();
            let normalization = 1.0 / sum;
            for weight in &mut weights {
                *weight *= normalization;
            }
            (lower - 3, weights.into_iter().map(f64::from).collect())
        }
        _ => (lower, vec![1.0 - fraction, fraction]),
    };
    let precision = if mode == 5 {
        match depth {
            MatDepth::U8 => 256.0,
            MatDepth::I8 | MatDepth::U16 | MatDepth::I16 => 65536.0,
            _ => 4_294_967_296.0,
        }
    } else if depth == MatDepth::U8 {
        2048.0
    } else {
        0.0
    };
    weights
        .into_iter()
        .enumerate()
        .map(|(offset, weight)| Tap {
            index: clamp(i64::from(start) + i64::try_from(offset).expect("eight filter taps")),
            weight: if precision > 0.0 {
                (weight * precision).round_ties_even() / precision
            } else {
                f64::from(weight as f32)
            },
        })
        .collect()
}
fn cubic(x: f32) -> f32 {
    let a = -0.75;
    if x <= 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    }
}
#[allow(clippy::cast_possible_truncation)]
fn lanczos(fraction: f64, offset: i32) -> f64 {
    if fraction.abs() < f64::from(f32::EPSILON) {
        return f64::from(offset == 0);
    }
    // sin(pi * (fraction - offset)) has a common sin(pi * fraction) factor.
    // Cancel it before normalization and retain the alternating sign.
    let angle =
        std::f64::consts::FRAC_PI_4 * (f64::from((fraction as f32) + 3.0) - f64::from(offset + 3));
    let sign = if offset % 2 == 0 { 1.0 } else { -1.0 };
    sign * angle.sin() / (angle * angle)
}
#[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
fn filtered(source: &Mat, input: &[u8], output: &mut [u8], g: &Geometry, mode: i32) {
    let area = mode == INTER_AREA && g.x >= 1.0 && g.y >= 1.0;
    let xs = (0..g.width)
        .map(|x| {
            let mut taps = axis(x, source.columns(), g.x, mode, area, source.depth());
            if matches!(mode, 1 | 3 | 5)
                && !area
                && taps.first().map(|tap| tap.index) == taps.last().map(|tap| tap.index)
            {
                taps.truncate(1);
                taps[0].weight = 1.0;
            }
            taps
        })
        .collect::<Vec<_>>();
    let ys = (0..g.height)
        .map(|y| axis(y, source.rows(), g.y, mode, area, source.depth()))
        .collect::<Vec<_>>();
    let channels = usize::from(source.channels());
    let bytes = source.depth().byte_width();
    let narrow = |value: f64| {
        if source.depth() == MatDepth::F64 || (source.depth() == MatDepth::U8 && !area) || mode == 5
        {
            value
        } else {
            f64::from(value as f32)
        }
    };
    for (y, vertical) in ys.iter().enumerate() {
        for (x, horizontal) in xs.iter().enumerate() {
            for channel in 0..channels {
                let mut value = 0.0;
                for ty in vertical {
                    let mut row = 0.0;
                    for tx in horizontal {
                        let index = ((ty.index as usize * source.columns() as usize
                            + tx.index as usize)
                            * channels
                            + channel)
                            * bytes;
                        row = narrow(
                            row + narrow(
                                decode(&input[index..index + bytes], source.depth()) * tx.weight,
                            ),
                        );
                    }
                    value = if source.depth() == MatDepth::U8 && matches!(mode, 1 | 3) && !area {
                        // Black-box probes resolve quarter-unit truncation of each vertical contribution.
                        value + (((row * 128.0).floor() / 128.0 * ty.weight) * 4.0).floor() / 4.0
                    } else {
                        narrow(value + narrow(row * ty.weight))
                    };
                }
                if (source.depth() == MatDepth::U8 && !area)
                    || mode == 5
                    || (area
                        && g.x == 2.0
                        && g.y == 2.0
                        && matches!(source.channels(), 1 | 3 | 4)
                        && !matches!(source.depth(), MatDepth::F32 | MatDepth::F64))
                {
                    value = (value + 0.5).floor();
                }
                let target = ((y * g.width as usize + x) * channels + channel) * bytes;
                encode(value, source.depth(), &mut output[target..target + bytes]);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_u8};

    #[test]
    fn nearest_neighbor_expands_pixels_without_interpolation() {
        let source = mat_from_u8(&[1, 2, 3, 4], 2, 2, 1).expect("source");
        let destination = mat_empty();
        resize_into(&source, &destination, 4, 2, 0.0, 0.0, INTER_NEAREST).expect("resize");
        assert_eq!(destination.rows(), 2);
        assert_eq!(destination.columns(), 4);
        assert_eq!(destination.compact_bytes(), [1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn scale_factors_resolve_zero_target_dimensions() {
        let source = mat_from_u8(&[1, 2, 3, 4], 2, 2, 1).expect("source");
        let destination = mat_empty();
        resize_into(&source, &destination, 0, 0, 2.0, 0.5, INTER_NEAREST).expect("resize");
        assert_eq!(destination.rows(), 1);
        assert_eq!(destination.columns(), 4);
        assert_eq!(destination.compact_bytes(), [1, 1, 2, 2]);
    }

    #[test]
    fn linear_uses_half_pixel_coordinates_and_nearest_even_rounding() {
        let source = mat_from_u8(&[0, 100, 150, 255], 2, 2, 1).expect("source");
        let destination = mat_empty();
        resize_into(&source, &destination, 3, 3, 0.0, 0.0, INTER_LINEAR).expect("resize");
        assert_eq!(
            destination.compact_bytes(),
            [0, 50, 100, 75, 126, 178, 150, 202, 255]
        );
    }

    #[test]
    fn area_averages_each_covered_source_pixel_when_shrinking() {
        let source = mat_from_u8(
            &[
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
            ],
            4,
            4,
            1,
        )
        .expect("source");
        let destination = mat_empty();
        resize_into(&source, &destination, 2, 2, 0.0, 0.0, INTER_AREA).expect("resize");
        assert_eq!(destination.compact_bytes(), [25, 45, 105, 125]);
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64, mat_from_i8, mat_from_u8};

    #[test]
    fn fractional_scales_are_not_replaced_by_rounded_size_ratios() {
        let source = mat_from_u8(&[10, 20, 30, 40, 50], 1, 5, 1).unwrap();
        let destination = mat_empty();
        resize_into(&source, &destination, -1, 2, 1.3, 1.0, 0).unwrap();
        assert_eq!(destination.compact_bytes(), [10, 10, 20, 30, 40, 40]);
    }
    #[test]
    fn exact_nearest_rounds_the_fixed_step_before_sampling() {
        let source = mat_from_u8(&[10, 20, 30, 40], 1, 4, 1).unwrap();
        let destination = mat_empty();
        resize_into(&source, &destination, 7, 1, 0.0, 0.0, 6).unwrap();
        assert_eq!(destination.compact_bytes(), [10, 10, 20, 20, 30, 40, 40]);
    }
    #[test]
    fn cubic_and_lanczos_reconstruct_the_impulse_midpoint() {
        let source = mat_from_f64(&[0.0, 1.0, 0.0, 0.0], 1, 4, 1).unwrap();
        let destination = mat_empty();
        resize_into(&source, &destination, 7, 1, 0.0, 0.0, 2).unwrap();
        assert_eq!(destination.to_f64_array().unwrap()[3], 0.59375);
        resize_into(&source, &destination, 7, 1, 0.0, 0.0, 4).unwrap();
        assert!((destination.to_f64_array().unwrap()[3] - 0.618_877_410_888_671_9).abs() < 1e-14);
    }
    #[test]
    fn nearest_overlap_reads_prior_writes_in_pixel_order() {
        let source = mat_from_u8(&(0..16).collect::<Vec<_>>(), 4, 4, 1).unwrap();
        let destination = source.roi(1, 1, 3, 3).unwrap();
        resize_into(&source, &destination, 3, 3, 0.0, 0.0, 0).unwrap();
        assert_eq!(destination.compact_bytes(), [0, 1, 2, 4, 0, 1, 8, 4, 0]);
    }
    #[test]
    fn unsupported_depth_rebinds_output_before_reporting_error() {
        let source = mat_from_i8(&[-2, 1, 4, 9], 2, 2, 1).unwrap();
        let destination = mat_empty();
        assert!(resize_into(&source, &destination, 3, 3, 0.0, 0.0, 1).is_err());
        assert_eq!(
            (
                destination.rows(),
                destination.columns(),
                destination.depth()
            ),
            (3, 3, MatDepth::I8)
        );
        assert_eq!(destination.compact_bytes(), [0; 9]);
    }
}
