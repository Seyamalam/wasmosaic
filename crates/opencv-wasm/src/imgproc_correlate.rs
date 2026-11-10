//! Correlation, derivative, pyramid, and corner-response filters.
//!
//! Behavior derives from the pinned OpenCV.js 4.13.0 browser build. Correlation
//! uses the kernel as written (OpenCV `filter2D` performs correlation, not
//! convolution); linear filters accept every depth except `I8` sources.

use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_border::{
        BORDER_CONSTANT, BORDER_ISOLATED, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_REPLICATE,
        border_index,
    },
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
pub(crate) enum CorrelateError {
    EmptySource,
    UnsupportedDepth(MatDepth),
    UnsupportedDestinationDepth(i32),
    ShapeMismatch,
    InvalidAnchor { x: i32, y: i32 },
    InvalidKernelSize { width: i32, height: i32 },
    InvalidDerivative { dx: i32, dy: i32 },
    InvalidBorderType(i32),
    Matrix(MatError),
}

impl fmt::Display for CorrelateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => {
                formatter.write_str("source matrix dimensions must be greater than zero")
            }
            Self::UnsupportedDepth(depth) => {
                write!(
                    formatter,
                    "operation does not support matrix depth {depth:?}"
                )
            }
            Self::UnsupportedDestinationDepth(depth) => {
                write!(formatter, "unsupported destination depth {depth}")
            }
            Self::ShapeMismatch => {
                formatter.write_str("kernel must be a non-empty single-channel matrix")
            }
            Self::InvalidAnchor { x, y } => {
                write!(formatter, "anchor point ({x}, {y}) lies outside the kernel")
            }
            Self::InvalidKernelSize { width, height } => {
                write!(formatter, "kernel size {width}x{height} is invalid")
            }
            Self::InvalidDerivative { dx, dy } => {
                write!(formatter, "derivative orders dx={dx}, dy={dy} are invalid")
            }
            Self::InvalidBorderType(border) => {
                write!(formatter, "border type {border} is unsupported")
            }
            Self::Matrix(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for CorrelateError {}

impl From<MatError> for CorrelateError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

fn require_source(source: &Mat) -> Result<(), CorrelateError> {
    if source.rows() == 0 || source.columns() == 0 {
        return Err(CorrelateError::EmptySource);
    }
    Ok(())
}

fn require_linear_depth(source: &Mat) -> Result<(), CorrelateError> {
    require_source(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 | MatDepth::F64 => Ok(()),
        depth => Err(CorrelateError::UnsupportedDepth(depth)),
    }
}

fn offset(row: u32, column: u32, columns: u32, channels: usize, channel: usize) -> usize {
    (row as usize * columns as usize + column as usize) * channels + channel
}

fn resolve_anchor(x: i32, y: i32, width: u32, height: u32) -> Result<(u32, u32), CorrelateError> {
    let x = if x < 0 { width / 2 } else { x as u32 };
    let y = if y < 0 { height / 2 } else { y as u32 };
    if x >= width || y >= height {
        return Err(CorrelateError::InvalidAnchor {
            x: x as i32,
            y: y as i32,
        });
    }
    Ok((x, y))
}

fn check_border(border_type: i32) -> Result<i32, CorrelateError> {
    match border_type & !BORDER_ISOLATED {
        BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_REFLECT_101 => {
            Ok(border_type & !BORDER_ISOLATED)
        }
        _ => Err(CorrelateError::InvalidBorderType(border_type)),
    }
}

fn map_destination_depth(source: &Mat, destination_depth: i32) -> Result<MatDepth, CorrelateError> {
    let out = if destination_depth < 0 {
        source.depth()
    } else {
        match destination_depth & 7 {
            0 => MatDepth::U8,
            2 => MatDepth::U16,
            3 => MatDepth::I16,
            5 => MatDepth::F32,
            6 => MatDepth::F64,
            _ => {
                return Err(CorrelateError::UnsupportedDestinationDepth(
                    destination_depth,
                ));
            }
        }
    };
    let allowed: &[MatDepth] = match source.depth() {
        MatDepth::U8 => &[
            MatDepth::U8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::F32,
            MatDepth::F64,
        ],
        MatDepth::U16 => &[MatDepth::U16, MatDepth::F32, MatDepth::F64],
        MatDepth::I16 => &[MatDepth::I16, MatDepth::F32, MatDepth::F64],
        MatDepth::F32 => &[MatDepth::F32, MatDepth::F64],
        MatDepth::F64 => &[MatDepth::F64],
        _ => &[],
    };
    if allowed.contains(&out) {
        Ok(out)
    } else {
        Err(CorrelateError::UnsupportedDestinationDepth(
            destination_depth,
        ))
    }
}

/// Wide destination table used by the Sobel-path filters (Laplacian ksize>3,
/// Scharr): any non-I8/I32 depth is accepted.
fn map_sobel_destination_depth(
    source: &Mat,
    destination_depth: i32,
) -> Result<MatDepth, CorrelateError> {
    if destination_depth < 0 {
        return Ok(source.depth());
    }
    match destination_depth & 7 {
        0 => Ok(MatDepth::U8),
        2 => Ok(MatDepth::U16),
        3 => Ok(MatDepth::I16),
        5 => Ok(MatDepth::F32),
        6 => Ok(MatDepth::F64),
        _ => Err(CorrelateError::UnsupportedDestinationDepth(
            destination_depth,
        )),
    }
}

fn read_kernel(kernel: &Mat) -> Result<Vec<f64>, CorrelateError> {
    if kernel.rows() == 0 || kernel.columns() == 0 || kernel.channels() != 1 {
        return Err(CorrelateError::ShapeMismatch);
    }
    let compact = kernel.compact_bytes();
    let depth = scalar_depth(kernel.depth());
    let count = (kernel.rows() * kernel.columns()) as usize;
    Ok((0..count)
        .map(|index| read_scalar(&compact, index, depth))
        .collect())
}

/// Correlates `source` with `kernel` and writes into `destination`.
///
/// `kernel` is a dense f64 matrix laid out row-major; `delta` shifts every
/// result before the destination-depth cast.
#[allow(clippy::too_many_arguments)]
fn correlate_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel: &[f64],
    kernel_width: u32,
    kernel_height: u32,
    anchor_x: i32,
    anchor_y: i32,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    correlate_into_depth(
        source,
        destination,
        destination_depth,
        kernel,
        kernel_width,
        kernel_height,
        anchor_x,
        anchor_y,
        delta,
        border_type,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn correlate_into_depth(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel: &[f64],
    kernel_width: u32,
    kernel_height: u32,
    anchor_x: i32,
    anchor_y: i32,
    delta: f64,
    border_type: i32,
    narrow_table: bool,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    let border = check_border(border_type)?;
    let out_depth = if narrow_table {
        map_destination_depth(source, destination_depth)?
    } else {
        map_sobel_destination_depth(source, destination_depth)?
    };
    let (ax, ay) = resolve_anchor(anchor_x, anchor_y, kernel_width, kernel_height)?;
    let depth = scalar_depth(source.depth());
    let out_scalar = scalar_depth(out_depth);
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let mut output = vec![0u8; input.len() / depth.width() * out_scalar.width()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for ky in 0..kernel_height {
                    let sample_y = i64::from(row) + i64::from(ky) - i64::from(ay);
                    let mapped_y = border_index(sample_y, rows, border);
                    for kx in 0..kernel_width {
                        let sample_x = i64::from(column) + i64::from(kx) - i64::from(ax);
                        let mapped_x = border_index(sample_x, columns, border);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            sum += kernel[(ky * kernel_width + kx) as usize]
                                * read_scalar(
                                    &input,
                                    offset(y, x, columns, channels, channel),
                                    depth,
                                );
                        }
                    }
                }
                let index = offset(row, column, columns, channels, channel);
                write_scalar(&mut output, index, out_scalar, sum + delta);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), out_depth)?;
    Ok(())
}

/// Two-pass separable correlation: `kernel_x` applies horizontally, then
/// `kernel_y` vertically. Intermediate accumulation is f64.
#[allow(clippy::too_many_arguments)]
fn separable_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel_x: &[f64],
    kernel_y: &[f64],
    anchor_x: i32,
    anchor_y: i32,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    let border = check_border(border_type)?;
    let out_depth = map_destination_depth(source, destination_depth)?;
    let (ax, _ay) = resolve_anchor(anchor_x, anchor_y, kernel_x.len() as u32, 1)?;
    let depth = scalar_depth(source.depth());
    let out_scalar = scalar_depth(out_depth);
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    // Horizontal pass into f64 scratch.
    let mut row_scratch = vec![0.0f64; (rows * columns) as usize * channels];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (kx, weight) in kernel_x.iter().enumerate() {
                    let sample_x = i64::from(column) + kx as i64 - i64::from(ax);
                    if let Some(x) = border_index(sample_x, columns, border) {
                        sum += weight
                            * read_scalar(
                                &input,
                                offset(row, x, columns, channels, channel),
                                depth,
                            );
                    }
                }
                row_scratch[offset(row, column, columns, channels, channel)] = sum;
            }
        }
    }
    // Vertical pass.
    let ay_default = kernel_y.len() as i32 / 2;
    let ay = if anchor_y < 0 { ay_default } else { anchor_y };
    if ay < 0 || ay >= kernel_y.len() as i32 {
        return Err(CorrelateError::InvalidAnchor {
            x: ax as i32,
            y: ay,
        });
    }
    let mut output = vec![0u8; row_scratch.len() * out_scalar.width()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (ky, weight) in kernel_y.iter().enumerate() {
                    let sample_y = i64::from(row) + ky as i64 - i64::from(ay);
                    if let Some(y) = border_index(sample_y, rows, border) {
                        sum += weight * row_scratch[offset(y, column, columns, channels, channel)];
                    }
                }
                let index = offset(row, column, columns, channels, channel);
                write_scalar(&mut output, index, out_scalar, sum + delta);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), out_depth)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_2d_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel: &Mat,
    anchor_x: i32,
    anchor_y: i32,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    let coefficients = read_kernel(kernel)?;
    correlate_into(
        source,
        destination,
        destination_depth,
        &coefficients,
        kernel.columns(),
        kernel.rows(),
        anchor_x,
        anchor_y,
        delta,
        border_type,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sep_filter_2d_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel_x: &Mat,
    kernel_y: &Mat,
    anchor_x: i32,
    anchor_y: i32,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    let column = read_kernel(kernel_x)?;
    let row = read_kernel(kernel_y)?;
    let _ = (kernel_x.rows(), kernel_y.columns());
    separable_into(
        source,
        destination,
        destination_depth,
        &column,
        &row,
        anchor_x,
        anchor_y,
        delta,
        border_type,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sqr_box_filter_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel_width: i32,
    kernel_height: i32,
    anchor_x: i32,
    anchor_y: i32,
    normalize: bool,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    if kernel_width <= 0 || kernel_height <= 0 {
        return Err(CorrelateError::InvalidKernelSize {
            width: kernel_width,
            height: kernel_height,
        });
    }
    let border = check_border(border_type)?;
    let out_depth = if destination_depth < 0 {
        MatDepth::F32
    } else {
        match destination_depth & 7 {
            0 => MatDepth::U8,
            2 => MatDepth::U16,
            3 => MatDepth::I16,
            4 => MatDepth::I32,
            5 => MatDepth::F32,
            6 => MatDepth::F64,
            _ => {
                return Err(CorrelateError::UnsupportedDestinationDepth(
                    destination_depth,
                ));
            }
        }
    };
    let (ax, ay) = resolve_anchor(
        anchor_x,
        anchor_y,
        kernel_width as u32,
        kernel_height as u32,
    )?;
    let depth = scalar_depth(source.depth());
    let out_scalar = scalar_depth(out_depth);
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let scale = if normalize {
        1.0 / (f64::from(kernel_width) * f64::from(kernel_height))
    } else {
        1.0
    };
    let float_accum = matches!(source.depth(), MatDepth::F32 | MatDepth::F64);
    let mut output = vec![0u8; input.len() / depth.width() * out_scalar.width()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for ky in 0..kernel_height as u32 {
                    let sample_y = i64::from(row) + i64::from(ky) - i64::from(ay);
                    let mapped_y = border_index(sample_y, rows, border);
                    for kx in 0..kernel_width as u32 {
                        let sample_x = i64::from(column) + i64::from(kx) - i64::from(ax);
                        let mapped_x = border_index(sample_x, columns, border);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            let value = read_scalar(
                                &input,
                                offset(y, x, columns, channels, channel),
                                depth,
                            );
                            if float_accum {
                                sum += f64::from((value * value) as f32);
                            } else {
                                sum += value * value;
                            }
                        }
                    }
                }
                let index = offset(row, column, columns, channels, channel);
                write_scalar(&mut output, index, out_scalar, sum * scale);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), out_depth)?;
    Ok(())
}

/// Stack blur: separable triangular kernel `[1..r+1..1]` normalized by
/// `(r+1)^2` per dimension, with replicated edges and truncation.
pub(crate) fn stack_blur_into(
    source: &Mat,
    destination: &Mat,
    kernel_width: i32,
    kernel_height: i32,
) -> Result<(), CorrelateError> {
    require_source(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 => {}
        depth => return Err(CorrelateError::UnsupportedDepth(depth)),
    }
    if kernel_width <= 0 || kernel_height <= 0 || kernel_width % 2 == 0 || kernel_height % 2 == 0 {
        return Err(CorrelateError::InvalidKernelSize {
            width: kernel_width,
            height: kernel_height,
        });
    }
    let weights = |size: i32| -> Vec<f64> {
        let radius = size / 2;
        (0..size)
            .map(|index| f64::from(radius + 1 - (index - radius).abs()))
            .collect()
    };
    let kernel_x = weights(kernel_width);
    let kernel_y = weights(kernel_height);
    let rx = kernel_width / 2;
    let ry = kernel_height / 2;
    let div_x_scratch = f64::from(rx + 1) * f64::from(rx + 1);
    let is_float = matches!(source.depth(), MatDepth::F32);
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let mut scratch = vec![0.0f64; (rows * columns) as usize * channels];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (kx, weight) in kernel_x.iter().enumerate() {
                    let sample_x = (i64::from(column) + kx as i64 - i64::from(rx))
                        .clamp(0, i64::from(columns) - 1) as u32;
                    sum += weight
                        * read_scalar(
                            &input,
                            offset(row, sample_x, columns, channels, channel),
                            depth,
                        );
                }
                let intermediate = sum / div_x_scratch;
                scratch[offset(row, column, columns, channels, channel)] = if is_float {
                    intermediate
                } else {
                    intermediate.trunc()
                };
            }
        }
    }
    let div_y = f64::from(ry + 1) * f64::from(ry + 1);
    let mut output = vec![0u8; input.len()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (ky, weight) in kernel_y.iter().enumerate() {
                    let sample_y = (i64::from(row) + ky as i64 - i64::from(ry))
                        .clamp(0, i64::from(rows) - 1) as u32;
                    sum += weight * scratch[offset(sample_y, column, columns, channels, channel)];
                }
                let value = sum / div_y;
                let index = offset(row, column, columns, channels, channel);
                match source.depth() {
                    MatDepth::F32 => write_scalar(&mut output, index, depth, value),
                    _ => write_scalar(&mut output, index, depth, value.trunc()),
                }
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), source.depth())?;
    Ok(())
}

/// Bilateral filter with the upstream default border (`BORDER_REFLECT_101`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bilateral_filter_into(
    source: &Mat,
    destination: &Mat,
    diameter: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_source(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::F32 => {}
        depth => return Err(CorrelateError::UnsupportedDepth(depth)),
    }
    let diameter = if diameter <= 0 {
        let radius = (sigma_space * 1.5).round().max(1.0);
        (radius * 2.0) as i32 + 1
    } else {
        diameter
    };
    let radius = diameter / 2;
    let border = check_border(border_type)?;
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let color_coeff = -0.5 / (sigma_color * sigma_color);
    let space_coeff = -0.5 / (sigma_space * sigma_space);
    let mut output = vec![0u8; input.len()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let center = read_scalar(
                    &input,
                    offset(row, column, columns, channels, channel),
                    depth,
                );
                let mut sum = 0.0f64;
                let mut weight_sum = 0.0f64;
                for ky in -radius..=radius {
                    let sample_y = i64::from(row) + i64::from(ky);
                    let mapped_y = border_index(sample_y, rows, border);
                    for kx in -radius..=radius {
                        let sample_x = i64::from(column) + i64::from(kx);
                        let mapped_x = border_index(sample_x, columns, border);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            let value = read_scalar(
                                &input,
                                offset(y, x, columns, channels, channel),
                                depth,
                            );
                            let spatial = ((i64::from(kx) * i64::from(kx)
                                + i64::from(ky) * i64::from(ky))
                                as f64)
                                * space_coeff;
                            let difference = value - center;
                            let weight = (spatial + difference * difference * color_coeff).exp();
                            sum += weight * value;
                            weight_sum += weight;
                        }
                    }
                }
                let index = offset(row, column, columns, channels, channel);
                write_scalar(
                    &mut output,
                    index,
                    depth,
                    if weight_sum == 0.0 {
                        center
                    } else {
                        sum / weight_sum
                    },
                );
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), source.depth())?;
    Ok(())
}

const PYRAMID_KERNEL: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

fn check_pyramid_border(border_type: i32) -> Result<i32, CorrelateError> {
    match border_type & !BORDER_ISOLATED {
        BORDER_REPLICATE | BORDER_REFLECT | BORDER_REFLECT_101 => {
            Ok(border_type & !BORDER_ISOLATED)
        }
        _ => Err(CorrelateError::InvalidBorderType(border_type)),
    }
}

/// Gaussian pyramid down-sample: separable 5x5 kernel, keep even coordinates.
pub(crate) fn pyr_down_into(
    source: &Mat,
    destination: &Mat,
    dst_rows: i32,
    dst_cols: i32,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    let border = check_pyramid_border(border_type)?;
    let out_rows = if dst_rows <= 0 {
        source.rows().div_ceil(2)
    } else {
        dst_rows as u32
    };
    let out_cols = if dst_cols <= 0 {
        source.columns().div_ceil(2)
    } else {
        dst_cols as u32
    };
    let expected_rows = source.rows().div_ceil(2);
    let expected_cols = source.columns().div_ceil(2);
    if out_rows > expected_rows
        || out_rows * 2 < source.rows() - 1
        || out_cols > expected_cols
        || out_cols * 2 < source.columns() - 1
    {
        return Err(CorrelateError::ShapeMismatch);
    }
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let mut output = vec![0u8; out_rows as usize * out_cols as usize * channels * depth.width()];
    for row in 0..out_rows {
        for column in 0..out_cols {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for ky in -2i64..=2 {
                    let sample_y = i64::from(row) * 2 + ky;
                    let mapped_y = border_index(sample_y, rows, border);
                    for kx in -2i64..=2 {
                        let sample_x = i64::from(column) * 2 + kx;
                        let mapped_x = border_index(sample_x, columns, border);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            sum += PYRAMID_KERNEL[(ky + 2) as usize]
                                * PYRAMID_KERNEL[(kx + 2) as usize]
                                * read_scalar(
                                    &input,
                                    offset(y, x, columns, channels, channel),
                                    depth,
                                );
                        }
                    }
                }
                let index = offset(row, column, out_cols, channels, channel);
                if matches!(source.depth(), MatDepth::F32 | MatDepth::F64) {
                    write_scalar(&mut output, index, depth, sum / 256.0);
                } else {
                    let value = ((sum as i64) + 128) >> 8;
                    write_scalar(&mut output, index, depth, value as f64);
                }
            }
        }
    }
    destination.write_output(
        output,
        out_rows,
        out_cols,
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

/// Gaussian pyramid up-sample: 2x upsize then the same separable 5x5 kernel.
pub(crate) fn pyr_up_into(
    source: &Mat,
    destination: &Mat,
    dst_rows: i32,
    dst_cols: i32,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    let border = check_pyramid_border(border_type)?;
    let out_rows = if dst_rows <= 0 {
        source.rows() * 2
    } else {
        dst_rows as u32
    };
    let out_cols = if dst_cols <= 0 {
        source.columns() * 2
    } else {
        dst_cols as u32
    };
    let rows_delta = i64::from(out_rows) - 2 * i64::from(source.rows());
    let cols_delta = i64::from(out_cols) - 2 * i64::from(source.columns());
    if rows_delta.abs() > 1 || cols_delta.abs() > 1 {
        return Err(CorrelateError::ShapeMismatch);
    }
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    // Horizontal pass: filter the zero-interleaved rows.
    let mut scratch = vec![0.0f64; (rows * out_cols) as usize * channels];
    for row in 0..rows {
        for column in 0..out_cols {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for kx in -2i64..=2 {
                    let source_x = i64::from(column) - kx;
                    if source_x % 2 != 0 {
                        continue;
                    }
                    if let Some(x) = border_index(source_x / 2, columns, border) {
                        sum += PYRAMID_KERNEL[(kx + 2) as usize]
                            * read_scalar(
                                &input,
                                offset(row, x, columns, channels, channel),
                                depth,
                            );
                    }
                }
                scratch[offset(row, column, out_cols, channels, channel)] = sum;
            }
        }
    }
    let mut output = vec![0u8; out_rows as usize * out_cols as usize * channels * depth.width()];
    for row in 0..out_rows {
        for column in 0..out_cols {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for ky in -2i64..=2 {
                    let scratch_y = i64::from(row) - ky;
                    if scratch_y % 2 != 0 {
                        continue;
                    }
                    if let Some(y) = border_index(scratch_y / 2, rows, border) {
                        sum += PYRAMID_KERNEL[(ky + 2) as usize]
                            * scratch[offset(y, column, out_cols, channels, channel)];
                    }
                }
                let index = offset(row, column, out_cols, channels, channel);
                if matches!(source.depth(), MatDepth::F32 | MatDepth::F64) {
                    write_scalar(&mut output, index, depth, sum / 64.0);
                } else {
                    let value = ((sum as i64) + 32) >> 6;
                    write_scalar(&mut output, index, depth, value as f64);
                }
            }
        }
    }
    destination.write_output(
        output,
        out_rows,
        out_cols,
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

const SCHARR_KERNEL: [f64; 3] = [-1.0, 0.0, 1.0];
const SCHARR_SMOOTH: [f64; 3] = [3.0, 10.0, 3.0];

fn sobel_kernels(ksize: u32) -> Result<(Vec<f64>, Vec<f64>), CorrelateError> {
    // Returns (derivative kernel, smoothing kernel) for ksize 1, 3, 5, 7.
    Ok(match ksize {
        1 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        5 => (
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        ),
        7 => (
            vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
            vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        ),
        other => {
            return Err(CorrelateError::InvalidKernelSize {
                width: other as i32,
                height: other as i32,
            });
        }
    })
}

/// Second-derivative kernel for `ksize` (used by Laplacian and corner ops).
fn sobel_second_kernel(ksize: u32) -> Vec<f64> {
    match ksize {
        1 | 3 => vec![1.0, -2.0, 1.0],
        5 => vec![1.0, 0.0, -2.0, 0.0, 1.0],
        _ => vec![1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0],
    }
}

/// Full 2D Sobel kernels for order (dx, dy) at `ksize`.
fn sobel_2d_kernel(dx: u32, dy: u32, ksize: u32) -> Result<Vec<f64>, CorrelateError> {
    let (derivative, smooth) = sobel_kernels(ksize)?;
    let second = sobel_second_kernel(ksize);
    let kx: &[f64] = match dx {
        0 => &smooth,
        1 => &derivative,
        _ => &second,
    };
    let ky: &[f64] = match dy {
        0 => &smooth,
        1 => &derivative,
        _ => &second,
    };
    let mut kernel = vec![0.0; kx.len() * ky.len()];
    for y in 0..ky.len() {
        for x in 0..kx.len() {
            kernel[y * kx.len() + x] = ky[y] * kx[x];
        }
    }
    Ok(kernel)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn laplacian_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    let small = matches!(ksize, 1 | 3);
    let kernel: Vec<f64> = match ksize {
        1 => vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        3 => vec![2.0, 0.0, 2.0, 0.0, -8.0, 0.0, 2.0, 0.0, 2.0],
        other if other == 5 || other == 7 => {
            let (_derivative, smooth) = sobel_kernels(other as u32)?;
            let second = sobel_second_kernel(other as u32);
            return laplacian_separable_into(
                source,
                destination,
                destination_depth,
                &second,
                &smooth,
                scale,
                delta,
                border_type,
            );
        }
        other => {
            return Err(CorrelateError::InvalidKernelSize {
                width: other,
                height: other,
            });
        }
    };
    let side = (kernel.len() as f64).sqrt() as u32;
    let coefficients: Vec<f64> = kernel.iter().map(|v| v * scale).collect();
    // The fixed-size path accepts only same-depth float outputs.
    if small && matches!(source.depth(), MatDepth::F32 | MatDepth::F64) {
        let requested = if destination_depth < 0 {
            source.depth()
        } else {
            match destination_depth & 7 {
                5 => MatDepth::F32,
                6 => MatDepth::F64,
                _ => MatDepth::I8,
            }
        };
        if requested != source.depth() {
            return Err(CorrelateError::UnsupportedDestinationDepth(
                destination_depth,
            ));
        }
    }
    correlate_into_depth(
        source,
        destination,
        destination_depth,
        &coefficients,
        side,
        side,
        -1,
        -1,
        delta,
        border_type,
        small,
    )
}

/// Laplacian for large apertures: separable Dxx + separable Dyy, matching the
/// upstream two-pass accumulation order.
#[allow(clippy::too_many_arguments)]
fn laplacian_separable_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    second: &[f64],
    smooth: &[f64],
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_linear_depth(source)?;
    let border = check_border(border_type)?;
    let out_depth = map_sobel_destination_depth(source, destination_depth)?;
    let depth = scalar_depth(source.depth());
    let out_scalar = scalar_depth(out_depth);
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let anchor_x = smooth.len() / 2;
    let anchor_y = second.len() / 2;
    // Pass 1: smooth along x, second derivative along y (f64 scratch).
    let mut scratch = vec![0.0f64; (rows * columns) as usize * channels];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (ky, weight) in second.iter().enumerate() {
                    let sample_y = i64::from(row) + ky as i64 - anchor_y as i64;
                    if let Some(y) = border_index(sample_y, rows, border) {
                        sum += weight
                            * read_scalar(
                                &input,
                                offset(y, column, columns, channels, channel),
                                depth,
                            );
                    }
                }
                scratch[offset(row, column, columns, channels, channel)] = sum;
            }
        }
    }
    let mut dyy = vec![0.0f64; scratch.len()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (kx, weight) in smooth.iter().enumerate() {
                    let sample_x = i64::from(column) + kx as i64 - anchor_x as i64;
                    if let Some(x) = border_index(sample_x, columns, border) {
                        sum += weight * scratch[offset(row, x, columns, channels, channel)];
                    }
                }
                dyy[offset(row, column, columns, channels, channel)] = sum;
            }
        }
    }
    // Pass 2: second derivative along x, smooth along y.
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (kx, weight) in second.iter().enumerate() {
                    let sample_x = i64::from(column) + kx as i64 - anchor_x as i64;
                    if let Some(x) = border_index(sample_x, columns, border) {
                        sum += weight
                            * read_scalar(
                                &input,
                                offset(row, x, columns, channels, channel),
                                depth,
                            );
                    }
                }
                scratch[offset(row, column, columns, channels, channel)] = sum;
            }
        }
    }
    let mut output = vec![0u8; scratch.len() * out_scalar.width()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for (ky, weight) in smooth.iter().enumerate() {
                    let sample_y = i64::from(row) + ky as i64 - anchor_y as i64;
                    if let Some(y) = border_index(sample_y, rows, border) {
                        sum += weight * scratch[offset(y, column, columns, channels, channel)];
                    }
                }
                let index = offset(row, column, columns, channels, channel);
                write_scalar(
                    &mut output,
                    index,
                    out_scalar,
                    (sum + dyy[index]) * scale + delta,
                );
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), out_depth)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scharr_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    let mut kernel = vec![0.0; 9];
    match (dx, dy) {
        (1, 0) => {
            for y in 0..3 {
                for x in 0..3 {
                    kernel[y * 3 + x] = SCHARR_SMOOTH[y] * SCHARR_KERNEL[x] * scale;
                }
            }
        }
        (0, 1) => {
            for y in 0..3 {
                for x in 0..3 {
                    kernel[y * 3 + x] = SCHARR_KERNEL[y] * SCHARR_SMOOTH[x] * scale;
                }
            }
        }
        _ => {
            return Err(CorrelateError::InvalidDerivative { dx, dy });
        }
    }
    correlate_into_depth(
        source,
        destination,
        destination_depth,
        &kernel,
        3,
        3,
        -1,
        -1,
        delta,
        border_type,
        false,
    )
}

/// Spatial gradient: Sobel dx/dy at ksize=3 into I16 destinations.
pub(crate) fn spatial_gradient_into(
    source: &Mat,
    destination_x: &Mat,
    destination_y: &Mat,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_source(source)?;
    if source.depth() != MatDepth::U8 {
        return Err(CorrelateError::UnsupportedDepth(source.depth()));
    }
    let kernel_x = sobel_2d_kernel(1, 0, 3)?;
    let kernel_y = sobel_2d_kernel(0, 1, 3)?;
    correlate_into(
        source,
        destination_x,
        3,
        &kernel_x,
        3,
        3,
        -1,
        -1,
        0.0,
        border_type,
    )?;
    correlate_into(
        source,
        destination_y,
        3,
        &kernel_y,
        3,
        3,
        -1,
        -1,
        0.0,
        border_type,
    )?;
    Ok(())
}

/// preCornerDetect: `dx²·Dyy + dy²·Dxx − 2·dx·dy·Dxy` on F32 scratch.
pub(crate) fn pre_corner_detect_into(
    source: &Mat,
    destination: &Mat,
    ksize: i32,
    border_type: i32,
) -> Result<(), CorrelateError> {
    require_source(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::F32 => {}
        depth => return Err(CorrelateError::UnsupportedDepth(depth)),
    }
    if ksize <= 0 || ksize % 2 == 0 {
        return Err(CorrelateError::InvalidKernelSize {
            width: ksize,
            height: ksize,
        });
    }
    let scratch = Mat::empty_output();
    let sobel = |dx: u32, dy: u32| -> Result<Vec<f64>, CorrelateError> {
        let kernel = sobel_2d_kernel(dx, dy, ksize as u32)?;
        correlate_into(
            source,
            &scratch,
            6,
            &kernel,
            ksize as u32,
            ksize as u32,
            -1,
            -1,
            0.0,
            border_type,
        )?;
        let bytes = scratch.compact_bytes();
        Ok((0..bytes.len() / 8)
            .map(|index| read_scalar(&bytes, index, ScalarDepth::F64))
            .collect())
    };
    let dx = sobel(1, 0)?;
    let dy = sobel(0, 1)?;
    let dxx = sobel(2, 0)?;
    let dxy = sobel(1, 1)?;
    let dyy = sobel(0, 2)?;
    let mut scale = 1.0 / f64::from(1u32 << (ksize - 1));
    if source.depth() == MatDepth::U8 {
        scale /= 255.0;
    }
    let scale3 = scale * scale * scale;
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let mut output = vec![0u8; dx.len() * 4];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let index = offset(row, column, columns, channels, channel);
                let value = (dx[index] * dx[index] * dyy[index] + dy[index] * dy[index] * dxx[index]
                    - 2.0 * dx[index] * dy[index] * dxy[index])
                    * scale3;
                write_scalar(&mut output, index, ScalarDepth::F32, value);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), MatDepth::F32)?;
    Ok(())
}

/// Corner response shared by cornerHarris and cornerMinEigenVal.
#[allow(clippy::too_many_arguments)]
fn corner_response_into(
    source: &Mat,
    destination: &Mat,
    block_size: i32,
    ksize: i32,
    k: f64,
    border_type: i32,
    min_eigen: bool,
) -> Result<(), CorrelateError> {
    require_source(source)?;
    let scratch = Mat::empty_output();
    let sobel = |dx: u32, dy: u32| -> Result<Vec<f64>, CorrelateError> {
        let kernel = sobel_2d_kernel(dx, dy, ksize as u32)?;
        correlate_into(
            source,
            &scratch,
            6,
            &kernel,
            ksize as u32,
            ksize as u32,
            -1,
            -1,
            0.0,
            border_type,
        )?;
        let bytes = scratch.compact_bytes();
        Ok((0..bytes.len() / 8)
            .map(|index| read_scalar(&bytes, index, ScalarDepth::F64))
            .collect())
    };
    let dx = sobel(1, 0)?;
    let dy = sobel(0, 1)?;
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    match source.depth() {
        MatDepth::U8 | MatDepth::F32 => {}
        depth => return Err(CorrelateError::UnsupportedDepth(depth)),
    }
    if block_size <= 0 {
        return Err(CorrelateError::InvalidKernelSize {
            width: block_size,
            height: block_size,
        });
    }
    let border = check_border(border_type)?;
    let mut cov_scale = 1.0 / f64::from(1u32 << (ksize - 1));
    if source.depth() == MatDepth::U8 {
        cov_scale /= 255.0;
    }
    let cov_scale = cov_scale * cov_scale / (block_size * block_size) as f64;
    let mut output = vec![0u8; dx.len() * 4];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sxx = 0.0f64;
                let mut sxy = 0.0f64;
                let mut syy = 0.0f64;
                for by in (-block_size / 2)..(block_size - block_size / 2) {
                    let sample_y = i64::from(row) + i64::from(by);
                    let mapped_y = border_index(sample_y, rows, border);
                    for bx in (-block_size / 2)..(block_size - block_size / 2) {
                        let sample_x = i64::from(column) + i64::from(bx);
                        let mapped_x = border_index(sample_x, columns, border);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            let index = offset(y, x, columns, channels, channel);
                            sxx += dx[index] * dx[index];
                            sxy += dx[index] * dy[index];
                            syy += dy[index] * dy[index];
                        }
                    }
                }
                let det = (sxx * syy - sxy * sxy) * cov_scale * cov_scale;
                let trace = (sxx + syy) * cov_scale;
                let response = if min_eigen {
                    (trace - (trace * trace - 4.0 * det).sqrt().max(0.0)) / 2.0
                } else {
                    det - k * trace * trace
                };
                let index = offset(row, column, columns, channels, channel);
                write_scalar(&mut output, index, ScalarDepth::F32, response);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), MatDepth::F32)?;
    Ok(())
}

pub(crate) fn corner_harris_into(
    source: &Mat,
    destination: &Mat,
    block_size: i32,
    ksize: i32,
    k: f64,
    border_type: i32,
) -> Result<(), CorrelateError> {
    corner_response_into(
        source,
        destination,
        block_size,
        ksize,
        k,
        border_type,
        false,
    )
}

pub(crate) fn corner_min_eigen_val_into(
    source: &Mat,
    destination: &Mat,
    block_size: i32,
    ksize: i32,
    border_type: i32,
) -> Result<(), CorrelateError> {
    corner_response_into(
        source,
        destination,
        block_size,
        ksize,
        0.0,
        border_type,
        true,
    )
}
