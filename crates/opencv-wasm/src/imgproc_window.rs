//! Depth-generic sliding-window kernels for morphological, box, and median filters.
//!
//! Behavior derives from the pinned OpenCV.js 4.13.0 browser build: erode/dilate accept
//! U8, U16, F32, and F64 sources; box filters accept U8, U16, I16, I32, F32, and F64;
//! median blur accepts U8, U16, and F32 for kernels up to 5x5 and U8 only beyond that.

use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_border::{
        BORDER_CONSTANT, BORDER_ISOLATED, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_REPLICATE,
        border_index,
    },
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
pub(crate) enum WindowError {
    EmptySource,
    UnsupportedDepth(MatDepth),
    UnsupportedDestinationDepth(i32),
    ShapeMismatch,
    InvalidAnchor { x: i32, y: i32 },
    InvalidKernelSize { width: i32, height: i32 },
    InvalidIterations(i32),
    InvalidBorderType(i32),
    Matrix(MatError),
}

impl fmt::Display for WindowError {
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
            Self::InvalidIterations(iterations) => {
                write!(formatter, "iteration count {iterations} must be positive")
            }
            Self::InvalidBorderType(border) => {
                write!(formatter, "border type {border} is unsupported")
            }
            Self::Matrix(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for WindowError {}

impl From<MatError> for WindowError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

pub(crate) const fn scalar_depth(depth: MatDepth) -> ScalarDepth {
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

fn require_shape(source: &Mat) -> Result<(), WindowError> {
    if source.rows() == 0 || source.columns() == 0 {
        return Err(WindowError::EmptySource);
    }
    Ok(())
}

fn element_offset(row: u32, column: u32, columns: u32, channels: usize, channel: usize) -> usize {
    (row as usize * columns as usize + column as usize) * channels + channel
}

fn resolve_anchor(x: i32, y: i32, width: u32, height: u32) -> Result<(i32, i32), WindowError> {
    let x = if x < 0 { (width / 2) as i32 } else { x };
    let y = if y < 0 { (height / 2) as i32 } else { y };
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return Err(WindowError::InvalidAnchor { x, y });
    }
    Ok((x, y))
}

fn validate_filter_border(border_type: i32) -> Result<(), WindowError> {
    if matches!(
        border_type & !BORDER_ISOLATED,
        BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_REFLECT_101
    ) {
        Ok(())
    } else {
        Err(WindowError::InvalidBorderType(border_type))
    }
}

fn require_morph_depth(source: &Mat) -> Result<(), WindowError> {
    require_shape(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 | MatDepth::F64 => Ok(()),
        depth => Err(WindowError::UnsupportedDepth(depth)),
    }
}

fn require_box_depth(source: &Mat) -> Result<(), WindowError> {
    require_shape(source)?;
    match source.depth() {
        MatDepth::U8
        | MatDepth::U16
        | MatDepth::I16
        | MatDepth::I32
        | MatDepth::F32
        | MatDepth::F64 => Ok(()),
        depth => Err(WindowError::UnsupportedDepth(depth)),
    }
}

fn require_median_depth(source: &Mat, kernel_size: i32) -> Result<(), WindowError> {
    require_shape(source)?;
    if kernel_size > 5 {
        return match source.depth() {
            MatDepth::U8 => Ok(()),
            depth => Err(WindowError::UnsupportedDepth(depth)),
        };
    }
    match source.depth() {
        MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 => Ok(()),
        depth => Err(WindowError::UnsupportedDepth(depth)),
    }
}

#[allow(clippy::too_many_arguments)]
fn morph_pass(
    input: &[u8],
    rows: u32,
    columns: u32,
    channels: usize,
    depth: ScalarDepth,
    kernel: &[u8],
    kernel_width: u32,
    kernel_height: u32,
    anchor_x: i32,
    anchor_y: i32,
    border_type: i32,
    border_value: &[f64],
    dilate: bool,
) -> Vec<u8> {
    let elements = (rows as usize) * (columns as usize) * channels;
    let mut output = vec![0u8; input.len()];
    let extreme = if dilate { f64::MIN } else { f64::MAX };
    let effective = border_type & !BORDER_ISOLATED;
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut best = extreme;
                for kernel_y in 0..kernel_height {
                    let sample_y = i64::from(row) + i64::from(kernel_y) - i64::from(anchor_y);
                    let mapped_y = border_index(sample_y, rows, effective);
                    for kernel_x in 0..kernel_width {
                        if kernel[(kernel_y * kernel_width + kernel_x) as usize] == 0 {
                            continue;
                        }
                        let sample_x =
                            i64::from(column) + i64::from(kernel_x) - i64::from(anchor_x);
                        let mapped_x = border_index(sample_x, columns, effective);
                        let sample = match (mapped_x, mapped_y) {
                            (Some(x), Some(y)) => read_scalar(
                                input,
                                element_offset(y, x, columns, channels, channel),
                                depth,
                            ),
                            _ => border_value[channel.min(border_value.len() - 1)],
                        };
                        best = if dilate {
                            best.max(sample)
                        } else {
                            best.min(sample)
                        };
                    }
                }
                let index = element_offset(row, column, columns, channels, channel);
                let best = match depth {
                    ScalarDepth::F32 => {
                        best.clamp(-3.402_823_466_385_288_6e38, 3.402_823_466_385_288_6e38)
                    }
                    ScalarDepth::F64 => best,
                    _ => best.clamp(-2_147_483_648.0, 2_147_483_647.0),
                };
                write_scalar(&mut output, index, depth, best);
            }
        }
    }
    let _ = elements;
    output
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn morphology_into(
    source: &Mat,
    destination: &Mat,
    kernel: &Mat,
    anchor_x: i32,
    anchor_y: i32,
    iterations: i32,
    border_type: i32,
    border_value: &[f64],
    default_border_value: bool,
    dilate: bool,
) -> Result<(), WindowError> {
    require_morph_depth(source)?;
    validate_filter_border(border_type)?;
    if iterations < 1 {
        return Err(WindowError::InvalidIterations(iterations));
    }
    if kernel.rows() == 0 || kernel.columns() == 0 || kernel.channels() != 1 {
        if kernel.rows() == 0 || kernel.columns() == 0 {
            let input = source.compact_bytes();
            return destination
                .write_output(
                    input,
                    source.rows(),
                    source.columns(),
                    source.channels(),
                    source.depth(),
                )
                .map_err(WindowError::from);
        }
        return Err(WindowError::ShapeMismatch);
    }
    let kernel_compact = kernel.compact_bytes();
    let kernel_depth = scalar_depth(kernel.depth());
    let kernel_count = (kernel.rows() * kernel.columns()) as usize;
    let kernel_mask: Vec<u8> = (0..kernel_count)
        .map(|index| u8::from(read_scalar(&kernel_compact, index, kernel_depth) != 0.0))
        .collect();
    if kernel.depth() != MatDepth::U8 && kernel_mask.iter().any(|&v| v == 0) {
        return Err(WindowError::ShapeMismatch);
    }
    let (anchor_x, anchor_y) = resolve_anchor(anchor_x, anchor_y, kernel.columns(), kernel.rows())?;
    let mut pad = if default_border_value {
        vec![if dilate { f64::MIN } else { f64::MAX }; 4]
    } else {
        let mut padded = border_value.to_vec();
        padded.resize(4, 0.0);
        if dilate {
            for value in &mut padded {
                *value = -*value;
            }
        }
        padded
    };
    pad.resize(4, if dilate { f64::MIN } else { f64::MAX });
    let mut input = source.compact_bytes();
    let depth = scalar_depth(source.depth());
    let channels = usize::from(source.channels());
    for _ in 0..iterations {
        input = morph_pass(
            &input,
            source.rows(),
            source.columns(),
            channels,
            depth,
            &kernel_mask,
            kernel.columns(),
            kernel.rows(),
            anchor_x,
            anchor_y,
            border_type,
            &pad,
            dilate,
        );
    }
    destination.write_output(
        input,
        source.rows(),
        source.columns(),
        source.channels(),
        source.depth(),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn box_filter_into(
    source: &Mat,
    destination: &Mat,
    destination_depth: i32,
    kernel_width: i32,
    kernel_height: i32,
    anchor_x: i32,
    anchor_y: i32,
    normalize: bool,
    border_type: i32,
) -> Result<(), WindowError> {
    require_box_depth(source)?;
    validate_filter_border(border_type)?;
    if kernel_width <= 0 || kernel_height <= 0 {
        return Err(WindowError::InvalidKernelSize {
            width: kernel_width,
            height: kernel_height,
        });
    }
    let kernel_width = kernel_width as u32;
    let kernel_height = kernel_height as u32;
    let (anchor_x, anchor_y) = resolve_anchor(anchor_x, anchor_y, kernel_width, kernel_height)?;
    let out_depth = if destination_depth < 0 {
        source.depth()
    } else {
        match destination_depth & 7 {
            0 => MatDepth::U8,
            2 => MatDepth::U16,
            3 => MatDepth::I16,
            4 => MatDepth::I32,
            5 => MatDepth::F32,
            6 => MatDepth::F64,
            _ => return Err(WindowError::UnsupportedDestinationDepth(destination_depth)),
        }
    };
    let source_depth = scalar_depth(source.depth());
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
    let effective = border_type & !BORDER_ISOLATED;
    let mut output = vec![0u8; input.len() / source_depth.width() * out_scalar.width()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let mut sum = 0.0f64;
                for ky in 0..kernel_height {
                    let sample_y = i64::from(row) + i64::from(ky) - i64::from(anchor_y);
                    let mapped_y = border_index(sample_y, rows, effective);
                    for kx in 0..kernel_width {
                        let sample_x = i64::from(column) + i64::from(kx) - i64::from(anchor_x);
                        let mapped_x = border_index(sample_x, columns, effective);
                        if let (Some(x), Some(y)) = (mapped_x, mapped_y) {
                            sum += read_scalar(
                                &input,
                                element_offset(y, x, columns, channels, channel),
                                source_depth,
                            );
                        }
                    }
                }
                let index = element_offset(row, column, columns, channels, channel);
                if normalize && out_depth == MatDepth::U8 && source.depth() == MatDepth::U8 {
                    let numerator = sum as i64;
                    let fixed_scale = 65536i64 / (kernel_width as i64 * kernel_height as i64);
                    let value = ((numerator * fixed_scale + 32768) >> 16).clamp(0, 255);
                    output[index] = value as u8;
                    continue;
                }
                let accumulated = if source.depth() == MatDepth::I32 {
                    f64::from((sum as i64) as i32)
                } else {
                    sum
                };
                write_box_scalar(&mut output, index, out_scalar, accumulated * scale);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), out_depth)?;
    Ok(())
}

fn write_box_scalar(data: &mut [u8], index: usize, depth: ScalarDepth, value: f64) {
    write_scalar(data, index, depth, value);
}

pub(crate) fn blur_into(
    source: &Mat,
    destination: &Mat,
    kernel_width: i32,
    kernel_height: i32,
    anchor_x: i32,
    anchor_y: i32,
    border_type: i32,
) -> Result<(), WindowError> {
    box_filter_into(
        source,
        destination,
        -1,
        kernel_width,
        kernel_height,
        anchor_x,
        anchor_y,
        true,
        border_type,
    )
}

pub(crate) fn median_blur_into(
    source: &Mat,
    destination: &Mat,
    kernel_size: i32,
) -> Result<(), WindowError> {
    if kernel_size <= 0 || kernel_size % 2 == 0 {
        return Err(WindowError::InvalidKernelSize {
            width: kernel_size,
            height: kernel_size,
        });
    }
    if kernel_size == 1 {
        require_shape(source)?;
        let input = source.compact_bytes();
        return destination
            .write_output(
                input,
                source.rows(),
                source.columns(),
                source.channels(),
                source.depth(),
            )
            .map_err(WindowError::from);
    }
    require_median_depth(source, kernel_size)?;
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows();
    let columns = source.columns();
    let kernel = kernel_size as u32;
    let anchor = (kernel / 2) as i32;
    let mut samples: Vec<f64> = Vec::with_capacity((kernel * kernel) as usize);
    let mut output = vec![0u8; input.len()];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                samples.clear();
                for ky in 0..kernel {
                    let sample_y = (i64::from(row) + i64::from(ky) - i64::from(anchor))
                        .clamp(0, i64::from(rows) - 1) as u32;
                    for kx in 0..kernel {
                        let sample_x = (i64::from(column) + i64::from(kx) - i64::from(anchor))
                            .clamp(0, i64::from(columns) - 1)
                            as u32;
                        samples.push(read_scalar(
                            &input,
                            element_offset(sample_y, sample_x, columns, channels, channel),
                            depth,
                        ));
                    }
                }
                samples.sort_by(f64::total_cmp);
                let median = samples[samples.len() / 2];
                let index = element_offset(row, column, columns, channels, channel);
                write_scalar(&mut output, index, depth, median);
            }
        }
    }
    destination.write_output(output, rows, columns, source.channels(), source.depth())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::mat_from_u8;

    fn kernel(width: u32, height: u32, data: &[u8]) -> Mat {
        mat_from_u8(data, height, width, 1).expect("kernel")
    }

    #[test]
    fn erode_takes_the_minimum_over_the_kernel_footprint() {
        let source = mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, 1).expect("mat");
        let destination = Mat::empty_output();
        morphology_into(
            &source,
            &destination,
            &kernel(3, 3, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
            -1,
            -1,
            1,
            BORDER_CONSTANT,
            &[],
            true,
            false,
        )
        .expect("erode");
        assert_eq!(destination.to_u8_array(), vec![1, 1, 2, 1, 1, 2, 4, 4, 5]);
    }

    #[test]
    fn dilate_takes_the_maximum_over_the_kernel_footprint() {
        let source = mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, 1).expect("mat");
        let destination = Mat::empty_output();
        morphology_into(
            &source,
            &destination,
            &kernel(3, 3, &[1, 1, 1, 1, 1, 1, 1, 1, 1]),
            -1,
            -1,
            1,
            BORDER_CONSTANT,
            &[],
            true,
            true,
        )
        .expect("dilate");
        assert_eq!(destination.to_u8_array(), vec![5, 6, 6, 8, 9, 9, 8, 9, 9]);
    }

    #[test]
    fn box_filter_normalizes_and_replicates_via_reflect_101() {
        let source = mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, 1).expect("mat");
        let destination = Mat::empty_output();
        blur_into(&source, &destination, 3, 3, -1, -1, BORDER_REFLECT_101).expect("blur");
        assert_eq!(destination.to_u8_array(), vec![4, 4, 4, 5, 5, 5, 6, 6, 6]);
    }
}
