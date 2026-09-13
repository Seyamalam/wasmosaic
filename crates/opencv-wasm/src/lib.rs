//! Browser-focused image processing operations for the `wasmosaic` package.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

mod core_algebra;
mod core_algebra_wasm;
mod core_channels;
mod core_channels_wasm;
mod core_float_math;
mod core_float_math_wasm;
mod core_layout;
mod core_layout_wasm;
mod core_lut_wasm;
mod core_norm;
mod core_norm_wasm;
mod core_numeric;
mod core_ops;
mod core_random;
mod core_random_wasm;
mod core_reductions;
mod core_reductions_wasm;
mod core_runtime;
mod core_runtime_wasm;
mod core_stats;
mod core_stats_wasm;
mod core_transform;
mod core_transform_wasm;
mod core_wasm;
mod features2d_akaze;
mod features2d_akaze_wasm;
mod features2d_gftt;
mod features2d_gftt_wasm;
mod features2d_kaze;
mod features2d_kaze_wasm;
mod features2d_mser;
mod features2d_mser_wasm;
mod features2d_orb;
mod features2d_orb_wasm;
mod features2d_threshold_detectors;
mod features2d_threshold_detectors_wasm;
mod imgproc_approx;
mod imgproc_border;
mod imgproc_border_wasm;
mod imgproc_canny;
mod imgproc_canny_wasm;
mod imgproc_color;
mod imgproc_color_wasm;
mod imgproc_contours;
mod imgproc_contours_wasm;
mod imgproc_filter;
mod imgproc_filter_wasm;
mod imgproc_geometry;
mod imgproc_geometry_wasm;
mod imgproc_helpers;
mod imgproc_helpers_wasm;
mod imgproc_histogram;
mod imgproc_histogram_wasm;
mod imgproc_perspective;
mod imgproc_resize;
mod imgproc_resize_wasm;
mod imgproc_sample;
mod imgproc_template;
mod imgproc_template_wasm;
mod imgproc_threshold;
mod imgproc_threshold_wasm;
mod imgproc_transform_matrices;
mod imgproc_transform_matrices_wasm;
mod imgproc_warp;
mod imgproc_warp_wasm;
mod mat;
mod mat_vector;
mod mutable_storage;
mod photo_tonemap;
mod photo_tonemap_wasm;

pub use mat::{Mat, MatDepth, mat_empty, mat_from_u8, mat_zeros_u8};

const RGBA_CHANNELS: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ImageError {
    EmptyDimensions,
    BufferSizeOverflow,
    IncorrectBufferLength { expected: usize, actual: usize },
}

impl fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => {
                formatter.write_str("image dimensions must be greater than zero")
            }
            Self::BufferSizeOverflow => {
                formatter.write_str("image dimensions exceed the WASM buffer limit")
            }
            Self::IncorrectBufferLength { expected, actual } => {
                write!(
                    formatter,
                    "RGBA buffer has {actual} bytes; expected {expected} bytes"
                )
            }
        }
    }
}

impl Error for ImageError {}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = grayscaleRgba)]
/// Converts RGBA pixels to grayscale with fixed-point BT.601 luma weights.
///
/// # Errors
///
/// Returns an error when dimensions are zero, exceed the WASM buffer limit, or do not match the
/// supplied byte length.
pub fn grayscale_rgba(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsError> {
    grayscale(data, width, height).map_err(JsError::from)
}

#[wasm_bindgen(js_name = invertRgba)]
/// Inverts the RGB channels of an RGBA image while preserving alpha.
///
/// # Errors
///
/// Returns an error when dimensions are zero, exceed the WASM buffer limit, or do not match the
/// supplied byte length.
pub fn invert_rgba(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsError> {
    invert(data, width, height).map_err(JsError::from)
}

#[wasm_bindgen(js_name = thresholdRgba)]
/// Applies an inclusive binary luma threshold while preserving alpha.
///
/// # Errors
///
/// Returns an error when dimensions are zero, exceed the WASM buffer limit, or do not match the
/// supplied byte length.
pub fn threshold_rgba(
    data: &[u8],
    width: u32,
    height: u32,
    threshold: u8,
) -> Result<Vec<u8>, JsError> {
    threshold_binary(data, width, height, threshold).map_err(JsError::from)
}

#[wasm_bindgen(js_name = resizeNearestRgba)]
/// Resizes an RGBA image with nearest-neighbor sampling.
///
/// # Errors
///
/// Returns an error when source or target dimensions are zero, exceed the WASM buffer limit, or do
/// not match the supplied byte length.
pub fn resize_nearest_rgba(
    data: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, JsError> {
    resize_nearest(data, width, height, target_width, target_height).map_err(JsError::from)
}

fn checked_buffer_length(width: u32, height: u32) -> Result<usize, ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::EmptyDimensions);
    }

    let pixels = width
        .checked_mul(height)
        .ok_or(ImageError::BufferSizeOverflow)?;
    let bytes = pixels
        .checked_mul(RGBA_CHANNELS)
        .ok_or(ImageError::BufferSizeOverflow)?;
    usize::try_from(bytes).map_err(|_| ImageError::BufferSizeOverflow)
}

fn validate_rgba(data: &[u8], width: u32, height: u32) -> Result<(), ImageError> {
    let expected = checked_buffer_length(width, height)?;
    if data.len() != expected {
        return Err(ImageError::IncorrectBufferLength {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

fn luminance(red: u8, green: u8, blue: u8) -> u8 {
    let weighted = u32::from(red) * 77 + u32::from(green) * 150 + u32::from(blue) * 29 + 128;
    u8::try_from(weighted >> 8).unwrap_or(u8::MAX)
}

fn grayscale(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, ImageError> {
    validate_rgba(data, width, height)?;
    let mut output = Vec::with_capacity(data.len());

    for pixel in data.chunks_exact(RGBA_CHANNELS as usize) {
        let gray = luminance(pixel[0], pixel[1], pixel[2]);
        output.extend_from_slice(&[gray, gray, gray, pixel[3]]);
    }

    Ok(output)
}

fn invert(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, ImageError> {
    validate_rgba(data, width, height)?;
    let mut output = Vec::with_capacity(data.len());

    for pixel in data.chunks_exact(RGBA_CHANNELS as usize) {
        output.extend_from_slice(&[255 - pixel[0], 255 - pixel[1], 255 - pixel[2], pixel[3]]);
    }

    Ok(output)
}

fn threshold_binary(
    data: &[u8],
    width: u32,
    height: u32,
    threshold: u8,
) -> Result<Vec<u8>, ImageError> {
    validate_rgba(data, width, height)?;
    let mut output = Vec::with_capacity(data.len());

    for pixel in data.chunks_exact(RGBA_CHANNELS as usize) {
        let value = if luminance(pixel[0], pixel[1], pixel[2]) >= threshold {
            255
        } else {
            0
        };
        output.extend_from_slice(&[value, value, value, pixel[3]]);
    }

    Ok(output)
}

fn resize_nearest(
    data: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, ImageError> {
    validate_rgba(data, width, height)?;
    let target_length = checked_buffer_length(target_width, target_height)?;
    let mut output = vec![0; target_length];

    for target_y in 0..target_height {
        let source_y = u64::from(target_y) * u64::from(height) / u64::from(target_height);
        for target_x in 0..target_width {
            let source_x = u64::from(target_x) * u64::from(width) / u64::from(target_width);
            let source_offset = usize::try_from(
                (source_y * u64::from(width) + source_x) * u64::from(RGBA_CHANNELS),
            )
            .map_err(|_| ImageError::BufferSizeOverflow)?;
            let target_offset = usize::try_from(
                (u64::from(target_y) * u64::from(target_width) + u64::from(target_x))
                    * u64::from(RGBA_CHANNELS),
            )
            .map_err(|_| ImageError::BufferSizeOverflow)?;
            output[target_offset..target_offset + 4]
                .copy_from_slice(&data[source_offset..source_offset + 4]);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_uses_integer_luminance_and_preserves_alpha() {
        let output = grayscale(&[255, 0, 0, 17], 1, 1).expect("valid image");
        assert_eq!(output, [77, 77, 77, 17]);
    }

    #[test]
    fn invert_preserves_alpha() {
        let output = invert(&[10, 20, 30, 40], 1, 1).expect("valid image");
        assert_eq!(output, [245, 235, 225, 40]);
    }

    #[test]
    fn threshold_is_inclusive() {
        let output = threshold_binary(&[100, 100, 100, 255], 1, 1, 100).expect("valid image");
        assert_eq!(output, [255, 255, 255, 255]);
    }

    #[test]
    fn resize_nearest_duplicates_source_pixels() {
        let source = [255, 0, 0, 255, 0, 0, 255, 255];
        let output = resize_nearest(&source, 2, 1, 4, 1).expect("valid image");
        assert_eq!(
            output,
            [
                255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255,
            ]
        );
    }

    #[test]
    fn invalid_buffer_length_is_rejected() {
        let error = grayscale(&[0, 0, 0], 1, 1).expect_err("invalid image must fail");
        assert_eq!(
            error,
            ImageError::IncorrectBufferLength {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn empty_dimensions_are_rejected() {
        let error = invert(&[], 0, 1).expect_err("empty image must fail");
        assert_eq!(error, ImageError::EmptyDimensions);
    }
}
