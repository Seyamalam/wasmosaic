//! Direct sliding-window matching from the documented correlation and distance formulae.
//! All channel means are separate; only the final score combines channels.

use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchMethod {
    SquaredDifference,
    NormalizedSquaredDifference,
    Correlation,
    NormalizedCorrelation,
    Coefficient,
    NormalizedCoefficient,
}

impl MatchMethod {
    fn from_code(code: i32) -> Result<Self, TemplateError> {
        match code {
            0 => Ok(Self::SquaredDifference),
            1 => Ok(Self::NormalizedSquaredDifference),
            2 => Ok(Self::Correlation),
            3 => Ok(Self::NormalizedCorrelation),
            4 => Ok(Self::Coefficient),
            5 => Ok(Self::NormalizedCoefficient),
            _ => Err(TemplateError::InvalidMethod(code)),
        }
    }

    fn centered(self) -> bool {
        matches!(self, Self::Coefficient | Self::NormalizedCoefficient)
    }

    fn normalized(self) -> bool {
        matches!(
            self,
            Self::NormalizedSquaredDifference
                | Self::NormalizedCorrelation
                | Self::NormalizedCoefficient
        )
    }

    fn squared_difference(self) -> bool {
        matches!(
            self,
            Self::SquaredDifference | Self::NormalizedSquaredDifference
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TemplateError {
    EmptyInput,
    InvalidMethod(i32),
    Matrix(MatError),
    MismatchedTypes,
    InvalidSize,
    InvalidMask,
    UnsupportedDepth,
    UnsupportedChannels,
}

impl fmt::Display for TemplateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("matchTemplate inputs must not be empty"),
            Self::InvalidMethod(method) => write!(formatter, "invalid template matching method {method}"),
            Self::Matrix(error) => error.fmt(formatter),
            Self::MismatchedTypes => formatter.write_str("matchTemplate image and template must have the same depth and channels"),
            Self::InvalidSize => formatter.write_str("matchTemplate template must fit inside the image in both dimensions"),
            Self::InvalidMask => formatter.write_str("matchTemplate mask must be U8 or F32, match the template size, and have one or the template's channel count"),
            Self::UnsupportedDepth => formatter.write_str("matchTemplate requires U8 or F32 inputs"),
            Self::UnsupportedChannels => formatter.write_str("matchTemplate supports one through four channels"),
        }
    }
}

impl Error for TemplateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for TemplateError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

pub(crate) fn match_template_into(
    image: &Mat,
    template: &Mat,
    destination: &Mat,
    method: i32,
    mask: Option<&Mat>,
) -> Result<(), TemplateError> {
    let method = MatchMethod::from_code(method)?;
    validate_inputs(image, template)?;
    let mask = mask.filter(|mask| mask.rows() != 0 && mask.columns() != 0);
    if let Some(mask) = mask {
        validate_mask(template, mask)?;
    }
    // The pinned unmasked binding accepts the reversed size relationship too.
    let (image, template) = if mask.is_none()
        && image.rows() <= template.rows()
        && image.columns() <= template.columns()
    {
        (template, image)
    } else {
        (image, template)
    };
    if template.rows() > image.rows() || template.columns() > image.columns() {
        return Err(TemplateError::InvalidSize);
    }

    let rows = image.rows() - template.rows() + 1;
    let columns = image.columns() - template.columns() + 1;
    let output_bytes = result_byte_length(rows, columns)?;
    // Snapshot before destination replacement or ROI writes, including exact aliases.
    let input = samples(image);
    let pattern = samples(template);
    let layout = WindowLayout {
        image_stride: image.columns() as usize * usize::from(image.channels()),
        template_stride: template.columns() as usize * usize::from(template.channels()),
        channels: usize::from(image.channels()),
    };
    let prepared = PreparedTemplate::new(&pattern, layout.channels, method, mask);
    let mut output = Vec::with_capacity(output_bytes);
    for row in 0..rows as usize {
        for column in 0..columns as usize {
            let origin = row * layout.image_stride + column * layout.channels;
            let score = prepared.score(&input[origin..], &layout, method);
            #[allow(clippy::cast_possible_truncation)] // The API's result depth is F32.
            output.extend_from_slice(&(score as f32).to_ne_bytes());
        }
    }
    destination.write_output(output, rows, columns, 1, MatDepth::F32)?;
    Ok(())
}

fn result_byte_length(rows: u32, columns: u32) -> Result<usize, TemplateError> {
    rows.checked_mul(columns)
        .and_then(|pixels| pixels.checked_mul(4))
        .map(|bytes| bytes as usize)
        .ok_or(TemplateError::Matrix(MatError::BufferSizeOverflow))
}

fn validate_inputs(image: &Mat, template: &Mat) -> Result<(), TemplateError> {
    if image.rows() == 0 || image.columns() == 0 || template.rows() == 0 || template.columns() == 0
    {
        return Err(TemplateError::EmptyInput);
    }
    if image.depth() != template.depth() || image.channels() != template.channels() {
        return Err(TemplateError::MismatchedTypes);
    }
    if !matches!(image.depth(), MatDepth::U8 | MatDepth::F32) {
        return Err(TemplateError::UnsupportedDepth);
    }
    if !(1..=4).contains(&image.channels()) {
        return Err(TemplateError::UnsupportedChannels);
    }
    Ok(())
}

fn validate_mask(template: &Mat, mask: &Mat) -> Result<(), TemplateError> {
    if mask.rows() != template.rows()
        || mask.columns() != template.columns()
        || (mask.channels() != 1 && mask.channels() != template.channels())
        || !matches!(mask.depth(), MatDepth::U8 | MatDepth::F32)
    {
        return Err(TemplateError::InvalidMask);
    }
    Ok(())
}

fn samples(matrix: &Mat) -> Vec<f32> {
    let bytes = matrix.compact_bytes();
    if matrix.depth() == MatDepth::U8 {
        return bytes.into_iter().map(f32::from).collect();
    }
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
}

struct WindowLayout {
    image_stride: usize,
    template_stride: usize,
    channels: usize,
}

struct PreparedTemplate {
    values: Vec<f64>,
    weights: Option<Vec<f64>>,
    weight_sums: [f64; 4],
    energy: f64,
}

impl PreparedTemplate {
    fn new(pattern: &[f32], channels: usize, method: MatchMethod, mask: Option<&Mat>) -> Self {
        let weights = mask.map(|mask| {
            let mask_samples = samples(mask);
            (0..pattern.len())
                .map(|index| {
                    let mask_index = if mask.channels() == 1 {
                        index / channels
                    } else {
                        index
                    };
                    let value = mask_samples[mask_index];
                    if mask.depth() == MatDepth::U8 {
                        f64::from(u8::from(value != 0.0))
                    } else {
                        f64::from(value)
                    }
                })
                .collect::<Vec<_>>()
        });
        let mut weight_sums = [0.0; 4];
        let mut means = [0.0; 4];
        for (index, &value) in pattern.iter().enumerate() {
            let weight = weights.as_ref().map_or(1.0, |weights| weights[index]);
            weight_sums[index % channels] += weight;
            means[index % channels] += f64::from(value) * weight;
        }
        for channel in 0..channels {
            means[channel] = if method.centered() {
                means[channel] / weight_sums[channel]
            } else {
                0.0
            };
        }
        let values: Vec<_> = pattern
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let weight = weights.as_ref().map_or(1.0, |weights| weights[index]);
                (f64::from(value) - means[index % channels]) * weight
            })
            .collect();
        let energy = values.iter().map(|value| value * value).sum();
        Self {
            values,
            weights,
            weight_sums,
            energy,
        }
    }

    fn score(&self, image: &[f32], layout: &WindowLayout, method: MatchMethod) -> f64 {
        if let Some(weights) = &self.weights {
            return self.masked_score(image, layout, method, weights);
        }
        let mut sums = [0.0; 4];
        let mut image_energy = 0.0;
        let mut numerator = 0.0;
        for (row, pattern_row) in self.values.chunks_exact(layout.template_stride).enumerate() {
            let input_row = &image[row * layout.image_stride..][..layout.template_stride];
            for (index, (&input, &pattern)) in input_row.iter().zip(pattern_row).enumerate() {
                let input = f64::from(input);
                if method.squared_difference() {
                    numerator += (input - pattern) * (input - pattern);
                } else {
                    numerator += input * pattern;
                }
                if method.normalized() {
                    image_energy += input * input;
                    if method.centered() {
                        sums[index % layout.channels] += input;
                    }
                }
            }
        }
        if !method.normalized() {
            return numerator;
        }
        if method == MatchMethod::NormalizedCoefficient {
            // The unmasked browser contract defines a constant template as a perfect match.
            if self.energy == 0.0 {
                return 1.0;
            }
            image_energy -= sums
                .iter()
                .zip(self.weight_sums)
                .take(layout.channels)
                .map(|(&sum, count)| sum * sum / count)
                .sum::<f64>();
        }
        let denominator = (image_energy.max(0.0) * self.energy).sqrt();
        if denominator == 0.0 {
            return if method.squared_difference() {
                1.0
            } else {
                0.0
            };
        }
        if method.squared_difference() {
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    }

    fn masked_score(
        &self,
        image: &[f32],
        layout: &WindowLayout,
        method: MatchMethod,
        weights: &[f64],
    ) -> f64 {
        let mut means = [0.0; 4];
        if method.centered() {
            for (row, weights_row) in weights.chunks_exact(layout.template_stride).enumerate() {
                let input_row = &image[row * layout.image_stride..][..layout.template_stride];
                for (index, (&input, &weight)) in input_row.iter().zip(weights_row).enumerate() {
                    means[index % layout.channels] += f64::from(input) * weight;
                }
            }
            for (mean, sum) in means.iter_mut().zip(self.weight_sums).take(layout.channels) {
                *mean /= sum;
            }
        }
        let mut numerator = 0.0;
        let mut image_energy = 0.0;
        for (row, pattern_row) in self.values.chunks_exact(layout.template_stride).enumerate() {
            let input_row = &image[row * layout.image_stride..][..layout.template_stride];
            let weights_row = &weights[row * layout.template_stride..];
            for (index, (&input, &pattern)) in input_row.iter().zip(pattern_row).enumerate() {
                let input =
                    (f64::from(input) - means[index % layout.channels]) * weights_row[index];
                numerator += if method.squared_difference() {
                    (input - pattern) * (input - pattern)
                } else {
                    input * pattern
                };
                image_energy += input * input;
            }
        }
        if method.normalized() {
            // Nonempty masks use the documented quotient, including NaN/Infinity at zero norms.
            numerator / (image_energy * self.energy).sqrt()
        } else {
            numerator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f32, mat_from_i16, mat_from_u8};

    fn scores(image: &Mat, template: &Mat, method: i32, mask: Option<&Mat>) -> Vec<f32> {
        let result = mat_empty();
        match_template_into(image, template, &result, method, mask).expect("matching");
        assert_eq!(result.depth(), MatDepth::F32);
        assert_eq!(result.channels(), 1);
        samples(&result)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn six_methods_match_hand_calculated_windows() {
        let image = mat_from_u8(&[1, 2, 3, 4], 1, 4, 1).expect("image");
        let template = mat_from_u8(&[1, 2], 1, 2, 1).expect("template");
        let expected = [
            [0.0, 2.0, 8.0],
            [0.0, 2.0 / 65.0_f32.sqrt(), 8.0 / 125.0_f32.sqrt()],
            [5.0, 8.0, 11.0],
            [1.0, 8.0 / 65.0_f32.sqrt(), 11.0 / 125.0_f32.sqrt()],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ];
        for (method, expected) in expected.iter().enumerate() {
            assert_close(
                &scores(
                    &image,
                    &template,
                    i32::try_from(method).expect("method"),
                    None,
                ),
                expected,
            );
        }
    }

    #[test]
    fn coefficient_removes_each_channels_offset() {
        let image = mat_from_f32(&[11.0, 80.0, 13.0, 180.0], 1, 2, 2).expect("image");
        let template = mat_from_f32(&[1.0, 100.0, 3.0, 200.0], 1, 2, 2).expect("template");
        assert_close(&scores(&image, &template, 4, None), &[5002.0]);
        assert_close(&scores(&image, &template, 5, None), &[1.0]);
        let reversed = mat_from_f32(&[3.0, 200.0, 1.0, 100.0], 1, 2, 2).expect("reversed");
        assert_close(&scores(&reversed, &template, 5, None), &[-1.0]);
    }

    #[test]
    fn masks_use_binary_selection_or_squared_float_weights() {
        let image = mat_from_u8(&[1, 2, 3], 1, 3, 1).expect("image");
        let template = mat_from_u8(&[1, 2], 1, 2, 1).expect("template");
        let binary = mat_from_u8(&[2, 255], 1, 2, 1).expect("binary mask");
        let weighted = mat_from_f32(&[0.25, 1.0], 1, 2, 1).expect("weighted mask");
        for method in 0..6 {
            assert_close(
                &scores(&image, &template, method, Some(&binary)),
                &scores(&image, &template, method, None),
            );
        }
        assert_close(
            &scores(&image, &template, 0, Some(&weighted)),
            &[0.0, 1.0625],
        );
        assert_close(
            &scores(&image, &template, 2, Some(&weighted)),
            &[4.0625, 6.125],
        );
        assert_close(
            &scores(&image, &template, 4, Some(&weighted)),
            &[0.08, 0.08],
        );
        assert_eq!(binary.compact_bytes(), [2, 255]);
        assert_eq!(samples(&weighted), [0.25, 1.0]);
    }

    #[test]
    fn one_channel_masks_broadcast_and_multichannel_masks_select_lanes() {
        let image = mat_from_u8(&[1, 10, 3, 20], 1, 2, 2).expect("image");
        let template = mat_from_u8(&[2, 8, 4, 24], 1, 2, 2).expect("template");
        let shared = mat_from_u8(&[0, 7], 1, 2, 1).expect("shared mask");
        let lanes = mat_from_u8(&[0, 0, 0, 7], 1, 2, 2).expect("lane mask");
        assert_close(&scores(&image, &template, 0, Some(&shared)), &[17.0]);
        assert_close(&scores(&image, &template, 0, Some(&lanes)), &[16.0]);
    }

    #[test]
    fn masked_and_unmasked_zero_norms_have_distinct_results() {
        let image = mat_from_u8(&[1, 2, 3], 1, 3, 1).expect("image");
        let zero = mat_from_u8(&[0, 0], 1, 2, 1).expect("zero template");
        let mask = mat_from_u8(&[1, 1], 1, 2, 1).expect("mask");
        assert_eq!(scores(&image, &zero, 1, None), [1.0, 1.0]);
        assert_eq!(scores(&image, &zero, 3, None), [0.0, 0.0]);
        assert_eq!(scores(&image, &zero, 5, None), [1.0, 1.0]);
        assert!(
            scores(&image, &zero, 1, Some(&mask))
                .iter()
                .all(|v| *v == f32::INFINITY)
        );
        for method in [3, 5] {
            assert!(
                scores(&image, &zero, method, Some(&mask))
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
        assert!(
            scores(&image, &mask, 4, Some(&zero))
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn strided_inputs_and_outputs_preserve_parent_padding() {
        let parent = mat_from_u8(
            &[99, 1, 2, 3, 99, 99, 4, 5, 6, 99, 99, 7, 8, 9, 99],
            3,
            5,
            1,
        )
        .expect("parent");
        let image = parent.roi(0, 1, 3, 3).expect("image ROI");
        let template = parent.roi(1, 2, 2, 2).expect("template ROI");
        let output_parent = mat_from_f32(&[-1.0; 12], 3, 4, 1).expect("output parent");
        let output = output_parent.roi(1, 1, 2, 2).expect("output ROI");
        match_template_into(&image, &template, &output, 0, None).expect("match ROIs");
        assert_eq!(samples(&output), [64.0, 36.0, 4.0, 0.0]);
        assert_eq!(
            samples(&output_parent),
            [
                -1.0, -1.0, -1.0, -1.0, -1.0, 64.0, 36.0, -1.0, -1.0, 4.0, 0.0, -1.0
            ]
        );
        assert_eq!(image.compact_bytes(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn destination_aliases_snapshot_inputs_before_replacement() {
        let image = mat_from_f32(&[1.0, 2.0, 3.0], 1, 3, 1).expect("image");
        let template = mat_from_f32(&[1.0, 2.0], 1, 2, 1).expect("template");
        match_template_into(&image, &template, &image, 0, None).expect("image alias");
        assert_eq!(samples(&image), [0.0, 2.0]);
        match_template_into(&image, &template, &template, 0, None).expect("template alias");
        assert_eq!(samples(&template), [1.0]);
    }

    #[test]
    fn unmasked_swapped_inputs_and_empty_mask_match() {
        let image = mat_from_u8(&[1, 2, 3], 1, 3, 1).expect("image");
        let template = mat_from_u8(&[1, 2], 1, 2, 1).expect("template");
        let empty = mat_empty();
        for method in 0..6 {
            assert_eq!(
                scores(&template, &image, method, None),
                scores(&image, &template, method, Some(&empty))
            );
        }
    }

    #[test]
    fn invalid_inputs_leave_destination_unchanged() {
        let image = mat_from_u8(&[1, 2, 3], 1, 3, 1).expect("image");
        let template = mat_from_u8(&[1, 2], 1, 2, 1).expect("template");
        let empty = mat_empty();
        let tall = mat_from_u8(&[1, 2], 2, 1, 1).expect("tall");
        let signed = mat_from_i16(&[1, 2], 1, 2, 1).expect("signed");
        let destination = mat_from_u8(&[42], 1, 1, 1).expect("destination");
        for error in [
            match_template_into(&image, &template, &destination, 6, None),
            match_template_into(&empty, &template, &destination, 0, None),
            match_template_into(&image, &tall, &destination, 0, None),
            match_template_into(&image, &signed, &destination, 0, None),
            match_template_into(&signed, &signed, &destination, 0, None),
            match_template_into(&image, &template, &destination, 0, Some(&image)),
            match_template_into(&image, &template, &destination, 0, Some(&signed)),
            match_template_into(&template, &image, &destination, 0, Some(&image)),
        ] {
            assert!(error.is_err());
            assert_eq!(destination.compact_bytes(), [42]);
            assert_eq!(destination.depth(), MatDepth::U8);
        }
    }

    #[test]
    fn result_size_checks_the_f32_wasm_byte_limit_before_allocating() {
        assert_eq!(result_byte_length(1, 10).expect("small output"), 40);
        assert!(result_byte_length(1, u32::MAX).is_err());
        assert!(result_byte_length(65_536, 65_536).is_err());
    }
}
