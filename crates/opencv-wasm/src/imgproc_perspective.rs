//! Original projective inverse mapping with nearest and 1/32-grid bilinear sampling.
use crate::{
    imgproc_border::border_index,
    imgproc_sample::{decode, encode},
    imgproc_warp::{WarpError, read_transform},
    mat::{Mat, MatDepth, MatError},
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_perspective_into(
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
    let interpolation = flags & 7;
    if !matches!(interpolation, 0 | 1 | 3) {
        return Err(WarpError::InvalidInterpolation(interpolation));
    }
    if interpolation != 0 && matches!(source.depth(), MatDepth::I8 | MatDepth::I32) {
        return Err(WarpError::UnsupportedDepth(source.depth()));
    }
    if !(0..=4).contains(&border_type) {
        return Err(WarpError::InvalidBorderType(border_type));
    }
    let coefficients = read_transform::<9>(transform, 3)?;
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(WarpError::NonFiniteTransform);
    }
    let inverse = if flags & 16 != 0 {
        coefficients
    } else {
        inverse_homography(coefficients)
    };
    let (width, height) = if width <= 0 || height <= 0 {
        (source.columns(), source.rows())
    } else {
        (width.unsigned_abs(), height.unsigned_abs())
    };
    let channels = usize::from(source.channels());
    let depth = source.depth();
    let scalar_width = depth.byte_width();
    let length = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul(scalar_width))
        .ok_or(MatError::BufferSizeOverflow)?;
    let input = source.compact_bytes();
    let mut output = vec![0; length];
    let mut constant = vec![0; 4 * scalar_width];
    for channel in 0..4 {
        encode(
            border_value.get(channel).copied().unwrap_or(0.0),
            depth,
            &mut constant[channel * scalar_width..(channel + 1) * scalar_width],
        );
    }
    let sampler = Sampler {
        input: &input,
        rows: source.rows(),
        columns: source.columns(),
        channels,
        depth,
        border_type,
        constant: &constant,
    };
    for y in 0..height {
        for x in 0..width {
            let denominator = inverse[6] * f64::from(x) + inverse[7] * f64::from(y) + inverse[8];
            let scale = if denominator == 0.0 {
                0.0
            } else {
                (if interpolation == 0 { 1.0 } else { 32.0 }) / denominator
            };
            let mapped_x = coordinate(
                (inverse[0] * f64::from(x) + inverse[1] * f64::from(y) + inverse[2]) * scale,
            );
            let mapped_y = coordinate(
                (inverse[3] * f64::from(x) + inverse[4] * f64::from(y) + inverse[5]) * scale,
            );
            let target = ((y as usize * width as usize + x as usize) * channels) * scalar_width;
            for channel in 0..channels {
                let bytes = &mut output
                    [target + channel * scalar_width..target + (channel + 1) * scalar_width];
                if interpolation == 0 {
                    bytes.copy_from_slice(sampler.at(
                        mapped_x.clamp(-32768, 32767),
                        mapped_y.clamp(-32768, 32767),
                        channel,
                    ));
                } else {
                    let value = sampler.linear(mapped_x, mapped_y, channel);
                    encode(value, depth, bytes);
                }
            }
        }
    }
    destination.write_output(output, height, width, source.channels(), depth)?;
    Ok(())
}

struct Sampler<'a> {
    input: &'a [u8],
    rows: u32,
    columns: u32,
    channels: usize,
    depth: MatDepth,
    border_type: i32,
    constant: &'a [u8],
}
impl Sampler<'_> {
    // These depths accumulate bilinear contributions in F32 in the pinned build.
    #[allow(clippy::cast_possible_truncation)]
    fn linear(&self, mapped_x: i64, mapped_y: i64, channel: usize) -> f64 {
        let left = (mapped_x >> 5).clamp(-32768, 32767);
        let top = (mapped_y >> 5).clamp(-32768, 32767);
        let u = f64::from(u8::try_from(mapped_x & 31).expect("five-bit fraction")) / 32.0;
        let v = f64::from(u8::try_from(mapped_y & 31).expect("five-bit fraction")) / 32.0;
        let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
        let pixels = [
            (left, top),
            (left + 1, top),
            (left, top + 1),
            (left + 1, top + 1),
        ];
        let mut value = 0.0;
        for ((sx, sy), weight) in pixels.into_iter().zip(weights) {
            let term = decode(self.at(sx, sy, channel), self.depth) * weight;
            value = if matches!(self.depth, MatDepth::F32 | MatDepth::I16 | MatDepth::U16) {
                f64::from((value as f32) + (term as f32))
            } else {
                value + term
            };
        }
        if self.depth == MatDepth::U8 {
            value = (value + 0.5).floor();
        }
        value
    }

    fn at(&self, x: i64, y: i64, channel: usize) -> &[u8] {
        let width = self.depth.byte_width();
        if let (Some(x), Some(y)) = (
            border_index(x, self.columns, self.border_type),
            border_index(y, self.rows, self.border_type),
        ) {
            let offset = ((y as usize * self.columns as usize + x as usize) * self.channels
                + channel)
                * width;
            &self.input[offset..offset + width]
        } else {
            &self.constant[(channel % 4) * width..(channel % 4 + 1) * width]
        }
    }
}

fn inverse_homography(m: [f64; 9]) -> [f64; 9] {
    let cofactors = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    let determinant = m[0] * cofactors[0] + m[1] * cofactors[3] + m[2] * cofactors[6];
    if determinant == 0.0 {
        [0.0; 9]
    } else {
        cofactors.map(|value| value * (1.0 / determinant))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn coordinate(value: f64) -> i64 {
    if !value.is_finite() {
        return i64::from(i32::MIN);
    }
    value
        .round_ties_even()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64, mat_from_u8, mat_from_u16};
    fn matrix(values: &[f64; 9]) -> Mat {
        mat_from_f64(values, 3, 3, 1).unwrap()
    }
    const IDENTITY: [f64; 9] = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    #[test]
    fn inverse_homography_undoes_a_projective_map() {
        let m = [1.2, 0.1, 4., -0.2, 0.9, 3., 0.02, 0.01, 1.];
        let inverse = inverse_homography(m);
        for row in 0..3 {
            for column in 0..3 {
                let product = (0..3)
                    .map(|k| m[row * 3 + k] * inverse[k * 3 + column])
                    .sum::<f64>();
                assert!((product - f64::from(row == column)).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn nearest_preserves_every_depth_and_channel_layout() {
        for depth in [
            MatDepth::U8,
            MatDepth::I8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::I32,
            MatDepth::F32,
            MatDepth::F64,
        ] {
            for channels in 1..=4 {
                let mut input = vec![0; 6 * usize::from(channels) * depth.byte_width()];
                for (index, bytes) in input.chunks_exact_mut(depth.byte_width()).enumerate() {
                    encode(
                        f64::from(u32::try_from(index).unwrap()) + 0.25,
                        depth,
                        bytes,
                    );
                }
                let source = Mat::from_owned_bytes(input.clone(), 2, 3, channels, depth).unwrap();
                let output = mat_empty();
                warp_perspective_into(&source, &output, &matrix(&IDENTITY), 3, 2, 0, 0, &[0.; 4])
                    .unwrap();
                assert_eq!(output.compact_bytes(), input);
            }
        }
    }
    #[test]
    fn integer_bilinear_rounding_matches_each_depth() {
        let transform = matrix(&[1., 0., 0.5, 0., 1., 0.5, 0., 0., 1.]);
        let u8_source = mat_from_u8(&[0, 10, 20, 30, 40, 50, 60, 70], 2, 4, 1).unwrap();
        let u16_source = mat_from_u16(&[0, 10, 20, 30, 40, 50, 60, 70], 2, 4, 1).unwrap();
        let output = mat_empty();
        warp_perspective_into(&u8_source, &output, &transform, 4, 2, 1, 0, &[9.; 4]).unwrap();
        assert_eq!(output.compact_bytes(), [7, 7, 12, 17, 15, 25, 35, 45]);
        warp_perspective_into(&u16_source, &output, &transform, 4, 2, 1, 0, &[9.; 4]).unwrap();
        let values = output
            .compact_bytes()
            .chunks_exact(2)
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [7, 7, 12, 17, 14, 25, 35, 45]);
    }
    #[test]
    fn input_alias_and_destination_roi_use_snapshots_and_shared_output() {
        let source = mat_from_u8(&[1, 2, 3, 4, 5, 6], 2, 3, 1).unwrap();
        let transform = matrix(&[1., 0., 1., 0., 1., 0., 0., 0., 1.]);
        warp_perspective_into(&source, &source, &transform, 3, 2, 0, 0, &[9.; 4]).unwrap();
        assert_eq!(source.compact_bytes(), [9, 1, 2, 9, 4, 5]);
        let parent = mat_from_u8(&[99; 20], 4, 5, 1).unwrap();
        let roi = parent.roi(1, 1, 2, 3).unwrap();
        warp_perspective_into(&source, &roi, &matrix(&IDENTITY), 3, 2, 0, 0, &[0.; 4]).unwrap();
        assert_eq!(&parent.compact_bytes()[6..9], &[9, 1, 2]);
        assert_eq!(parent.compact_bytes()[5], 99);
    }
    #[test]
    fn singular_maps_sample_the_origin_and_empty_size_uses_source_size() {
        let source = mat_from_u8(&[3, 4, 5, 6], 2, 2, 1).unwrap();
        let output = mat_empty();
        warp_perspective_into(&source, &output, &matrix(&[0.; 9]), 0, 0, 1, 0, &[9.; 4]).unwrap();
        assert_eq!(output.compact_bytes(), [3; 4]);
    }
    #[test]
    fn validation_preserves_an_existing_destination() {
        let source = mat_from_u8(&[1, 2, 3, 4], 2, 2, 1).unwrap();
        let output = mat_from_u8(&[99], 1, 1, 1).unwrap();
        assert!(
            warp_perspective_into(&source, &output, &matrix(&IDENTITY), 2, 2, 5, 0, &[0.; 4])
                .is_err()
        );
        assert!(
            warp_perspective_into(&source, &output, &matrix(&IDENTITY), 2, 2, 0, 5, &[0.; 4])
                .is_err()
        );
        assert_eq!(output.compact_bytes(), [99]);
    }
}
