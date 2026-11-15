use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

pub(crate) const COLOR_BGR2_BGRA: i32 = 0;
pub(crate) const COLOR_BGRA2_BGR: i32 = 1;
pub(crate) const COLOR_BGR2_RGBA: i32 = 2;
pub(crate) const COLOR_RGBA2_BGR: i32 = 3;
pub(crate) const COLOR_BGR2_RGB: i32 = 4;
pub(crate) const COLOR_BGRA2_RGBA: i32 = 5;
pub(crate) const COLOR_BGR2_GRAY: i32 = 6;
pub(crate) const COLOR_RGB2_GRAY: i32 = 7;
pub(crate) const COLOR_GRAY2_BGR: i32 = 8;
pub(crate) const COLOR_GRAY2_BGRA: i32 = 9;
pub(crate) const COLOR_BGRA2_GRAY: i32 = 10;
pub(crate) const COLOR_RGBA2_GRAY: i32 = 11;

const BLUE_TO_GRAY: u32 = 1_868;
const GREEN_TO_GRAY: u32 = 9_617;
const RED_TO_GRAY: u32 = 4_899;
const GRAY_SHIFT: u32 = 14;
const GRAY_ROUND: u32 = 1 << (GRAY_SHIFT - 1);

const BLUE_TO_GRAY_16: u64 = 3_735;
const GREEN_TO_GRAY_16: u64 = 19_235;
const RED_TO_GRAY_16: u64 = 9_798;
const GRAY_SHIFT_16: u32 = 15;
const GRAY_ROUND_16: u64 = 1 << (GRAY_SHIFT_16 - 1);

const RED_TO_GRAY_F: f32 = 0.299;
const GREEN_TO_GRAY_F: f32 = 0.587;
const BLUE_TO_GRAY_F: f32 = 0.114;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColorError {
    InvalidDestinationChannels(i32),
    InvalidSourceChannels { expected: u16, actual: u16 },
    Matrix(MatError),
    UnsupportedCode(i32),
    UnsupportedDepth(MatDepth),
}

impl fmt::Display for ColorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDestinationChannels(channels) => {
                write!(
                    formatter,
                    "cvtColor destination channels must be 0, 3, or 4; received {channels}"
                )
            }
            Self::InvalidSourceChannels { expected, actual } => write!(
                formatter,
                "cvtColor source has {actual} channels; conversion requires {expected}"
            ),
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedCode(code) => write!(formatter, "unsupported cvtColor code {code}"),
            Self::UnsupportedDepth(depth) => {
                write!(formatter, "cvtColor depth {depth:?} is not implemented")
            }
        }
    }
}

impl Error for ColorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for ColorError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

pub(crate) fn cvt_color_into(
    source: &Mat,
    destination: &Mat,
    code: i32,
    destination_channels: i32,
) -> Result<(), ColorError> {
    let specification = Conversion::from_code(code)?;
    require_source_channels(source, specification.source_channels())?;
    let width = match source.depth() {
        MatDepth::U8 => 1,
        MatDepth::U16 => 2,
        MatDepth::F32 => 4,
        other => return Err(ColorError::UnsupportedDepth(other)),
    };
    let output_channels = specification.output_channels(destination_channels)?;
    let source_bytes = source.compact_bytes();
    let output = specification.convert(&source_bytes, output_channels, width, source.depth());
    destination.write_output(
        output,
        source.rows(),
        source.columns(),
        output_channels,
        source.depth(),
    )?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conversion {
    AddAlpha {
        swap_blue_red: bool,
    },
    DropAlpha {
        swap_blue_red: bool,
    },
    Gray {
        blue_index: usize,
        source_channels: u16,
    },
    GrayToColor {
        default_channels: u16,
    },
    SwapBlueRed {
        channels: u16,
    },
}

impl Conversion {
    fn from_code(code: i32) -> Result<Self, ColorError> {
        match code {
            COLOR_BGR2_BGRA => Ok(Self::AddAlpha {
                swap_blue_red: false,
            }),
            COLOR_BGRA2_BGR => Ok(Self::DropAlpha {
                swap_blue_red: false,
            }),
            COLOR_BGR2_RGBA => Ok(Self::AddAlpha {
                swap_blue_red: true,
            }),
            COLOR_RGBA2_BGR => Ok(Self::DropAlpha {
                swap_blue_red: true,
            }),
            COLOR_BGR2_RGB => Ok(Self::SwapBlueRed { channels: 3 }),
            COLOR_BGRA2_RGBA => Ok(Self::SwapBlueRed { channels: 4 }),
            COLOR_BGR2_GRAY => Ok(Self::Gray {
                blue_index: 0,
                source_channels: 3,
            }),
            COLOR_RGB2_GRAY => Ok(Self::Gray {
                blue_index: 2,
                source_channels: 3,
            }),
            COLOR_GRAY2_BGR => Ok(Self::GrayToColor {
                default_channels: 3,
            }),
            COLOR_GRAY2_BGRA => Ok(Self::GrayToColor {
                default_channels: 4,
            }),
            COLOR_BGRA2_GRAY => Ok(Self::Gray {
                blue_index: 0,
                source_channels: 4,
            }),
            COLOR_RGBA2_GRAY => Ok(Self::Gray {
                blue_index: 2,
                source_channels: 4,
            }),
            _ => Err(ColorError::UnsupportedCode(code)),
        }
    }

    const fn source_channels(self) -> u16 {
        match self {
            Self::AddAlpha { .. } | Self::SwapBlueRed { channels: 3 } => 3,
            Self::DropAlpha { .. } | Self::SwapBlueRed { channels: 4 } => 4,
            Self::Gray {
                source_channels, ..
            } => source_channels,
            Self::GrayToColor { .. } => 1,
            Self::SwapBlueRed { .. } => unreachable!(),
        }
    }

    fn output_channels(self, requested: i32) -> Result<u16, ColorError> {
        let default = match self {
            Self::AddAlpha { .. } | Self::SwapBlueRed { channels: 4 } => 4,
            Self::DropAlpha { .. } | Self::SwapBlueRed { channels: 3 } => 3,
            Self::Gray { .. } => return Ok(1),
            Self::GrayToColor { default_channels } => default_channels,
            Self::SwapBlueRed { .. } => unreachable!(),
        };
        match requested {
            0 => Ok(default),
            3 => Ok(3),
            4 => Ok(4),
            value => Err(ColorError::InvalidDestinationChannels(value)),
        }
    }

    fn convert(
        self,
        source: &[u8],
        output_channels: u16,
        width: usize,
        depth: MatDepth,
    ) -> Vec<u8> {
        let source_channels = usize::from(self.source_channels());
        let output_channels_usize = usize::from(output_channels);
        let pixels = source.len() / (source_channels * width);
        let mut output = Vec::with_capacity(pixels * output_channels_usize * width);
        for pixel in source.chunks_exact(source_channels * width) {
            match self {
                Self::AddAlpha { swap_blue_red } | Self::DropAlpha { swap_blue_red } => {
                    append_color_pixel(&mut output, pixel, output_channels, swap_blue_red, width, depth);
                }
                Self::SwapBlueRed { .. } => {
                    append_color_pixel(&mut output, pixel, output_channels, true, width, depth);
                }
                Self::Gray {
                    blue_index,
                    source_channels: _,
                } => {
                    let red_index = 2 - blue_index;
                    let blue = channel(pixel, blue_index, width, depth);
                    let green = channel(pixel, 1, width, depth);
                    let red = channel(pixel, red_index, width, depth);
                    write_gray(&mut output, red, green, blue, depth);
                }
                Self::GrayToColor { .. } => {
                    let gray = &pixel[..width];
                    output.extend_from_slice(gray);
                    output.extend_from_slice(gray);
                    output.extend_from_slice(gray);
                    if output_channels == 4 {
                        output.extend_from_slice(alpha_bytes(depth));
                    }
                }
            }
        }
        output
    }
}

fn channel(pixel: &[u8], index: usize, width: usize, depth: MatDepth) -> f64 {
    let offset = index * width;
    match depth {
        MatDepth::U8 => f64::from(pixel[offset]),
        MatDepth::U16 => f64::from(u16::from_ne_bytes(
            pixel[offset..offset + 2].try_into().expect("u16 channel"),
        )),
        MatDepth::F32 => f64::from(f32::from_ne_bytes(
            pixel[offset..offset + 4].try_into().expect("f32 channel"),
        )),
        _ => unreachable!("unsupported depth filtered at entry"),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn write_gray(output: &mut Vec<u8>, red: f64, green: f64, blue: f64, depth: MatDepth) {
    match depth {
        MatDepth::U8 => {
            let weighted = (red as u32) * RED_TO_GRAY
                + (green as u32) * GREEN_TO_GRAY
                + (blue as u32) * BLUE_TO_GRAY
                + GRAY_ROUND;
            output.push(u8::try_from(weighted >> GRAY_SHIFT).unwrap_or(u8::MAX));
        }
        MatDepth::U16 => {
            let weighted = (red as u64) * RED_TO_GRAY_16
                + (green as u64) * GREEN_TO_GRAY_16
                + (blue as u64) * BLUE_TO_GRAY_16
                + GRAY_ROUND_16;
            let value = u16::try_from(weighted >> GRAY_SHIFT_16).unwrap_or(u16::MAX);
            output.extend_from_slice(&value.to_ne_bytes());
        }
        MatDepth::F32 => {
            let weighted = (red as f32) * RED_TO_GRAY_F
                + (green as f32) * GREEN_TO_GRAY_F
                + (blue as f32) * BLUE_TO_GRAY_F;
            output.extend_from_slice(&weighted.to_ne_bytes());
        }
        _ => unreachable!("unsupported depth filtered at entry"),
    }
}

const ALPHA_U16: [u8; 2] = u16::MAX.to_ne_bytes();
const ALPHA_F32: [u8; 4] = 1.0f32.to_ne_bytes();

fn alpha_bytes(depth: MatDepth) -> &'static [u8] {
    match depth {
        MatDepth::U8 => &[u8::MAX],
        MatDepth::U16 => &ALPHA_U16,
        MatDepth::F32 => &ALPHA_F32,
        _ => unreachable!("unsupported depth filtered at entry"),
    }
}

fn require_source_channels(source: &Mat, expected: u16) -> Result<(), ColorError> {
    let actual = source.channels();
    if actual != expected {
        return Err(ColorError::InvalidSourceChannels { expected, actual });
    }
    Ok(())
}

fn append_color_pixel(
    output: &mut Vec<u8>,
    pixel: &[u8],
    channels: u16,
    swap: bool,
    width: usize,
    depth: MatDepth,
) {
    if swap {
        output.extend_from_slice(&pixel[2 * width..3 * width]);
        output.extend_from_slice(&pixel[width..2 * width]);
        output.extend_from_slice(&pixel[..width]);
    } else {
        output.extend_from_slice(&pixel[..3 * width]);
    }
    if channels == 4 {
        match pixel.get(3 * width..4 * width) {
            Some(alpha) => output.extend_from_slice(alpha),
            None => output.extend_from_slice(alpha_bytes(depth)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f32, mat_from_u16, mat_from_u8};

    #[test]
    fn rgba_to_gray_matches_pinned_u8_rounding() {
        let source = mat_from_u8(&[255, 0, 0, 7, 1, 2, 3, 4], 1, 2, 4).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_RGBA2_GRAY, 0).expect("conversion");
        assert_eq!(destination.channels(), 1);
        assert_eq!(destination.compact_bytes(), [76, 2]);
    }

    #[test]
    fn rgba_to_bgra_preserves_alpha() {
        let source = mat_from_u8(&[1, 2, 3, 4], 1, 1, 4).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGRA2_RGBA, 0).expect("conversion");
        assert_eq!(destination.compact_bytes(), [3, 2, 1, 4]);
    }

    #[test]
    fn gray_expansion_uses_an_opaque_alpha_channel() {
        let source = mat_from_u8(&[7, 9], 1, 2, 1).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_GRAY2_BGRA, 0).expect("conversion");
        assert_eq!(destination.compact_bytes(), [7, 7, 7, 255, 9, 9, 9, 255]);
    }

    #[test]
    fn conversion_is_safe_in_place() {
        let matrix = mat_from_u8(&[1, 2, 3, 4], 1, 1, 4).expect("source");
        cvt_color_into(&matrix, &matrix, COLOR_RGBA2_GRAY, 0).expect("conversion");
        assert_eq!(matrix.channels(), 1);
        assert_eq!(matrix.compact_bytes(), [2]);
    }

    #[test]
    fn u16_gray_uses_the_wide_lookup_coefficients() {
        let source =
            mat_from_u16(&[2570, 5140, 7710, 1, 2, 3], 1, 2, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGR2_GRAY, 0).expect("conversion");
        assert_eq!(destination.depth(), MatDepth::U16);
        assert_eq!(
            destination.compact_bytes(),
            [5616u16, 2]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<u8>>()
        );
    }

    #[test]
    fn u16_gray_preserves_equal_channels() {
        let source = mat_from_u16(&[1000, 1000, 1000], 1, 1, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGR2_GRAY, 0).expect("conversion");
        assert_eq!(destination.compact_bytes(), 1000u16.to_ne_bytes());
    }

    #[test]
    fn f32_gray_uses_float_weights() {
        let source = mat_from_f32(&[0.1, 0.5, 0.9], 1, 1, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGR2_GRAY, 0).expect("conversion");
        let value = f32::from_ne_bytes(destination.compact_bytes().try_into().expect("f32"));
        assert!((value - 0.574).abs() < 1e-6);
    }

    #[test]
    fn f32_alpha_fill_is_one() {
        let source = mat_from_f32(&[0.1, 0.2, 0.3], 1, 1, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGR2_RGBA, 0).expect("conversion");
        assert_eq!(destination.channels(), 4);
        let bytes = destination.compact_bytes();
        let alpha = f32::from_ne_bytes(bytes[12..16].try_into().expect("alpha"));
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn u16_shuffle_preserves_alpha() {
        let source = mat_from_u16(&[1, 2, 3, 4], 1, 1, 4).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGRA2_RGBA, 0).expect("conversion");
        assert_eq!(destination.depth(), MatDepth::U16);
        assert_eq!(
            destination.compact_bytes(),
            [3u16, 2, 1, 4]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<u8>>()
        );
    }
}
