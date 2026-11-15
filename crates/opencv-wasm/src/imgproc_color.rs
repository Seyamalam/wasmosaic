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
pub(crate) const COLOR_BGR2XYZ: i32 = 32;
pub(crate) const COLOR_RGB2XYZ: i32 = 33;
pub(crate) const COLOR_XYZ2BGR: i32 = 34;
pub(crate) const COLOR_XYZ2RGB: i32 = 35;
pub(crate) const COLOR_BGR2_YCR_CB: i32 = 36;
pub(crate) const COLOR_RGB2_YCR_CB: i32 = 37;
pub(crate) const COLOR_YCR_CB2BGR: i32 = 38;
pub(crate) const COLOR_YCR_CB2RGB: i32 = 39;
pub(crate) const COLOR_BGR2YUV: i32 = 82;
pub(crate) const COLOR_RGB2YUV: i32 = 83;
pub(crate) const COLOR_YUV2BGR: i32 = 84;
pub(crate) const COLOR_YUV2RGB: i32 = 85;

const BLUE_TO_GRAY: u32 = 1_868;
const GREEN_TO_GRAY: u32 = 9_617;
const RED_TO_GRAY: u32 = 4_899;
const GRAY_SHIFT: u32 = 14;
const GRAY_ROUND: u32 = 1 << (GRAY_SHIFT - 1);

// U8 linear color transforms use a single descale after a 32-bit dot product. Coefficients are
// pinned-implementation lookups recovered through dense black-box probing, not textbook values.
const LINEAR_SHIFT: i32 = 14;
const LINEAR_ROUND: i32 = 1 << (LINEAR_SHIFT - 1);
const CHROMA_SHIFT: i32 = 13;
const CHROMA_ROUND: i32 = 1 << (CHROMA_SHIFT - 1);
const CHROMA_CENTER: i32 = 128;

// RGB to XYZ rows apply to (blue, green, red) input order.
const XYZ_FROM_BGR: [[i32; 3]; 3] = [
    [2_956, 5_860, 6_756],
    [1_184, 11_716, 3_484],
    [15_568, 1_952, 316],
];
// XYZ to BGR rows apply to (x, y, z) input order.
const BGR_FROM_XYZ: [[i32; 3]; 3] = [
    [912, -3_344, 17_324],
    [-15_880, 30_736, 680],
    [53_092, -25_184, -8_168],
];
// Forward chroma is a pure function of the source channel minus the rounded luma byte.
const YCR_CB_CR: i32 = 11_682;
const YCR_CB_CB: i32 = 9_241;
const YUV_U: i32 = 8_061;
const YUV_V: i32 = 14_369;
// Inverse chroma adds a descaled delta from the 128-centered source channel to luma.
const YCR_CB_TO_B: i32 = 14_524;
const YCR_CB_TO_R: i32 = 11_493;
const YCR_CB_TO_G_CR: i32 = -5_849;
const YCR_CB_TO_G_CB: i32 = -2_818;
const YUV_TO_B: i32 = 16_646;
const YUV_TO_R: i32 = 9_340;
const YUV_TO_G_U: i32 = -6_472;
const YUV_TO_G_V: i32 = -9_519;

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
    if !specification.accepts_source_channels(source.channels()) {
        return Err(ColorError::InvalidSourceChannels {
            expected: specification.source_channels(),
            actual: source.channels(),
        });
    }
    let depth = source.depth();
    if specification.is_linear() && depth != MatDepth::U8 {
        return Err(ColorError::UnsupportedDepth(depth));
    }
    let width = match depth {
        MatDepth::U8 => 1,
        MatDepth::U16 => 2,
        MatDepth::F32 => 4,
        other => return Err(ColorError::UnsupportedDepth(other)),
    };
    let output_channels = specification.output_channels(destination_channels)?;
    let source_bytes = source.compact_bytes();
    let output = specification.convert(
        &source_bytes,
        source.channels(),
        output_channels,
        width,
        source.depth(),
    );
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
    Linear {
        swapped: bool,
        kind: LinearKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearKind {
    Xyz,
    XyzInverse,
    YCrCb,
    YCrCbInverse,
    Yuv,
    YuvInverse,
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
            COLOR_BGR2XYZ | COLOR_RGB2XYZ => Ok(Self::Linear {
                swapped: code == COLOR_RGB2XYZ,
                kind: LinearKind::Xyz,
            }),
            COLOR_XYZ2BGR | COLOR_XYZ2RGB => Ok(Self::Linear {
                swapped: code == COLOR_XYZ2RGB,
                kind: LinearKind::XyzInverse,
            }),
            COLOR_BGR2_YCR_CB | COLOR_RGB2_YCR_CB => Ok(Self::Linear {
                swapped: code == COLOR_RGB2_YCR_CB,
                kind: LinearKind::YCrCb,
            }),
            COLOR_YCR_CB2BGR | COLOR_YCR_CB2RGB => Ok(Self::Linear {
                swapped: code == COLOR_YCR_CB2RGB,
                kind: LinearKind::YCrCbInverse,
            }),
            COLOR_BGR2YUV | COLOR_RGB2YUV => Ok(Self::Linear {
                swapped: code == COLOR_RGB2YUV,
                kind: LinearKind::Yuv,
            }),
            COLOR_YUV2BGR | COLOR_YUV2RGB => Ok(Self::Linear {
                swapped: code == COLOR_YUV2RGB,
                kind: LinearKind::YuvInverse,
            }),
            _ => Err(ColorError::UnsupportedCode(code)),
        }
    }

    const fn is_linear(self) -> bool {
        matches!(self, Self::Linear { .. })
    }

    const fn source_channels(self) -> u16 {
        match self {
            Self::AddAlpha { .. } | Self::SwapBlueRed { channels: 3 } => 3,
            Self::DropAlpha { .. } | Self::SwapBlueRed { channels: 4 } => 4,
            Self::Gray {
                source_channels, ..
            } => source_channels,
            Self::GrayToColor { .. } => 1,
            Self::Linear { .. } => 3,
            Self::SwapBlueRed { .. } => unreachable!(),
        }
    }

    fn accepts_source_channels(self, actual: u16) -> bool {
        match self {
            Self::Linear {
                kind: LinearKind::Xyz | LinearKind::YCrCb | LinearKind::Yuv,
                ..
            } => actual == 3 || actual == 4,
            _ => actual == self.source_channels(),
        }
    }

    fn output_channels(self, requested: i32) -> Result<u16, ColorError> {
        let default = match self {
            Self::AddAlpha { .. } | Self::SwapBlueRed { channels: 4 } => 4,
            Self::DropAlpha { .. } | Self::SwapBlueRed { channels: 3 } => 3,
            Self::Gray { .. } => return Ok(1),
            Self::GrayToColor { default_channels } => default_channels,
            Self::Linear { kind, .. } => match kind {
                // Forward transforms always emit three channels; inverse transforms grow alpha.
                LinearKind::Xyz | LinearKind::YCrCb | LinearKind::Yuv => {
                    return match requested {
                        0 | 3 | 4 => Ok(3),
                        value => Err(ColorError::InvalidDestinationChannels(value)),
                    };
                }
                _ => 3,
            },
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
        source_channels: u16,
        output_channels: u16,
        width: usize,
        depth: MatDepth,
    ) -> Vec<u8> {
        let source_channels = usize::from(source_channels);
        let output_channels_usize = usize::from(output_channels);
        let pixels = source.len() / (source_channels * width);
        let mut output = Vec::with_capacity(pixels * output_channels_usize * width);
        for pixel in source.chunks_exact(source_channels * width) {
            match self {
                Self::AddAlpha { swap_blue_red } | Self::DropAlpha { swap_blue_red } => {
                    append_color_pixel(
                        &mut output,
                        pixel,
                        output_channels,
                        swap_blue_red,
                        width,
                        depth,
                    );
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
                Self::Linear { swapped, kind } => {
                    let [b, g, r] = linear_pixel(kind, swapped, pixel);
                    output.extend_from_slice(&[b, g, r]);
                    if output_channels == 4 {
                        output.push(u8::MAX);
                    }
                }
            }
        }
        output
    }
}

fn linear_pixel(kind: LinearKind, swapped: bool, pixel: &[u8]) -> [u8; 3] {
    let c0 = i32::from(pixel[0]);
    let c1 = i32::from(pixel[1]);
    let c2 = i32::from(pixel[2]);
    let (blue, green, red) = if swapped { (c2, c1, c0) } else { (c0, c1, c2) };
    match kind {
        LinearKind::Xyz => XYZ_FROM_BGR.map(|row| {
            clamp_u8(descale(
                blue * row[0] + green * row[1] + red * row[2],
                LINEAR_SHIFT,
            ))
        }),
        LinearKind::XyzInverse => {
            let bgr = BGR_FROM_XYZ.map(|row| {
                clamp_u8(descale(
                    c0 * row[0] + c1 * row[1] + c2 * row[2],
                    LINEAR_SHIFT,
                ))
            });
            if swapped {
                [bgr[2], bgr[1], bgr[0]]
            } else {
                bgr
            }
        }
        LinearKind::YCrCb | LinearKind::Yuv => {
            let luma = ((blue as u32) * BLUE_TO_GRAY
                + (green as u32) * GREEN_TO_GRAY
                + (red as u32) * RED_TO_GRAY
                + GRAY_ROUND) as i32
                >> GRAY_SHIFT;
            let (chroma_a, chroma_b) = if kind == LinearKind::YCrCb {
                (red - luma, blue - luma)
            } else {
                (blue - luma, red - luma)
            };
            let (ca, cb) = if kind == LinearKind::YCrCb {
                (YCR_CB_CR, YCR_CB_CB)
            } else {
                (YUV_U, YUV_V)
            };
            [
                clamp_u8(luma),
                clamp_u8(CHROMA_CENTER + descale(chroma_a * ca, LINEAR_SHIFT)),
                clamp_u8(CHROMA_CENTER + descale(chroma_b * cb, LINEAR_SHIFT)),
            ]
        }
        LinearKind::YCrCbInverse | LinearKind::YuvInverse => {
            let luma = c0;
            let (chroma_a, chroma_b) = (c1 - CHROMA_CENTER, c2 - CHROMA_CENTER);
            // YCrCb orders its deltas (Cr, Cb); YUV orders them (U, V).
            let (b_delta, b_coef, g_a, g_b, r_delta, r_coef, g_shift) =
                if kind == LinearKind::YCrCbInverse {
                    (
                        chroma_b,
                        YCR_CB_TO_B,
                        YCR_CB_TO_G_CR,
                        YCR_CB_TO_G_CB,
                        chroma_a,
                        YCR_CB_TO_R,
                        CHROMA_SHIFT,
                    )
                } else {
                    (
                        chroma_a,
                        YUV_TO_B,
                        YUV_TO_G_U,
                        YUV_TO_G_V,
                        chroma_b,
                        YUV_TO_R,
                        LINEAR_SHIFT,
                    )
                };
            let bgr = [
                clamp_u8(luma + descale(b_delta * b_coef, CHROMA_SHIFT)),
                clamp_u8(descale(
                    luma * (1 << g_shift) + chroma_a * g_a + chroma_b * g_b,
                    g_shift,
                )),
                clamp_u8(luma + descale(r_delta * r_coef, CHROMA_SHIFT)),
            ];
            if swapped {
                [bgr[2], bgr[1], bgr[0]]
            } else {
                bgr
            }
        }
    }
}

fn descale(value: i32, shift: i32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, i32::from(u8::MAX)) as u8
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
    use crate::mat::{mat_empty, mat_from_f32, mat_from_u8, mat_from_u16};

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
        let source = mat_from_u16(&[2570, 5140, 7710, 1, 2, 3], 1, 2, 3).expect("source");
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

    fn linear_expect(code: i32, pixel: [u8; 3], expected: [u8; 3]) {
        let source = mat_from_u8(&pixel, 1, 1, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, code, 0).expect("conversion");
        assert_eq!(destination.compact_bytes(), expected, "code {code}");
    }

    #[test]
    fn xyz_round_trips_match_pinned_u8() {
        // Pinned OpenCV.js 4.13.0 outputs for a (10, 20, 30) source pixel.
        linear_expect(COLOR_BGR2XYZ, [10, 20, 30], [21, 21, 12]);
        linear_expect(COLOR_RGB2XYZ, [10, 20, 30], [17, 19, 31]);
        linear_expect(COLOR_XYZ2BGR, [10, 20, 30], [28, 29, 0]);
        linear_expect(COLOR_XYZ2RGB, [10, 20, 30], [0, 29, 28]);
        linear_expect(COLOR_XYZ2BGR, [100, 110, 120], [110, 114, 95]);
        linear_expect(COLOR_XYZ2RGB, [100, 110, 120], [95, 114, 110]);
    }

    #[test]
    fn ycr_cb_matches_pinned_u8() {
        linear_expect(COLOR_BGR2_YCR_CB, [10, 20, 30], [22, 134, 121]);
        linear_expect(COLOR_RGB2_YCR_CB, [10, 20, 30], [18, 122, 135]);
        linear_expect(COLOR_YCR_CB2BGR, [10, 20, 30], [0, 121, 0]);
        linear_expect(COLOR_YCR_CB2RGB, [10, 20, 30], [0, 121, 0]);
        linear_expect(COLOR_YCR_CB2BGR, [100, 110, 120], [86, 116, 75]);
        linear_expect(COLOR_YCR_CB2RGB, [100, 110, 120], [75, 116, 86]);
    }

    #[test]
    fn yuv_matches_pinned_u8() {
        linear_expect(COLOR_BGR2YUV, [10, 20, 30], [22, 122, 135]);
        linear_expect(COLOR_RGB2YUV, [10, 20, 30], [18, 134, 121]);
        linear_expect(COLOR_YUV2BGR, [10, 20, 30], [0, 110, 0]);
        linear_expect(COLOR_YUV2RGB, [10, 20, 30], [0, 110, 0]);
        linear_expect(COLOR_YUV2BGR, [100, 110, 120], [63, 112, 91]);
        linear_expect(COLOR_YUV2RGB, [100, 110, 120], [91, 112, 63]);
    }

    #[test]
    fn forward_linear_accepts_alpha_and_always_emits_three_channels() {
        let source = mat_from_u8(&[10, 20, 30, 40], 1, 1, 4).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_BGR2_YCR_CB, 4).expect("conversion");
        assert_eq!(destination.channels(), 3);
        assert_eq!(destination.compact_bytes(), [22, 134, 121]);
    }

    #[test]
    fn inverse_linear_grows_opaque_alpha_on_request() {
        let source = mat_from_u8(&[10, 20, 30], 1, 1, 3).expect("source");
        let destination = mat_empty();
        cvt_color_into(&source, &destination, COLOR_XYZ2BGR, 4).expect("conversion");
        assert_eq!(destination.compact_bytes(), [28, 29, 0, 255]);
    }

    #[test]
    fn linear_rejects_non_u8_depth_and_alpha_source() {
        let wide = mat_from_u16(&[10, 20, 30], 1, 1, 3).expect("source");
        let destination = mat_empty();
        assert!(matches!(
            cvt_color_into(&wide, &destination, COLOR_BGR2XYZ, 0),
            Err(ColorError::UnsupportedDepth(MatDepth::U16))
        ));
        let alpha = mat_from_u8(&[10, 20, 30, 40], 1, 1, 4).expect("source");
        assert!(matches!(
            cvt_color_into(&alpha, &destination, COLOR_XYZ2BGR, 0),
            Err(ColorError::InvalidSourceChannels {
                expected: 3,
                actual: 4
            })
        ));
    }
}
