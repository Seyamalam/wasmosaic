//! Scalar encoding at the resampling boundary.
use crate::mat::MatDepth;

pub(crate) fn decode(bytes: &[u8], depth: MatDepth) -> f64 {
    match depth {
        MatDepth::U8 => f64::from(bytes[0]),
        MatDepth::I8 => f64::from(i8::from_ne_bytes([bytes[0]])),
        MatDepth::U16 => f64::from(u16::from_ne_bytes(bytes.try_into().expect("u16"))),
        MatDepth::I16 => f64::from(i16::from_ne_bytes(bytes.try_into().expect("i16"))),
        MatDepth::I32 => f64::from(i32::from_ne_bytes(bytes.try_into().expect("i32"))),
        MatDepth::F32 => f64::from(f32::from_ne_bytes(bytes.try_into().expect("f32"))),
        MatDepth::F64 => f64::from_ne_bytes(bytes.try_into().expect("f64")),
    }
}
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn encode(value: f64, depth: MatDepth, output: &mut [u8]) {
    let rounded = value.round_ties_even();
    match depth {
        MatDepth::U8 => output[0] = rounded.clamp(0.0, 255.0) as u8,
        MatDepth::I8 => output.copy_from_slice(&(rounded.clamp(-128.0, 127.0) as i8).to_ne_bytes()),
        MatDepth::U16 => {
            output.copy_from_slice(&(rounded.clamp(0.0, 65535.0) as u16).to_ne_bytes());
        }
        MatDepth::I16 => {
            output.copy_from_slice(&(rounded.clamp(-32768.0, 32767.0) as i16).to_ne_bytes());
        }
        MatDepth::I32 => output.copy_from_slice(
            &(if !value.is_finite()
                || rounded < f64::from(i32::MIN)
                || rounded > f64::from(i32::MAX)
            {
                i32::MIN
            } else {
                rounded as i32
            })
            .to_ne_bytes(),
        ),
        MatDepth::F32 => output.copy_from_slice(&(value as f32).to_ne_bytes()),
        MatDepth::F64 => output.copy_from_slice(&value.to_ne_bytes()),
    }
}
