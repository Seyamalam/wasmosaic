//! OpenCV-compatible lookup-table transforms.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
enum LutError {
    UnsupportedSourceDepth(MatDepth),
    IncorrectTableLength { actual: u64 },
    IncorrectTableChannels { source: u16, table: u16 },
    IncompatibleDestination,
    SizeOverflow,
    Matrix(MatError),
}

impl fmt::Display for LutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSourceDepth(depth) => write!(
                formatter,
                "LUT source depth {depth:?} is unsupported; expected U8 or I8"
            ),
            Self::IncorrectTableLength { actual } => write!(
                formatter,
                "LUT must contain exactly 256 entries; found {actual}"
            ),
            Self::IncorrectTableChannels { source, table } => write!(
                formatter,
                "LUT channels must be 1 or match source channels ({source}); found {table}"
            ),
            Self::IncompatibleDestination => formatter.write_str(
                "LUT destination must match source rows, columns, and channels and table depth",
            ),
            Self::SizeOverflow => formatter.write_str("LUT output size exceeds the WASM limit"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for LutError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for LutError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Applies a 256-entry lookup table and returns a newly allocated matrix.
///
/// The source must have U8 or I8 depth. A table may have one channel, in which case it is reused
/// for every source channel, or the same channel count as the source. The output depth is the table
/// depth.
///
/// # Errors
///
/// Returns an error for an unsupported source depth or an invalid table shape/channel count.
#[wasm_bindgen(js_name = matLut)]
pub fn mat_lut(source: &Mat, table: &Mat) -> Result<Mat, JsError> {
    lut_mat(source, table).map_err(JsError::from)
}

/// Applies a lookup table into an existing matrix, including strided regions of interest.
///
/// Validation and lookup complete before destination storage is changed, making aliased source,
/// table, and destination matrices deterministic.
///
/// # Errors
///
/// Returns an error for invalid inputs or a destination whose shape, channels, or depth differ
/// from the required output.
#[wasm_bindgen(js_name = matLutInto)]
pub fn mat_lut_into(source: &Mat, table: &Mat, destination: &Mat) -> Result<(), JsError> {
    lut_into_mat(source, table, destination).map_err(JsError::from)
}

fn lut_mat(source: &Mat, table: &Mat) -> Result<Mat, LutError> {
    let bytes = lookup_bytes(source, table)?;
    Mat::from_owned_bytes(
        bytes,
        source.rows(),
        source.columns(),
        source.channels(),
        table.depth(),
    )
    .map_err(LutError::from)
}

fn lut_into_mat(source: &Mat, table: &Mat, destination: &Mat) -> Result<(), LutError> {
    let bytes = lookup_bytes(source, table)?;
    destination.write_output(
        bytes,
        source.rows(),
        source.columns(),
        source.channels(),
        table.depth(),
    )?;
    Ok(())
}

fn lookup_bytes(source: &Mat, table: &Mat) -> Result<Vec<u8>, LutError> {
    if !matches!(source.depth(), MatDepth::U8 | MatDepth::I8) {
        return Err(LutError::UnsupportedSourceDepth(source.depth()));
    }
    let table_entries = u64::from(table.rows()) * u64::from(table.columns());
    if table_entries != 256 {
        return Err(LutError::IncorrectTableLength {
            actual: table_entries,
        });
    }
    if table.channels() != 1 && table.channels() != source.channels() {
        return Err(LutError::IncorrectTableChannels {
            source: source.channels(),
            table: table.channels(),
        });
    }

    let source_bytes = source.compact_bytes();
    let table_bytes = table.compact_bytes();
    let scalar_width = table.depth().byte_width();
    let output_length = source_bytes
        .len()
        .checked_mul(scalar_width)
        .ok_or(LutError::SizeOverflow)?;
    let source_channels = usize::from(source.channels());
    let table_channels = usize::from(table.channels());
    let mut output = vec![0; output_length];

    for (scalar_offset, (source_value, destination_scalar)) in source_bytes
        .into_iter()
        .zip(output.chunks_exact_mut(scalar_width))
        .enumerate()
    {
        let index = usize::from(source_value);
        let channel = scalar_offset % source_channels;
        let table_channel = if table_channels == 1 { 0 } else { channel };
        let table_scalar = index
            .checked_mul(table_channels)
            .and_then(|offset| offset.checked_add(table_channel))
            .ok_or(LutError::SizeOverflow)?;
        let byte_start = table_scalar
            .checked_mul(scalar_width)
            .ok_or(LutError::SizeOverflow)?;
        let byte_end = byte_start
            .checked_add(scalar_width)
            .ok_or(LutError::SizeOverflow)?;
        destination_scalar.copy_from_slice(&table_bytes[byte_start..byte_end]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(bytes: Vec<u8>, rows: u32, columns: u32, channels: u16, depth: MatDepth) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, channels, depth).expect("valid test matrix")
    }

    #[test]
    fn unsigned_source_uses_a_single_channel_table_for_every_channel() {
        let source = matrix(vec![0, 1, 255, 2], 1, 2, 2, MatDepth::U8);
        let table = matrix((0_u8..=255).rev().collect(), 1, 256, 1, MatDepth::U8);

        let output = lut_mat(&source, &table).expect("valid LUT");

        assert_eq!(output.to_u8_array(), [255, 254, 0, 253]);
        assert_eq!(output.channels(), 2);
    }

    #[test]
    fn signed_source_indexes_the_table_by_its_raw_byte_and_preserves_table_depth() {
        let source = matrix(vec![128, 255, 0, 127], 1, 4, 1, MatDepth::I8);
        let values = (0_i16..256).collect::<Vec<_>>();
        let table_bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let table = matrix(table_bytes, 256, 1, 1, MatDepth::I16);

        let output = lut_mat(&source, &table).expect("valid signed LUT");

        assert_eq!(
            output.to_i16_array().expect("I16 output"),
            [128, 255, 0, 127]
        );
        assert_eq!(output.depth(), MatDepth::I16);
    }

    #[test]
    fn multi_channel_table_selects_the_corresponding_channel() {
        let source = matrix(vec![0, 1, 2, 3], 1, 2, 2, MatDepth::U8);
        let mut table_bytes = Vec::with_capacity(512);
        for index in 0_u8..=255 {
            table_bytes.extend_from_slice(&[index, 255 - index]);
        }
        let table = matrix(table_bytes, 16, 16, 2, MatDepth::U8);

        let output = lut_mat(&source, &table).expect("valid channel LUT");

        assert_eq!(output.to_u8_array(), [0, 254, 2, 252]);
    }

    #[test]
    fn reads_strided_source_and_writes_through_a_strided_destination() {
        let source_parent = matrix((0_u8..18).collect(), 3, 3, 2, MatDepth::U8);
        let source = source_parent.roi(0, 1, 2, 2).expect("source ROI");
        let table = matrix((0_u8..=255).rev().collect(), 256, 1, 1, MatDepth::U8);
        let destination_parent = matrix(vec![0; 18], 3, 3, 2, MatDepth::U8);
        let destination = destination_parent.roi(1, 0, 2, 2).expect("destination ROI");

        lut_into_mat(&source, &table, &destination).expect("LUT into ROI");

        assert_eq!(
            destination.to_u8_array(),
            [253, 252, 251, 250, 247, 246, 245, 244]
        );
        assert_eq!(
            destination_parent.to_u8_array(),
            [
                0, 0, 0, 0, 0, 0, 253, 252, 251, 250, 0, 0, 247, 246, 245, 244, 0, 0
            ]
        );
    }

    #[test]
    fn supports_every_table_depth_without_numeric_conversion() {
        let source = matrix(vec![0, 255], 1, 2, 1, MatDepth::U8);
        for depth in [
            MatDepth::U8,
            MatDepth::I8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::I32,
            MatDepth::F32,
            MatDepth::F64,
        ] {
            let width = depth.byte_width();
            let mut bytes = vec![0; 256 * width];
            bytes[..width].fill(0x12);
            bytes[255 * width..].fill(0xAB);
            let table = matrix(bytes, 1, 256, 1, depth);

            let output = lut_mat(&source, &table).expect("supported table depth");

            assert_eq!(output.depth(), depth);
            assert_eq!(
                output.compact_bytes(),
                [vec![0x12; width], vec![0xAB; width]].concat()
            );
        }
    }

    #[test]
    fn validates_every_input_before_mutating_destination() {
        let source = matrix(vec![0, 1], 1, 2, 1, MatDepth::U8);
        let short_table = matrix(vec![0; 255], 1, 255, 1, MatDepth::U8);
        let destination = matrix(vec![9, 9], 1, 2, 1, MatDepth::U8);

        let error = lut_into_mat(&source, &short_table, &destination).expect_err("invalid table");

        assert_eq!(error, LutError::IncorrectTableLength { actual: 255 });
        assert_eq!(destination.to_u8_array(), [9, 9]);
    }

    #[test]
    fn rejects_non_byte_sources_and_unrelated_table_channel_counts() {
        let non_byte_source = matrix(vec![0; 4], 1, 2, 1, MatDepth::U16);
        let one_channel_table = matrix(vec![0; 256], 1, 256, 1, MatDepth::U8);
        assert_eq!(
            lut_mat(&non_byte_source, &one_channel_table).expect_err("non-byte source"),
            LutError::UnsupportedSourceDepth(MatDepth::U16)
        );

        let source = matrix(vec![0; 6], 1, 2, 3, MatDepth::U8);
        let two_channel_table = matrix(vec![0; 512], 1, 256, 2, MatDepth::U8);
        assert_eq!(
            lut_mat(&source, &two_channel_table).expect_err("unrelated channel count"),
            LutError::IncorrectTableChannels {
                source: 3,
                table: 2
            }
        );
    }

    #[test]
    fn permits_exact_in_place_lookup_without_read_after_write_corruption() {
        let source = matrix((0_u8..=255).collect(), 16, 16, 1, MatDepth::U8);
        let table = matrix((0_u8..=255).rev().collect(), 1, 256, 1, MatDepth::U8);

        lut_into_mat(&source, &table, &source).expect("in-place LUT");

        assert_eq!(source.to_u8_array(), (0_u8..=255).rev().collect::<Vec<_>>());
    }
}
