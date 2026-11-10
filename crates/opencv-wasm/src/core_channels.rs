//! Byte-exact channel layout kernels for compact, interleaved matrices.

use std::{error::Error, fmt};

/// A compact matrix whose scalar values are stored as uninterpreted native bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMatrix {
    bytes: Vec<u8>,
    rows: u32,
    columns: u32,
    channels: u16,
    element_byte_width: u8,
}

impl ChannelMatrix {
    /// Creates a compact interleaved matrix after validating its complete byte length.
    ///
    /// # Errors
    ///
    /// Returns an error for empty layout components, byte-length overflow, or mismatched storage.
    pub fn new(
        bytes: Vec<u8>,
        rows: u32,
        columns: u32,
        channels: u16,
        element_byte_width: u8,
    ) -> Result<Self, ChannelError> {
        let expected = checked_byte_length(rows, columns, channels, element_byte_width)?;
        if bytes.len() != expected {
            return Err(ChannelError::IncorrectByteLength {
                expected,
                actual: bytes.len(),
            });
        }

        Ok(Self {
            bytes,
            rows,
            columns,
            channels,
            element_byte_width,
        })
    }

    /// Returns the compact interleaved bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the number of matrix rows.
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the number of matrix columns.
    #[must_use]
    pub const fn columns(&self) -> u32 {
        self.columns
    }

    /// Returns the number of interleaved channels.
    #[must_use]
    pub const fn channels(&self) -> u16 {
        self.channels
    }

    /// Returns the number of bytes occupied by one scalar value.
    #[must_use]
    pub const fn element_byte_width(&self) -> u8 {
        self.element_byte_width
    }

    fn scalar_offset(&self, pixel: usize, channel: u16) -> usize {
        (pixel * usize::from(self.channels) + usize::from(channel))
            * usize::from(self.element_byte_width)
    }

    fn pixel_count(&self) -> usize {
        self.rows as usize * self.columns as usize
    }
}

/// One channel transfer used by [`mix_channels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelMapping {
    /// Index into the source matrix slice.
    pub source_index: usize,
    /// Zero-based channel within the selected source matrix.
    pub source_channel: u16,
    /// Index into the destination matrix slice.
    pub destination_index: usize,
    /// Zero-based channel within the selected destination matrix.
    pub destination_channel: u16,
}

/// Errors reported by checked channel kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// A row or column count was zero.
    EmptyDimensions,
    /// A matrix had no channels.
    EmptyChannels,
    /// A scalar element had a zero byte width.
    EmptyElement,
    /// An operation requiring sources received none.
    EmptySources,
    /// An operation requiring destinations received none.
    EmptyDestinations,
    /// Dimensions could not be represented as a byte length.
    BufferSizeOverflow,
    /// The requested output allocation failed.
    AllocationFailed,
    /// Supplied storage did not match the matrix layout.
    IncorrectByteLength {
        /// Required compact byte length.
        expected: usize,
        /// Supplied byte length.
        actual: usize,
    },
    /// Matrices did not have the same rows and columns.
    IncompatibleShape,
    /// Matrices used different byte widths for scalar elements.
    IncompatibleElementByteWidth,
    /// An operation requiring a single-channel source received another layout.
    SourceMustHaveOneChannel,
    /// A mapping referred to a missing source matrix.
    SourceIndexOutOfBounds {
        /// Invalid matrix index.
        index: usize,
    },
    /// A mapping referred to a missing destination matrix.
    DestinationIndexOutOfBounds {
        /// Invalid matrix index.
        index: usize,
    },
    /// A mapping referred to a missing source channel.
    SourceChannelOutOfBounds {
        /// Source matrix index.
        index: usize,
        /// Invalid channel index.
        channel: u16,
    },
    /// A mapping referred to a missing destination channel.
    DestinationChannelOutOfBounds {
        /// Destination matrix index.
        index: usize,
        /// Invalid channel index.
        channel: u16,
    },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ChannelError {}

/// Splits an interleaved matrix into compact single-channel planes.
///
/// # Errors
///
/// Returns an error if output sizing overflows or allocation fails.
pub fn split(source: &ChannelMatrix) -> Result<Vec<ChannelMatrix>, ChannelError> {
    let plane_len = checked_byte_length(source.rows, source.columns, 1, source.element_byte_width)?;
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(usize::from(source.channels))
        .map_err(|_| ChannelError::AllocationFailed)?;
    for channel in 0..source.channels {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plane_len)
            .map_err(|_| ChannelError::AllocationFailed)?;
        copy_channel_to_vec(source, channel, &mut bytes);
        planes.push(ChannelMatrix::new(
            bytes,
            source.rows,
            source.columns,
            1,
            source.element_byte_width,
        )?);
    }
    Ok(planes)
}

/// Merges compatible planes into one compact interleaved matrix.
///
/// Each source may carry multiple channels; its channel run stays contiguous in the output, so a
/// three-channel plane followed by a single-channel plane produces four-channel pixels ordered
/// `a0, a1, a2, b0`.
///
/// # Errors
///
/// Returns an error for empty input, incompatible planes, too many channels, overflow, or an
/// allocation failure.
pub fn merge(planes: &[ChannelMatrix]) -> Result<ChannelMatrix, ChannelError> {
    let first = planes.first().ok_or(ChannelError::EmptySources)?;
    ensure_compatible(planes, first)?;
    let mut channels = 0_u32;
    for plane in planes {
        channels = channels
            .checked_add(u32::from(plane.channels))
            .ok_or(ChannelError::BufferSizeOverflow)?;
    }
    let channels = u16::try_from(channels).map_err(|_| ChannelError::BufferSizeOverflow)?;
    let length = checked_byte_length(
        first.rows,
        first.columns,
        channels,
        first.element_byte_width,
    )?;
    let mut bytes = zeroed_bytes(length)?;
    let width = usize::from(first.element_byte_width);
    for pixel in 0..first.pixel_count() {
        let mut destination_start = pixel * usize::from(channels) * width;
        for plane in planes {
            let run = usize::from(plane.channels) * width;
            let source_start = pixel * run;
            bytes[destination_start..destination_start + run]
                .copy_from_slice(&plane.bytes[source_start..source_start + run]);
            destination_start += run;
        }
    }
    ChannelMatrix::new(
        bytes,
        first.rows,
        first.columns,
        channels,
        first.element_byte_width,
    )
}

/// Copies one channel into a compact single-channel matrix.
///
/// # Errors
///
/// Returns an error if the channel is invalid, output sizing overflows, or allocation fails.
pub fn extract_channel(
    source: &ChannelMatrix,
    channel: u16,
) -> Result<ChannelMatrix, ChannelError> {
    validate_source_channel(source, 0, channel)?;
    let length = checked_byte_length(source.rows, source.columns, 1, source.element_byte_width)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| ChannelError::AllocationFailed)?;
    copy_channel_to_vec(source, channel, &mut bytes);
    ChannelMatrix::new(
        bytes,
        source.rows,
        source.columns,
        1,
        source.element_byte_width,
    )
}

/// Returns a destination copy with a single-channel source inserted at `destination_channel`.
///
/// # Errors
///
/// Returns an error if the source is not single-channel, layouts are incompatible, or the
/// destination channel is invalid.
pub fn insert_channel(
    source: &ChannelMatrix,
    destination: &ChannelMatrix,
    destination_channel: u16,
) -> Result<ChannelMatrix, ChannelError> {
    if source.channels != 1 {
        return Err(ChannelError::SourceMustHaveOneChannel);
    }
    ensure_pair_compatible(source, destination)?;
    validate_destination_channel(destination, 0, destination_channel)?;
    let mut output = destination.clone();
    transfer_channel(source, 0, &mut output, destination_channel);
    Ok(output)
}

/// Applies channel mappings across one or more sources and destinations.
///
/// All inputs and mappings are checked before any destination is mutated.
///
/// # Errors
///
/// Returns an error for empty slices, incompatible layouts, or any invalid matrix or channel index.
pub fn mix_channels(
    sources: &[ChannelMatrix],
    destinations: &mut [ChannelMatrix],
    mappings: &[ChannelMapping],
) -> Result<(), ChannelError> {
    let first = sources.first().ok_or(ChannelError::EmptySources)?;
    if destinations.is_empty() {
        return Err(ChannelError::EmptyDestinations);
    }
    ensure_compatible(sources, first)?;
    ensure_compatible(destinations, first)?;

    for mapping in mappings {
        let source =
            sources
                .get(mapping.source_index)
                .ok_or(ChannelError::SourceIndexOutOfBounds {
                    index: mapping.source_index,
                })?;
        let destination = destinations.get(mapping.destination_index).ok_or(
            ChannelError::DestinationIndexOutOfBounds {
                index: mapping.destination_index,
            },
        )?;
        validate_source_channel(source, mapping.source_index, mapping.source_channel)?;
        validate_destination_channel(
            destination,
            mapping.destination_index,
            mapping.destination_channel,
        )?;
    }

    for mapping in mappings {
        transfer_channel(
            &sources[mapping.source_index],
            mapping.source_channel,
            &mut destinations[mapping.destination_index],
            mapping.destination_channel,
        );
    }
    Ok(())
}

fn checked_byte_length(
    rows: u32,
    columns: u32,
    channels: u16,
    element_byte_width: u8,
) -> Result<usize, ChannelError> {
    if rows == 0 || columns == 0 {
        return Err(ChannelError::EmptyDimensions);
    }
    if channels == 0 {
        return Err(ChannelError::EmptyChannels);
    }
    if element_byte_width == 0 {
        return Err(ChannelError::EmptyElement);
    }
    usize::try_from(rows)
        .ok()
        .and_then(|value| value.checked_mul(columns as usize))
        .and_then(|value| value.checked_mul(usize::from(channels)))
        .and_then(|value| value.checked_mul(usize::from(element_byte_width)))
        .ok_or(ChannelError::BufferSizeOverflow)
}

fn zeroed_bytes(length: usize) -> Result<Vec<u8>, ChannelError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| ChannelError::AllocationFailed)?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn ensure_compatible(
    matrices: &[ChannelMatrix],
    expected: &ChannelMatrix,
) -> Result<(), ChannelError> {
    for matrix in matrices {
        ensure_pair_compatible(matrix, expected)?;
    }
    Ok(())
}

fn ensure_pair_compatible(left: &ChannelMatrix, right: &ChannelMatrix) -> Result<(), ChannelError> {
    if left.rows != right.rows || left.columns != right.columns {
        return Err(ChannelError::IncompatibleShape);
    }
    if left.element_byte_width != right.element_byte_width {
        return Err(ChannelError::IncompatibleElementByteWidth);
    }
    Ok(())
}

fn validate_source_channel(
    matrix: &ChannelMatrix,
    index: usize,
    channel: u16,
) -> Result<(), ChannelError> {
    if channel >= matrix.channels {
        return Err(ChannelError::SourceChannelOutOfBounds { index, channel });
    }
    Ok(())
}

fn validate_destination_channel(
    matrix: &ChannelMatrix,
    index: usize,
    channel: u16,
) -> Result<(), ChannelError> {
    if channel >= matrix.channels {
        return Err(ChannelError::DestinationChannelOutOfBounds { index, channel });
    }
    Ok(())
}

fn copy_channel_to_vec(source: &ChannelMatrix, channel: u16, output: &mut Vec<u8>) {
    let width = usize::from(source.element_byte_width);
    for pixel in 0..source.pixel_count() {
        let start = source.scalar_offset(pixel, channel);
        output.extend_from_slice(&source.bytes[start..start + width]);
    }
}

fn transfer_channel(
    source: &ChannelMatrix,
    source_channel: u16,
    destination: &mut ChannelMatrix,
    destination_channel: u16,
) {
    let width = usize::from(source.element_byte_width);
    for pixel in 0..source.pixel_count() {
        let source_start = source.scalar_offset(pixel, source_channel);
        let destination_start = destination.scalar_offset(pixel, destination_channel);
        destination.bytes[destination_start..destination_start + width]
            .copy_from_slice(&source.bytes[source_start..source_start + width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(bytes: &[u8], rows: u32, columns: u32, channels: u16) -> ChannelMatrix {
        ChannelMatrix::new(bytes.to_vec(), rows, columns, channels, 1).unwrap()
    }

    #[test]
    fn split_and_merge_u8_multichannel() {
        let source = matrix(&[1, 10, 100, 2, 20, 200], 1, 2, 3);
        let planes = split(&source).unwrap();
        assert_eq!(
            planes.iter().map(ChannelMatrix::bytes).collect::<Vec<_>>(),
            vec![&[1, 2], &[10, 20], &[100, 200]]
        );
        assert_eq!(merge(&planes).unwrap(), source);
    }

    #[test]
    fn multibyte_values_are_copied_without_interpretation() {
        let source = ChannelMatrix::new(
            vec![0x01, 0x02, 0xa1, 0xa2, 0x03, 0x04, 0xa3, 0xa4],
            1,
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(
            extract_channel(&source, 1).unwrap().bytes(),
            &[0xa1, 0xa2, 0xa3, 0xa4]
        );
        assert_eq!(merge(&split(&source).unwrap()).unwrap(), source);
    }

    #[test]
    fn insert_replaces_only_selected_channel() {
        let source = matrix(&[7, 8], 1, 2, 1);
        let destination = matrix(&[1, 2, 3, 4, 5, 6], 1, 2, 3);
        assert_eq!(
            insert_channel(&source, &destination, 1).unwrap().bytes(),
            &[1, 7, 3, 4, 8, 6]
        );
        assert_eq!(destination.bytes(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn mix_routes_multiple_matrices_and_preserves_unmapped_channels() {
        let sources = vec![matrix(&[1, 2, 3, 4], 1, 2, 2), matrix(&[9, 8], 1, 2, 1)];
        let mut destinations = vec![
            matrix(&[0, 10, 0, 20], 1, 2, 2),
            matrix(&[30, 0, 40, 0], 1, 2, 2),
        ];
        let mappings = [
            ChannelMapping {
                source_index: 0,
                source_channel: 1,
                destination_index: 0,
                destination_channel: 0,
            },
            ChannelMapping {
                source_index: 1,
                source_channel: 0,
                destination_index: 1,
                destination_channel: 1,
            },
        ];
        mix_channels(&sources, &mut destinations, &mappings).unwrap();
        assert_eq!(destinations[0].bytes(), &[2, 10, 4, 20]);
        assert_eq!(destinations[1].bytes(), &[30, 9, 40, 8]);
    }

    #[test]
    fn invalid_mapping_is_transactional() {
        let sources = vec![matrix(&[1], 1, 1, 1)];
        let original = matrix(&[8], 1, 1, 1);
        let mut destinations = vec![original.clone()];
        let error = mix_channels(
            &sources,
            &mut destinations,
            &[ChannelMapping {
                source_index: 0,
                source_channel: 1,
                destination_index: 0,
                destination_channel: 0,
            }],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ChannelError::SourceChannelOutOfBounds {
                index: 0,
                channel: 1
            }
        );
        assert_eq!(destinations[0], original);
    }

    #[test]
    fn rejects_empty_invalid_and_incompatible_inputs() {
        assert_eq!(
            ChannelMatrix::new(vec![], 0, 1, 1, 1),
            Err(ChannelError::EmptyDimensions)
        );
        assert_eq!(
            ChannelMatrix::new(vec![], 1, 1, 0, 1),
            Err(ChannelError::EmptyChannels)
        );
        assert_eq!(
            ChannelMatrix::new(vec![], 1, 1, 1, 0),
            Err(ChannelError::EmptyElement)
        );
        assert_eq!(
            ChannelMatrix::new(vec![], 1, 1, 1, 1),
            Err(ChannelError::IncorrectByteLength {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(merge(&[]), Err(ChannelError::EmptySources));

        let one = matrix(&[1], 1, 1, 1);
        let different_shape = matrix(&[1, 2], 1, 2, 1);
        assert_eq!(
            merge(&[one.clone(), different_shape]),
            Err(ChannelError::IncompatibleShape)
        );
        let different_width = ChannelMatrix::new(vec![1, 2], 1, 1, 1, 2).unwrap();
        assert_eq!(
            insert_channel(&one, &different_width, 0),
            Err(ChannelError::IncompatibleElementByteWidth)
        );
        assert_eq!(
            extract_channel(&one, 2),
            Err(ChannelError::SourceChannelOutOfBounds {
                index: 0,
                channel: 2
            })
        );
    }

    #[test]
    fn rejects_overflow_before_reading_the_buffer() {
        let result = ChannelMatrix::new(vec![], u32::MAX, u32::MAX, u16::MAX, u8::MAX);
        assert_eq!(result, Err(ChannelError::BufferSizeOverflow));
    }
}
