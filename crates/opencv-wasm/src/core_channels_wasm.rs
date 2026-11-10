//! Matrix-aware WebAssembly adapters for depth-agnostic channel kernels.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    core_channels::{
        ChannelError, ChannelMapping, ChannelMatrix, extract_channel, insert_channel, merge,
        mix_channels, split,
    },
    mat::{Mat, MatDepth, MatError},
    mat_vector::MatVector,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChannelWasmError {
    ElementWidthOverflow(usize),
    IncompatibleDepth {
        expected: MatDepth,
        actual: MatDepth,
    },
    Kernel(ChannelError),
    Matrix(MatError),
    OddMappingLength(usize),
}

impl fmt::Display for ChannelWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementWidthOverflow(width) => {
                write!(
                    formatter,
                    "matrix scalar width {width} exceeds the channel-kernel limit"
                )
            }
            Self::IncompatibleDepth { expected, actual } => {
                write!(
                    formatter,
                    "matrix depth {actual:?} does not match {expected:?}"
                )
            }
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
            Self::OddMappingLength(length) => write!(
                formatter,
                "mixChannels mapping has {length} values; expected source/destination channel pairs"
            ),
        }
    }
}

impl Error for ChannelWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::ElementWidthOverflow(_)
            | Self::IncompatibleDepth { .. }
            | Self::OddMappingLength(_) => None,
        }
    }
}

impl From<ChannelError> for ChannelWasmError {
    fn from(error: ChannelError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for ChannelWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Splits an interleaved matrix into an array of single-channel matrices.
///
/// # Errors
///
/// Returns an error when output sizing overflows or allocation fails.
#[wasm_bindgen(js_name = matSplit)]
pub fn mat_split(source: &Mat) -> Result<Box<[Mat]>, JsError> {
    split_mat(source)
        .map(Vec::into_boxed_slice)
        .map_err(JsError::from)
}

/// Merges two compatible single-channel matrices into an interleaved matrix.
///
/// # Errors
///
/// Returns an error for incompatible layouts or allocation failure.
#[wasm_bindgen(js_name = matMerge)]
pub fn mat_merge(first: &Mat, second: &Mat) -> Result<Mat, JsError> {
    merge_mats(&[first, second]).map_err(JsError::from)
}

/// Merges three compatible single-channel matrices into an interleaved matrix.
///
/// # Errors
///
/// Returns an error for incompatible layouts or allocation failure.
#[wasm_bindgen(js_name = matMerge3)]
pub fn mat_merge_three(first: &Mat, second: &Mat, third: &Mat) -> Result<Mat, JsError> {
    merge_mats(&[first, second, third]).map_err(JsError::from)
}

/// Merges four compatible single-channel matrices into an interleaved matrix.
///
/// # Errors
///
/// Returns an error for incompatible layouts or allocation failure.
#[wasm_bindgen(js_name = matMerge4)]
pub fn mat_merge_four(
    first: &Mat,
    second: &Mat,
    third: &Mat,
    fourth: &Mat,
) -> Result<Mat, JsError> {
    merge_mats(&[first, second, third, fourth]).map_err(JsError::from)
}

/// Extracts one zero-based channel into a new single-channel matrix.
///
/// # Errors
///
/// Returns an error when `channel` is outside the source channel range or allocation fails.
#[wasm_bindgen(js_name = matExtractChannel)]
pub fn mat_extract_channel(source: &Mat, channel: u16) -> Result<Mat, JsError> {
    extract_channel_mat(source, channel).map_err(JsError::from)
}

/// Inserts a single-channel source into one channel of an existing destination matrix.
///
/// Writes through destination regions update their parent and overlapping regions.
///
/// # Errors
///
/// Returns an error for incompatible shapes or depths, a multi-channel source, or an invalid
/// destination channel. Validation completes before destination bytes are changed.
#[wasm_bindgen(js_name = matInsertChannel)]
pub fn mat_insert_channel(
    source: &Mat,
    destination: &Mat,
    destination_channel: u16,
) -> Result<(), JsError> {
    insert_channel_mat(source, destination, destination_channel).map_err(JsError::from)
}

/// Routes selected channels from one matrix into an existing destination matrix.
///
/// `from_to` contains flattened source/destination channel pairs. The adapter implements the
/// single-source, single-destination form and preserves every destination channel not named by a
/// mapping. Source and destination may alias because both are snapshotted before the write.
///
/// # Errors
///
/// Returns an error for an odd mapping length, incompatible matrix metadata, or an out-of-range
/// channel. Validation completes before destination bytes change.
#[wasm_bindgen(js_name = matMixChannels)]
pub fn mat_mix_channels(source: &Mat, destination: &Mat, from_to: &[u16]) -> Result<(), JsError> {
    mix_channels_mat(source, destination, from_to).map_err(JsError::from)
}

fn channel_matrix(source: &Mat) -> Result<ChannelMatrix, ChannelWasmError> {
    let width = u8::try_from(source.depth().byte_width())
        .map_err(|_| ChannelWasmError::ElementWidthOverflow(source.depth().byte_width()))?;
    ChannelMatrix::new(
        source.compact_bytes(),
        source.rows(),
        source.columns(),
        source.channels(),
        width,
    )
    .map_err(ChannelWasmError::from)
}

fn mat_from_channel(source: &ChannelMatrix, depth: MatDepth) -> Result<Mat, ChannelWasmError> {
    debug_assert_eq!(usize::from(source.element_byte_width()), depth.byte_width());
    let rows = source.rows();
    let columns = source.columns();
    let channels = source.channels();
    Mat::from_owned_bytes(source.bytes().to_vec(), rows, columns, channels, depth)
        .map_err(ChannelWasmError::from)
}

fn split_mat(source: &Mat) -> Result<Vec<Mat>, ChannelWasmError> {
    split(&channel_matrix(source)?)?
        .into_iter()
        .map(|plane| mat_from_channel(&plane, source.depth()))
        .collect()
}

fn merge_mats(planes: &[&Mat]) -> Result<Mat, ChannelWasmError> {
    let first = planes.first().ok_or(ChannelError::EmptySources)?;
    for plane in planes {
        if plane.depth() != first.depth() {
            return Err(ChannelWasmError::IncompatibleDepth {
                expected: first.depth(),
                actual: plane.depth(),
            });
        }
    }
    let kernels = planes
        .iter()
        .map(|matrix| channel_matrix(matrix))
        .collect::<Result<Vec<_>, _>>()?;
    mat_from_channel(&merge(&kernels)?, first.depth())
}

fn extract_channel_mat(source: &Mat, channel: u16) -> Result<Mat, ChannelWasmError> {
    let output = extract_channel(&channel_matrix(source)?, channel)?;
    mat_from_channel(&output, source.depth())
}

fn insert_channel_mat(
    source: &Mat,
    destination: &Mat,
    destination_channel: u16,
) -> Result<(), ChannelWasmError> {
    if source.depth() != destination.depth() {
        return Err(ChannelWasmError::IncompatibleDepth {
            expected: destination.depth(),
            actual: source.depth(),
        });
    }
    let output = insert_channel(
        &channel_matrix(source)?,
        &channel_matrix(destination)?,
        destination_channel,
    )?;
    destination.write_compact_bytes(output.bytes())?;
    Ok(())
}

fn mix_channels_mat(
    source: &Mat,
    destination: &Mat,
    from_to: &[u16],
) -> Result<(), ChannelWasmError> {
    if from_to.len() % 2 != 0 {
        return Err(ChannelWasmError::OddMappingLength(from_to.len()));
    }
    if source.depth() != destination.depth() {
        return Err(ChannelWasmError::IncompatibleDepth {
            expected: destination.depth(),
            actual: source.depth(),
        });
    }
    let source_kernel = channel_matrix(source)?;
    let mut destinations = [channel_matrix(destination)?];
    let mappings = from_to
        .chunks_exact(2)
        .map(|pair| ChannelMapping {
            source_index: 0,
            source_channel: pair[0],
            destination_index: 0,
            destination_channel: pair[1],
        })
        .collect::<Vec<_>>();
    mix_channels(&[source_kernel], &mut destinations, &mappings)?;
    destination.write_compact_bytes(destinations[0].bytes())?;
    Ok(())
}

/// Splits an interleaved matrix into single-channel planes stored in `destination`.
///
/// The vector contents are replaced, matching upstream `cv.split(source, vector)` semantics.
///
/// # Errors
///
/// Returns an error when output sizing overflows or allocation fails.
#[wasm_bindgen(js_name = matSplitVecInto)]
pub fn mat_split_vec_into(source: &Mat, destination: &MatVector) -> Result<(), JsError> {
    let planes = split_mat(source).map_err(JsError::from)?;
    destination.replace(planes);
    Ok(())
}

/// Merges every matrix in `sources` into one interleaved destination.
///
/// Sources may carry different channel counts; each source contributes its channel run
/// contiguously per pixel. The destination is reallocated to the merged layout.
///
/// # Errors
///
/// Returns an error for an empty vector, incompatible layouts or depths, or allocation failure.
#[wasm_bindgen(js_name = matMergeVecInto)]
pub fn mat_merge_vec_into(sources: &MatVector, destination: &Mat) -> Result<(), JsError> {
    let mats = sources.mats();
    let merged = merge_mats(&mats.iter().collect::<Vec<_>>()).map_err(JsError::from)?;
    destination
        .write_output(
            merged.compact_bytes(),
            merged.rows(),
            merged.columns(),
            merged.channels(),
            merged.depth(),
        )
        .map_err(ChannelWasmError::from)
        .map_err(JsError::from)
}

/// Routes channels between matrix vectors using flattened channel index pairs.
///
/// Each `from_to` pair maps a source channel index (counted across the source vector) to a
/// destination channel index (counted across the destination vector). Destination matrices must
/// already carry the final layout; channels not named by a mapping are preserved. Empty
/// destination matrices keep their empty layout.
///
/// # Errors
///
/// Returns an error for an odd mapping length, empty sources or destinations, incompatible
/// layouts or depths, or an out-of-range channel index. Validation completes before any
/// destination byte changes.
#[wasm_bindgen(js_name = matMixChannelsVec)]
pub fn mat_mix_channels_vec(
    sources: &MatVector,
    destinations: &MatVector,
    from_to: &[u16],
) -> Result<(), JsError> {
    mix_channels_vecs(sources, destinations, from_to).map_err(JsError::from)
}

fn mix_channels_vecs(
    sources: &MatVector,
    destinations: &MatVector,
    from_to: &[u16],
) -> Result<(), ChannelWasmError> {
    if from_to.len() % 2 != 0 {
        return Err(ChannelWasmError::OddMappingLength(from_to.len()));
    }
    let source_mats = sources.mats();
    let destination_mats = destinations.mats();
    let first = source_mats
        .first()
        .ok_or(ChannelError::EmptySources)?
        .clone();
    if destination_mats.is_empty() {
        return Err(ChannelError::EmptyDestinations.into());
    }
    for source in &source_mats {
        if source.depth() != first.depth() {
            return Err(ChannelWasmError::IncompatibleDepth {
                expected: first.depth(),
                actual: source.depth(),
            });
        }
        if source.rows() != first.rows() || source.columns() != first.columns() {
            return Err(ChannelWasmError::Kernel(ChannelError::IncompatibleShape));
        }
    }
    for destination in &destination_mats {
        if destination.depth() != first.depth() {
            return Err(ChannelWasmError::IncompatibleDepth {
                expected: first.depth(),
                actual: destination.depth(),
            });
        }
        if (destination.rows() != 0 || destination.columns() != 0)
            && (destination.rows() != first.rows() || destination.columns() != first.columns())
        {
            return Err(ChannelWasmError::Kernel(ChannelError::IncompatibleShape));
        }
    }

    let source_kernels = source_mats
        .iter()
        .map(channel_matrix)
        .collect::<Result<Vec<_>, _>>()?;
    let mut destination_kernels = Vec::new();
    for destination in &destination_mats {
        if destination.rows() == 0 && destination.columns() == 0 {
            // An empty destination keeps its layout but still occupies channel slots; give the
            // kernel a scratch buffer with the shared shape so mappings resolve.
            let scratch = usize::try_from(first.rows())
                .ok()
                .and_then(|rows| rows.checked_mul(first.columns() as usize))
                .and_then(|pixels| pixels.checked_mul(usize::from(destination.channels())))
                .and_then(|elements| {
                    elements.checked_mul(usize::from(source_kernels[0].element_byte_width()))
                })
                .ok_or(ChannelError::BufferSizeOverflow)?;
            destination_kernels.push(ChannelMatrix::new(
                vec![0; scratch],
                first.rows(),
                first.columns(),
                destination.channels(),
                source_kernels[0].element_byte_width(),
            )?);
        } else {
            destination_kernels.push(channel_matrix(destination)?);
        }
    }

    let offsets = |mats: &[Mat]| -> Vec<u16> {
        let mut result = Vec::with_capacity(mats.len());
        let mut offset = 0_u16;
        for mat in mats {
            result.push(offset);
            offset = offset.saturating_add(mat.channels());
        }
        result
    };
    let resolve = |flat: u16, mats: &[Mat], offsets: &[u16]| -> Option<(usize, u16)> {
        for (index, (mat, offset)) in mats.iter().zip(offsets).enumerate() {
            if flat < *offset {
                break;
            }
            if flat - offset < mat.channels() {
                return Some((index, flat - offset));
            }
        }
        None
    };
    let source_offsets = offsets(&source_mats);
    let destination_offsets = offsets(&destination_mats);
    let mut mappings = Vec::with_capacity(from_to.len() / 2);
    for pair in from_to.chunks_exact(2) {
        let (source_index, source_channel) = resolve(pair[0], &source_mats, &source_offsets)
            .ok_or(ChannelError::SourceChannelOutOfBounds {
                index: usize::MAX,
                channel: pair[0],
            })?;
        let (destination_index, destination_channel) =
            resolve(pair[1], &destination_mats, &destination_offsets).ok_or(
                ChannelError::DestinationChannelOutOfBounds {
                    index: usize::MAX,
                    channel: pair[1],
                },
            )?;
        mappings.push(ChannelMapping {
            source_index,
            source_channel,
            destination_index,
            destination_channel,
        });
    }

    mix_channels(&source_kernels, &mut destination_kernels, &mappings)?;
    for (mat, kernel) in destination_mats.iter().zip(destination_kernels.iter()) {
        if mat.rows() == 0 && mat.columns() == 0 {
            continue;
        }
        mat.write_compact_bytes(kernel.bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(bytes: Vec<u8>, rows: u32, columns: u32, channels: u16, depth: MatDepth) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, channels, depth).expect("valid test matrix")
    }

    #[test]
    fn split_and_merge_compact_a_strided_region() {
        let parent = matrix((1..=18).collect(), 3, 3, 2, MatDepth::U8);
        let region = parent.roi(0, 1, 3, 2).expect("valid strided region");
        assert!(!region.is_continuous());

        let planes = split_mat(&region).expect("split region");
        assert_eq!(planes[0].to_u8_array(), [3, 5, 9, 11, 15, 17]);
        assert_eq!(planes[1].to_u8_array(), [4, 6, 10, 12, 16, 18]);

        let plane_refs = planes.iter().collect::<Vec<_>>();
        let merged = merge_mats(&plane_refs).expect("merge planes");
        assert_eq!(merged.to_u8_array(), region.to_u8_array());
        assert_eq!(merged.depth(), MatDepth::U8);
        assert_eq!(merged.channels(), 2);
    }

    #[test]
    fn every_scalar_depth_round_trips_without_reinterpreting_bytes() {
        for depth in [
            MatDepth::U8,
            MatDepth::I8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::I32,
            MatDepth::F32,
            MatDepth::F64,
        ] {
            let bytes = (0..depth.byte_width() * 4)
                .map(|index| u8::try_from(index + 1).expect("small test value"))
                .collect::<Vec<_>>();
            let source = matrix(bytes.clone(), 1, 2, 2, depth);
            let planes = split_mat(&source).expect("split");
            let plane_refs = planes.iter().collect::<Vec<_>>();
            let output = merge_mats(&plane_refs).expect("merge");
            assert_eq!(output.depth(), depth);
            assert_eq!(output.to_u8_array(), bytes);
        }
    }

    #[test]
    fn extract_preserves_depth_and_selected_channel() {
        let source = matrix(
            [1_i16, 10, 2, 20]
                .into_iter()
                .flat_map(i16::to_ne_bytes)
                .collect(),
            1,
            2,
            2,
            MatDepth::I16,
        );
        let output = extract_channel_mat(&source, 1).expect("extract channel");
        assert_eq!(output.depth(), MatDepth::I16);
        assert_eq!(output.channels(), 1);
        assert_eq!(output.to_i16_array().expect("I16 output"), [10, 20]);
    }

    #[test]
    fn insert_writes_through_a_strided_destination_region() {
        let parent = matrix(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            2,
            3,
            2,
            MatDepth::U8,
        );
        let region = parent.roi(0, 1, 2, 2).expect("valid strided region");
        let source = matrix(vec![20, 30, 40, 50], 2, 2, 1, MatDepth::U8);

        insert_channel_mat(&source, &region, 1).expect("insert channel");

        assert_eq!(region.to_u8_array(), [3, 20, 5, 30, 9, 40, 11, 50]);
        assert_eq!(
            parent.to_u8_array(),
            [1, 2, 3, 20, 5, 30, 7, 8, 9, 40, 11, 50]
        );
    }

    #[test]
    fn incompatible_depth_is_rejected_before_mutation() {
        let source = matrix(vec![1, 0], 1, 1, 1, MatDepth::U16);
        let destination = matrix(vec![8, 9], 1, 1, 1, MatDepth::I16);

        assert!(matches!(
            insert_channel_mat(&source, &destination, 0),
            Err(ChannelWasmError::IncompatibleDepth {
                expected: MatDepth::I16,
                actual: MatDepth::U16
            })
        ));
        assert_eq!(destination.to_u8_array(), [8, 9]);
    }

    #[test]
    fn merge_rejects_same_width_different_depths() {
        let unsigned = matrix(vec![1, 0], 1, 1, 1, MatDepth::U16);
        let signed = matrix(vec![2, 0], 1, 1, 1, MatDepth::I16);
        assert!(matches!(
            merge_mats(&[&unsigned, &signed]),
            Err(ChannelWasmError::IncompatibleDepth { .. })
        ));
    }

    #[test]
    fn mix_channels_routes_selected_lanes_and_preserves_others() {
        let source = matrix(vec![1, 10, 100, 2, 20, 200], 1, 2, 3, MatDepth::U8);
        let destination = matrix(vec![7, 8, 9, 70, 80, 90], 1, 2, 3, MatDepth::U8);

        mix_channels_mat(&source, &destination, &[2, 0, 0, 2]).expect("valid routing");

        assert_eq!(destination.to_u8_array(), [100, 8, 1, 200, 80, 2]);
    }

    #[test]
    fn mix_channels_supports_aliasing_and_rejects_invalid_maps_atomically() {
        let matrix = matrix(vec![1, 10, 100, 2, 20, 200], 1, 2, 3, MatDepth::U8);

        mix_channels_mat(&matrix, &matrix, &[0, 2, 2, 0]).expect("aliased routing");
        assert_eq!(matrix.to_u8_array(), [100, 10, 1, 200, 20, 2]);

        let before = matrix.to_u8_array();
        assert!(matches!(
            mix_channels_mat(&matrix, &matrix, &[0, 1, 2]),
            Err(ChannelWasmError::OddMappingLength(3))
        ));
        assert_eq!(matrix.to_u8_array(), before);
    }
}

#[cfg(test)]
mod vec_tests {
    use super::*;
    use crate::mat_vector::mat_vector_new;

    #[test]
    fn mix_channels_vec_routes_flat_pairs() {
        let sources = mat_vector_new();
        sources
            .push_back(&Mat::from_owned_bytes((0..12).collect(), 2, 2, 3, MatDepth::U8).unwrap());
        sources.push_back(
            &Mat::from_owned_bytes(vec![40, 41, 42, 43], 2, 2, 1, MatDepth::U8).unwrap(),
        );
        let destinations = mat_vector_new();
        destinations.push_back(&Mat::from_owned_bytes(vec![0; 4], 2, 2, 1, MatDepth::U8).unwrap());
        destinations.push_back(&Mat::from_owned_bytes(vec![0; 8], 2, 2, 2, MatDepth::U8).unwrap());
        mix_channels_vecs(&sources, &destinations, &[2, 0, 3, 1, 0, 2]).unwrap();
        assert_eq!(
            destinations.get(0).unwrap().compact_bytes(),
            vec![2, 5, 8, 11]
        );
        assert_eq!(
            destinations.get(1).unwrap().compact_bytes(),
            vec![40, 0, 41, 3, 42, 6, 43, 9]
        );
    }
}
