//! Matrix-aware WebAssembly adapters for depth-agnostic layout kernels.

use std::{error::Error, fmt};

use wasm_bindgen::prelude::*;

use crate::{
    core_layout::{
        ByteMatrix, LayoutError, MatLayout, OwnedByteMatrix, flip_bytes, hconcat_bytes,
        repeat_bytes, rotate_bytes, transpose_bytes, vconcat_bytes,
    },
    mat::{Mat, MatDepth, MatError},
    mat_vector::MatVector,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum LayoutWasmError {
    IncompatibleSourceDepth {
        index: usize,
        expected: MatDepth,
        actual: MatDepth,
    },
    ElementWidthOverflow(usize),
    InPlaceRepeat,
    Kernel(LayoutError),
    Matrix(MatError),
}

impl fmt::Display for LayoutWasmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleSourceDepth {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "source matrix {index} has depth {actual:?}; expected {expected:?}"
            ),
            Self::ElementWidthOverflow(width) => {
                write!(
                    formatter,
                    "matrix scalar width {width} exceeds the layout limit"
                )
            }
            Self::InPlaceRepeat => {
                formatter.write_str("repeat source and destination must be different matrices")
            }
            Self::Kernel(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for LayoutWasmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Matrix(error) => Some(error),
            Self::ElementWidthOverflow(_)
            | Self::InPlaceRepeat
            | Self::IncompatibleSourceDepth { .. } => None,
        }
    }
}

impl From<LayoutError> for LayoutWasmError {
    fn from(error: LayoutError) -> Self {
        Self::Kernel(error)
    }
}

impl From<MatError> for LayoutWasmError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Flips a matrix vertically, horizontally, or across both axes.
///
/// The accepted codes match `OpenCV`: `0` flips rows, positive values flip columns, and negative
/// values flip both.
///
/// # Errors
///
/// Returns an error when the output cannot be allocated.
#[wasm_bindgen(js_name = matFlip)]
pub fn mat_flip(source: &Mat, flip_code: i32) -> Result<Mat, JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        return Ok(Mat::empty_output());
    }
    apply_layout(source, |matrix| flip_bytes(matrix, flip_code)).map_err(JsError::from)
}

/// Flips a matrix into a caller-provided destination.
///
/// # Errors
///
/// Returns an error for output allocation failure.
#[wasm_bindgen(js_name = matFlipInto)]
pub fn mat_flip_into(source: &Mat, destination: &Mat, flip_code: i32) -> Result<(), JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        destination.write_empty_output();
        return Ok(());
    }
    if destination
        .try_write_shared_flip(source, flip_code)
        .map_err(JsError::from)?
    {
        return Ok(());
    }
    apply_layout_into(source, destination, |matrix| flip_bytes(matrix, flip_code))
        .map_err(JsError::from)
}

/// Transposes a matrix while preserving its scalar depth and interleaved channels.
///
/// # Errors
///
/// Returns an error when the output shape or allocation exceeds the WASM matrix limit.
#[wasm_bindgen(js_name = matTranspose)]
pub fn mat_transpose(source: &Mat) -> Result<Mat, JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        return Ok(crate::mat::mat_empty());
    }
    apply_layout(source, transpose_bytes).map_err(JsError::from)
}

/// Transposes a matrix into a caller-provided destination.
///
/// # Errors
///
/// Returns an error when the output cannot be allocated.
#[wasm_bindgen(js_name = matTransposeInto)]
pub fn mat_transpose_into(source: &Mat, destination: &Mat) -> Result<(), JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        return destination
            .write_output(Vec::new(), 0, 0, 1, MatDepth::U8)
            .map_err(JsError::from);
    }
    if destination
        .try_write_shared_transpose(source)
        .map_err(JsError::from)?
    {
        return Ok(());
    }
    apply_layout_into(source, destination, transpose_bytes).map_err(JsError::from)
}

/// Rotates a matrix by 90 degrees clockwise, 180 degrees, or 90 degrees counterclockwise.
///
/// The accepted codes match `OpenCV`: `0`, `1`, and `2`, respectively.
///
/// # Errors
///
/// Returns an error when `rotate_code` is outside `0..=2`, or when the output cannot be
/// allocated.
#[wasm_bindgen(js_name = matRotate)]
pub fn mat_rotate(source: &Mat, rotate_code: i32) -> Result<Mat, JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        validate_rotate_code(rotate_code).map_err(JsError::from)?;
        return Ok(Mat::empty_output());
    }
    apply_layout(source, |matrix| rotate_bytes(matrix, rotate_code)).map_err(JsError::from)
}

/// Rotates a matrix into a caller-provided destination.
///
/// # Errors
///
/// Returns an error for an invalid rotation code or output allocation failure.
#[wasm_bindgen(js_name = matRotateInto)]
pub fn mat_rotate_into(source: &Mat, destination: &Mat, rotate_code: i32) -> Result<(), JsError> {
    if source.rows() == 0 && source.columns() == 0 {
        validate_rotate_code(rotate_code).map_err(JsError::from)?;
        if destination.rows() != 0 || destination.columns() != 0 {
            destination.write_empty_output();
        }
        return Ok(());
    }
    if destination
        .try_write_shared_rotate(source, rotate_code)
        .map_err(JsError::from)?
    {
        return Ok(());
    }
    apply_layout_into(source, destination, |matrix| {
        rotate_bytes(matrix, rotate_code)
    })
    .map_err(JsError::from)
}

fn validate_rotate_code(rotate_code: i32) -> Result<(), LayoutWasmError> {
    match rotate_code {
        0..=2 => Ok(()),
        code => Err(LayoutError::InvalidRotateCode(code).into()),
    }
}

/// Tiles a matrix by positive row and column repeat counts.
///
/// # Errors
///
/// Returns an error when either count is not positive or the repeated shape exceeds the WASM
/// matrix limit.
#[wasm_bindgen(js_name = matRepeat)]
pub fn mat_repeat(source: &Mat, row_repeats: i32, column_repeats: i32) -> Result<Mat, JsError> {
    validate_repeat_counts(row_repeats, column_repeats)?;
    if source.rows() == 0 && source.columns() == 0 {
        return Ok(Mat::empty_output());
    }
    apply_layout(source, |matrix| {
        repeat_bytes(matrix, row_repeats, column_repeats)
    })
    .map_err(JsError::from)
}

/// Repeats a matrix into a caller-provided destination.
///
/// # Errors
///
/// Returns an error for invalid repeat counts or output allocation failure.
#[wasm_bindgen(js_name = matRepeatInto)]
pub fn mat_repeat_into(
    source: &Mat,
    destination: &Mat,
    row_repeats: i32,
    column_repeats: i32,
) -> Result<(), JsError> {
    repeat_into_mat(source, destination, row_repeats, column_repeats).map_err(JsError::from)
}

fn repeat_into_mat(
    source: &Mat,
    destination: &Mat,
    row_repeats: i32,
    column_repeats: i32,
) -> Result<(), LayoutWasmError> {
    validate_repeat_counts(row_repeats, column_repeats)?;
    if std::ptr::eq(source, destination) {
        return Err(LayoutWasmError::InPlaceRepeat);
    }
    if source.rows() == 0 && source.columns() == 0 {
        destination.write_empty_output();
        return Ok(());
    }
    let row_repeat_count = u32::try_from(row_repeats).expect("positive repeat count fits u32");
    let column_repeat_count =
        u32::try_from(column_repeats).expect("positive repeat count fits u32");
    if destination.try_write_shared_repeat(source, row_repeat_count, column_repeat_count)? {
        return Ok(());
    }
    apply_layout_into(source, destination, |matrix| {
        repeat_bytes(matrix, row_repeats, column_repeats)
    })
}

fn validate_repeat_counts(row_repeats: i32, column_repeats: i32) -> Result<(), LayoutWasmError> {
    if row_repeats <= 0 || column_repeats <= 0 {
        return Err(LayoutError::InvalidRepeatCount {
            rows: row_repeats,
            columns: column_repeats,
        }
        .into());
    }
    Ok(())
}

/// Horizontally concatenates two matrices with identical rows, channels, and depth.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matHconcat2)]
pub fn mat_hconcat2(first: &Mat, second: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second], hconcat_bytes).map_err(JsError::from)
}

/// Horizontally concatenates three compatible matrices.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matHconcat3)]
pub fn mat_hconcat3(first: &Mat, second: &Mat, third: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second, third], hconcat_bytes).map_err(JsError::from)
}

/// Horizontally concatenates four compatible matrices.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matHconcat4)]
pub fn mat_hconcat4(first: &Mat, second: &Mat, third: &Mat, fourth: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second, third, fourth], hconcat_bytes).map_err(JsError::from)
}

/// Vertically concatenates two matrices with identical columns, channels, and depth.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matVconcat2)]
pub fn mat_vconcat2(first: &Mat, second: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second], vconcat_bytes).map_err(JsError::from)
}

/// Vertically concatenates three compatible matrices.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matVconcat3)]
pub fn mat_vconcat3(first: &Mat, second: &Mat, third: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second, third], vconcat_bytes).map_err(JsError::from)
}

/// Vertically concatenates four compatible matrices.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matVconcat4)]
pub fn mat_vconcat4(first: &Mat, second: &Mat, third: &Mat, fourth: &Mat) -> Result<Mat, JsError> {
    apply_concat(&[first, second, third, fourth], vconcat_bytes).map_err(JsError::from)
}

/// Horizontally concatenates every matrix in `sources` into `destination`.
///
/// All sources must share rows, channels, and depth. An empty vector is a no-op and leaves the
/// destination untouched, matching upstream `cv.hconcat` behavior.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matHconcatVecInto)]
pub fn mat_hconcat_vec_into(sources: &MatVector, destination: &Mat) -> Result<(), JsError> {
    concat_vec_into(sources, destination, hconcat_bytes).map_err(JsError::from)
}

/// Vertically concatenates every matrix in `sources` into `destination`.
///
/// All sources must share columns, channels, and depth. An empty vector is a no-op and leaves the
/// destination untouched, matching upstream `cv.vconcat` behavior.
///
/// # Errors
///
/// Returns an error when source metadata is incompatible or the output cannot be allocated.
#[wasm_bindgen(js_name = matVconcatVecInto)]
pub fn mat_vconcat_vec_into(sources: &MatVector, destination: &Mat) -> Result<(), JsError> {
    concat_vec_into(sources, destination, vconcat_bytes).map_err(JsError::from)
}

fn concat_vec_into(
    sources: &MatVector,
    destination: &Mat,
    kernel: impl FnOnce(&[ByteMatrix<'_>]) -> Result<OwnedByteMatrix, LayoutError>,
) -> Result<(), LayoutWasmError> {
    let mats = sources.mats();
    if mats.is_empty() {
        return Ok(());
    }
    let depth = mats[0].depth();
    for (index, source) in mats.iter().enumerate().skip(1) {
        if source.depth() != depth {
            return Err(LayoutWasmError::IncompatibleSourceDepth {
                index,
                expected: depth,
                actual: source.depth(),
            });
        }
    }
    let element_width = u8::try_from(depth.byte_width())
        .map_err(|_| LayoutWasmError::ElementWidthOverflow(depth.byte_width()))?;
    let snapshots = mats
        .iter()
        .map(|source| source.compact_bytes())
        .collect::<Vec<_>>();
    let layouts = mats
        .iter()
        .map(|source| {
            MatLayout::new(
                source.rows(),
                source.columns(),
                source.channels(),
                element_width,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let matrices = snapshots
        .iter()
        .zip(layouts)
        .map(|(bytes, layout)| ByteMatrix::new(bytes, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let output = kernel(&matrices)?;
    let (bytes, layout) = output.into_parts();
    destination.write_output(
        bytes,
        layout.rows(),
        layout.columns(),
        layout.channels(),
        depth,
    )?;
    Ok(())
}

fn apply_concat(
    sources: &[&Mat],
    kernel: impl FnOnce(&[ByteMatrix<'_>]) -> Result<OwnedByteMatrix, LayoutError>,
) -> Result<Mat, LayoutWasmError> {
    let depth = sources
        .first()
        .expect("fixed-arity concat adapters always provide sources")
        .depth();
    for (index, source) in sources.iter().enumerate().skip(1) {
        if source.depth() != depth {
            return Err(LayoutWasmError::IncompatibleSourceDepth {
                index,
                expected: depth,
                actual: source.depth(),
            });
        }
    }

    let element_width = u8::try_from(depth.byte_width())
        .map_err(|_| LayoutWasmError::ElementWidthOverflow(depth.byte_width()))?;
    let snapshots = sources
        .iter()
        .map(|source| source.compact_bytes())
        .collect::<Vec<_>>();
    let layouts = sources
        .iter()
        .map(|source| {
            MatLayout::new(
                source.rows(),
                source.columns(),
                source.channels(),
                element_width,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let matrices = snapshots
        .iter()
        .zip(layouts)
        .map(|(bytes, layout)| ByteMatrix::new(bytes, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let output = kernel(&matrices)?;
    let (bytes, layout) = output.into_parts();
    Mat::from_owned_bytes(
        bytes,
        layout.rows(),
        layout.columns(),
        layout.channels(),
        depth,
    )
    .map_err(LayoutWasmError::from)
}

fn apply_layout(
    source: &Mat,
    kernel: impl FnOnce(ByteMatrix<'_>) -> Result<OwnedByteMatrix, LayoutError>,
) -> Result<Mat, LayoutWasmError> {
    let bytes = source.compact_bytes();
    let element_width = u8::try_from(source.depth().byte_width())
        .map_err(|_| LayoutWasmError::ElementWidthOverflow(source.depth().byte_width()))?;
    let layout = MatLayout::new(
        source.rows(),
        source.columns(),
        source.channels(),
        element_width,
    )?;
    let output = kernel(ByteMatrix::new(&bytes, layout)?)?;
    let (bytes, layout) = output.into_parts();
    Mat::from_owned_bytes(
        bytes,
        layout.rows(),
        layout.columns(),
        layout.channels(),
        source.depth(),
    )
    .map_err(LayoutWasmError::from)
}

fn apply_layout_into(
    source: &Mat,
    destination: &Mat,
    kernel: impl FnOnce(ByteMatrix<'_>) -> Result<OwnedByteMatrix, LayoutError>,
) -> Result<(), LayoutWasmError> {
    // Snapshot and compute before inspecting or writing the destination. This makes overlapping
    // source and destination regions, including exact in-place operations, deterministic.
    let bytes = source.compact_bytes();
    let element_width = u8::try_from(source.depth().byte_width())
        .map_err(|_| LayoutWasmError::ElementWidthOverflow(source.depth().byte_width()))?;
    let source_layout = MatLayout::new(
        source.rows(),
        source.columns(),
        source.channels(),
        element_width,
    )?;
    let output = kernel(ByteMatrix::new(&bytes, source_layout)?)?;
    let (bytes, layout) = output.into_parts();

    destination.write_output(
        bytes,
        layout.rows(),
        layout.columns(),
        layout.channels(),
        source.depth(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::MatDepth;

    fn u8_matrix(bytes: Vec<u8>, rows: u32, columns: u32, channels: u16) -> Mat {
        Mat::from_owned_bytes(bytes, rows, columns, channels, MatDepth::U8)
            .expect("valid U8 test matrix")
    }

    #[test]
    fn transpose_preserves_metadata_and_compacts_a_strided_region() {
        let parent = u8_matrix((1..=18).collect(), 3, 3, 2);
        let region = parent.roi(0, 1, 3, 2).expect("valid strided region");
        assert!(!region.is_continuous());

        let output = apply_layout(&region, transpose_bytes).expect("transpose region");

        assert_eq!(output.rows(), 2);
        assert_eq!(output.columns(), 3);
        assert_eq!(output.channels(), 2);
        assert_eq!(output.depth(), MatDepth::U8);
        assert!(output.is_continuous());
        assert_eq!(
            output.to_u8_array(),
            [3, 4, 9, 10, 15, 16, 5, 6, 11, 12, 17, 18]
        );
    }

    #[test]
    fn transpose_accepts_the_canonical_empty_matrix() {
        let source = crate::mat::mat_empty();
        let allocated = mat_transpose(&source).expect("transpose empty matrix");
        let destination = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);

        mat_transpose_into(&source, &destination).expect("transpose empty into destination");

        for output in [&allocated, &destination] {
            assert_eq!(
                (output.rows(), output.columns(), output.channels()),
                (0, 0, 1)
            );
            assert_eq!(output.depth(), MatDepth::U8);
            assert!(!output.is_continuous());
            assert!(output.to_u8_array().is_empty());
        }
    }

    #[test]
    fn flip_marks_empty_outputs_continuous_without_changing_fresh_empty_matrices() {
        let source = crate::mat::mat_empty();
        let allocated = mat_flip(&source, 1).expect("flip empty matrix");
        let destination = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);

        assert!(!source.is_continuous());
        mat_flip_into(&source, &destination, 1).expect("flip empty into destination");

        for output in [&allocated, &destination] {
            assert_eq!(
                (output.rows(), output.columns(), output.channels()),
                (0, 0, 1)
            );
            assert_eq!(output.depth(), MatDepth::U8);
            assert!(output.is_continuous());
            assert!(output.to_u8_array().is_empty());
        }
    }

    #[test]
    fn rotate_preserves_a_fresh_empty_destination_but_releases_a_populated_one() {
        let source = crate::mat::mat_empty();
        let allocated = mat_rotate(&source, 0).expect("rotate empty matrix");
        let fresh_destination = crate::mat::mat_empty();
        let populated_destination = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);

        mat_rotate_into(&source, &fresh_destination, 0)
            .expect("rotate empty into fresh empty destination");
        mat_rotate_into(&source, &populated_destination, 0)
            .expect("rotate empty into populated destination");

        assert!(!fresh_destination.is_continuous());
        for output in [&allocated, &populated_destination] {
            assert_eq!(
                (output.rows(), output.columns(), output.channels()),
                (0, 0, 1)
            );
            assert_eq!(output.depth(), MatDepth::U8);
            assert!(output.is_continuous());
            assert!(output.to_u8_array().is_empty());
        }
    }

    #[test]
    fn repeat_marks_positive_count_empty_outputs_continuous() {
        let source = crate::mat::mat_empty();
        let allocated = mat_repeat(&source, 2, 3).expect("repeat empty matrix");
        let destination = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);

        mat_repeat_into(&source, &destination, 2, 3).expect("repeat empty into destination");

        for output in [&allocated, &destination] {
            assert_eq!(
                (output.rows(), output.columns(), output.channels()),
                (0, 0, 1)
            );
            assert_eq!(output.depth(), MatDepth::U8);
            assert!(output.is_continuous());
            assert!(output.to_u8_array().is_empty());
        }
    }

    #[test]
    fn repeat_rejects_the_exact_same_matrix_handle_without_mutation() {
        for (row_repeats, column_repeats) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            let matrix = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);

            assert!(matches!(
                repeat_into_mat(&matrix, &matrix, row_repeats, column_repeats),
                Err(LayoutWasmError::InPlaceRepeat)
            ));
            assert_eq!(matrix.to_u8_array(), [1, 2, 3, 4]);
        }
    }

    #[test]
    fn repeat_matches_opencv_live_traversal_for_overlapping_regions() {
        let horizontal_parent = u8_matrix((1..=6).collect(), 1, 6, 1);
        let horizontal_source = horizontal_parent
            .roi(0, 0, 1, 2)
            .expect("valid horizontal source region");
        let horizontal_destination = horizontal_parent
            .roi(0, 1, 1, 4)
            .expect("valid horizontal destination region");

        repeat_into_mat(&horizontal_source, &horizontal_destination, 1, 2)
            .expect("repeat horizontal overlap");

        assert_eq!(horizontal_parent.to_u8_array(), [1, 1, 1, 1, 1, 6]);
        assert_eq!(horizontal_destination.to_u8_array(), [1, 1, 1, 1]);

        let vertical_parent = u8_matrix((1..=10).collect(), 5, 2, 1);
        let vertical_source = vertical_parent
            .roi(0, 0, 2, 2)
            .expect("valid vertical source region");
        let vertical_destination = vertical_parent
            .roi(1, 0, 4, 2)
            .expect("valid vertical destination region");

        repeat_into_mat(&vertical_source, &vertical_destination, 2, 1)
            .expect("repeat vertical overlap");

        assert_eq!(
            vertical_parent.to_u8_array(),
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        );
        assert_eq!(vertical_destination.to_u8_array(), [1, 2, 1, 2, 1, 2, 1, 2]);

        let combined_parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let combined_source = combined_parent
            .roi(0, 0, 2, 2)
            .expect("valid combined source region");
        let combined_destination = combined_parent
            .roi(1, 1, 4, 4)
            .expect("valid combined destination region");

        repeat_into_mat(&combined_source, &combined_destination, 2, 2)
            .expect("repeat combined overlap");

        assert_eq!(
            combined_destination.to_u8_array(),
            [1, 2, 1, 2, 7, 1, 7, 1, 1, 2, 1, 2, 7, 1, 7, 1]
        );
    }

    #[test]
    fn repeat_shared_traversal_moves_multibyte_multichannel_pixels_atomically() {
        let values = [1.0_f64, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0];
        let parent = Mat::from_owned_bytes(
            values.into_iter().flat_map(f64::to_ne_bytes).collect(),
            1,
            5,
            2,
            MatDepth::F64,
        )
        .expect("valid F64C2 parent");
        let source = parent.roi(0, 0, 1, 2).expect("valid F64C2 source");
        let destination = parent.roi(0, 1, 1, 4).expect("valid F64C2 destination");

        repeat_into_mat(&source, &destination, 1, 2).expect("repeat F64C2 overlap");

        assert_eq!(
            destination.to_f64_array().expect("F64 output"),
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        );
    }

    #[test]
    fn transpose_matches_opencv_for_overlapping_regions_on_one_parent() {
        let parent = u8_matrix((1..=20).collect(), 4, 5, 1);
        let source = parent.roi(0, 0, 2, 3).expect("valid source region");
        let destination = parent
            .roi(0, 1, 3, 2)
            .expect("valid overlapping destination region");

        mat_transpose_into(&source, &destination).expect("transpose overlapping regions");

        assert_eq!(
            parent.to_u8_array(),
            [
                1, 1, 6, 4, 5, 6, 1, 1, 9, 10, 11, 6, 1, 14, 15, 16, 17, 18, 19, 20
            ]
        );
        assert_eq!(source.to_u8_array(), [1, 1, 6, 6, 1, 1]);
        assert_eq!(destination.to_u8_array(), [1, 6, 1, 1, 6, 1]);
    }

    #[test]
    fn horizontal_flip_matches_opencv_for_overlapping_regions_on_one_parent() {
        let parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let source = parent.roi(0, 0, 2, 3).expect("valid source region");
        let destination = parent
            .roi(0, 1, 2, 3)
            .expect("valid overlapping destination region");

        mat_flip_into(&source, &destination, 1).expect("flip overlapping regions");

        assert_eq!(
            parent.to_u8_array(),
            [
                1, 3, 3, 1, 5, 6, 7, 9, 9, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30
            ]
        );
        assert_eq!(source.to_u8_array(), [1, 3, 3, 7, 9, 9]);
        assert_eq!(destination.to_u8_array(), [3, 3, 1, 9, 9, 7]);
    }

    #[test]
    fn vertical_flip_matches_opencv_for_overlapping_regions_on_one_parent() {
        let parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let source = parent.roi(0, 0, 3, 3).expect("valid source region");
        let destination = parent
            .roi(1, 0, 3, 3)
            .expect("valid overlapping destination region");

        mat_flip_into(&source, &destination, 0).expect("flip overlapping regions");

        assert_eq!(
            parent.to_u8_array(),
            [
                1, 2, 3, 4, 5, 6, 13, 14, 15, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 22, 23,
                24, 25, 26, 27, 28, 29, 30
            ]
        );
        assert_eq!(source.to_u8_array(), [1, 2, 3, 13, 14, 15, 13, 14, 15]);
        assert_eq!(destination.to_u8_array(), [13, 14, 15, 13, 14, 15, 1, 2, 3]);
    }

    #[test]
    fn negative_flip_matches_opencv_for_diagonally_overlapping_regions() {
        let parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let source = parent.roi(0, 0, 3, 3).expect("valid source region");
        let destination = parent
            .roi(1, 1, 3, 3)
            .expect("valid diagonally overlapping destination region");

        mat_flip_into(&source, &destination, -1).expect("flip overlapping regions");

        assert_eq!(
            parent.to_u8_array(),
            [
                1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 11, 12, 13, 14, 13, 7, 17, 18, 19, 3, 2, 1, 23,
                24, 25, 26, 27, 28, 29, 30
            ]
        );
        assert_eq!(source.to_u8_array(), [1, 2, 3, 7, 15, 14, 13, 14, 13]);
        assert_eq!(destination.to_u8_array(), [15, 14, 13, 14, 13, 7, 3, 2, 1]);
    }

    #[test]
    fn negative_flip_matches_opencv_for_axis_shifted_overlaps() {
        let horizontal_parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let horizontal_source = horizontal_parent
            .roi(0, 0, 3, 3)
            .expect("valid source region");
        let horizontal_destination = horizontal_parent
            .roi(0, 1, 3, 3)
            .expect("valid horizontally shifted destination");
        mat_flip_into(&horizontal_source, &horizontal_destination, -1)
            .expect("flip horizontal overlap");
        assert_eq!(
            horizontal_destination.to_u8_array(),
            [13, 1, 13, 7, 7, 7, 1, 13, 1]
        );

        let vertical_parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let vertical_source = vertical_parent
            .roi(0, 0, 3, 3)
            .expect("valid source region");
        let vertical_destination = vertical_parent
            .roi(1, 0, 3, 3)
            .expect("valid vertically shifted destination");
        mat_flip_into(&vertical_source, &vertical_destination, -1).expect("flip vertical overlap");
        assert_eq!(
            vertical_destination.to_u8_array(),
            [15, 14, 13, 15, 14, 13, 3, 2, 1]
        );
    }

    #[test]
    fn negative_flip_matches_opencv_for_even_diagonal_overlap() {
        let parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let source = parent.roi(0, 0, 2, 2).expect("valid source region");
        let destination = parent
            .roi(1, 1, 2, 2)
            .expect("valid diagonally overlapping destination region");

        mat_flip_into(&source, &destination, -1).expect("flip overlapping regions");

        assert_eq!(destination.to_u8_array(), [7, 7, 2, 1]);
    }

    #[test]
    fn rotate_matches_opencv_live_composition_for_overlapping_regions() {
        for rotate_code in [0, 2] {
            let parent = u8_matrix((1..=36).collect(), 6, 6, 1);
            let source = parent.roi(0, 0, 2, 3).expect("valid source region");
            let destination = parent
                .roi(0, 1, 3, 2)
                .expect("valid overlapping destination region");

            mat_rotate_into(&source, &destination, rotate_code).expect("rotate overlapping region");

            assert_eq!(destination.to_u8_array(), [7, 1, 1, 1, 1, 7]);
            assert_eq!(
                parent.to_u8_array(),
                [
                    1, 7, 1, 4, 5, 6, 7, 1, 1, 10, 11, 12, 13, 1, 7, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
                ]
            );
        }

        let parent = u8_matrix((1..=36).collect(), 6, 6, 1);
        let source = parent.roi(0, 0, 2, 3).expect("valid source region");
        let destination = parent
            .roi(0, 1, 2, 3)
            .expect("valid overlapping destination region");

        mat_rotate_into(&source, &destination, 1).expect("rotate overlapping region");

        assert_eq!(destination.to_u8_array(), [7, 1, 7, 1, 7, 1]);
        assert_eq!(
            parent.to_u8_array(),
            [
                1, 7, 1, 7, 5, 6, 7, 1, 7, 1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
            ]
        );
    }

    #[test]
    fn flip_keeps_exact_same_views_on_the_snapshot_safe_path() {
        let matrix = u8_matrix((1..=6).collect(), 2, 3, 1);
        mat_flip_into(&matrix, &matrix, 1).expect("flip root matrix in place");
        assert_eq!(matrix.to_u8_array(), [3, 2, 1, 6, 5, 4]);

        let both_axes = u8_matrix((1..=6).collect(), 2, 3, 1);
        mat_flip_into(&both_axes, &both_axes, -1).expect("flip both axes in place");
        assert_eq!(both_axes.to_u8_array(), [6, 5, 4, 3, 2, 1]);

        let parent = u8_matrix((1..=20).collect(), 4, 5, 1);
        let source = parent.roi(1, 1, 2, 3).expect("valid source region");
        let destination = parent
            .roi(1, 1, 2, 3)
            .expect("separate header for the same view");
        mat_flip_into(&source, &destination, 0).expect("flip exact ROI view in place");

        assert_eq!(
            parent.to_u8_array(),
            [
                1, 2, 3, 4, 5, 6, 12, 13, 14, 10, 11, 7, 8, 9, 15, 16, 17, 18, 19, 20
            ]
        );
        assert_eq!(source.to_u8_array(), [12, 13, 14, 7, 8, 9]);
        assert_eq!(destination.to_u8_array(), [12, 13, 14, 7, 8, 9]);
    }

    #[test]
    fn flip_detaches_an_incompatible_roi_destination() {
        let parent = u8_matrix((1..=30).collect(), 5, 6, 1);
        let source = parent.roi(0, 0, 2, 3).expect("valid source region");
        let destination = parent
            .roi(2, 3, 3, 2)
            .expect("valid incompatible destination region");
        let old_alias = parent
            .roi(2, 3, 3, 2)
            .expect("alias of the old destination view");
        let before = parent.to_u8_array();

        mat_flip_into(&source, &destination, 1).expect("flip into incompatible ROI");

        assert_eq!(parent.to_u8_array(), before);
        assert_eq!((destination.rows(), destination.columns()), (2, 3));
        assert_eq!(destination.to_u8_array(), [3, 2, 1, 9, 8, 7]);
        assert_eq!(old_alias.to_u8_array(), [16, 17, 22, 23, 28, 29]);
    }

    #[test]
    fn transpose_keeps_exact_in_place_matrices_on_the_safe_path() {
        let square = u8_matrix((1..=9).collect(), 3, 3, 1);
        mat_transpose_into(&square, &square).expect("transpose square in place");
        assert_eq!(square.to_u8_array(), [1, 4, 7, 2, 5, 8, 3, 6, 9]);

        let rectangular = u8_matrix((1..=6).collect(), 2, 3, 1);
        mat_transpose_into(&rectangular, &rectangular).expect("transpose rectangle in place");
        assert_eq!((rectangular.rows(), rectangular.columns()), (3, 2));
        assert_eq!(rectangular.to_u8_array(), [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn flip_keeps_f64_scalar_bytes_together() {
        let values = [1.25_f64, -2.5, 3.75, 4.5];
        let bytes = values
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>();
        let source =
            Mat::from_owned_bytes(bytes, 2, 2, 1, MatDepth::F64).expect("valid F64 test matrix");

        let output =
            apply_layout(&source, |matrix| flip_bytes(matrix, 1)).expect("horizontal flip");

        assert_eq!(output.depth(), MatDepth::F64);
        assert_eq!(output.channels(), 1);
        assert_eq!(
            output.to_f64_array().expect("F64 output"),
            [-2.5, 1.25, 4.5, 3.75]
        );
    }

    #[test]
    fn every_scalar_depth_passes_through_the_adapter() {
        let depths = [
            MatDepth::U8,
            MatDepth::I8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::I32,
            MatDepth::F32,
            MatDepth::F64,
        ];

        for depth in depths {
            let width = depth.byte_width();
            let bytes = (0..width * 2)
                .map(|index| u8::try_from(index + 1).expect("test element fits in U8"))
                .collect::<Vec<_>>();
            let source = Mat::from_owned_bytes(bytes.clone(), 1, 2, 1, depth)
                .expect("valid matrix for scalar depth");

            let output = apply_layout(&source, |matrix| flip_bytes(matrix, 1))
                .expect("layout adapter accepts scalar depth");
            let mut expected = bytes[width..].to_vec();
            expected.extend_from_slice(&bytes[..width]);

            assert_eq!(output.depth(), depth);
            assert_eq!(output.to_u8_array(), expected);
        }
    }

    #[test]
    fn rotate_public_adapter_preserves_depth_in_place_and_destination_replacement() {
        let depths = [
            MatDepth::U8,
            MatDepth::I8,
            MatDepth::U16,
            MatDepth::I16,
            MatDepth::I32,
            MatDepth::F32,
            MatDepth::F64,
        ];

        for depth in depths {
            let width = depth.byte_width();
            let bytes = (0..width * 4)
                .map(|index| u8::try_from(index + 1).expect("test byte fits in U8"))
                .collect::<Vec<_>>();
            let source = Mat::from_owned_bytes(bytes.clone(), 2, 2, 1, depth)
                .expect("valid matrix for scalar depth");

            let output = mat_rotate(&source, 1).expect("rotate accepts scalar depth");
            let expected = bytes
                .chunks_exact(width)
                .rev()
                .flatten()
                .copied()
                .collect::<Vec<_>>();

            assert_eq!(output.depth(), depth);
            assert_eq!(output.to_u8_array(), expected);
        }

        let in_place = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        mat_rotate_into(&in_place, &in_place, 0).expect("rotate rectangle in place");
        assert_eq!((in_place.rows(), in_place.columns()), (3, 2));
        assert_eq!(in_place.to_u8_array(), [4, 1, 5, 2, 6, 3]);

        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        let destination = Mat::from_owned_bytes(vec![0; 8], 2, 2, 1, MatDepth::U16)
            .expect("valid incompatible destination");
        mat_rotate_into(&source, &destination, 2).expect("rotate rebinds destination");
        assert_eq!((destination.rows(), destination.columns()), (3, 2));
        assert_eq!(destination.depth(), MatDepth::U8);
        assert_eq!(destination.to_u8_array(), [3, 6, 2, 5, 1, 4]);
    }

    #[test]
    fn rotate_and_repeat_return_expected_shapes() {
        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);

        let rotated =
            apply_layout(&source, |matrix| rotate_bytes(matrix, 0)).expect("clockwise rotation");
        assert_eq!((rotated.rows(), rotated.columns()), (3, 2));
        assert_eq!(rotated.to_u8_array(), [4, 1, 5, 2, 6, 3]);

        let repeated =
            apply_layout(&source, |matrix| repeat_bytes(matrix, 2, 1)).expect("row repeat");
        assert_eq!((repeated.rows(), repeated.columns()), (4, 3));
        assert_eq!(repeated.to_u8_array(), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn adapters_reject_invalid_codes_and_repeat_counts() {
        let source = u8_matrix(vec![1], 1, 1, 1);

        assert!(matches!(
            apply_layout(&source, |matrix| rotate_bytes(matrix, -1)),
            Err(LayoutWasmError::Kernel(LayoutError::InvalidRotateCode(-1)))
        ));
        assert!(matches!(
            apply_layout(&source, |matrix| repeat_bytes(matrix, 0, 2)),
            Err(LayoutWasmError::Kernel(LayoutError::InvalidRepeatCount {
                rows: 0,
                columns: 2
            }))
        ));
    }

    #[test]
    fn into_supports_exact_in_place_aliasing() {
        let matrix = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);

        apply_layout_into(&matrix, &matrix, |source| flip_bytes(source, -1))
            .expect("in-place flip");

        assert_eq!(matrix.to_u8_array(), [6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn into_writes_through_a_strided_destination_region() {
        let source = u8_matrix(vec![1, 2, 3, 4], 2, 2, 1);
        let parent = u8_matrix(vec![0; 16], 4, 4, 1);
        let destination = parent.roi(1, 1, 2, 2).expect("valid destination ROI");
        assert!(!destination.is_continuous());

        apply_layout_into(&source, &destination, transpose_bytes).expect("transpose into ROI");

        assert_eq!(destination.to_u8_array(), [1, 3, 2, 4]);
        assert_eq!(
            parent.to_u8_array(),
            [0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 4, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn into_rebinds_each_incompatible_ordinary_destination() {
        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        let wrong_shape = u8_matrix(vec![9; 6], 2, 3, 1);
        let wrong_channels = u8_matrix(vec![8; 6], 3, 1, 2);
        let wrong_depth = Mat::from_owned_bytes(vec![7; 12], 3, 2, 1, MatDepth::U16)
            .expect("valid wrong-depth destination");

        for destination in [&wrong_shape, &wrong_channels, &wrong_depth] {
            apply_layout_into(&source, destination, transpose_bytes)
                .expect("ordinary destination header can rebind");
            assert_eq!((destination.rows(), destination.columns()), (3, 2));
            assert_eq!(destination.channels(), 1);
            assert_eq!(destination.depth(), MatDepth::U8);
            assert_eq!(destination.to_u8_array(), [1, 4, 2, 5, 3, 6]);
        }
    }

    #[test]
    fn into_detaches_incompatible_roi_without_changing_parent() {
        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        let parent = u8_matrix(vec![9; 12], 3, 4, 1);
        let destination = parent.roi(0, 0, 2, 3).expect("valid ROI");
        let before = parent.to_u8_array();

        apply_layout_into(&source, &destination, transpose_bytes)
            .expect("incompatible ROI header detaches");
        assert_eq!(parent.to_u8_array(), before);
        assert_eq!((destination.rows(), destination.columns()), (3, 2));
        assert_eq!(destination.to_u8_array(), [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn rotate_and_repeat_into_write_expected_outputs() {
        let source = u8_matrix(vec![1, 2, 3, 4, 5, 6], 2, 3, 1);
        let rotated = u8_matrix(vec![0; 6], 3, 2, 1);
        let repeated = u8_matrix(vec![0; 12], 4, 3, 1);

        apply_layout_into(&source, &rotated, |matrix| rotate_bytes(matrix, 0))
            .expect("rotate into destination");
        apply_layout_into(&source, &repeated, |matrix| repeat_bytes(matrix, 2, 1))
            .expect("repeat into destination");

        assert_eq!(rotated.to_u8_array(), [4, 1, 5, 2, 6, 3]);
        assert_eq!(repeated.to_u8_array(), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn horizontal_concat_compacts_two_strided_rois_without_mutating_inputs() {
        let left_parent = u8_matrix(vec![1, 2, 3, 4, 5, 6, 7, 8], 2, 4, 1);
        let right_parent = u8_matrix(vec![9, 10, 11, 12, 13, 14, 15, 16], 2, 4, 1);
        let left = left_parent.roi(0, 1, 2, 2).expect("left ROI");
        let right = right_parent.roi(0, 0, 2, 1).expect("right ROI");
        let left_before = left_parent.to_u8_array();
        let right_before = right_parent.to_u8_array();

        let output = apply_concat(&[&left, &right], hconcat_bytes).expect("horizontal concat");

        assert_eq!((output.rows(), output.columns()), (2, 3));
        assert_eq!(output.to_u8_array(), [2, 3, 9, 6, 7, 13]);
        assert_eq!(left_parent.to_u8_array(), left_before);
        assert_eq!(right_parent.to_u8_array(), right_before);
    }

    #[test]
    fn vertical_concat_supports_four_all_depth_channel_matrices() {
        let values = [
            [1.25_f64, -2.5],
            [3.75_f64, 4.5],
            [-6.0_f64, 7.25],
            [8.5_f64, -9.75],
        ];
        let matrices = values.map(|row| {
            Mat::from_owned_bytes(
                row.into_iter().flat_map(f64::to_ne_bytes).collect(),
                1,
                1,
                2,
                MatDepth::F64,
            )
            .expect("valid F64 matrix")
        });

        let output = apply_concat(
            &[&matrices[0], &matrices[1], &matrices[2], &matrices[3]],
            vconcat_bytes,
        )
        .expect("vertical concat");

        assert_eq!(
            (output.rows(), output.columns(), output.channels()),
            (4, 1, 2)
        );
        assert_eq!(output.depth(), MatDepth::F64);
        assert_eq!(
            output.to_f64_array().expect("F64 output"),
            [1.25, -2.5, 3.75, 4.5, -6.0, 7.25, 8.5, -9.75]
        );
    }

    #[test]
    fn concat_rejects_equal_width_but_different_depths() {
        let unsigned =
            Mat::from_owned_bytes(vec![1, 0], 1, 1, 1, MatDepth::U16).expect("valid U16 matrix");
        let signed =
            Mat::from_owned_bytes(vec![1, 0], 1, 1, 1, MatDepth::I16).expect("valid I16 matrix");

        assert!(matches!(
            apply_concat(&[&unsigned, &signed], hconcat_bytes),
            Err(LayoutWasmError::IncompatibleSourceDepth {
                index: 1,
                expected: MatDepth::U16,
                actual: MatDepth::I16
            })
        ));
    }

    #[test]
    fn concat_rejects_shape_and_channel_mismatches() {
        let first = u8_matrix(vec![1, 2], 1, 2, 1);
        let wrong_rows = u8_matrix(vec![3, 4, 5, 6], 2, 2, 1);
        let wrong_columns = u8_matrix(vec![3, 4], 2, 1, 1);
        let wrong_channels = u8_matrix(vec![3, 4, 5, 6], 1, 2, 2);

        for result in [
            apply_concat(&[&first, &wrong_rows], hconcat_bytes),
            apply_concat(&[&first, &wrong_columns], vconcat_bytes),
            apply_concat(&[&first, &wrong_channels], hconcat_bytes),
        ] {
            assert!(matches!(
                result,
                Err(LayoutWasmError::Kernel(LayoutError::IncompatibleSource {
                    index: 1,
                    ..
                }))
            ));
        }
    }
}
