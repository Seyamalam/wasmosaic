use std::{cell::RefCell, error::Error, fmt, rc::Rc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MutableStorageError {
    EmptyDimensions,
    SizeOverflow,
    IncorrectBufferLength { expected: usize, actual: usize },
    RegionOutOfBounds,
}

impl fmt::Display for MutableStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => formatter.write_str("storage dimensions must be nonzero"),
            Self::SizeOverflow => {
                formatter.write_str("storage dimensions exceed addressable memory")
            }
            Self::IncorrectBufferLength { expected, actual } => write!(
                formatter,
                "storage buffer has {actual} bytes; expected {expected} bytes"
            ),
            Self::RegionOutOfBounds => {
                formatter.write_str("storage region extends outside its parent")
            }
        }
    }
}

impl Error for MutableStorageError {}

/// Mutable byte storage shared by a matrix and each of its ROI views.
///
/// This module assumes the browser's normal single-JavaScript-agent execution model. `Rc` and
/// `RefCell` keep mutation safe without atomics or `unsafe` Rust. The type is intentionally neither
/// `Send` nor `Sync`; a future threaded WASM adapter can replace the private owner with synchronized
/// storage without changing this interface.
#[derive(Debug, Clone)]
pub(crate) struct MutableStorage {
    data: Rc<RefCell<Vec<u8>>>,
    rows: usize,
    row_bytes: usize,
    row_stride: usize,
    offset: usize,
}

impl MutableStorage {
    pub(crate) fn write_nearest_from_shared(
        &self,
        source: &Self,
        pixel_bytes: usize,
        columns: &[u32],
        rows: &[u32],
    ) -> Result<(), MutableStorageError> {
        if !self.shares_allocation_with(source)
            || self.rows != rows.len()
            || self.row_bytes
                != columns
                    .len()
                    .checked_mul(pixel_bytes)
                    .ok_or(MutableStorageError::SizeOverflow)?
            || rows.iter().any(|&row| row as usize >= source.rows)
            || columns
                .iter()
                .any(|&column| (column as usize + 1) * pixel_bytes > source.row_bytes)
        {
            return Err(MutableStorageError::RegionOutOfBounds);
        }
        let mut data = self.data.borrow_mut();
        for (y, &source_y) in rows.iter().enumerate() {
            for (x, &source_x) in columns.iter().enumerate() {
                let from = source.offset
                    + source_y as usize * source.row_stride
                    + source_x as usize * pixel_bytes;
                let to = self.offset + y * self.row_stride + x * pixel_bytes;
                data.copy_within(from..from + pixel_bytes, to);
            }
        }
        Ok(())
    }

    pub(crate) fn shares_allocation_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }

    pub(crate) fn describes_same_view_as(&self, other: &Self) -> bool {
        self.shares_allocation_with(other)
            && self.rows == other.rows
            && self.row_bytes == other.row_bytes
            && self.row_stride == other.row_stride
            && self.offset == other.offset
    }

    pub(crate) fn write_transpose_from_shared(
        &self,
        source: &Self,
        pixel_bytes: usize,
    ) -> Result<(), MutableStorageError> {
        let expected_source_row_bytes = self
            .rows
            .checked_mul(pixel_bytes)
            .ok_or(MutableStorageError::SizeOverflow)?;
        let expected_destination_row_bytes = source
            .rows
            .checked_mul(pixel_bytes)
            .ok_or(MutableStorageError::SizeOverflow)?;
        if !self.shares_allocation_with(source)
            || source.row_bytes != expected_source_row_bytes
            || self.row_bytes != expected_destination_row_bytes
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: expected_destination_row_bytes,
                actual: self.row_bytes,
            });
        }

        let mut data = self.data.borrow_mut();
        let mut pixel = vec![0; pixel_bytes];
        for destination_row in 0..self.rows {
            for destination_column in 0..source.rows {
                let source_start = source.offset
                    + destination_column * source.row_stride
                    + destination_row * pixel_bytes;
                pixel.copy_from_slice(&data[source_start..source_start + pixel_bytes]);
                let destination_start = self.offset
                    + destination_row * self.row_stride
                    + destination_column * pixel_bytes;
                data[destination_start..destination_start + pixel_bytes].copy_from_slice(&pixel);
            }
        }
        Ok(())
    }

    pub(crate) fn write_horizontal_flip_from_shared(
        &self,
        source: &Self,
        pixel_bytes: usize,
    ) -> Result<(), MutableStorageError> {
        if !self.shares_allocation_with(source)
            || self.rows != source.rows
            || self.row_bytes != source.row_bytes
            || pixel_bytes == 0
            || self.row_bytes % pixel_bytes != 0
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: source.row_bytes,
                actual: self.row_bytes,
            });
        }

        let columns = self.row_bytes / pixel_bytes;
        let mut data = self.data.borrow_mut();
        let mut left_pixel = vec![0; pixel_bytes];
        let mut right_pixel = vec![0; pixel_bytes];
        for row in 0..self.rows {
            let source_row = source.offset + row * source.row_stride;
            let destination_row = self.offset + row * self.row_stride;
            for left_column in 0..columns / 2 {
                let right_column = columns - left_column - 1;
                let source_left = source_row + left_column * pixel_bytes;
                let source_right = source_row + right_column * pixel_bytes;
                left_pixel.copy_from_slice(&data[source_left..source_left + pixel_bytes]);
                right_pixel.copy_from_slice(&data[source_right..source_right + pixel_bytes]);

                let destination_left = destination_row + left_column * pixel_bytes;
                let destination_right = destination_row + right_column * pixel_bytes;
                data[destination_right..destination_right + pixel_bytes]
                    .copy_from_slice(&left_pixel);
                data[destination_left..destination_left + pixel_bytes]
                    .copy_from_slice(&right_pixel);
            }
            if columns % 2 == 1 {
                let center = columns / 2;
                let source_center = source_row + center * pixel_bytes;
                left_pixel.copy_from_slice(&data[source_center..source_center + pixel_bytes]);
                let destination_center = destination_row + center * pixel_bytes;
                data[destination_center..destination_center + pixel_bytes]
                    .copy_from_slice(&left_pixel);
            }
        }
        Ok(())
    }

    pub(crate) fn write_vertical_flip_from_shared(
        &self,
        source: &Self,
        pixel_bytes: usize,
    ) -> Result<(), MutableStorageError> {
        if !self.shares_allocation_with(source)
            || self.rows != source.rows
            || self.row_bytes != source.row_bytes
            || pixel_bytes == 0
            || self.row_bytes % pixel_bytes != 0
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: source.row_bytes,
                actual: self.row_bytes,
            });
        }

        let columns = self.row_bytes / pixel_bytes;
        let mut data = self.data.borrow_mut();
        let mut top_pixel = vec![0; pixel_bytes];
        let mut bottom_pixel = vec![0; pixel_bytes];
        for top in 0..self.rows / 2 {
            let bottom = self.rows - top - 1;
            let source_top_row = source.offset + top * source.row_stride;
            let source_bottom_row = source.offset + bottom * source.row_stride;
            let destination_top_row = self.offset + top * self.row_stride;
            let destination_bottom_row = self.offset + bottom * self.row_stride;
            for column in 0..columns {
                let source_top = source_top_row + column * pixel_bytes;
                let source_bottom = source_bottom_row + column * pixel_bytes;
                top_pixel.copy_from_slice(&data[source_top..source_top + pixel_bytes]);
                bottom_pixel.copy_from_slice(&data[source_bottom..source_bottom + pixel_bytes]);

                let destination_top = destination_top_row + column * pixel_bytes;
                let destination_bottom = destination_bottom_row + column * pixel_bytes;
                data[destination_bottom..destination_bottom + pixel_bytes]
                    .copy_from_slice(&top_pixel);
                data[destination_top..destination_top + pixel_bytes].copy_from_slice(&bottom_pixel);
            }
        }
        if self.rows % 2 == 1 {
            let center = self.rows / 2;
            let source_center_row = source.offset + center * source.row_stride;
            let destination_center_row = self.offset + center * self.row_stride;
            for column in 0..columns {
                let source_center = source_center_row + column * pixel_bytes;
                top_pixel.copy_from_slice(&data[source_center..source_center + pixel_bytes]);
                let destination_center = destination_center_row + column * pixel_bytes;
                data[destination_center..destination_center + pixel_bytes]
                    .copy_from_slice(&top_pixel);
            }
        }
        Ok(())
    }

    pub(crate) fn write_repeat_from_shared(
        &self,
        source: &Self,
        pixel_bytes: usize,
        row_repeats: usize,
        column_repeats: usize,
    ) -> Result<(), MutableStorageError> {
        let expected_rows = source
            .rows
            .checked_mul(row_repeats)
            .ok_or(MutableStorageError::SizeOverflow)?;
        let expected_row_bytes = source
            .row_bytes
            .checked_mul(column_repeats)
            .ok_or(MutableStorageError::SizeOverflow)?;
        if !self.shares_allocation_with(source)
            || self.rows != expected_rows
            || self.row_bytes != expected_row_bytes
            || pixel_bytes == 0
            || source.row_bytes % pixel_bytes != 0
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: expected_row_bytes,
                actual: self.row_bytes,
            });
        }

        let source_columns = source.row_bytes / pixel_bytes;
        let destination_columns = self.row_bytes / pixel_bytes;
        let mut data = self.data.borrow_mut();
        let mut pixel = vec![0; pixel_bytes];
        for destination_row in 0..self.rows {
            let source_row = source.offset + destination_row % source.rows * source.row_stride;
            let destination_row_start = self.offset + destination_row * self.row_stride;
            for destination_column in 0..destination_columns {
                let source_start = source_row + destination_column % source_columns * pixel_bytes;
                pixel.copy_from_slice(&data[source_start..source_start + pixel_bytes]);
                let destination_start = destination_row_start + destination_column * pixel_bytes;
                data[destination_start..destination_start + pixel_bytes].copy_from_slice(&pixel);
            }
        }
        Ok(())
    }

    pub(crate) fn write_unary_scalars_from_shared(
        &self,
        source: &Self,
        scalar_width: usize,
        mut operation: impl FnMut(&[u8], &mut [u8]),
    ) -> Result<(), MutableStorageError> {
        if !self.shares_allocation_with(source)
            || self.rows != source.rows
            || self.row_bytes != source.row_bytes
            || scalar_width == 0
            || self.row_bytes % scalar_width != 0
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: source.row_bytes,
                actual: self.row_bytes,
            });
        }

        let scalars_per_row = self.row_bytes / scalar_width;
        let mut data = self.data.borrow_mut();
        let mut input = vec![0; scalar_width];
        let mut output = vec![0; scalar_width];
        for row in 0..self.rows {
            let source_row = source.offset + row * source.row_stride;
            let destination_row = self.offset + row * self.row_stride;
            for scalar in 0..scalars_per_row {
                let source_start = source_row + scalar * scalar_width;
                input.copy_from_slice(&data[source_start..source_start + scalar_width]);
                operation(&input, &mut output);
                let destination_start = destination_row + scalar * scalar_width;
                data[destination_start..destination_start + scalar_width].copy_from_slice(&output);
            }
        }
        Ok(())
    }

    pub(crate) fn write_bitwise_not_from_shared(
        &self,
        source: &Self,
    ) -> Result<(), MutableStorageError> {
        if !self.shares_allocation_with(source)
            || self.rows != source.rows
            || self.row_bytes != source.row_bytes
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: source.row_bytes,
                actual: self.row_bytes,
            });
        }

        let mut data = self.data.borrow_mut();
        let mut input = [0; 2];
        for row in 0..self.rows {
            let source_row = source.offset + row * source.row_stride;
            let destination_row = self.offset + row * self.row_stride;
            let mut byte = 0;
            while byte + 1 < self.row_bytes {
                input.copy_from_slice(&data[source_row + byte..source_row + byte + 2]);
                data[destination_row + byte] = !input[0];
                data[destination_row + byte + 1] = !input[1];
                byte += 2;
            }
            if byte < self.row_bytes {
                let input = data[source_row + byte];
                data[destination_row + byte] = !input;
            }
        }
        Ok(())
    }

    pub(crate) fn write_binary_scalars_from_shared(
        &self,
        first: &Self,
        second: &Self,
        scalar_width: usize,
        mut operation: impl FnMut(&[u8], &[u8], &mut [u8]),
    ) -> Result<(), MutableStorageError> {
        if self.rows != first.rows
            || self.rows != second.rows
            || self.row_bytes != first.row_bytes
            || self.row_bytes != second.row_bytes
            || scalar_width == 0
            || self.row_bytes % scalar_width != 0
        {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: first.row_bytes,
                actual: self.row_bytes,
            });
        }

        let shares_first = self.shares_allocation_with(first);
        let shares_second = self.shares_allocation_with(second);
        if !shares_first && !shares_second {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected: first.row_bytes,
                actual: self.row_bytes,
            });
        }

        let scalars_per_row = self.row_bytes / scalar_width;
        let mut first_input = vec![0; scalar_width];
        let mut second_input = vec![0; scalar_width];
        let mut output = vec![0; scalar_width];
        if shares_first && shares_second {
            let mut data = self.data.borrow_mut();
            for row in 0..self.rows {
                let first_row = first.offset + row * first.row_stride;
                let second_row = second.offset + row * second.row_stride;
                let destination_row = self.offset + row * self.row_stride;
                for scalar in 0..scalars_per_row {
                    let first_start = first_row + scalar * scalar_width;
                    let second_start = second_row + scalar * scalar_width;
                    first_input.copy_from_slice(&data[first_start..first_start + scalar_width]);
                    second_input.copy_from_slice(&data[second_start..second_start + scalar_width]);
                    operation(&first_input, &second_input, &mut output);
                    let destination_start = destination_row + scalar * scalar_width;
                    data[destination_start..destination_start + scalar_width]
                        .copy_from_slice(&output);
                }
            }
            return Ok(());
        }

        if shares_first {
            let mut shared_data = self.data.borrow_mut();
            let second_data = second.data.borrow();
            for row in 0..self.rows {
                let first_row = first.offset + row * first.row_stride;
                let second_row = second.offset + row * second.row_stride;
                let destination_row = self.offset + row * self.row_stride;
                for scalar in 0..scalars_per_row {
                    let first_start = first_row + scalar * scalar_width;
                    let second_start = second_row + scalar * scalar_width;
                    first_input
                        .copy_from_slice(&shared_data[first_start..first_start + scalar_width]);
                    second_input
                        .copy_from_slice(&second_data[second_start..second_start + scalar_width]);
                    operation(&first_input, &second_input, &mut output);
                    let destination_start = destination_row + scalar * scalar_width;
                    shared_data[destination_start..destination_start + scalar_width]
                        .copy_from_slice(&output);
                }
            }
            return Ok(());
        }

        let first_data = first.data.borrow();
        let mut shared_data = self.data.borrow_mut();
        for row in 0..self.rows {
            let first_row = first.offset + row * first.row_stride;
            let second_row = second.offset + row * second.row_stride;
            let destination_row = self.offset + row * self.row_stride;
            for scalar in 0..scalars_per_row {
                let first_start = first_row + scalar * scalar_width;
                let second_start = second_row + scalar * scalar_width;
                first_input.copy_from_slice(&first_data[first_start..first_start + scalar_width]);
                second_input
                    .copy_from_slice(&shared_data[second_start..second_start + scalar_width]);
                operation(&first_input, &second_input, &mut output);
                let destination_start = destination_row + scalar * scalar_width;
                shared_data[destination_start..destination_start + scalar_width]
                    .copy_from_slice(&output);
            }
        }
        Ok(())
    }

    pub(crate) fn from_compact(
        data: Vec<u8>,
        rows: usize,
        row_bytes: usize,
    ) -> Result<Self, MutableStorageError> {
        let expected = checked_area(rows, row_bytes)?;
        if data.len() != expected {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            data: Rc::new(RefCell::new(data)),
            rows,
            row_bytes,
            row_stride: row_bytes,
            offset: 0,
        })
    }

    pub(crate) fn region(
        &self,
        row: usize,
        byte_column: usize,
        rows: usize,
        row_bytes: usize,
    ) -> Result<Self, MutableStorageError> {
        checked_area(rows, row_bytes)?;
        let row_end = row
            .checked_add(rows)
            .ok_or(MutableStorageError::RegionOutOfBounds)?;
        let column_end = byte_column
            .checked_add(row_bytes)
            .ok_or(MutableStorageError::RegionOutOfBounds)?;
        if row_end > self.rows || column_end > self.row_bytes {
            return Err(MutableStorageError::RegionOutOfBounds);
        }

        let row_offset = row
            .checked_mul(self.row_stride)
            .ok_or(MutableStorageError::RegionOutOfBounds)?;
        let offset = self
            .offset
            .checked_add(row_offset)
            .and_then(|value| value.checked_add(byte_column))
            .ok_or(MutableStorageError::RegionOutOfBounds)?;

        Ok(Self {
            data: Rc::clone(&self.data),
            rows,
            row_bytes,
            row_stride: self.row_stride,
            offset,
        })
    }

    pub(crate) fn compact_bytes(&self) -> Vec<u8> {
        let data = self.data.borrow();
        if self.row_bytes == self.row_stride {
            let length = self
                .rows
                .checked_mul(self.row_bytes)
                .expect("validated storage dimensions remain valid");
            return data[self.offset..self.offset + length].to_vec();
        }

        let mut output = Vec::with_capacity(
            self.rows
                .checked_mul(self.row_bytes)
                .expect("validated storage dimensions remain valid"),
        );
        for row in 0..self.rows {
            let start = self.offset + row * self.row_stride;
            output.extend_from_slice(&data[start..start + self.row_bytes]);
        }
        output
    }

    pub(crate) const fn row_stride(&self) -> usize {
        self.row_stride
    }

    pub(crate) const fn is_continuous(&self) -> bool {
        self.rows <= 1 || self.row_bytes == self.row_stride
    }

    pub(crate) fn write_from_compact(&self, source: &[u8]) -> Result<(), MutableStorageError> {
        let expected = checked_area(self.rows, self.row_bytes)?;
        if source.len() != expected {
            return Err(MutableStorageError::IncorrectBufferLength {
                expected,
                actual: source.len(),
            });
        }

        let mut data = self.data.borrow_mut();
        for row in 0..self.rows {
            let source_start = row * self.row_bytes;
            let destination_start = self.offset + row * self.row_stride;
            data[destination_start..destination_start + self.row_bytes]
                .copy_from_slice(&source[source_start..source_start + self.row_bytes]);
        }
        Ok(())
    }
}

fn checked_area(rows: usize, row_bytes: usize) -> Result<usize, MutableStorageError> {
    if rows == 0 || row_bytes == 0 {
        return Err(MutableStorageError::EmptyDimensions);
    }
    rows.checked_mul(row_bytes)
        .ok_or(MutableStorageError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destination_write_updates_parent_and_overlapping_views() {
        let parent = MutableStorage::from_compact(
            vec![
                1, 2, 3, 4, // row 0
                5, 6, 7, 8, // row 1
                9, 10, 11, 12, // row 2
            ],
            3,
            4,
        )
        .expect("valid parent storage");
        let destination = parent.region(0, 1, 2, 2).expect("valid destination region");
        let overlapping = parent.region(1, 0, 2, 3).expect("valid overlapping region");

        destination
            .write_from_compact(&[20, 21, 22, 23])
            .expect("source matches destination shape");

        assert_eq!(
            parent.compact_bytes(),
            vec![1, 20, 21, 4, 5, 22, 23, 8, 9, 10, 11, 12]
        );
        assert_eq!(overlapping.compact_bytes(), vec![5, 22, 23, 9, 10, 11]);
    }

    #[test]
    fn rejected_write_leaves_destination_unchanged() {
        let destination = MutableStorage::from_compact(vec![1, 2, 3, 4], 2, 2)
            .expect("valid destination storage");

        let error = destination
            .write_from_compact(&[9, 8, 7])
            .expect_err("short sources must be rejected");

        assert_eq!(
            error,
            MutableStorageError::IncorrectBufferLength {
                expected: 4,
                actual: 3,
            }
        );
        assert_eq!(destination.compact_bytes(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn nested_region_keeps_parent_stride_after_parent_is_dropped() {
        let nested = {
            let parent = MutableStorage::from_compact(
                vec![
                    1, 2, 3, 4, 5, // row 0
                    6, 7, 8, 9, 10, // row 1
                    11, 12, 13, 14, 15, // row 2
                ],
                3,
                5,
            )
            .expect("valid parent storage");
            parent
                .region(1, 1, 2, 4)
                .expect("valid first region")
                .region(0, 1, 2, 2)
                .expect("valid nested region")
        };

        nested
            .write_from_compact(&[80, 90, 120, 130])
            .expect("valid compact source");

        assert_eq!(nested.compact_bytes(), vec![80, 90, 120, 130]);
    }

    #[test]
    fn constructor_and_region_reject_invalid_geometry() {
        assert_eq!(
            MutableStorage::from_compact(Vec::new(), 0, 2).expect_err("zero rows must be rejected"),
            MutableStorageError::EmptyDimensions
        );

        let parent = MutableStorage::from_compact(vec![0; 12], 3, 4).expect("valid parent storage");
        assert_eq!(
            parent
                .region(2, 0, 2, 1)
                .expect_err("row range must stay inside parent"),
            MutableStorageError::RegionOutOfBounds
        );
        assert_eq!(
            parent
                .region(0, 3, 1, 2)
                .expect_err("byte range must stay inside parent"),
            MutableStorageError::RegionOutOfBounds
        );
        assert_eq!(
            parent
                .region(0, 0, usize::MAX, 1)
                .expect_err("overflowing row range must be rejected"),
            MutableStorageError::RegionOutOfBounds
        );
    }
}
