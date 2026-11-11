use std::{error::Error, fmt};

use crate::imgproc_sample::{decode, encode};
use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClaheError {
    Matrix(MatError),
    UnsupportedDepth(MatDepth),
    UnsupportedChannels(u16),
    EmptySource,
}

impl fmt::Display for ClaheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth(depth) => {
                write!(formatter, "CLAHE depth {depth:?} is not implemented")
            }
            Self::UnsupportedChannels(channels) => {
                write!(formatter, "CLAHE requires a single-channel matrix, got {channels}")
            }
            Self::EmptySource => formatter.write_str("CLAHE requires a non-empty source"),
        }
    }
}

impl Error for ClaheError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for ClaheError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Contrast-limited adaptive histogram equalization.
///
/// Reference: Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
/// Graphics Gems IV. The image is split into a grid of tiles; each tile's
/// histogram is clipped above a limit and the excess redistributed across all
/// bins; per-pixel results bilinearly blend the four surrounding tile LUTs.
pub(crate) fn clahe_apply_into(
    source: &Mat,
    destination: &Mat,
    clip_limit: f64,
    tiles_x: u32,
    tiles_y: u32,
) -> Result<(), ClaheError> {
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    let depth = source.depth();
    if source.channels() != 1 {
        return Err(ClaheError::UnsupportedChannels(source.channels()));
    }
    if !matches!(depth, MatDepth::U8 | MatDepth::U16) {
        return Err(ClaheError::UnsupportedDepth(depth));
    }
    if rows == 0 || columns == 0 {
        return Err(ClaheError::EmptySource);
    }
    // A tile must cover at least one pixel; oversized grids collapse to the
    // image dimensions rather than producing empty tiles.
    let tiles_x = (tiles_x.max(1) as usize).min(columns);
    let tiles_y = (tiles_y.max(1) as usize).min(rows);
    let histogram_size = if depth == MatDepth::U8 { 256usize } else { 65536usize };
    let element_width = depth.byte_width();

    let input = source.compact_bytes();
    let value_at = |index: usize| -> usize {
        decode(&input[index * element_width..(index + 1) * element_width], depth) as usize
    };

    // Per-tile lookup tables, flattened as luts[(ty * tiles_x + tx) * histogram_size + bin].
    let mut luts = vec![0u32; tiles_x * tiles_y * histogram_size];
    for ty in 0..tiles_y {
        let y0 = ty * rows / tiles_y;
        let y1 = (ty + 1) * rows / tiles_y;
        for tx in 0..tiles_x {
            let x0 = tx * columns / tiles_x;
            let x1 = (tx + 1) * columns / tiles_x;
            let tile_pixels = (y1 - y0) * (x1 - x0);
            let lut_offset = (ty * tiles_x + tx) * histogram_size;

            let mut histogram = vec![0u32; histogram_size];
            for row in y0..y1 {
                for column in x0..x1 {
                    histogram[value_at(row * columns + column)] += 1;
                }
            }

            if clip_limit >= 1.0 && tile_pixels > 0 {
                let limit = ((clip_limit * tile_pixels as f64 / histogram_size as f64) as u64)
                    .max(1);
                let mut excess = 0u64;
                for count in histogram.iter_mut() {
                    if u64::from(*count) > limit {
                        excess += u64::from(*count) - limit;
                        *count = limit as u32;
                    }
                }
                if excess > 0 {
                    let bonus = (excess / histogram_size as u64) as u32;
                    let mut residual = (excess % histogram_size as u64) as usize;
                    if bonus > 0 {
                        for count in histogram.iter_mut() {
                            *count += bonus;
                        }
                    }
                    if residual > 0 {
                        // Spread the leftover counts evenly across the bins.
                        let stride = histogram_size / residual;
                        let mut bin = 0usize;
                        while residual > 0 {
                            histogram[bin] += 1;
                            bin += stride;
                            residual -= 1;
                        }
                    }
                }
            }

            // Cumulative distribution scaled to the output range.
            let scale = (histogram_size - 1) as f32 / tile_pixels.max(1) as f32;
            let mut cumulative = 0u32;
            for bin in 0..histogram_size {
                cumulative += histogram[bin];
                luts[lut_offset + bin] =
                    (cumulative as f32 * scale).round_ties_even() as u32;
            }
        }
    }

    // Bilinear interpolation between neighboring tile LUTs.
    let mut output = vec![0u8; input.len()];
    for row in 0..rows {
        let mut y_map = row as f32 * tiles_y as f32 / rows as f32 - 0.5;
        if y_map < 0.0 {
            y_map = 0.0;
        }
        if y_map > (tiles_y - 1) as f32 {
            y_map = (tiles_y - 1) as f32;
        }
        let ty0 = y_map.floor() as usize;
        let ty1 = (ty0 + 1).min(tiles_y - 1);
        let fy = y_map - ty0 as f32;
        for column in 0..columns {
            let mut x_map = column as f32 * tiles_x as f32 / columns as f32 - 0.5;
            if x_map < 0.0 {
                x_map = 0.0;
            }
            if x_map > (tiles_x - 1) as f32 {
                x_map = (tiles_x - 1) as f32;
            }
            let tx0 = x_map.floor() as usize;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let fx = x_map - tx0 as f32;

            let value = value_at(row * columns + column);
            let base = tiles_y * 0 + ty0;
            let v00 = luts[(base * tiles_x + tx0) * histogram_size + value] as f32;
            let v01 = luts[(base * tiles_x + tx1) * histogram_size + value] as f32;
            let v10 = luts[(ty1 * tiles_x + tx0) * histogram_size + value] as f32;
            let v11 = luts[(ty1 * tiles_x + tx1) * histogram_size + value] as f32;
            let top = v00 * (1.0 - fx) + v01 * fx;
            let bottom = v10 * (1.0 - fx) + v11 * fx;
            let result = top * (1.0 - fy) + bottom * fy;
            encode(
                f64::from(result),
                depth,
                &mut output[(row * columns + column) * element_width
                    ..(row * columns + column + 1) * element_width],
            );
        }
    }

    destination.write_output(output, source.rows(), source.columns(), 1, depth)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_from_u8, mat_from_u16};

    fn mat_empty() -> Mat {
        crate::mat::mat_empty()
    }

    fn u8_values(mat: &Mat) -> Vec<u8> {
        mat.compact_bytes().to_vec()
    }

    fn u16_values(mat: &Mat) -> Vec<u16> {
        mat.compact_bytes()
            .chunks_exact(2)
            .map(|b| u16::from_ne_bytes([b[0], b[1]]))
            .collect()
    }

    #[test]
    fn clahe_matches_pinned_u8_ramp() {
        let source = mat_from_u8(
            &(0u32..64).map(|i| (i * 4) as u8).collect::<Vec<u8>>(),
            8,
            8,
            1,
        )
        .expect("source");
        let destination = mat_empty();
        clahe_apply_into(&source, &destination, 10.0, 4, 4).expect("apply");
        let expected: [u8; 64] = [
            64, 128, 96, 128, 96, 128, 96, 128, 191, 255, 223, 255, 223, 255, 223, 255, 160, 192,
            176, 192, 176, 192, 176, 192, 191, 255, 223, 255, 223, 255, 223, 255, 160, 192, 176,
            192, 176, 192, 176, 192, 191, 255, 223, 255, 223, 255, 223, 255, 160, 192, 176, 192,
            176, 192, 176, 192, 191, 255, 223, 255, 223, 255, 223, 255,
        ];
        assert_eq!(u8_values(&destination), expected.to_vec());
    }

    #[test]
    fn clahe_matches_pinned_u16_ramp() {
        let source = mat_from_u16(
            &(0u32..64).map(|i| (i * 100) as u16).collect::<Vec<u16>>(),
            8,
            8,
            1,
        )
        .expect("source");
        let destination = mat_empty();
        clahe_apply_into(&source, &destination, 10.0, 4, 4).expect("apply");
        let expected: [u16; 64] = [
            16384, 32768, 24576, 32768, 24576, 32768, 24576, 32768, 49151, 65535, 57343, 65535,
            57343, 65535, 57343, 65535, 40960, 49152, 45056, 49152, 45056, 49152, 45056, 49152,
            49151, 65535, 57343, 65535, 57343, 65535, 57343, 65535, 40960, 49152, 45056, 49152,
            45056, 49152, 45056, 49152, 49151, 65535, 57343, 65535, 57343, 65535, 57343, 65535,
            40960, 49152, 45056, 49152, 45056, 49152, 45056, 49152, 49151, 65535, 57343, 65535,
            57343, 65535, 57343, 65535,
        ];
        assert_eq!(u16_values(&destination), expected.to_vec());
    }

    #[test]
    fn clahe_matches_pinned_u8_quadratic() {
        let source = mat_from_u8(
            &(0u32..256).map(|i| ((i * i) % 256) as u8).collect::<Vec<u8>>(),
            16,
            16,
            1,
        )
        .expect("source");
        let destination = mat_empty();
        clahe_apply_into(&source, &destination, 5.0, 2, 2).expect("apply");
        let expected: [u8; 256] = [
            8, 12, 16, 20, 28, 40, 51, 62, 72, 92, 112, 131, 155, 183, 207, 231, 8, 44, 80, 116,
            155, 195, 239, 30, 72, 124, 175, 227, 24, 88, 143, 203, 8, 76, 143, 211, 28, 100, 175,
            251, 72, 159, 239, 64, 155, 243, 80, 171, 8, 108, 207, 56, 155, 255, 112, 219, 72,
            187, 49, 163, 24, 147, 12, 135, 8, 135, 16, 147, 28, 163, 51, 187, 72, 219, 112, 255,
            155, 52, 207, 108, 8, 171, 80, 243, 155, 68, 239, 159, 72, 251, 175, 100, 24, 211,
            143, 76, 8, 203, 143, 88, 28, 227, 175, 124, 72, 30, 239, 195, 155, 116, 80, 40, 8,
            231, 207, 183, 155, 131, 112, 92, 72, 62, 49, 36, 24, 16, 12, 8, 8, 12, 16, 20, 28,
            40, 51, 62, 72, 92, 112, 131, 155, 183, 207, 231, 8, 44, 80, 116, 155, 195, 239, 30,
            72, 124, 175, 227, 24, 88, 143, 203, 8, 76, 143, 211, 28, 100, 175, 251, 72, 159,
            239, 64, 155, 243, 80, 171, 8, 108, 207, 56, 155, 255, 112, 219, 72, 187, 49, 163,
            24, 147, 12, 135, 8, 135, 16, 147, 28, 163, 51, 187, 72, 219, 112, 255, 155, 52, 207,
            108, 8, 171, 80, 243, 155, 68, 239, 159, 72, 251, 175, 100, 24, 211, 143, 76, 8, 203,
            143, 88, 28, 227, 175, 124, 72, 30, 239, 195, 155, 116, 80, 40, 8, 231, 207, 183,
            155, 131, 112, 92, 72, 62, 49, 36, 24, 16, 12, 8,
        ];
        assert_eq!(u8_values(&destination), expected.to_vec());
    }
}
