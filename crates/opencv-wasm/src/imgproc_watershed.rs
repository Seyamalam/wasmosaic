use std::{collections::VecDeque, error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WatershedError {
    Matrix(MatError),
    UnsupportedImage,
    UnsupportedMarkers,
    SizeMismatch,
}

impl fmt::Display for WatershedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedImage => {
                formatter.write_str("watershed requires an 8-bit 3-channel image")
            }
            Self::UnsupportedMarkers => {
                formatter.write_str("watershed requires a single-channel 32SC1 marker matrix")
            }
            Self::SizeMismatch => {
                formatter.write_str("watershed image and markers must share dimensions")
            }
        }
    }
}

impl Error for WatershedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for WatershedError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

const WSHED: i32 = -1;

/// Marker-controlled watershed segmentation.
///
/// Reference: Vincent & Soille, "Watersheds in Digital Spaces", IEEE PAMI
/// 1991. Interior pixels are grouped into intensity levels (the first
/// channel of the 8-bit color image) and flooded from low to high. At each
/// level, unlabeled pixels take the single adjacent label or become
/// watershed pixels when two or more labels meet; a seed activated beside
/// two or more already-established regions is itself a watershed pixel.
/// The one-pixel frame is always watershed.
pub(crate) fn watershed_into(image: &Mat, markers: &Mat) -> Result<(), WatershedError> {
    if image.depth() != MatDepth::U8 || image.channels() != 3 {
        return Err(WatershedError::UnsupportedImage);
    }
    if markers.depth() != MatDepth::I32 || markers.channels() != 1 {
        return Err(WatershedError::UnsupportedMarkers);
    }
    let rows = image.rows() as usize;
    let columns = image.columns() as usize;
    if markers.rows() as usize != rows || markers.columns() as usize != columns {
        return Err(WatershedError::SizeMismatch);
    }

    let marker_bytes = markers.compact_bytes();
    if rows < 3 || columns < 3 {
        // Every pixel lies on the frame; unlabeled pixels become watershed.
        let mut out = Vec::with_capacity(marker_bytes.len());
        for chunk in marker_bytes.chunks_exact(4) {
            let label = i32::from_ne_bytes(chunk.try_into().expect("i32"));
            out.extend_from_slice(&if label == 0 { WSHED } else { label }.to_ne_bytes());
        }
        markers.write_compact_bytes(&out)?;
        return Ok(());
    }

    let image_bytes = image.compact_bytes();
    let mut labels: Vec<i32> = marker_bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32")))
        .collect();

    // The image frame is always a watershed boundary.
    for column in 0..columns {
        labels[column] = WSHED;
        labels[(rows - 1) * columns + column] = WSHED;
    }
    for row in 0..rows {
        labels[row * columns] = WSHED;
        labels[row * columns + columns - 1] = WSHED;
    }

    // Interior pixels grouped by intensity level.
    let mut order: Vec<usize> = (0..rows * columns)
        .filter(|&index| {
            let row = index / columns;
            let column = index % columns;
            row > 0 && row + 1 < rows && column > 0 && column + 1 < columns
        })
        .collect();
    order.sort_by_key(|&index| image_bytes[index * 3]);

    // Axial (4-connected) neighbors of an interior pixel.
    let neighbor4 = |index: usize| -> [usize; 4] {
        let row = index / columns;
        let column = index % columns;
        [
            (row - 1) * columns + column,
            (row + 1) * columns + column,
            row * columns + column - 1,
            row * columns + column + 1,
        ]
    };

    // Count the distinct positive labels among axial neighbors; returns the
    // single label when exactly one is present.
    let distinct_labels = |labels: &[i32], index: usize| -> (usize, i32) {
        let mut count = 0usize;
        let mut single = 0i32;
        let mut seen = [0i32; 4];
        let mut seen_len = 0usize;
        for neighbor in neighbor4(index) {
            let label = labels[neighbor];
            if label <= 0 {
                continue;
            }
            if !seen[..seen_len].contains(&label) {
                seen[seen_len] = label;
                seen_len += 1;
                if count == 0 {
                    single = label;
                }
                count += 1;
            }
        }
        (count, single)
    };

    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut queued = vec![false; rows * columns];
    let mut cursor = 0usize;
    while cursor < order.len() {
        let level = image_bytes[order[cursor] * 3];
        let mut end = cursor;
        while end < order.len() && image_bytes[order[end] * 3] == level {
            end += 1;
        }
        let pixels = &order[cursor..end];

        // Seeds activated beside two or more already-established labels
        // (assigned at earlier levels) collapse to watershed pixels.
        for &index in pixels {
            if labels[index] <= 0 {
                continue;
            }
            let mut count = 0usize;
            let mut seen = [0i32; 4];
            let mut seen_len = 0usize;
            for neighbor in neighbor4(index) {
                let label = labels[neighbor];
                if label <= 0 || image_bytes[neighbor * 3] >= level {
                    continue;
                }
                if !seen[..seen_len].contains(&label) {
                    seen[seen_len] = label;
                    seen_len += 1;
                    count += 1;
                }
            }
            if count >= 2 {
                labels[index] = WSHED;
            }
        }

        // Frontier queue: every established label plus surviving level seeds,
        // seeded in scan order.
        queue.clear();
        queued.fill(false);
        for (index, label) in labels.iter().enumerate() {
            if *label > 0 {
                queued[index] = true;
                queue.push_back(index);
            }
        }

        while let Some(index) = queue.pop_front() {
            if labels[index] > 0 {
                for neighbor in neighbor4(index) {
                    if labels[neighbor] == 0
                        && image_bytes[neighbor * 3] == level
                        && !queued[neighbor]
                    {
                        queued[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            } else if labels[index] == 0 {
                let (count, single) = distinct_labels(&labels, index);
                if count >= 2 {
                    labels[index] = WSHED;
                } else if count == 1 {
                    labels[index] = single;
                    for neighbor in neighbor4(index) {
                        if labels[neighbor] == 0
                            && image_bytes[neighbor * 3] == level
                            && !queued[neighbor]
                        {
                            queued[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        cursor = end;
    }

    let mut out = Vec::with_capacity(marker_bytes.len());
    for label in &labels {
        out.extend_from_slice(&label.to_ne_bytes());
    }
    markers.write_compact_bytes(&out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_from_i32, mat_from_u8};

    fn i32_values(mat: &Mat) -> Vec<i32> {
        mat.compact_bytes()
            .chunks_exact(4)
            .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn run(intensities: &[u8], markers: &[i32], rows: u32, columns: u32) -> Vec<i32> {
        let mut image_data = Vec::with_capacity(intensities.len() * 3);
        for &value in intensities {
            image_data.extend_from_slice(&[value, 0, 0]);
        }
        let image = mat_from_u8(&image_data, rows, columns, 3).expect("image");
        let markers_mat = mat_from_i32(markers, rows, columns, 1).expect("markers");
        watershed_into(&image, &markers_mat).expect("watershed");
        i32_values(&markers_mat)
    }

    #[test]
    fn watershed_matches_pinned_two_seed_checkerboard() {
        // 4x4 image: even indices intensity 10, odd 200; seeds 1 at (1,1),(2,1)
        // and 2 at (1,2),(2,2). Pinned OpenCV.js keeps every seed.
        let intensities: Vec<u8> = (0..16).map(|i| if i % 2 == 0 { 10 } else { 200 }).collect();
        let markers: Vec<i32> = vec![
            0, 0, 0, 0,
            0, 1, 2, 0,
            0, 1, 2, 0,
            0, 0, 0, 0,
        ];
        let expected: Vec<i32> = vec![
            -1, -1, -1, -1,
            -1, 1, 2, -1,
            -1, 1, 2, -1,
            -1, -1, -1, -1,
        ];
        assert_eq!(run(&intensities, &markers, 4, 4), expected);
    }

    #[test]
    fn watershed_matches_pinned_wide_contest() {
        // 5x5 uniform image; seeds 1 at (1,1), 2 at (3,3). Pinned result puts
        // the boundary on the diagonal frontier between the regions.
        let intensities = vec![100u8; 25];
        let markers: Vec<i32> = vec![
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 2, 0,
            0, 0, 0, 0, 0,
        ];
        let expected: Vec<i32> = vec![
            -1, -1, -1, -1, -1,
            -1, 1, 1, -1, -1,
            -1, 1, -1, 2, -1,
            -1, -1, 2, 2, -1,
            -1, -1, -1, -1, -1,
        ];
        assert_eq!(run(&intensities, &markers, 5, 5), expected);
    }

    #[test]
    fn watershed_seed_beside_two_regions_collapses() {
        // Gradient levels 15/18/27/30; the last seed at (2,2) activates beside
        // labels 1 and 2 from earlier levels and becomes watershed.
        let intensities: Vec<u8> = vec![
            0, 0, 0, 0,
            0, 15, 18, 0,
            0, 27, 30, 0,
            0, 0, 0, 0,
        ];
        let markers: Vec<i32> = vec![
            0, 0, 0, 0,
            0, 1, 2, 0,
            0, 1, 2, 0,
            0, 0, 0, 0,
        ];
        let expected: Vec<i32> = vec![
            -1, -1, -1, -1,
            -1, 1, 2, -1,
            -1, 1, -1, -1,
            -1, -1, -1, -1,
        ];
        assert_eq!(run(&intensities, &markers, 4, 4), expected);
    }

    #[test]
    fn watershed_without_seeds_leaves_zero_and_frames() {
        let intensities = vec![7u8; 16];
        let markers = vec![0i32; 16];
        let expected: Vec<i32> = vec![
            -1, -1, -1, -1,
            -1, 0, 0, -1,
            -1, 0, 0, -1,
            -1, -1, -1, -1,
        ];
        assert_eq!(run(&intensities, &markers, 4, 4), expected);
    }
}
