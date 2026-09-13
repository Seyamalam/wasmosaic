use std::{collections::VecDeque, error::Error, fmt};

use crate::{
    mat::{Mat, MatDepth, MatError},
    mat_vector::MatVector,
};

pub(crate) const RETR_EXTERNAL: i32 = 0;
pub(crate) const RETR_LIST: i32 = 1;
pub(crate) const RETR_CCOMP: i32 = 2;
pub(crate) const RETR_TREE: i32 = 3;
pub(crate) const CHAIN_APPROX_NONE: i32 = 1;
pub(crate) const CHAIN_APPROX_SIMPLE: i32 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FindContoursError {
    InvalidMethod(i32),
    InvalidMode(i32),
    Matrix(MatError),
    RequiresSingleChannel,
    UnsupportedDepth(MatDepth),
}

impl fmt::Display for FindContoursError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMethod(value) => write!(formatter, "unsupported contour method {value}"),
            Self::InvalidMode(value) => write!(formatter, "unsupported contour mode {value}"),
            Self::Matrix(error) => error.fmt(formatter),
            Self::RequiresSingleChannel => {
                formatter.write_str("findContours requires a single-channel source")
            }
            Self::UnsupportedDepth(depth) => {
                write!(
                    formatter,
                    "findContours source depth {depth:?} is not implemented"
                )
            }
        }
    }
}

impl Error for FindContoursError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for FindContoursError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn find_contours_into(
    source: &Mat,
    contours: &MatVector,
    hierarchy: &Mat,
    mode: i32,
    method: i32,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), FindContoursError> {
    if source.depth() != MatDepth::U8 {
        return Err(FindContoursError::UnsupportedDepth(source.depth()));
    }
    if source.channels() != 1 {
        return Err(FindContoursError::RequiresSingleChannel);
    }
    if !matches!(mode, RETR_EXTERNAL | RETR_LIST | RETR_CCOMP | RETR_TREE) {
        return Err(FindContoursError::InvalidMode(mode));
    }
    if !matches!(method, CHAIN_APPROX_NONE | CHAIN_APPROX_SIMPLE) {
        return Err(FindContoursError::InvalidMethod(method));
    }

    if source.rows() == 0 || source.columns() == 0 {
        hierarchy.write_output(Vec::new(), 0, 0, 1, MatDepth::U8)?;
        contours.replace(Vec::new());
        return Ok(());
    }

    // A zero border joins every exterior background pixel into region zero.
    // Foreground uses eight-connectivity; background uses four-connectivity,
    // so diagonal contacts do not incorrectly join an enclosed hole to outside.
    let compact = source.compact_bytes();
    let rows = (source.rows() as usize)
        .checked_add(2)
        .ok_or(MatError::BufferSizeOverflow)?;
    let columns = (source.columns() as usize)
        .checked_add(2)
        .ok_or(MatError::BufferSizeOverflow)?;
    let length = rows
        .checked_mul(columns)
        .ok_or(MatError::BufferSizeOverflow)?;
    let mut input = vec![0; length];
    for row in 0..source.rows() as usize {
        let width = source.columns() as usize;
        input[(row + 1) * columns + 1..(row + 1) * columns + 1 + width]
            .copy_from_slice(&compact[row * width..(row + 1) * width]);
    }
    let (labels, regions) = label_regions(&input, rows, columns);
    let (order, hierarchy_values) = retrieval_layout(&labels, &regions, mode)?;
    let mut found = Vec::with_capacity(order.len());
    for id in order {
        let region = &regions[id];
        let x = i32::try_from(region.seed % columns).map_err(|_| MatError::BufferSizeOverflow)?;
        let y = i32::try_from(region.seed / columns).map_err(|_| MatError::BufferSizeOverflow)?;
        let (start, backtrack) = if region.filled {
            ((x, y), (x - 1, y))
        } else {
            ((x - 1, y), (x, y))
        };
        let mut points = trace_boundary(&input, rows, columns, start, backtrack);
        if method == CHAIN_APPROX_SIMPLE {
            points = simplify_chain(&points);
        }
        let mut bytes = Vec::with_capacity(points.len() * 8);
        for (x, y) in points {
            bytes.extend_from_slice(&((x - 1).wrapping_add(offset_x)).to_ne_bytes());
            bytes.extend_from_slice(&((y - 1).wrapping_add(offset_y)).to_ne_bytes());
        }
        let count = u32::try_from(bytes.len() / 8).map_err(|_| MatError::BufferSizeOverflow)?;
        found.push(Mat::from_owned_bytes(bytes, count, 1, 2, MatDepth::I32)?);
    }
    if hierarchy_values.is_empty() {
        hierarchy.write_output(Vec::new(), 0, 0, 1, MatDepth::U8)?;
    } else {
        let mut hierarchy_bytes = Vec::with_capacity(hierarchy_values.len() * 4);
        for value in hierarchy_values {
            hierarchy_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        hierarchy.write_output(
            hierarchy_bytes,
            1,
            u32::try_from(found.len()).map_err(|_| MatError::BufferSizeOverflow)?,
            4,
            MatDepth::I32,
        )?;
    }
    contours.replace(found);
    Ok(())
}

struct Region {
    seed: usize,
    filled: bool,
}

fn retrieval_layout(
    labels: &[usize],
    regions: &[Region],
    mode: i32,
) -> Result<(Vec<usize>, Vec<i32>), MatError> {
    let mut children = vec![Vec::new(); regions.len()];
    let mut parents = vec![0; regions.len()];
    for id in 1..regions.len() {
        let region = &regions[id];
        let enclosing = labels[region.seed - 1];
        if mode == RETR_EXTERNAL && enclosing != 0 {
            continue;
        }
        let parent = if mode == RETR_LIST || (mode == RETR_CCOMP && region.filled) {
            0
        } else {
            enclosing
        };
        parents[id] = parent;
        children[parent].push(id);
    }
    // Reverse discovery order among siblings, then visit parents before children.
    let mut pending = children[0].clone();
    let mut order = Vec::new();
    while let Some(id) = pending.pop() {
        order.push(id);
        pending.extend_from_slice(&children[id]);
    }
    let mut positions = vec![-1; regions.len()];
    for (position, &id) in order.iter().enumerate() {
        positions[id] = i32::try_from(position).map_err(|_| MatError::BufferSizeOverflow)?;
    }
    let mut hierarchy_rows = vec![[-1; 4]; order.len()];
    for siblings in &children {
        for (index, &id) in siblings.iter().rev().enumerate() {
            let position =
                usize::try_from(positions[id]).expect("retrieved region has an output position");
            hierarchy_rows[position] = [
                if index + 1 < siblings.len() {
                    positions[siblings[siblings.len() - index - 2]]
                } else {
                    -1
                },
                if index == 0 {
                    -1
                } else {
                    positions[siblings[siblings.len() - index]]
                },
                children[id].last().map_or(-1, |&child| positions[child]),
                positions[parents[id]],
            ];
        }
    }
    let hierarchy_values = hierarchy_rows.into_iter().flatten().collect::<Vec<_>>();
    Ok((order, hierarchy_values))
}

fn label_regions(input: &[u8], rows: usize, columns: usize) -> (Vec<usize>, Vec<Region>) {
    let mut labels = vec![usize::MAX; input.len()];
    let mut regions = Vec::new();
    let mut queue = VecDeque::new();
    for seed in 0..input.len() {
        if labels[seed] != usize::MAX {
            continue;
        }
        let filled = input[seed] != 0;
        let id = regions.len();
        regions.push(Region { seed, filled });
        labels[seed] = id;
        queue.push_back(seed);
        while let Some(index) = queue.pop_front() {
            let (row, column) = (index / columns, index % columns);
            for y in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                for x in column.saturating_sub(1)..=(column + 1).min(columns - 1) {
                    if !filled && y != row && x != column {
                        continue;
                    }
                    let neighbor = y * columns + x;
                    if labels[neighbor] == usize::MAX && (input[neighbor] != 0) == filled {
                        labels[neighbor] = id;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    (labels, regions)
}

fn trace_boundary(
    input: &[u8],
    rows: usize,
    columns: usize,
    start: (i32, i32),
    mut backtrack: (i32, i32),
) -> Vec<(i32, i32)> {
    let mut points = vec![start];
    let mut current = start;
    let Some((first, first_backtrack)) = next_boundary(input, rows, columns, current, backtrack)
    else {
        return points;
    };
    current = first;
    backtrack = first_backtrack;
    let maximum_steps = input.len().saturating_mul(8).max(8);
    for _ in 0..maximum_steps {
        let Some((next, next_backtrack)) = next_boundary(input, rows, columns, current, backtrack)
        else {
            break;
        };
        if current == start && next == first {
            break;
        }
        points.push(current);
        current = next;
        backtrack = next_backtrack;
    }
    points
}

fn next_boundary(
    input: &[u8],
    rows: usize,
    columns: usize,
    current: (i32, i32),
    backtrack: (i32, i32),
) -> Option<((i32, i32), (i32, i32))> {
    const DIRECTIONS: [(i32, i32); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];
    let relative = (backtrack.0 - current.0, backtrack.1 - current.1);
    let start_index = DIRECTIONS
        .iter()
        .position(|&direction| direction == relative)
        .unwrap_or(0);
    for scan in 0..8 {
        let direction_index = (start_index + scan) % 8;
        let direction = DIRECTIONS[direction_index];
        let candidate = (current.0 + direction.0, current.1 + direction.1);
        if foreground(input, rows, columns, candidate) {
            let previous = DIRECTIONS[(direction_index + 7) % 8];
            return Some((candidate, (current.0 + previous.0, current.1 + previous.1)));
        }
    }
    None
}

fn foreground(input: &[u8], rows: usize, columns: usize, point: (i32, i32)) -> bool {
    let (Ok(x), Ok(y)) = (usize::try_from(point.0), usize::try_from(point.1)) else {
        return false;
    };
    y < rows && x < columns && input[y * columns + x] != 0
}

fn simplify_chain(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut output = Vec::new();
    for index in 0..points.len() {
        let previous = points[(index + points.len() - 1) % points.len()];
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        let incoming = (
            (current.0 - previous.0).signum(),
            (current.1 - previous.1).signum(),
        );
        let outgoing = ((next.0 - current.0).signum(), (next.1 - current.1).signum());
        if incoming != outgoing {
            output.push(current);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mat::mat_from_u8, mat_vector::mat_vector_new};

    fn values(matrix: &Mat) -> Vec<i32> {
        matrix
            .compact_bytes()
            .chunks_exact(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn nested() -> Mat {
        let pixels = (0..121)
            .map(|i| {
                let (x, y) = (i % 11, i / 11);
                u8::from(
                    (1..=9).contains(&x)
                        && (1..=9).contains(&y)
                        && (!(3..=7).contains(&x) || !(3..=7).contains(&y) || (x == 5 && y == 5)),
                )
            })
            .collect::<Vec<_>>();
        mat_from_u8(&pixels, 11, 11, 1).unwrap()
    }
    #[test]
    fn retrieval_modes_preserve_nested_holes_and_islands() {
        let source = nested();
        for (mode, expected) in [
            (RETR_EXTERNAL, vec![-1, -1, -1, -1]),
            (RETR_TREE, vec![-1, -1, 1, -1, -1, -1, 2, 0, -1, -1, -1, 1]),
            (RETR_CCOMP, vec![1, -1, -1, -1, -1, 0, 2, -1, -1, -1, -1, 1]),
            (RETR_LIST, vec![1, -1, -1, -1, 2, 0, -1, -1, -1, 1, -1, -1]),
        ] {
            let contours = mat_vector_new();
            let hierarchy = crate::mat::mat_empty();
            find_contours_into(
                &source,
                &contours,
                &hierarchy,
                mode,
                CHAIN_APPROX_SIMPLE,
                0,
                0,
            )
            .unwrap();
            assert_eq!(values(&hierarchy), expected);
            if mode == RETR_TREE {
                assert_eq!(
                    values(&contours.get(1).unwrap()),
                    [2, 3, 3, 2, 7, 2, 8, 3, 8, 7, 7, 8, 3, 8, 2, 7]
                );
                assert_eq!(values(&contours.get(2).unwrap()), [5, 5]);
            }
        }
    }
    #[test]
    fn source_hierarchy_alias_uses_input_snapshot() {
        let source = nested();
        let contours = mat_vector_new();
        find_contours_into(
            &source,
            &contours,
            &source,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            -2,
            7,
        )
        .unwrap();
        assert_eq!(contours.size(), 3);
        assert_eq!(values(&contours.get(2).unwrap()), [3, 12]);
        assert_eq!(
            values(&source),
            [-1, -1, 1, -1, -1, -1, 2, 0, -1, -1, -1, 1]
        );
    }
    #[test]
    fn empty_input_clears_previous_outputs_and_invalid_mode_preserves_them() {
        let source = nested();
        let contours = mat_vector_new();
        let hierarchy = crate::mat::mat_empty();
        find_contours_into(
            &source,
            &contours,
            &hierarchy,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            0,
            0,
        )
        .unwrap();
        let saved = hierarchy.compact_bytes();
        assert!(
            find_contours_into(&source, &contours, &hierarchy, 4, CHAIN_APPROX_SIMPLE, 0, 0)
                .is_err()
        );
        assert_eq!(hierarchy.compact_bytes(), saved);
        assert_eq!(contours.size(), 3);
        find_contours_into(
            &crate::mat::mat_empty(),
            &contours,
            &hierarchy,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            0,
            0,
        )
        .unwrap();
        assert_eq!(contours.size(), 0);
        assert_eq!(hierarchy.rows(), 0);
    }
    #[test]
    fn simple_external_rectangle_matches_browser_point_order() {
        let source = mat_from_u8(
            &[
                0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0,
                0, 0, 0,
            ],
            5,
            5,
            1,
        )
        .expect("source");
        let contours = mat_vector_new();
        let hierarchy = crate::mat::mat_empty();
        find_contours_into(
            &source,
            &contours,
            &hierarchy,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            0,
            0,
        )
        .expect("contours");
        assert_eq!(contours.size(), 1);
        let contour = contours.get(0).expect("first contour");
        let points = contour
            .compact_bytes()
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
            .collect::<Vec<_>>();
        assert_eq!(points, [1, 1, 1, 3, 3, 3, 3, 1]);
        let hierarchy_values = hierarchy
            .compact_bytes()
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
            .collect::<Vec<_>>();
        assert_eq!(hierarchy_values, [-1, -1, -1, -1]);
    }
}
