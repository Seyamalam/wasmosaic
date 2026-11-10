//! Small, independently authored image-processing geometry and window helpers.

use std::{error::Error, fmt};

pub(crate) const MORPH_RECT: i32 = 0;
pub(crate) const MORPH_CROSS: i32 = 1;
pub(crate) const MORPH_ELLIPSE: i32 = 2;
pub(crate) const MORPH_DIAMOND: i32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HelperError {
    EmptySize,
    WindowTooSmall,
    SizeOverflow,
    InvalidShape(i32),
    InvalidAxes {
        x: i32,
        y: i32,
    },
    InvalidArc {
        start: i32,
        end: i32,
    },
    InvalidAngularStep(i32),
    CoordinateOverflow,
    InvalidRectangle {
        width: u32,
        height: u32,
    },
    InvalidAnchor {
        x: i32,
        y: i32,
        columns: u32,
        rows: u32,
    },
}

impl fmt::Display for HelperError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySize => formatter.write_str("dimensions must be greater than zero"),
            Self::WindowTooSmall => {
                formatter.write_str("Hanning window dimensions must each be at least two")
            }
            Self::SizeOverflow => formatter.write_str("output exceeds the WASM size limit"),
            Self::InvalidShape(shape) => write!(formatter, "unsupported structuring shape {shape}"),
            Self::InvalidAxes { x, y } => {
                write!(
                    formatter,
                    "ellipse axes must be non-negative; received ({x}, {y})"
                )
            }
            Self::InvalidArc { start, end } => write!(
                formatter,
                "ellipse arc must satisfy 0 <= start <= end <= 360; received {start} to {end}"
            ),
            Self::InvalidAngularStep(delta) => write!(
                formatter,
                "ellipse angular step must be in 1..=180; received {delta}"
            ),
            Self::CoordinateOverflow => formatter
                .write_str("generated point lies outside the signed 32-bit coordinate range"),
            Self::InvalidRectangle { width, height } => write!(
                formatter,
                "clipping rectangle dimensions must be positive; received {width} by {height}"
            ),
            Self::InvalidAnchor {
                x,
                y,
                columns,
                rows,
            } => write!(
                formatter,
                "anchor ({x}, {y}) lies outside the {columns} by {rows} kernel"
            ),
        }
    }
}

pub(crate) fn hanning_window(columns: u32, rows: u32) -> Result<Vec<f64>, HelperError> {
    if columns < 2 || rows < 2 {
        return Err(HelperError::WindowTooSmall);
    }
    let length = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(columns)
                .ok()
                .and_then(|columns| rows.checked_mul(columns))
        })
        .and_then(|length| length.checked_mul(size_of::<f64>()).map(|_| length))
        .filter(|&length| length <= u32::MAX as usize / size_of::<f64>())
        .ok_or(HelperError::SizeOverflow)?;

    let mut output = Vec::with_capacity(length);
    for row in 0..rows {
        let vertical = hanning_coefficient(row, rows);
        for column in 0..columns {
            output.push((vertical * hanning_coefficient(column, columns)).sqrt());
        }
    }
    Ok(output)
}

fn hanning_coefficient(index: u32, length: u32) -> f64 {
    let angle = std::f64::consts::TAU * f64::from(index) / f64::from(length - 1);
    0.5 * (1.0 - angle.cos())
}

impl Error for HelperError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Point {
    pub(crate) x: i32,
    pub(crate) y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

pub(crate) fn clip_line(
    rectangle: Rect,
    start: Point,
    end: Point,
) -> Result<Option<(Point, Point)>, HelperError> {
    if rectangle.width == 0 || rectangle.height == 0 {
        return Err(HelperError::InvalidRectangle {
            width: rectangle.width,
            height: rectangle.height,
        });
    }
    let left = rectangle.x;
    let top = rectangle.y;
    let right = i32::try_from(i64::from(left) + i64::from(rectangle.width) - 1)
        .map_err(|_| HelperError::CoordinateOverflow)?;
    let bottom = i32::try_from(i64::from(top) + i64::from(rectangle.height) - 1)
        .map_err(|_| HelperError::CoordinateOverflow)?;

    let x0 = f64::from(start.x);
    let y0 = f64::from(start.y);
    let delta_x = f64::from(end.x) - x0;
    let delta_y = f64::from(end.y) - y0;
    let mut lower = 0.0;
    let mut upper = 1.0;
    for (direction, distance) in [
        (-delta_x, x0 - f64::from(left)),
        (delta_x, f64::from(right) - x0),
        (-delta_y, y0 - f64::from(top)),
        (delta_y, f64::from(bottom) - y0),
    ] {
        if !clip_parameter(direction, distance, &mut lower, &mut upper) {
            return Ok(None);
        }
    }

    let clipped_start = Point {
        x: rounded_clamped_coordinate(x0 + lower * delta_x, left, right),
        y: rounded_clamped_coordinate(y0 + lower * delta_y, top, bottom),
    };
    let clipped_end = Point {
        x: rounded_clamped_coordinate(x0 + upper * delta_x, left, right),
        y: rounded_clamped_coordinate(y0 + upper * delta_y, top, bottom),
    };
    Ok(Some((clipped_start, clipped_end)))
}

fn clip_parameter(direction: f64, distance: f64, lower: &mut f64, upper: &mut f64) -> bool {
    if direction == 0.0 {
        return distance >= 0.0;
    }
    let ratio = distance / direction;
    if direction < 0.0 {
        if ratio > *upper {
            return false;
        }
        *lower = lower.max(ratio);
    } else {
        if ratio < *lower {
            return false;
        }
        *upper = upper.min(ratio);
    }
    true
}

#[allow(clippy::cast_possible_truncation)]
fn rounded_clamped_coordinate(value: f64, minimum: i32, maximum: i32) -> i32 {
    value
        .round_ties_even()
        .clamp(f64::from(minimum), f64::from(maximum)) as i32
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ellipse_poly(
    center: Point,
    axis_x: i32,
    axis_y: i32,
    rotation_degrees: i32,
    arc_start: i32,
    arc_end: i32,
    delta: i32,
) -> Result<Vec<Point>, HelperError> {
    if axis_x < 0 || axis_y < 0 {
        return Err(HelperError::InvalidAxes {
            x: axis_x,
            y: axis_y,
        });
    }
    let (arc_start, arc_end) = if arc_start <= arc_end {
        (arc_start, arc_end)
    } else {
        (arc_end, arc_start)
    };
    if !(0..=360).contains(&arc_start) || !(arc_start..=360).contains(&arc_end) {
        return Err(HelperError::InvalidArc {
            start: arc_start,
            end: arc_end,
        });
    }
    if !(1..=180).contains(&delta) {
        return Err(HelperError::InvalidAngularStep(delta));
    }
    if arc_start == arc_end {
        return Ok(vec![Point { x: 0, y: 0 }, Point { x: 0, y: 0 }]);
    }

    let rotation = f64::from(rotation_degrees.rem_euclid(360)).to_radians();
    let (rotation_sine, rotation_cosine) = rotation.sin_cos();
    let capacity = usize::try_from((arc_end - arc_start) / delta + 2)
        .expect("validated arc point count is positive and bounded");
    let mut points = Vec::with_capacity(capacity);
    let mut degrees = arc_start;
    loop {
        let angle = f64::from(degrees).to_radians();
        let (sine, cosine) = angle.sin_cos();
        let local_x = f64::from(axis_x) * cosine;
        let local_y = f64::from(axis_y) * sine;
        let x = f64::from(center.x) + local_x * rotation_cosine - local_y * rotation_sine;
        let y = f64::from(center.y) + local_x * rotation_sine + local_y * rotation_cosine;
        let point = Point {
            x: checked_i32_coordinate(x)?,
            y: checked_i32_coordinate(y)?,
        };
        if points.last() != Some(&point) {
            points.push(point);
        }
        if degrees == arc_end {
            break;
        }
        degrees = degrees.saturating_add(delta).min(arc_end);
    }
    if points.len() == 1 {
        points.push(points[0]);
    }
    Ok(points)
}

#[allow(clippy::cast_possible_truncation)]
fn checked_i32_coordinate(value: f64) -> Result<i32, HelperError> {
    let rounded = value.round_ties_even();
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        Err(HelperError::CoordinateOverflow)
    } else {
        Ok(rounded as i32)
    }
}

pub(crate) struct StructuringElement {
    pub(crate) bytes: Vec<u8>,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
}

pub(crate) fn structuring_element(
    shape: i32,
    columns: u32,
    rows: u32,
    anchor_x: i32,
    anchor_y: i32,
) -> Result<StructuringElement, HelperError> {
    if rows == 0 || columns == 0 {
        return Err(HelperError::EmptySize);
    }
    if !matches!(
        shape,
        MORPH_RECT | MORPH_CROSS | MORPH_ELLIPSE | MORPH_DIAMOND
    ) {
        return Err(HelperError::InvalidShape(shape));
    }
    let normalized_anchor_x = normalize_anchor(anchor_x, columns);
    let normalized_anchor_y = normalize_anchor(anchor_y, rows);
    let (Some(anchor_x), Some(anchor_y)) = (normalized_anchor_x, normalized_anchor_y) else {
        return Err(HelperError::InvalidAnchor {
            x: anchor_x,
            y: anchor_y,
            columns,
            rows,
        });
    };
    let length = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(columns)
                .ok()
                .and_then(|columns| rows.checked_mul(columns))
        })
        .filter(|&length| u32::try_from(length).is_ok())
        .ok_or(HelperError::SizeOverflow)?;
    let mut bytes = vec![0; length];

    match shape {
        MORPH_RECT => bytes.fill(1),
        MORPH_CROSS => {
            let columns_usize = columns as usize;
            for row in 0..rows {
                if row == anchor_y {
                    let start = row as usize * columns_usize;
                    bytes[start..start + columns_usize].fill(1);
                } else {
                    bytes[row as usize * columns_usize + anchor_x as usize] = 1;
                }
            }
        }
        MORPH_ELLIPSE => fill_ellipse(&mut bytes, columns, rows),
        MORPH_DIAMOND => fill_diamond(&mut bytes, columns, rows),
        _ => unreachable!("shape was validated"),
    }

    Ok(StructuringElement {
        bytes,
        rows,
        columns,
    })
}

fn fill_diamond(bytes: &mut [u8], columns: u32, rows: u32) {
    let center_x = columns / 2;
    let center_y = rows / 2;
    let columns_usize = columns as usize;
    for row in 0..rows {
        let vertical_distance = row.abs_diff(center_y);
        let half_width = center_x.saturating_sub(vertical_distance);
        let first = center_x - half_width;
        let last = (center_x + half_width).min(columns - 1);
        let start = row as usize * columns_usize;
        bytes[start + first as usize..=start + last as usize].fill(1);
    }
}

fn normalize_anchor(value: i32, size: u32) -> Option<u32> {
    if value == -1 {
        return Some(size / 2);
    }
    u32::try_from(value)
        .ok()
        .filter(|&coordinate| coordinate < size)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn fill_ellipse(bytes: &mut [u8], columns: u32, rows: u32) {
    let radius_y = f64::from(rows / 2);
    let radius_x = f64::from(columns / 2);
    let center_y = i64::from(rows / 2);
    let center_x = i64::from(columns / 2);
    let columns_usize = columns as usize;

    for row in 0..rows {
        let offset_y = i64::from(row) - center_y;
        let half_width = if radius_y == 0.0 {
            radius_x
        } else {
            let normalized_y = offset_y as f64 / radius_y;
            radius_x * (1.0 - normalized_y * normalized_y).max(0.0).sqrt()
        };
        let half_width = half_width.round_ties_even() as i64;
        let first = (center_x - half_width).max(0) as usize;
        let last = (center_x + half_width).min(i64::from(columns) - 1) as usize;
        let start = row as usize * columns_usize;
        bytes[start + first..=start + last].fill(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structuring_elements_match_worked_rect_cross_and_ellipse_fixtures() {
        let rect = structuring_element(MORPH_RECT, 3, 2, -1, -1).expect("valid rectangle");
        assert_eq!(rect.bytes, [1, 1, 1, 1, 1, 1]);

        let cross = structuring_element(MORPH_CROSS, 3, 3, 0, 1).expect("valid cross");
        assert_eq!(cross.bytes, [1, 0, 0, 1, 1, 1, 1, 0, 0]);

        let ellipse = structuring_element(MORPH_ELLIPSE, 5, 5, -1, -1).expect("valid ellipse");
        assert_eq!(
            ellipse.bytes,
            [
                0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0
            ]
        );

        let diamond = structuring_element(MORPH_DIAMOND, 5, 5, -1, -1).expect("valid diamond");
        assert_eq!(
            diamond.bytes,
            [
                0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0
            ]
        );
    }

    #[test]
    fn hanning_window_matches_the_pinned_cosine_square_root_bits() {
        let window = hanning_window(4, 3).expect("valid window");
        assert_eq!(
            window.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
            [
                0,
                0,
                0,
                0,
                0,
                0x3FEB_B67A_E858_4CAA,
                0x3FEB_B67A_E858_4CAC,
                0,
                0,
                0,
                0,
                0,
            ]
        );
    }

    #[test]
    fn ellipse_poly_samples_a_rotated_arc_and_keeps_the_exact_end_angle() {
        let points = ellipse_poly(Point { x: 3, y: -2 }, 10, 5, 0, 0, 90, 45).expect("valid arc");
        assert_eq!(
            points,
            [
                Point { x: 13, y: -2 },
                Point { x: 10, y: 2 },
                Point { x: 3, y: 3 }
            ]
        );

        let degenerate =
            ellipse_poly(Point { x: 0, y: 0 }, 10, 5, 90, 0, 0, 10).expect("valid empty arc");
        assert_eq!(degenerate, [Point { x: 0, y: 0 }, Point { x: 0, y: 0 }]);
    }

    #[test]
    fn clip_line_uses_the_last_pixel_as_the_inclusive_rectangle_edge() {
        let rect = Rect {
            x: 10,
            y: 20,
            width: 5,
            height: 4,
        };
        assert_eq!(
            clip_line(rect, Point { x: 8, y: 21 }, Point { x: 16, y: 21 })
                .expect("valid rectangle"),
            Some((Point { x: 10, y: 21 }, Point { x: 14, y: 21 }))
        );
        assert_eq!(
            clip_line(rect, Point { x: 8, y: 18 }, Point { x: 14, y: 24 })
                .expect("valid rectangle"),
            Some((Point { x: 10, y: 20 }, Point { x: 13, y: 23 }))
        );
        assert_eq!(
            clip_line(rect, Point { x: 0, y: 0 }, Point { x: 1, y: 1 }).expect("valid rectangle"),
            None
        );
    }

    #[test]
    fn helpers_reject_invalid_geometry_before_allocating_output() {
        assert!(matches!(
            structuring_element(MORPH_RECT, 3, 3, 3, 0),
            Err(HelperError::InvalidAnchor { .. })
        ));
        assert!(matches!(
            structuring_element(MORPH_RECT, u32::MAX, 2, -1, -1),
            Err(HelperError::SizeOverflow)
        ));
        assert_eq!(hanning_window(1, 3), Err(HelperError::WindowTooSmall));
        assert!(matches!(
            ellipse_poly(Point { x: 0, y: 0 }, 3, 2, 0, 370, 400, 1),
            Err(HelperError::InvalidArc { .. })
        ));
        assert_eq!(
            ellipse_poly(Point { x: 0, y: 0 }, 3, 2, 0, 0, 90, 0),
            Err(HelperError::InvalidAngularStep(0))
        );
        assert_eq!(
            ellipse_poly(Point { x: i32::MAX, y: 0 }, 1, 1, 0, 0, 90, 1),
            Err(HelperError::CoordinateOverflow)
        );
        assert_eq!(
            clip_line(
                Rect {
                    x: i32::MAX,
                    y: 0,
                    width: 2,
                    height: 1,
                },
                Point { x: 0, y: 0 },
                Point { x: 1, y: 0 },
            ),
            Err(HelperError::CoordinateOverflow)
        );
    }
}
