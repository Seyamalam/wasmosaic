//! Drawing primitives: lines, rectangles, circles, ellipses, polygons, markers.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, write_scalar},
    imgproc_window::scalar_depth,
    mat::Mat,
};

#[derive(Debug)]
pub(crate) enum DrawError {
    InvalidInput(&'static str),
    OutputError(String),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(r) => write!(f, "invalid input: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}
impl Error for DrawError {}

fn put_pixel(buf: &mut [u8], cols: u32, ch: usize, depth: ScalarDepth, x: i64, y: i64, color: &[f64]) {
    if x < 0 || y < 0 || x >= i64::from(cols) || y * i64::from(cols) + x >= (buf.len() / depth.width() / ch) as i64 {
        return;
    }
    let rows = buf.len() / depth.width() / ch / cols as usize;
    if y >= rows as i64 {
        return;
    }
    for c in 0..ch {
        let idx = ((y as u32 * cols + x as u32) as usize) * ch + c;
        write_scalar(buf, idx, depth, color[c.min(color.len() - 1)]);
    }
}

/// Bresenham line rasterization (8-connected, inclusive endpoints).
fn draw_line_pixels(
    buf: &mut [u8],
    cols: u32,
    ch: usize,
    depth: ScalarDepth,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    color: &[f64],
) {
    let (mut x, mut y) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let (sx, sy) = ((x1 - x0).signum(), (y1 - y0).signum());
    put_pixel(buf, cols, ch, depth, x, y, color);
    if dx > dy {
        let mut err = dx / 2;
        while x != x1 {
            x += sx;
            err -= dy;
            if err < 0 {
                y += sy;
                err += dx;
            }
            put_pixel(buf, cols, ch, depth, x, y, color);
        }
    } else {
        let mut err = dy / 2;
        while y != y1 {
            y += sy;
            err -= dx;
            if err < 0 {
                x += sx;
                err += dy;
            }
            put_pixel(buf, cols, ch, depth, x, y, color);
        }
    }
}

/// Thick line: the segment widened into a filled axis-bounded strip.
#[allow(clippy::too_many_arguments)]
fn draw_thick_line(
    buf: &mut [u8],
    cols: u32,
    ch: usize,
    depth: ScalarDepth,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    color: &[f64],
    thickness: i32,
) {
    // Pinned behavior: thick segments fill a centered perpendicular band with
    // rounded caps. For axis-aligned lines the band is exact.
    let t = thickness.max(1);
    let half = (t - 1) / 2;
    if x0 == x1 || y0 == y1 {
        for yy in (y0.min(y1) - i64::from(half))..=(y0.max(y1) + i64::from(t - 1 - half)) {
            for xx in (x0.min(x1) - i64::from(half))..=(x0.max(x1) + i64::from(t - 1 - half)) {
                if x0 == x1 && (yy < y0.min(y1) || yy > y0.max(y1)) {
                    continue;
                }
                if y0 == y1 && (xx < x0.min(x1) || xx > x0.max(x1)) {
                    continue;
                }
                put_pixel(buf, cols, ch, depth, xx, yy, color);
            }
        }
    } else {
        // Oblique thick lines: rasterize the centerline plus perpendicular offsets.
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (nx, ny) = (-dy / len, dx / len);
        let half_f = t as f64 / 2.0;
        for offset_i in 0..t {
            let off = f64::from(offset_i) - (half_f - 0.5);
            let ox = (nx * off).round() as i64;
            let oy = (ny * off).round() as i64;
            draw_line_pixels(buf, cols, ch, depth, x0 + ox, y0 + oy, x1 + ox, y1 + oy, color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn line_into(
    image: &Mat,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    color: &[f64],
    thickness: i32,
    _line_type: i32,
    _shift: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    if thickness <= 1 {
        draw_line_pixels(&mut buf, image.columns(), image.channels() as usize, depth, x0, y0, x1, y1, color);
    } else {
        draw_thick_line(&mut buf, image.columns(), image.channels() as usize, depth, x0, y0, x1, y1, color, thickness);
    }
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rectangle_into(
    image: &Mat,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    color: &[f64],
    thickness: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    if thickness < 0 {
        for y in y0.min(y1)..=y0.max(y1) {
            for x in x0.min(x1)..=x0.max(x1) {
                put_pixel(&mut buf, cols, ch, depth, x, y, color);
            }
        }
    } else {
        for _ in 0..thickness.max(1) {
            draw_line_pixels(&mut buf, cols, ch, depth, x0, y0, x1, y0, color);
            draw_line_pixels(&mut buf, cols, ch, depth, x1, y0, x1, y1, color);
            draw_line_pixels(&mut buf, cols, ch, depth, x1, y1, x0, y1, color);
            draw_line_pixels(&mut buf, cols, ch, depth, x0, y1, x0, y0, color);
        }
    }
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// Midpoint circle rasterization.
fn draw_circle_pixels(
    buf: &mut [u8],
    cols: u32,
    ch: usize,
    depth: ScalarDepth,
    cx: i64,
    cy: i64,
    radius: i64,
    color: &[f64],
) {
    let (mut x, mut y) = (0i64, radius);
    let mut d = 1 - radius;
    while x <= y {
        for (px, py) in [
            (cx + x, cy + y), (cx - x, cy + y), (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x), (cx + y, cy - x), (cx - y, cy - x),
        ] {
            put_pixel(buf, cols, ch, depth, px, py, color);
        }
        if d < 0 {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y -= 1;
        }
        x += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn circle_into(
    image: &Mat,
    cx: i64,
    cy: i64,
    radius: i64,
    color: &[f64],
    thickness: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    if thickness < 0 {
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                let (dx, dy) = (x - cx, y - cy);
                if dx * dx + dy * dy <= radius * radius {
                    put_pixel(&mut buf, cols, ch, depth, x, y, color);
                }
            }
        }
    } else {
        draw_circle_pixels(&mut buf, cols, ch, depth, cx, cy, radius, color);
    }
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// Parametric ellipse rasterization for axis-aligned cases.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ellipse_into(
    image: &Mat,
    cx: f64,
    cy: f64,
    ax: f64,
    ay: f64,
    angle_deg: f64,
    start_deg: f64,
    end_deg: f64,
    color: &[f64],
    thickness: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    let a = angle_deg.to_radians();
    let (sa, ca) = a.sin_cos();
    let n = ((ax.max(ay) * 8.0) as i64).max(64);
    let (s, e) = (start_deg.to_radians(), end_deg.to_radians());
    let span = e - s;
    let mut prev: Option<(i64, i64)> = None;
    for i in 0..=n {
        let t = s + span * (i as f64 / n as f64);
        let (st, ct) = t.sin_cos();
        let x = cx + ax * ct * ca - ay * st * sa;
        let y = cy + ax * ct * sa + ay * st * ca;
        let (px, py) = (x.round() as i64, y.round() as i64);
        if let Some((lx, ly)) = prev {
            if (px, py) != (lx, ly) {
                draw_line_pixels(&mut buf, cols, ch, depth, lx, ly, px, py, color);
            }
        } else {
            put_pixel(&mut buf, cols, ch, depth, px, py, color);
        }
        prev = Some((px, py));
    }
    let _ = thickness;
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// Scanline polygon fill from integer vertex lists.
pub(crate) fn fill_poly_into(
    image: &Mat,
    points: &[i64],
    closed_polylines: bool,
    color: &[f64],
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols, rows) = (image.channels() as usize, image.columns(), image.rows());
    let n = points.len() / 2;
    if n < 2 {
        return Err(DrawError::InvalidInput("expected at least two vertices"));
    }
    let pts: Vec<(i64, i64)> = (0..n).map(|i| (points[i * 2], points[i * 2 + 1])).collect();
    let (ymin, ymax) = (pts.iter().map(|p| p.1).min().unwrap(), pts.iter().map(|p| p.1).max().unwrap());
    for y in ymin.max(0)..=ymax.min(i64::from(rows) - 1) {
        // Collect x-intersections of each edge with scanline y+0.5.
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..n {
            let (x1, y1) = pts[i];
            let (x2, y2) = pts[(i + 1) % n];
            if (y1 <= y && y2 > y) || (y2 <= y && y1 > y) {
                let t = (y - y1) as f64 / (y2 - y1) as f64;
                xs.push(x1 as f64 + t * (x2 - x1) as f64);
            }
        }
        xs.sort_by(f64::total_cmp);
        for pair in xs.chunks_exact(2) {
            let (xa, xb) = (pair[0].ceil() as i64, pair[1].floor() as i64);
            for x in xa..=xb {
                put_pixel(&mut buf, cols, ch, depth, x, y, color);
            }
        }
    }
    if closed_polylines {
        for i in 0..n {
            let (x1, y1) = pts[i];
            let (x2, y2) = pts[(i + 1) % n];
            draw_line_pixels(&mut buf, cols, ch, depth, x1, y1, x2, y2, color);
        }
    }
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// drawMarker: cross/X/star/diamond/square marker glyphs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_marker_into(
    image: &Mat,
    x: i64,
    y: i64,
    color: &[f64],
    marker_type: i32,
    marker_size: i32,
    thickness: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    let r = i64::from(marker_size.max(1)) / 2 - 1;
    match marker_type {
        1 => {
            draw_line_pixels(&mut buf, cols, ch, depth, x - r, y - r, x + r, y + r, color);
            draw_line_pixels(&mut buf, cols, ch, depth, x + r, y - r, x - r, y + r, color);
        }
        _ => {
            draw_line_pixels(&mut buf, cols, ch, depth, x, y - r, x, y + r, color);
            draw_line_pixels(&mut buf, cols, ch, depth, x - r, y, x + r, y, color);
        }
    }
    let _ = thickness;
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// polylines: connect successive vertices, optionally closing the loop.
pub(crate) fn polylines_into(
    image: &Mat,
    points: &[i64],
    closed: bool,
    color: &[f64],
    thickness: i32,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    let n = points.len() / 2;
    let edges = if closed { n } else { n.saturating_sub(1) };
    for i in 0..edges {
        let (x1, y1) = (points[i * 2], points[i * 2 + 1]);
        let j = (i + 1) % n;
        let (x2, y2) = (points[j * 2], points[j * 2 + 1]);
        if thickness <= 1 {
            draw_line_pixels(&mut buf, cols, ch, depth, x1, y1, x2, y2, color);
        } else {
            draw_thick_line(&mut buf, cols, ch, depth, x1, y1, x2, y2, color, thickness);
        }
    }
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// arrowedLine: a line plus a triangular arrowhead at the end point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn arrowed_line_into(
    image: &Mat,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    color: &[f64],
    thickness: i32,
    _line_type: i32,
    _shift: i32,
    tip_length: f64,
) -> Result<(), DrawError> {
    let depth = scalar_depth(image.depth());
    let mut buf = image.compact_bytes();
    let (ch, cols) = (image.channels() as usize, image.columns());
    draw_line_pixels(&mut buf, cols, ch, depth, x0, y0, x1, y1, color);
    // Arrowhead: two short lines fanning back from the tip.
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = ((dx * dx + dy * dy) as f64).sqrt().max(1e-9);
    let head = if tip_length > 0.0 { tip_length * len } else { len * 0.1 };
    let (ux, uy) = (dx as f64 / len, dy as f64 / len);
    // Fan angle ~30 degrees each side.
    let fan = 0.577350269; // tan(30)
    let (nx, ny) = (-uy, ux);
    let (bx, by) = (x1 as f64 - ux * head, y1 as f64 - uy * head);
    draw_line_pixels(
        &mut buf, cols, ch, depth,
        (bx + nx * head * fan).round() as i64,
        (by + ny * head * fan).round() as i64,
        x1, y1, color,
    );
    draw_line_pixels(
        &mut buf, cols, ch, depth,
        (bx - nx * head * fan).round() as i64,
        (by - ny * head * fan).round() as i64,
        x1, y1, color,
    );
    let _ = thickness;
    image
        .write_output(buf, image.rows(), image.columns(), image.channels(), image.depth())
        .map_err(|e| DrawError::OutputError(e.to_string()))?;
    Ok(())
}

/// fillConvexPoly: scanline fill specialized for convex vertex sets.
pub(crate) fn fill_convex_poly_into(
    image: &Mat,
    points: &[i64],
    color: &[f64],
) -> Result<(), DrawError> {
    fill_poly_into(image, points, true, color)
}
