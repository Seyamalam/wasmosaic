//! Map-based resampling, sub-pixel extraction, demosaicing, and histograms.
use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar, write_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth},
};

#[derive(Debug)]
pub(crate) enum MapsError {
    UnsupportedDepth(MatDepth),
    InvalidInput(&'static str),
    OutputError(String),
}

impl fmt::Display for MapsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDepth(d) => write!(f, "unsupported matrix depth {d:?}"),
            Self::InvalidInput(r) => write!(f, "invalid input: {r}"),
            Self::OutputError(m) => write!(f, "{m}"),
        }
    }
}
impl Error for MapsError {}

const INTER_NEAREST: i32 = 0;
const INTER_LINEAR: i32 = 1;

/// remap: dst(y,x) = src(map2(y,x), map1(y,x)) with NN/bilinear sampling.
pub(crate) fn remap_into(
    source: &Mat,
    destination: &Mat,
    map1: &Mat,
    map2: &Mat,
    interpolation: i32,
    border_value: f64,
) -> Result<(), MapsError> {
    let depth = scalar_depth(source.depth());
    if map1.depth() != MatDepth::F32 || map2.depth() != MatDepth::F32 {
        return Err(MapsError::UnsupportedDepth(map1.depth()));
    }
    if map1.rows() != map2.rows() || map1.columns() != map2.columns() {
        return Err(MapsError::InvalidInput("map sizes differ"));
    }
    let input = source.compact_bytes();
    let m1 = map1.compact_bytes();
    let m2 = map2.compact_bytes();
    let (sr, sc, ch) = (source.rows(), source.columns(), source.channels() as usize);
    let (or, oc) = (map1.rows(), map1.columns());
    let n = or as usize * oc as usize;
    let mut output = vec![0u8; n * ch * depth.width()];
    let sample = |fy: f32, fx: f32, c: usize| -> f64 {
        match interpolation {
            INTER_NEAREST => {
                let (iy, ix) = (fy.round() as i64, fx.round() as i64);
                if iy < 0 || iy >= i64::from(sr) || ix < 0 || ix >= i64::from(sc) {
                    border_value
                } else {
                    read_scalar(&input, ((iy as u32 * sc + ix as u32) as usize) * ch + c, depth)
                }
            }
            _ => {
                let (y0, x0) = (fy.floor() as i64, fx.floor() as i64);
                let (wy, wx) = (f64::from(fy) - y0 as f64, f64::from(fx) - x0 as f64);
                let mut acc = 0.0;
                for (dy, wy_) in [(0i64, 1.0 - wy), (1i64, wy)] {
                    for (dx, wx_) in [(0i64, 1.0 - wx), (1i64, wx)] {
                        let (sy, sx) = (y0 + dy, x0 + dx);
                        let v = if sy < 0 || sy >= i64::from(sr) || sx < 0 || sx >= i64::from(sc) {
                            border_value
                        } else {
                            read_scalar(&input, ((sy as u32 * sc + sx as u32) as usize) * ch + c, depth)
                        };
                        acc += wy_ * wx_ * v;
                    }
                }
                acc
            }
        }
    };
    for j in 0..n {
        let fx = read_scalar(&m1, j, ScalarDepth::F32) as f32;
        let fy = read_scalar(&m2, j, ScalarDepth::F32) as f32;
        for c in 0..ch {
            let v = sample(fy, fx, c);
            write_scalar(&mut output, j * ch + c, depth, v);
        }
    }
    destination
        .write_output(output, or, oc, source.channels(), source.depth())
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    Ok(())
}

/// convertMaps: F32 xy maps to CV_16SC2 fixed-point + CV_16UC1 fractions.
pub(crate) fn convert_maps_into(
    map1: &Mat,
    map2: &Mat,
    dst1: &Mat,
    dst2: &Mat,
) -> Result<(), MapsError> {
    if map1.depth() != MatDepth::F32 || map2.depth() != MatDepth::F32 {
        return Err(MapsError::UnsupportedDepth(map1.depth()));
    }
    let m1 = map1.compact_bytes();
    let m2 = map2.compact_bytes();
    let n = m1.len() / 4;
    let mut out1 = vec![0u8; n * 4];
    let mut out2 = vec![0u8; n * 2];
    for i in 0..n {
        let fx = read_scalar(&m1, i, ScalarDepth::F32);
        let fy = read_scalar(&m2, i, ScalarDepth::F32);
        // Fixed-point: integer part + 5-bit fraction of the interpolation weight.
        let ix = fx.floor() as i64;
        let iy = fy.floor() as i64;
        let fx_frac = fx - ix as f64;
        let fy_frac = fy - iy as f64;
        // cv converts map1->CV_16SC2(x,y) and map2->CV_16UC1(fixed weights).
        let xw = (fx_frac * 32.0 + 0.5) as i64;
        let yw = (fy_frac * 32.0 + 0.5) as i64;
        let bilinear_bits = xw + yw * 32; // pinned 5-bit fraction pair packing
        write_scalar(&mut out1, i * 2, ScalarDepth::I16, ix as f64);
        write_scalar(&mut out1, i * 2 + 1, ScalarDepth::I16, iy as f64);
        write_scalar(&mut out2, i, ScalarDepth::U16, bilinear_bits as f64);
    }
    dst1.write_output(out1, map1.rows(), map1.columns(), 2, MatDepth::I16)
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    dst2.write_output(out2, map1.rows(), map1.columns(), 1, MatDepth::U16)
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    Ok(())
}

/// getRectSubPix: bilinear patch centered at (cx, cy).
pub(crate) fn get_rect_sub_pix_into(
    source: &Mat,
    destination: &Mat,
    width: i32,
    height: i32,
    center_x: f64,
    center_y: f64,
) -> Result<(), MapsError> {
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let (sr, sc, ch) = (source.rows(), source.columns(), source.channels() as usize);
    let n = width as usize * height as usize;
    let mut output = vec![0u8; n * ch * depth.width()];
    for y in 0..height {
        for x in 0..width {
            // Source position: patch pixel (x,y) maps to center + (x - w/2, y - h/2).
            let fx = center_x + f64::from(x) - (f64::from(width) - 1.0) / 2.0;
            let fy = center_y + f64::from(y) - (f64::from(height) - 1.0) / 2.0;
            let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
            let (wx, wy) = (fx - x0 as f64 as f64, fy - y0 as f64 as f64);
            for c in 0..ch {
                let mut acc = 0.0;
                for (dy, wy_) in [(0i64, 1.0 - wy), (1i64, wy)] {
                    for (dx, wx_) in [(0i64, 1.0 - wx), (1i64, wx)] {
                        let (sy, sx) = (y0 + dy, x0 + dx);
                        let v = if sy < 0 || sy >= i64::from(sr) || sx < 0 || sx >= i64::from(sc) {
                            0.0
                        } else {
                            read_scalar(&input, ((sy as u32 * sc + sx as u32) as usize) * ch + c, depth)
                        };
                        acc += wy_ * wx_ * v;
                    }
                }
                // Pinned U8 rounding is half-away-from-zero.
                let rounded = if matches!(depth, ScalarDepth::U8 | ScalarDepth::I8 | ScalarDepth::U16 | ScalarDepth::I16 | ScalarDepth::I32) {
                    acc.round() + if acc.fract().abs() == 0.5 { acc.signum() * 0.0 } else { 0.0 }
                } else {
                    acc
                };
                let rounded = if acc.fract().abs() == 0.5 { acc.trunc() + acc.signum() } else { rounded };
                write_scalar(&mut output, ((y * width + x) as usize) * ch + c, depth, rounded);
            }
        }
    }
    destination
        .write_output(output, height as u32, width as u32, source.channels(), source.depth())
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    Ok(())
}

/// demosaicing: Bayer mosaic to BGR (BG/GB/RG/GR layouts, VNG bilinear).
pub(crate) fn demosaicing_into(
    source: &Mat,
    destination: &Mat,
    code: i32,
) -> Result<(), MapsError> {
    if source.channels() != 1 {
        return Err(MapsError::InvalidInput("expected a single-channel mosaic"));
    }
    let depth = scalar_depth(source.depth());
    if !matches!(depth, ScalarDepth::U8 | ScalarDepth::U16) {
        return Err(MapsError::UnsupportedDepth(source.depth()));
    }
    // Bayer layout offset: BG(46), GB(47), RG(48), GR(49) → row/col parity of green.
    // COLOR_BayerBG2BGR = 46, BG2BGR_VNG = 62; support the linear variants.
    let green_parity = match code {
        46 | 62 => (0, 1), // B G / G R / R G / G B — green at (0,1)
        47 | 63 => (0, 0), // G B / B G / R G / G R — green at (0,0)
        48 | 64 => (0, 0), // R G / G B / B G / G R — green at (0,0)
        49 | 65 => (0, 1), // G R / B G / B G / R G — green at (0,1)
        _ => return Err(MapsError::InvalidInput("unsupported demosaic code")),
    };
    let input = source.compact_bytes();
    let (rows, cols) = (source.rows(), source.columns());
    let mut output = vec![0u8; (rows * cols) as usize * 3 * depth.width()];
    let at = |y: i64, x: i64| -> f64 {
        if y < 0 || y >= i64::from(rows) || x < 0 || x >= i64::from(cols) {
            0.0
        } else {
            read_scalar(&input, (y as u32 * cols + x as u32) as usize, depth)
        }
    };
    for y in 0..rows as i64 {
        for x in 0..cols as i64 {
            let (py, px) = ((y % 2) as u32, (x % 2) as u32);
            let is_green = (py + px) % 2 == (green_parity.0 + green_parity.1) % 2;
            // Determine the color of this pixel's site: for the BG layout, the
            // pixel at green_parity is green, the diagonal neighbors are the
            // "other" color, and orthogonal are the second color.
            let v = at(y, x);
            let (b, g, r);
            if is_green {
                g = v;
                // green site: same-row neighbors = one color, same-col = other.
                // For BG2BGR: green rows have B on even cols and R on odd cols.
                let (h_color, _v_color) = if green_parity.1 == 1 {
                    // green at odd column in row 0 → rows alternate B-then-R
                    if py == 0 { ("b", "r") } else { ("r", "b") }
                } else if py == 0 {
                    ("r", "b")
                } else {
                    ("b", "r")
                };
                let h_avg = (at(y, x - 1) + at(y, x + 1)) / 2.0;
                let v_avg = (at(y - 1, x) + at(y + 1, x)) / 2.0;
                if h_color == "b" {
                    b = h_avg;
                    r = v_avg;
                } else {
                    r = h_avg;
                    b = v_avg;
                }
            } else {
                // Red or blue site.
                // For BG layout: (0,0)=B, (1,1)=R for green_parity.1==1.
                let site_is_blue = if green_parity.1 == 1 {
                    (py == 0 && px == 0) || (py == 1 && px == 1)
                } else {
                    (py == 0 && px == 1) || (py == 1 && px == 0)
                };
                let diag = (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1)) / 4.0;
                let orth = (at(y - 1, x) + at(y + 1, x) + at(y, x - 1) + at(y, x + 1)) / 4.0;
                if site_is_blue {
                    b = v;
                    g = orth;
                    r = diag;
                } else {
                    r = v;
                    g = orth;
                    b = diag;
                }
            }
            let idx = ((y as u32 * cols + x as u32) as usize) * 3;
            write_scalar(&mut output, idx, depth, b);
            write_scalar(&mut output, idx + 1, depth, g);
            write_scalar(&mut output, idx + 2, depth, r);
        }
    }
    destination
        .write_output(output, rows, cols, 3, source.depth())
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    Ok(())
}

/// calcHist: bin counting over F32/U8 single-channel inputs.
pub(crate) fn calc_hist_into(
    source: &Mat,
    destination: &Mat,
    hist_size: i32,
    range_low: f64,
    range_high: f64,
    accumulate: bool,
) -> Result<(), MapsError> {
    if source.channels() != 1 {
        return Err(MapsError::InvalidInput("single channel only"));
    }
    let depth = scalar_depth(source.depth());
    if hist_size <= 0 || range_high <= range_low {
        return Err(MapsError::InvalidInput("bad range"));
    }
    let input = source.compact_bytes();
    let n = input.len() / depth.width();
    let mut hist = vec![0f64; hist_size as usize];
    if accumulate {
        let existing = destination.compact_bytes();
        let d = scalar_depth(destination.depth());
        for (i, h) in hist.iter_mut().enumerate() {
            if i * d.width() + d.width() <= existing.len() {
                *h = read_scalar(&existing, i, d);
            }
        }
    }
    let scale = f64::from(hist_size) / (range_high - range_low);
    for i in 0..n {
        let v = read_scalar(&input, i, depth);
        if v < range_low || v >= range_high {
            continue;
        }
        let bin = ((v - range_low) * scale) as usize;
        hist[bin.min(hist_size as usize - 1)] += 1.0;
    }
    let mut output = vec![0u8; hist.len() * 4];
    for (i, v) in hist.iter().enumerate() {
        write_scalar(&mut output, i, ScalarDepth::F32, *v);
    }
    destination
        .write_output(output, hist.len() as u32, 1, 1, MatDepth::F32)
        .map_err(|e| MapsError::OutputError(e.to_string()))?;
    Ok(())
}

/// compareHist: method 0 correlation, 1 chi-square, 2 intersection, 3 Bhattacharyya.
pub(crate) fn compare_hist_into(
    h1: &Mat,
    h2: &Mat,
    method: i32,
) -> Result<f64, MapsError> {
    let depth = scalar_depth(h1.depth());
    if !matches!(depth, ScalarDepth::F32 | ScalarDepth::F64) {
        return Err(MapsError::UnsupportedDepth(h1.depth()));
    }
    let b1 = h1.compact_bytes();
    let b2 = h2.compact_bytes();
    let n = (b1.len() / depth.width()).min(b2.len() / scalar_depth(h2.depth()).width());
    let v1: Vec<f64> = (0..n).map(|i| read_scalar(&b1, i, depth)).collect();
    let v2: Vec<f64> = (0..n).map(|i| read_scalar(&b2, i, depth)).collect();
    match method {
        0 => {
            // Pearson correlation.
            let m1 = v1.iter().sum::<f64>() / n as f64;
            let m2 = v2.iter().sum::<f64>() / n as f64;
            let mut num = 0.0;
            let mut d1 = 0.0;
            let mut d2 = 0.0;
            for i in 0..n {
                let (a, b) = (v1[i] - m1, v2[i] - m2);
                num += a * b;
                d1 += a * a;
                d2 += b * b;
            }
            if d1 * d2 < 1e-30 {
                return Ok(1.0);
            }
            Ok(num / (d1 * d2).sqrt())
        }
        1 => {
            let mut sum = 0.0;
            for i in 0..n {
                if v1[i].abs() > 1e-12 {
                    sum += (v1[i] - v2[i]) * (v1[i] - v2[i]) / v1[i];
                }
            }
            Ok(sum)
        }
        2 => Ok(v1.iter().zip(&v2).map(|(a, b)| a.min(*b)).sum()),
        3 => {
            let s1: f64 = v1.iter().sum();
            let s2: f64 = v2.iter().sum();
            if s1.abs() < 1e-12 || s2.abs() < 1e-12 {
                return Ok(0.0);
            }
            let mut bc = 0.0;
            for i in 0..n {
                bc += (v1[i] * v2[i]).sqrt();
            }
            Ok((1.0 - bc / (s1 * s2).sqrt()).max(0.0).sqrt())
        }
        _ => Err(MapsError::InvalidInput("unsupported compare method")),
    }
}
