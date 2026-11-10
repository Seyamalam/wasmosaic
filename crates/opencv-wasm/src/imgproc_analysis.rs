//! Integral images, raster and contour moments, Hu invariants, and shape matching.
//!
//! Behavior derives from the pinned OpenCV.js 4.13.0 browser build: integral images are
//! offered as `integral` (sum only) and `integral2` (sum plus squared sum); `moments`
//! accepts single-channel sources of any depth; `matchShapes` consumes two-dimensional
//! point sequences and compares log-transformed Hu invariants.

use std::{error::Error, fmt};

use crate::{
    core_norm::{ScalarDepth, read_scalar},
    imgproc_window::scalar_depth,
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
pub(crate) enum AnalysisError {
    EmptySource,
    UnsupportedDepth(MatDepth),
    UnsupportedSumDepth(i32),
    UnsupportedMomentDepth,
    UnsupportedMatchMethod(i32),
    UnsupportedContourShape,
    InvalidFontFace(i32),
    Matrix(MatError),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => f.write_str("source matrix dimensions must be positive"),
            Self::UnsupportedDepth(depth) => {
                write!(f, "operation does not support matrix depth {depth:?}")
            }
            Self::UnsupportedSumDepth(depth) => {
                write!(f, "unsupported integral output depth {depth}")
            }
            Self::UnsupportedMomentDepth => f.write_str("moments require a single-channel source"),
            Self::UnsupportedMatchMethod(method) => {
                write!(f, "unsupported shape-matching method {method}")
            }
            Self::UnsupportedContourShape => {
                f.write_str("contours must be two-dimensional point sequences")
            }
            Self::InvalidFontFace(face) => write!(f, "font face {face} is unsupported"),
            Self::Matrix(error) => write!(f, "{error}"),
        }
    }
}

impl Error for AnalysisError {}

impl From<MatError> for AnalysisError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

fn require_shape(source: &Mat) -> Result<(), AnalysisError> {
    if source.rows() == 0 || source.columns() == 0 {
        return Err(AnalysisError::EmptySource);
    }
    Ok(())
}

fn require_integral_depth(source: &Mat) -> Result<(), AnalysisError> {
    require_shape(source)?;
    match source.depth() {
        MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 | MatDepth::F64 => Ok(()),
        depth => Err(AnalysisError::UnsupportedDepth(depth)),
    }
}

fn sum_depth(source: &Mat, requested: i32, squared: bool) -> Result<MatDepth, AnalysisError> {
    if requested < 0 {
        return Ok(match (squared, source.depth()) {
            (true, _) => MatDepth::F64,
            (false, MatDepth::U8) => MatDepth::I32,
            (false, _) => MatDepth::F64,
        });
    }
    let candidate = match requested {
        4 => MatDepth::I32,
        5 => MatDepth::F32,
        6 => MatDepth::F64,
        other => return Err(AnalysisError::UnsupportedSumDepth(other)),
    };
    let allowed = match (source.depth(), squared) {
        (MatDepth::U8, false) => matches!(candidate, MatDepth::I32 | MatDepth::F32 | MatDepth::F64),
        (MatDepth::U8, true) => matches!(candidate, MatDepth::F32 | MatDepth::F64),
        (MatDepth::U16 | MatDepth::I16, _) => candidate == MatDepth::F64,
        (MatDepth::F32, false) => matches!(candidate, MatDepth::F32 | MatDepth::F64),
        (MatDepth::F32, true) => candidate == MatDepth::F64,
        (MatDepth::F64, _) => candidate == MatDepth::F64,
        _ => false,
    };
    if allowed {
        Ok(candidate)
    } else {
        Err(AnalysisError::UnsupportedSumDepth(requested))
    }
}

fn integral_pass(
    input: &[u8],
    rows: u32,
    columns: u32,
    channels: usize,
    depth: ScalarDepth,
    squared: bool,
) -> Vec<f64> {
    let out_rows = rows as usize + 1;
    let out_columns = columns as usize + 1;
    let mut sums = vec![0.0f64; out_rows * out_columns * channels];
    for row in 0..rows {
        for column in 0..columns {
            for channel in 0..channels {
                let source = read_scalar(
                    input,
                    (row as usize * columns as usize + column as usize) * channels + channel,
                    depth,
                );
                let value = if squared { source * source } else { source };
                let at = |r: usize, c: usize, ch: usize| -> usize {
                    (r * out_columns + c) * channels + ch
                };
                let index = at(row as usize + 1, column as usize + 1, channel);
                sums[index] = value
                    + sums[at(row as usize, column as usize + 1, channel)]
                    + sums[at(row as usize + 1, column as usize, channel)]
                    - sums[at(row as usize, column as usize, channel)];
            }
        }
    }
    sums
}

fn write_sums(
    sums: &[f64],
    destination: &Mat,
    rows: u32,
    columns: u32,
    channels: u16,
    depth: MatDepth,
) -> Result<(), AnalysisError> {
    let scalar = scalar_depth(depth);
    let mut output = vec![0u8; sums.len() * scalar.width()];
    for (index, &value) in sums.iter().enumerate() {
        crate::core_norm::write_scalar(&mut output, index, scalar, value);
    }
    destination.write_output(output, rows, columns, channels, depth)?;
    Ok(())
}

pub(crate) fn integral_into(
    source: &Mat,
    sum: &Mat,
    requested_depth: i32,
) -> Result<(), AnalysisError> {
    require_integral_depth(source)?;
    let depth = sum_depth(source, requested_depth, false)?;
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let sums = integral_pass(
        &input,
        source.rows(),
        source.columns(),
        channels,
        scalar_depth(source.depth()),
        false,
    );
    write_sums(
        &sums,
        sum,
        source.rows() + 1,
        source.columns() + 1,
        source.channels(),
        depth,
    )
}

pub(crate) fn integral2_into(
    source: &Mat,
    sum: &Mat,
    squared_sum: &Mat,
    requested_sum_depth: i32,
    requested_squared_depth: i32,
) -> Result<(), AnalysisError> {
    require_integral_depth(source)?;
    let depth = sum_depth(source, requested_sum_depth, false)?;
    let squared_depth = if requested_squared_depth < 0 {
        MatDepth::F64
    } else {
        sum_depth(source, requested_squared_depth, true)?
    };
    let input = source.compact_bytes();
    let channels = usize::from(source.channels());
    let scalar = scalar_depth(source.depth());
    let sums = integral_pass(
        &input,
        source.rows(),
        source.columns(),
        channels,
        scalar,
        false,
    );
    let squared = integral_pass(
        &input,
        source.rows(),
        source.columns(),
        channels,
        scalar,
        true,
    );
    write_sums(
        &sums,
        sum,
        source.rows() + 1,
        source.columns() + 1,
        source.channels(),
        depth,
    )?;
    write_sums(
        &squared,
        squared_sum,
        source.rows() + 1,
        source.columns() + 1,
        source.channels(),
        squared_depth,
    )
}

/// Raw + central + normalized moments, matching the 24-field `cv.Moments` record.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Moments {
    pub(crate) values: [f64; 24],
}

pub(crate) fn raster_moments(source: &Mat, binary: bool) -> Result<Moments, AnalysisError> {
    require_shape(source)?;
    if !binary {
        match source.depth() {
            MatDepth::U8 | MatDepth::U16 | MatDepth::I16 | MatDepth::F32 | MatDepth::F64 => {}
            _ => return Err(AnalysisError::UnsupportedMomentDepth),
        }
    }
    if source.channels() != 1 {
        return Err(AnalysisError::UnsupportedMomentDepth);
    }
    let depth = scalar_depth(source.depth());
    let input = source.compact_bytes();
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    let mut spatial = [0.0f64; 10];
    for row in 0..rows {
        for column in 0..columns {
            let mut value = read_scalar(&input, row * columns + column, depth);
            if binary && value != 0.0 {
                value = 1.0;
            }
            if value == 0.0 {
                continue;
            }
            let x = column as f64;
            let y = row as f64;
            spatial[0] += value;
            spatial[1] += x * value;
            spatial[2] += y * value;
            spatial[3] += x * x * value;
            spatial[4] += x * y * value;
            spatial[5] += y * y * value;
            spatial[6] += x * x * x * value;
            spatial[7] += x * x * y * value;
            spatial[8] += x * y * y * value;
            spatial[9] += y * y * y * value;
        }
    }
    Ok(finish_moments(spatial))
}

fn finish_moments(m: [f64; 10]) -> Moments {
    let [m00, m10, m01, m20, m11, m02, m30, m21, m12, m03] = m;
    let (cx, cy) = if m00 != 0.0 {
        (m10 / m00, m01 / m00)
    } else {
        (0.0, 0.0)
    };
    let mu20 = m20 - cx * m10;
    let mu11 = m11 - cy * m10;
    let mu02 = m02 - cy * m01;
    let mu30 = m30 - cx * (3.0 * mu20 + cx * m10);
    let mu21 = m21 - cx * (2.0 * mu11 + cx * m01) - cy * mu20;
    let mu12 = m12 - cy * (2.0 * mu11 + cy * m10) - cx * mu02;
    let mu03 = m03 - cy * (3.0 * mu02 + cy * m01);
    let nu = |mu: f64, order: u32| -> f64 {
        if m00 == 0.0 {
            return 0.0;
        }
        let power = -(f64::from(order) / 2.0 + 1.0);
        mu * m00.abs().powf(power)
    };
    Moments {
        values: [
            m00,
            m10,
            m01,
            m20,
            m11,
            m02,
            m30,
            m21,
            m12,
            m03,
            mu20,
            mu11,
            mu02,
            mu30,
            mu21,
            mu12,
            mu03,
            nu(mu20, 2),
            nu(mu11, 2),
            nu(mu02, 2),
            nu(mu30, 3),
            nu(mu21, 3),
            nu(mu12, 3),
            nu(mu03, 3),
        ],
    }
}

/// Green's-theorem spatial moments for a closed polygonal point sequence.
pub(crate) fn contour_moments(source: &Mat) -> Result<Moments, AnalysisError> {
    require_shape(source)?;
    let points = contour_points(source)?;
    let n = points.len();
    if n == 0 {
        return Ok(Moments::default());
    }
    let mut m = [0.0f64; 10];
    for index in 0..n {
        let (x0, y0) = points[index];
        let (x1, y1) = points[(index + 1) % n];
        let a = x0 * y1 - x1 * y0;
        m[0] += a * 0.5;
        m[1] += a * (x0 + x1) / 6.0;
        m[2] += a * (y0 + y1) / 6.0;
        m[3] += a * (x0 * x0 + x0 * x1 + x1 * x1) / 12.0;
        m[5] += a * (y0 * y0 + y0 * y1 + y1 * y1) / 12.0;
        m[4] += a * (x0 * (2.0 * y0 + y1) + x1 * (y0 + 2.0 * y1)) / 24.0;
        m[6] += a * (x0 * x0 + x0 * x1 + x1 * x1) * (x0 + x1) / 20.0;
        m[7] += a
            * (x0 * x0 * (3.0 * y0 + y1) + 2.0 * x0 * x1 * (y0 + y1) + x1 * x1 * (y0 + 3.0 * y1))
            / 60.0;
        m[8] += a
            * (y0 * y0 * (3.0 * x0 + x1) + 2.0 * y0 * y1 * (x0 + x1) + y1 * y1 * (x0 + 3.0 * x1))
            / 60.0;
        m[9] += a * (y0 * y0 + y0 * y1 + y1 * y1) * (y0 + y1) / 20.0;
    }
    m[0] = m[0].abs();
    Ok(finish_moments(m))
}

fn contour_points(source: &Mat) -> Result<Vec<(f64, f64)>, AnalysisError> {
    let depth = scalar_depth(source.depth());
    let bytes = source.compact_bytes();
    let channels = usize::from(source.channels());
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    if channels == 2 && (rows == 1 || columns == 1) {
        let count = rows * columns;
        return Ok((0..count)
            .map(|index| {
                (
                    read_scalar(&bytes, index * 2, depth),
                    read_scalar(&bytes, index * 2 + 1, depth),
                )
            })
            .collect());
    }
    if channels == 1 && (rows == 1 || columns == 1) && (rows * columns) % 2 == 0 {
        let count = rows * columns / 2;
        return Ok((0..count)
            .map(|index| {
                (
                    read_scalar(&bytes, index * 2, depth),
                    read_scalar(&bytes, index * 2 + 1, depth),
                )
            })
            .collect());
    }
    if channels == 1 && columns == 2 {
        return Ok((0..rows)
            .map(|row| {
                (
                    read_scalar(&bytes, row * 2, depth),
                    read_scalar(&bytes, row * 2 + 1, depth),
                )
            })
            .collect());
    }
    Err(AnalysisError::UnsupportedContourShape)
}

pub(crate) fn hu_values(moments: &Moments) -> [f64; 7] {
    let v = &moments.values;
    let (nu20, nu11, nu02, nu30, nu21, nu12, nu03) =
        (v[17], v[18], v[19], v[20], v[21], v[22], v[23]);
    let t0 = nu30 + nu12;
    let t1 = nu21 + nu03;
    let q0 = nu30 - 3.0 * nu12;
    let q1 = 3.0 * nu21 - nu03;
    [
        nu20 + nu02,
        (nu20 - nu02).powi(2) + 4.0 * nu11.powi(2),
        q0.powi(2) + q1.powi(2),
        t0.powi(2) + t1.powi(2),
        q0 * t0 * (t0.powi(2) - 3.0 * t1.powi(2)) + q1 * t1 * (3.0 * t0.powi(2) - t1.powi(2)),
        (nu20 - nu02) * (t0.powi(2) - t1.powi(2)) + 4.0 * nu11 * t0 * t1,
        q1 * t0 * (t0.powi(2) - 3.0 * t1.powi(2)) - q0 * t1 * (3.0 * t0.powi(2) - t1.powi(2)),
    ]
}

pub(crate) fn hu_moments_into(moments: &[f64], destination: &Mat) -> Result<(), AnalysisError> {
    if moments.len() != 24 {
        return Err(AnalysisError::UnsupportedMomentDepth);
    }
    let mut values = [0.0; 24];
    values.copy_from_slice(moments);
    let hu = hu_values(&Moments { values });
    let mut output = Vec::with_capacity(56);
    for value in hu {
        output.extend_from_slice(&value.to_ne_bytes());
    }
    destination.write_output(output, 7, 1, 1, MatDepth::F64)?;
    Ok(())
}

pub(crate) fn match_shapes(
    first: &Mat,
    second: &Mat,
    method: i32,
    _parameter: f64,
) -> Result<f64, AnalysisError> {
    if !(1..=3).contains(&method) {
        return Err(AnalysisError::UnsupportedMatchMethod(method));
    }
    let hu_first = hu_values(&shape_moments(first)?);
    let hu_second = hu_values(&shape_moments(second)?);
    let mut score = 0.0;
    for index in 0..7 {
        if hu_first[index] == 0.0 || hu_second[index] == 0.0 {
            continue;
        }
        let a = hu_first[index].signum() * hu_first[index].abs().log10();
        let b = hu_second[index].signum() * hu_second[index].abs().log10();
        score += match method {
            1 => (1.0 / a - 1.0 / b).abs(),
            2 => (a - b).abs(),
            _ => {
                if a == 0.0 {
                    0.0
                } else {
                    (a - b).abs() / a.abs()
                }
            }
        };
    }
    Ok(score)
}

fn shape_moments(source: &Mat) -> Result<Moments, AnalysisError> {
    if source.channels() == 1 && source.rows() > 1 && source.columns() > 1 {
        return raster_moments(source, false);
    }
    contour_moments(source)
}

const FONT_FACE_DENOMINATORS: [f64; 8] = [21.0, 9.0, 21.0, 21.0, 21.0, 13.0, 21.0, 21.0];

pub(crate) fn font_scale_from_height(
    font_face: i32,
    pixel_height: i32,
    thickness: i32,
) -> Result<f64, AnalysisError> {
    let face = font_face & !16;
    if !(0..8).contains(&face) {
        return Err(AnalysisError::InvalidFontFace(font_face));
    }
    let adjustment = (f64::from(thickness.max(-1)) + 1.0) / 2.0;
    Ok((f64::from(pixel_height) - adjustment) / FONT_FACE_DENOMINATORS[face as usize])
}
