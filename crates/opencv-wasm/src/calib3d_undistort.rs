use std::{error::Error, fmt};

use crate::core_norm::{ScalarDepth, read_scalar, write_scalar};
use crate::imgproc_window::scalar_depth;
use crate::mat::{Mat, MatDepth, MatError};
use crate::calib3d_distort;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UndistortError {
    Matrix(MatError),
    UnsupportedDepth,
    UnsupportedCamera,
    UnsupportedRectify,
    UnsupportedDistortion,
    UnsupportedMapType,
    InvalidSize,
}

impl fmt::Display for UndistortError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth => {
                formatter.write_str("calibration inputs require an F32 or F64 depth")
            }
            Self::UnsupportedCamera => formatter.write_str("camera matrices require 3x3"),
            Self::UnsupportedRectify => {
                formatter.write_str("rectification requires a 3x3 matrix")
            }
            Self::UnsupportedDistortion => {
                formatter.write_str("distortion coefficients require 4, 5, 8, 12, or 14 elements")
            }
            Self::UnsupportedMapType => {
                formatter.write_str("map type requires CV_32FC1, CV_32FC2, or CV_16SC2")
            }
            Self::InvalidSize => formatter.write_str("map size must be positive"),
        }
    }
}

impl Error for UndistortError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for UndistortError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

const MAP_F32C1: i32 = 5;
const MAP_F32C2: i32 = 13;
const MAP_FIXED: i32 = 11;

fn read_elements(source: &Mat) -> Result<Vec<f64>, UndistortError> {
    let element_size = match source.depth() {
        MatDepth::F64 => 8,
        MatDepth::F32 => 4,
        _ => return Err(UndistortError::UnsupportedDepth),
    };
    let bytes = source.compact_bytes();
    let count = bytes.len() / element_size;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * element_size;
        values.push(if element_size == 8 {
            f64::from_ne_bytes(bytes[start..start + 8].try_into().expect("f64"))
        } else {
            f32::from_ne_bytes(bytes[start..start + 4].try_into().expect("f32")) as f64
        });
    }
    Ok(values)
}

fn element_count(source: &Mat) -> usize {
    source.rows() as usize * source.columns() as usize * source.channels() as usize
}

fn read_camera(source: &Mat) -> Result<[f64; 9], UndistortError> {
    if source.channels() != 1 || source.rows() != 3 || source.columns() != 3 {
        return Err(UndistortError::UnsupportedCamera);
    }
    let values = read_elements(source)?;
    Ok([
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        values[8],
    ])
}

fn read_distortion(source: &Mat) -> Result<([f64; 14], usize), UndistortError> {
    let length = element_count(source);
    if length == 0 {
        return Ok(([0.0; 14], 0));
    }
    let values = read_elements(source)?;
    let mut d = [0.0f64; 14];
    match values.len() {
        4 | 5 | 8 | 12 | 14 => {
            for (index, value) in values.iter().enumerate() {
                d[index] = *value;
            }
            Ok((d, values.len()))
        }
        _ => Err(UndistortError::UnsupportedDistortion),
    }
}

fn invert3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let mut out = [0.0f64; 9];
    out[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    out[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    out[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    out[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    out[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    out[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    out[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    out[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    out[8] = (m[0] * m[4] - m[1] * m[3]) / det;
    out
}

/// Computes the source coordinate for one output pixel: back-project
/// through the new camera, apply the rectification inverse, normalize,
/// distort, and project through the source camera.
fn map_pixel(
    u: usize,
    v: usize,
    camera: &[f64; 9],
    new_camera: &[f64; 9],
    inv_rotation: &[f64; 9],
    coeffs: &[f64; 14],
    dist_len: usize,
) -> (f32, f32) {
    let xp = (u as f64 - new_camera[2]) / new_camera[0];
    let yp = (v as f64 - new_camera[5]) / new_camera[4];
    let tx = inv_rotation[0] * xp + inv_rotation[1] * yp + inv_rotation[2];
    let ty = inv_rotation[3] * xp + inv_rotation[4] * yp + inv_rotation[5];
    let tz = inv_rotation[6] * xp + inv_rotation[7] * yp + inv_rotation[8];
    let x = tx / tz;
    let y = ty / tz;
    let (xd, yd) = calib3d_distort::distort_normalized(x, y, coeffs, dist_len);
    (
        (camera[0] * xd + camera[2]) as f32,
        (camera[4] * yd + camera[5]) as f32,
    )
}

/// Computes dense backward maps for `cv.initUndistortRectifyMap` in the
/// CV_32FC1, CV_32FC2, and CV_16SC2 forms.
pub(crate) fn init_undistort_rectify_map_into(
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rotation: &Mat,
    new_camera_matrix: &Mat,
    width: u32,
    height: u32,
    m1type: i32,
    map1: &Mat,
    map2: &Mat,
) -> Result<(), UndistortError> {
    if width == 0 || height == 0 {
        return Err(UndistortError::InvalidSize);
    }
    let camera = read_camera(camera_matrix)?;
    let (coeffs, dist_len) = read_distortion(dist_coeffs)?;
    let inv_rotation = if element_count(rotation) == 0 {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    } else if rotation.channels() == 1 && rotation.rows() == 3 && rotation.columns() == 3 {
        invert3(&read_camera(rotation)?)
    } else {
        return Err(UndistortError::UnsupportedRectify);
    };
    let new_camera = if element_count(new_camera_matrix) == 0 {
        // Default: the camera matrix with the principal point recentered
        // on the map, matching the pinned binding's empty-newCameraMatrix
        // behavior.
        let mut centered = camera;
        centered[2] = (width as f64 - 1.0) / 2.0;
        centered[5] = (height as f64 - 1.0) / 2.0;
        centered
    } else {
        read_camera(new_camera_matrix)?
    };
    if !(m1type == MAP_F32C1 || m1type == MAP_F32C2 || m1type == MAP_FIXED) {
        return Err(UndistortError::UnsupportedMapType);
    }

    let n = width as usize * height as usize;
    let mut ux = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    for v in 0..height as usize {
        for u in 0..width as usize {
            let (sx, sy) = map_pixel(u, v, &camera, &new_camera, &inv_rotation, &coeffs, dist_len);
            ux.push(sx);
            vy.push(sy);
        }
    }

    match m1type {
        MAP_F32C1 => {
            let mut b1 = Vec::with_capacity(n * 4);
            let mut b2 = Vec::with_capacity(n * 4);
            for index in 0..n {
                b1.extend_from_slice(&ux[index].to_ne_bytes());
                b2.extend_from_slice(&vy[index].to_ne_bytes());
            }
            map1.write_output(b1, height, width, 1, MatDepth::F32)?;
            map2.write_output(b2, height, width, 1, MatDepth::F32)?;
        }
        MAP_F32C2 => {
            let mut b1 = Vec::with_capacity(n * 8);
            for index in 0..n {
                b1.extend_from_slice(&ux[index].to_ne_bytes());
                b1.extend_from_slice(&vy[index].to_ne_bytes());
            }
            map1.write_output(b1, height, width, 2, MatDepth::F32)?;
            map2.write_empty_output();
        }
        _ => {
            // Fixed-point CV_16SC2 + CV_16UC1: integer part plus packed
            // 5-bit bilinear fractions.
            let mut b1 = Vec::with_capacity(n * 4);
            let mut b2 = Vec::with_capacity(n * 2);
            for index in 0..n {
                let mut ix = ux[index].floor() as i64;
                let mut iy = vy[index].floor() as i64;
                let mut xw = ((f64::from(ux[index]) - ix as f64) * 32.0 + 0.5) as u64;
                let mut yw = ((f64::from(vy[index]) - iy as f64) * 32.0 + 0.5) as u64;
                // A fraction rounding to 32 carries into the integer part.
                if xw == 32 {
                    ix += 1;
                    xw = 0;
                }
                if yw == 32 {
                    iy += 1;
                    yw = 0;
                }
                b1.extend_from_slice(&(ix as i16).to_ne_bytes());
                b1.extend_from_slice(&(iy as i16).to_ne_bytes());
                b2.extend_from_slice(&((xw + yw * 32) as u16).to_ne_bytes());
            }
            map1.write_output(b1, height, width, 2, MatDepth::I16)?;
            map2.write_output(b2, height, width, 1, MatDepth::U16)?;
        }
    }
    Ok(())
}

/// `cv.undistort`: builds the fixed-point rectification map and resamples
/// with border-constant fixed-point bilinear interpolation, matching the
/// upstream fixed-point path.
pub(crate) fn undistort_into(
    source: &Mat,
    destination: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    new_camera_matrix: &Mat,
) -> Result<(), UndistortError> {
    let camera = read_camera(camera_matrix)?;
    let (coeffs, dist_len) = read_distortion(dist_coeffs)?;
    let new_camera = if element_count(new_camera_matrix) == 0 {
        camera
    } else {
        read_camera(new_camera_matrix)?
    };
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let sr = source.rows() as usize;
    let sc = source.columns() as usize;
    let ch = source.channels() as usize;
    let input = source.compact_bytes();
    let input_depth = source.depth();
    let depth = scalar_depth(input_depth);

    let mut output = vec![0u8; sr * sc * ch * depth.width()];
    for v in 0..sr {
        for u in 0..sc {
            let (fx32, fy32) = map_pixel(u, v, &camera, &new_camera, &identity, &coeffs, dist_len);
            let fx = f64::from(fx32);
            let fy = f64::from(fy32);
            let ix = fx.floor() as i64;
            let iy = fy.floor() as i64;
            let xw = ((fx - ix as f64) * 32.0 + 0.5) as i64;
            let yw = ((fy - iy as f64) * 32.0 + 0.5) as i64;
            for c in 0..ch {
                let value = fixed_sample(&input, sr, sc, ch, ix, iy, xw, yw, c, input_depth, depth);
                write_scalar(&mut output, (v * sc + u) * ch + c, depth, value);
            }
        }
    }
    destination.write_output(output, source.rows(), source.columns(), source.channels(), input_depth)?;
    Ok(())
}

/// Fixed-point bilinear sampling with 5-bit fractions and constant-zero
/// borders, rounding as `(acc + 512) >> 10` for integer depths.
fn fixed_sample(
    input: &[u8],
    sr: usize,
    sc: usize,
    ch: usize,
    ix: i64,
    iy: i64,
    xw: i64,
    yw: i64,
    c: usize,
    depth: MatDepth,
    scalar: ScalarDepth,
) -> f64 {
    let tap = |y: i64, x: i64| -> f64 {
        if y < 0 || y >= sr as i64 || x < 0 || x >= sc as i64 {
            0.0
        } else {
            read_scalar(input, ((y as usize) * sc + (x as usize)) * ch + c, scalar)
        }
    };
    match depth {
        MatDepth::U8 | MatDepth::I8 | MatDepth::U16 | MatDepth::I16 | MatDepth::I32 => {
            let acc = (32 - xw) * (32 - yw) * tap(iy, ix) as i64
                + xw * (32 - yw) * tap(iy, ix + 1) as i64
                + (32 - xw) * yw * tap(iy + 1, ix) as i64
                + xw * yw * tap(iy + 1, ix + 1) as i64;
            let value = (acc + 512) >> 10;
            value as f64
        }
        _ => {
            let wx0 = (32 - xw) as f64 / 32.0;
            let wx1 = xw as f64 / 32.0;
            let wy0 = (32 - yw) as f64 / 32.0;
            let wy1 = yw as f64 / 32.0;
            wy0 * wx0 * tap(iy, ix)
                + wy0 * wx1 * tap(iy, ix + 1)
                + wy1 * wx0 * tap(iy + 1, ix)
                + wy1 * wx1 * tap(iy + 1, ix + 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64};

    fn f32_values(mat: &Mat) -> Vec<f32> {
        mat.compact_bytes()
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().expect("f32")))
            .collect()
    }

    fn camera() -> Mat {
        mat_from_f64(&[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0], 3, 3, 1)
            .expect("camera")
    }

    #[test]
    fn rectifies_identity_maps_without_distortion() {
        let dist = mat_empty();
        let rotation = mat_empty();
        let map1 = mat_empty();
        let map2 = mat_empty();
        init_undistort_rectify_map_into(
            &camera(), &dist, &rotation, &camera(), 3, 2, MAP_F32C1, &map1, &map2,
        )
        .expect("maps");
        assert_eq!(map1.depth(), MatDepth::F32);
        let x = f32_values(&map1);
        let y = f32_values(&map2);
        for u in 0..3usize {
            assert!((x[u] - u as f32).abs() < 1e-6, "{x:?}");
            assert!((y[u]).abs() < 1e-6, "{y:?}");
        }
    }

    #[test]
    fn matches_pinned_distorted_map() {
        let dist = mat_from_f64(&[0.1, 0.01, 0.001, 0.002, 0.0], 1, 5, 1).expect("dist");
        let new_camera =
            mat_from_f64(&[450.0, 0.0, 320.0, 0.0, 450.0, 240.0, 0.0, 0.0, 1.0], 3, 3, 1)
                .expect("nk");
        let map1 = mat_empty();
        let map2 = mat_empty();
        init_undistort_rectify_map_into(
            &camera(), &dist, &mat_empty(), &new_camera, 8, 6, MAP_F32C1, &map1, &map2,
        )
        .expect("maps");
        let x = f32_values(&map1);
        let y = f32_values(&map2);
        assert!((x[0] - -63.687_809).abs() < 1e-4, "{}", x[0]);
        assert!((y[0] - -47.963_387).abs() < 1e-4, "{}", y[0]);
        assert!((x[1] - -62.363_136).abs() < 1e-4, "{}", x[1]);
    }

    #[test]
    fn rotates_through_inverse_rectification() {
        let dist = mat_empty();
        let rotation =
            mat_from_f64(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 3, 1).expect("r");
        let map1 = mat_empty();
        let map2 = mat_empty();
        init_undistort_rectify_map_into(
            &camera(), &dist, &rotation, &camera(), 3, 2, MAP_F32C1, &map1, &map2,
        )
        .expect("maps");
        let x = f32_values(&map1);
        let y = f32_values(&map2);
        assert!((x[0] - 80.0).abs() < 1e-4, "{}", x[0]);
        assert!((y[0] - 560.0).abs() < 1e-4, "{}", y[0]);
    }

    #[test]
    fn emits_fixed_point_maps() {
        let dist = mat_empty();
        let map1 = mat_empty();
        let map2 = mat_empty();
        init_undistort_rectify_map_into(
            &camera(), &dist, &mat_empty(), &camera(), 3, 2, MAP_FIXED, &map1, &map2,
        )
        .expect("maps");
        assert_eq!(map1.depth(), MatDepth::I16);
        assert_eq!(map1.channels(), 2);
        assert_eq!(map2.depth(), MatDepth::U16);
        let bytes = map1.compact_bytes();
        let coords: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_ne_bytes(b.try_into().expect("i16")))
            .collect();
        assert_eq!(&coords[..6], &[0, 0, 1, 0, 2, 0]);
    }

    #[test]
    fn combined_float_map_uses_two_channels() {
        let dist = mat_empty();
        let map1 = mat_empty();
        let map2 = mat_empty();
        init_undistort_rectify_map_into(
            &camera(), &dist, &mat_empty(), &camera(), 3, 2, MAP_F32C2, &map1, &map2,
        )
        .expect("maps");
        assert_eq!(map1.channels(), 2);
        assert_eq!(map1.depth(), MatDepth::F32);
        assert_eq!(map2.rows(), 0);
        let values = f32_values(&map1);
        assert!((values[0]).abs() < 1e-6 && (values[1]).abs() < 1e-6, "{values:?}");
        assert!((values[2] - 1.0).abs() < 1e-6, "{values:?}");
    }

    #[test]
    fn undistorts_u8_with_fixed_point_resampling() {
        let source = crate::mat::mat_from_u8(
            &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
            4,
            4,
            1,
        )
        .expect("src");
        let small_camera =
            mat_from_f64(&[4.0, 0.0, 1.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0], 3, 3, 1).expect("k");
        let dist = mat_from_f64(&[0.05, -0.02, 0.001, 0.0, 0.0], 1, 5, 1).expect("dist");
        let out = mat_empty();
        undistort_into(&source, &out, &small_camera, &dist, &mat_empty()).expect("undistort");
        let expected = [
            9u8, 20, 30, 38, 50, 60, 70, 80, 90, 100, 110, 120, 122, 140, 150, 150,
        ];
        assert_eq!(out.compact_bytes(), expected);
    }
}
