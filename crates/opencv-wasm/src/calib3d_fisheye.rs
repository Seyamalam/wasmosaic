use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FisheyeError {
    Matrix(MatError),
    UnsupportedDepth,
    UnsupportedPoints,
    UnsupportedVector,
    UnsupportedCamera,
    UnsupportedRectify,
    UnsupportedDistortion,
    UnsupportedMapType,
    InvalidSize,
}

impl fmt::Display for FisheyeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth => {
                formatter.write_str("calibration inputs require an F32 or F64 depth")
            }
            Self::UnsupportedPoints => {
                formatter.write_str("object points require a 3-channel vector")
            }
            Self::UnsupportedVector => {
                formatter.write_str("rotation and translation vectors require 3 elements")
            }
            Self::UnsupportedCamera => formatter.write_str("camera matrices require 3x3"),
            Self::UnsupportedRectify => {
                formatter.write_str("rectification requires a 3x3 matrix")
            }
            Self::UnsupportedDistortion => {
                formatter.write_str("fisheye distortion requires exactly 4 coefficients")
            }
            Self::UnsupportedMapType => {
                formatter.write_str("fisheye map type requires CV_32FC1 or CV_16SC2")
            }
            Self::InvalidSize => formatter.write_str("map size must be positive"),
        }
    }
}

impl Error for FisheyeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for FisheyeError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

const MAP_F32C1: i32 = 5;
const MAP_FIXED: i32 = 11;

fn read_elements(source: &Mat) -> Result<Vec<f64>, FisheyeError> {
    let element_size = match source.depth() {
        MatDepth::F64 => 8,
        MatDepth::F32 => 4,
        _ => return Err(FisheyeError::UnsupportedDepth),
    };
    let bytes = source.compact_bytes();
    let count = bytes.len() / element_size;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * element_size;
        values.push(if element_size == 8 {
            f64::from_ne_bytes(bytes[start..start + 8].try_into().expect("f64"))
        } else {
            f64::from(f32::from_ne_bytes(
                bytes[start..start + 4].try_into().expect("f32"),
            ))
        });
    }
    Ok(values)
}

fn element_count(source: &Mat) -> usize {
    source.rows() as usize * source.columns() as usize * source.channels() as usize
}

fn read_camera(source: &Mat) -> Result<[f64; 9], FisheyeError> {
    if source.channels() != 1 || source.rows() != 3 || source.columns() != 3 {
        return Err(FisheyeError::UnsupportedCamera);
    }
    let values = read_elements(source)?;
    Ok([
        values[0], values[1], values[2], values[3], values[4], values[5], values[6],
        values[7], values[8],
    ])
}

/// Reads a 3-element rotation or translation vector; any other shape,
/// including a 3x3 rotation matrix, rejects like the pinned binding.
fn read_vector3(source: &Mat) -> Result<[f64; 3], FisheyeError> {
    if element_count(source) != 3 || source.channels() != 1 {
        return Err(FisheyeError::UnsupportedVector);
    }
    let values = read_elements(source)?;
    Ok([values[0], values[1], values[2]])
}

/// Reads exactly four fisheye distortion coefficients.
fn read_distortion4(source: &Mat) -> Result<[f64; 4], FisheyeError> {
    if element_count(source) != 4 || source.channels() != 1 {
        return Err(FisheyeError::UnsupportedDistortion);
    }
    let values = read_elements(source)?;
    Ok([values[0], values[1], values[2], values[3]])
}

fn invert3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    [
        (m[4] * m[8] - m[5] * m[7]) / det,
        (m[2] * m[7] - m[1] * m[8]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        (m[5] * m[6] - m[3] * m[8]) / det,
        (m[0] * m[8] - m[2] * m[6]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        (m[3] * m[7] - m[4] * m[6]) / det,
        (m[1] * m[6] - m[0] * m[7]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ]
}

fn multiply3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0f64; 9];
    for row in 0..3 {
        for column in 0..3 {
            out[row * 3 + column] = a[row * 3] * b[column]
                + a[row * 3 + 1] * b[3 + column]
                + a[row * 3 + 2] * b[6 + column];
        }
    }
    out
}

/// Equidistant distortion angle: `theta_d = theta (1 + k1 t2 + k2 t4 +
/// k3 t6 + k4 t8)` where `theta = atan(r)` and `t*` are powers of theta.
/// Returns `(theta_d, d theta_d / d theta, d theta_d / d k_i)`.
fn fisheye_angle(r: f64, coeffs: &[f64; 4]) -> (f64, f64, [f64; 4]) {
    let theta = r.atan();
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    let (k1, k2, k3, k4) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    let theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
    let theta_d_dt =
        1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8;
    let theta_d_dk = [theta * t2, theta * t4, theta * t6, theta * t8];
    (theta_d, theta_d_dt, theta_d_dk)
}

/// Forward equidistant scale for the rectification map: maps the
/// normalized undistorted radius `r` to the distorted radius `theta_d`.
fn fisheye_scale(r: f64, theta_d: f64) -> f64 {
    if r > 0.0 { theta_d / r } else { 1.0 }
}

/// `cv::fisheye::projectPoints`: projects 3-channel object points through
/// the equidistant model. `u = fx (x_d + alpha y_d) + cx`,
/// `v = fy y_d + cy`. Optionally writes the 2N x 15 F64 Jacobian over
/// `[fx, fy, cx, cy, k1..k4, rvec, tvec, alpha]`.
///
/// Reference: the equidistant projection model documented for
/// `cv::fisheye::projectPoints`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fisheye_project_points_into(
    object_points: &Mat,
    image_points: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    alpha: f64,
    jacobian: &Mat,
    with_jacobian: bool,
) -> Result<(), FisheyeError> {
    let depth = object_points.depth();
    if depth != MatDepth::F32 && depth != MatDepth::F64 {
        return Err(FisheyeError::UnsupportedDepth);
    }
    if object_points.channels() != 3 {
        return Err(FisheyeError::UnsupportedPoints);
    }
    let point_values = read_elements(object_points)?;
    let mut points: Vec<[f64; 3]> = Vec::new();
    for chunk in point_values.chunks_exact(3) {
        points.push([chunk[0], chunk[1], chunk[2]]);
    }
    let count = points.len();

    let rv = read_vector3(rvec)?;
    let tv = read_vector3(tvec)?;
    let camera = read_camera(camera_matrix)?;
    let d = read_distortion4(dist_coeffs)?;
    let (fx, fy, cx, cy) = (camera[0], camera[4], camera[2], camera[5]);

    let rotation = crate::calib3d_rodrigues::rotation_from_vector(rv);
    let dr_drvec = crate::calib3d_rodrigues::rotation_jacobian(rv[0], rv[1], rv[2]);

    let mut projected = Vec::with_capacity(count * 2);
    let mut jac_rows = if with_jacobian {
        Vec::with_capacity(count * 30)
    } else {
        Vec::new()
    };

    for point in &points {
        let xc = rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2]
            + tv[0];
        let yc = rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2]
            + tv[1];
        let zc = rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2]
            + tv[2];
        let a = xc / zc;
        let b = yc / zc;
        let r = (a * a + b * b).sqrt();
        let (theta_d, theta_d_dt, theta_d_dk) = fisheye_angle(r, &d);
        let scale = fisheye_scale(r, theta_d);
        let xd = a * scale;
        let yd = b * scale;

        projected.push(fx * (xd + alpha * yd) + cx);
        projected.push(fy * yd + cy);

        if with_jacobian {
            // s(r) = theta_d(theta) / r with theta = atan r, so
            // s' = theta_d_dt / (r (1 + r^2)) - theta_d / r^2.
            let ds_dr = if r > 0.0 {
                theta_d_dt / (r * (1.0 + r * r)) - theta_d / (r * r)
            } else {
                0.0
            };
            // d(xd)/da, d(xd)/db, d(yd)/da, d(yd)/db through r.
            let dr_da = if r > 0.0 { a / r } else { 0.0 };
            let dr_db = if r > 0.0 { b / r } else { 0.0 };
            let dxd_da = scale + a * ds_dr * dr_da;
            let dxd_db = a * ds_dr * dr_db;
            let dyd_da = b * ds_dr * dr_da;
            let dyd_db = scale + b * ds_dr * dr_db;
            // (a, b) = (xc/zc, yc/zc)
            let inv_z = 1.0 / zc;
            let inv_z2 = inv_z * inv_z;
            // d(xd)/d(xc, yc, zc) and d(yd)/d(xc, yc, zc)
            let dxd_dxyz = [
                dxd_da * inv_z,
                dxd_db * inv_z,
                -(dxd_da * xc + dxd_db * yc) * inv_z2,
            ];
            let dyd_dxyz = [
                dyd_da * inv_z,
                dyd_db * inv_z,
                -(dyd_da * xc + dyd_db * yc) * inv_z2,
            ];
            // d(xc, yc, zc)/d rvec = dR/dr * point, chained through the
            // shared Rodrigues rotation Jacobian.
            let mut dxc_drvec = [0.0f64; 3];
            let mut dyc_drvec = [0.0f64; 3];
            let mut dzc_drvec = [0.0f64; 3];
            for column in 0..3 {
                for row in 0..3 {
                    // jac[p * 9 + (row * 3 + k)] = d R[row][k] / d r_p.
                    let base = column * 9 + row * 3;
                    let value = dr_drvec[base] * point[0]
                        + dr_drvec[base + 1] * point[1]
                        + dr_drvec[base + 2] * point[2];
                    match row {
                        0 => dxc_drvec[column] = value,
                        1 => dyc_drvec[column] = value,
                        _ => dzc_drvec[column] = value,
                    }
                }
            }
            let du_row = |dxd: f64, dyd: f64| fx * (dxd + alpha * dyd);
            let dv_row = |_dxd: f64, dyd: f64| fy * dyd;

            let mut du = [0.0f64; 15];
            let mut dv = [0.0f64; 15];
            du[0] = xd + alpha * yd;
            dv[1] = yd;
            du[2] = 1.0;
            dv[3] = 1.0;
            for i in 0..4 {
                // d(xd)/dk_i = a * theta_d_dk_i / r, d(yd)/dk_i = b * ... / r
                let (dxd_dk, dyd_dk) = if r > 0.0 {
                    (a * theta_d_dk[i] / r, b * theta_d_dk[i] / r)
                } else {
                    (0.0, 0.0)
                };
                du[4 + i] = du_row(dxd_dk, dyd_dk);
                dv[4 + i] = dv_row(dxd_dk, dyd_dk);
            }
            for i in 0..3 {
                let dxd = dxd_dxyz[0] * dxc_drvec[i]
                    + dxd_dxyz[1] * dyc_drvec[i]
                    + dxd_dxyz[2] * dzc_drvec[i];
                let dyd = dyd_dxyz[0] * dxc_drvec[i]
                    + dyd_dxyz[1] * dyc_drvec[i]
                    + dyd_dxyz[2] * dzc_drvec[i];
                du[8 + i] = du_row(dxd, dyd);
                dv[8 + i] = dv_row(dxd, dyd);
                let dxd_t = dxd_dxyz[i];
                let dyd_t = dyd_dxyz[i];
                du[11 + i] = du_row(dxd_t, dyd_t);
                dv[11 + i] = dv_row(dxd_t, dyd_t);
            }
            du[14] = fx * yd;
            jac_rows.extend_from_slice(&du);
            jac_rows.extend_from_slice(&dv);
        }
    }

    let (rows, columns) = (object_points.rows(), object_points.columns());
    let out_depth = match depth {
        MatDepth::F32 => MatDepth::F32,
        _ => MatDepth::F64,
    };
    let element_size = if out_depth == MatDepth::F32 { 4 } else { 8 };
    let mut bytes = Vec::with_capacity(count * 2 * element_size);
    match out_depth {
        MatDepth::F32 => {
            for value in &projected {
                bytes.extend_from_slice(&(*value as f32).to_ne_bytes());
            }
        }
        _ => {
            for value in &projected {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
    }
    image_points.write_output(bytes, rows, columns, 2, out_depth)?;

    if with_jacobian {
        let mut jac_bytes = Vec::with_capacity(count * 30 * 8);
        for value in &jac_rows {
            jac_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        jacobian.write_output(jac_bytes, (count * 2) as u32, 15, 1, MatDepth::F64)?;
    }
    Ok(())
}

/// `cv::fisheye::initUndistortRectifyMap`: dense backward maps through the
/// forward equidistant model in the CV_32FC1 and CV_16SC2 (+CV_16UC1)
/// forms. An empty `P` resolves to the identity matrix like the pinned
/// binding, and an empty `R` rectifies as identity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fisheye_init_undistort_rectify_map_into(
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rotation: &Mat,
    new_camera_matrix: &Mat,
    width: u32,
    height: u32,
    m1type: i32,
    map1: &Mat,
    map2: &Mat,
) -> Result<(), FisheyeError> {
    if width == 0 || height == 0 {
        return Err(FisheyeError::InvalidSize);
    }
    let camera = read_camera(camera_matrix)?;
    let d = read_distortion4(dist_coeffs)?;
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let rectification = if element_count(rotation) == 0 {
        identity
    } else if rotation.channels() == 1 && rotation.rows() == 3 && rotation.columns() == 3 {
        invert3(&read_camera(rotation)?)
    } else {
        return Err(FisheyeError::UnsupportedRectify);
    };
    let projection_inverse = if element_count(new_camera_matrix) == 0 {
        identity
    } else {
        invert3(&read_camera(new_camera_matrix)?)
    };
    if !(m1type == MAP_F32C1 || m1type == MAP_FIXED) {
        return Err(FisheyeError::UnsupportedMapType);
    }
    let ir = multiply3(&rectification, &projection_inverse);

    let n = width as usize * height as usize;
    let mut ux = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    for v in 0..height as usize {
        for u in 0..width as usize {
            let (uf, vf) = (u as f64, v as f64);
            let tx = ir[0] * uf + ir[1] * vf + ir[2];
            let ty = ir[3] * uf + ir[4] * vf + ir[5];
            let tz = ir[6] * uf + ir[7] * vf + ir[8];
            let x = tx / tz;
            let y = ty / tz;
            let r = (x * x + y * y).sqrt();
            let (theta_d, _, _) = fisheye_angle(r, &d);
            let scale = fisheye_scale(r, theta_d);
            ux.push((camera[0] * (x * scale) + camera[2]) as f32);
            vy.push((camera[4] * (y * scale) + camera[5]) as f32);
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
        _ => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64};

    fn f64_mat(rows: u32, columns: u32, channels: u16, values: &[f64]) -> Mat {
        mat_from_f64(values, rows, columns, channels).expect("mat")
    }

    fn empty() -> Mat {
        mat_empty()
    }

    fn camera() -> Mat {
        mat_from_f64(
            &[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            3,
            3,
            1,
        )
        .expect("camera")
    }

    fn distortion() -> Mat {
        f64_mat(1, 4, 1, &[0.01, 0.03, -0.02, 0.005])
    }

    fn f32_values(mat: &Mat) -> Vec<f32> {
        mat.compact_bytes()
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().expect("f32")))
            .collect()
    }

    fn f64_values(mat: &Mat) -> Vec<f64> {
        mat.compact_bytes()
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().expect("f64")))
            .collect()
    }

    #[test]
    fn projects_equidistant_points() {
        let object = f64_mat(1, 3, 3, &[0.1, 0.2, 1.0, -0.3, 0.05, 1.5, 0.0, 0.0, 2.0]);
        let image = empty();
        let zero3 = f64_mat(3, 1, 1, &[0.0, 0.0, 0.0]);
        fisheye_project_points_into(
            &object, &image, &zero3, &zero3, &camera(), &distortion(), 0.0, &empty(), false,
        )
        .expect("project");
        let values = f64_values(&image);
        // Pinned values from the differential probe.
        let expected = [
            369.2179589750708,
            338.43591795014163,
            221.29343131726142,
            256.45109478045646,
            320.0,
            240.0,
        ];
        for (index, want) in expected.iter().enumerate() {
            assert!(
                (values[index] - want).abs() < 1e-9,
                "index {index}: {} vs {want}",
                values[index]
            );
        }
    }

    #[test]
    fn alpha_adds_skew_to_u() {
        let object = f64_mat(1, 2, 3, &[0.1, 0.2, 1.0, -0.3, 0.05, 1.5]);
        let image = empty();
        let zero3 = f64_mat(3, 1, 1, &[0.0, 0.0, 0.0]);
        fisheye_project_points_into(
            &object, &image, &zero3, &zero3, &camera(), &distortion(), 0.5, &empty(), false,
        )
        .expect("project");
        let values = f64_values(&image);
        assert!((values[0] - 418.43591795014163).abs() < 1e-9);
        assert!((values[2] - 229.51897870748962).abs() < 1e-9);
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let object = f64_mat(1, 2, 3, &[0.1, 0.2, 1.0, -0.3, 0.05, 1.5]);
        let rvec = f64_mat(3, 1, 1, &[0.1, -0.2, 0.15]);
        let tvec = f64_mat(3, 1, 1, &[10.0, -5.0, 30.0]);
        let image = empty();
        let jac = empty();
        fisheye_project_points_into(
            &object, &image, &rvec, &tvec, &camera(), &distortion(), 0.0, &jac, true,
        )
        .expect("project");
        assert_eq!(jac.rows(), 4);
        assert_eq!(jac.columns(), 15);
        assert_eq!(jac.depth(), MatDepth::F64);
        let values = f64_values(&jac);
        // Pinned first-row entries from finite-difference probes.
        let row0_expected = [
            (0, 0.3064157336580513),
            (2, 1.0),
            (4, 17.848142652029086),
            (8, -0.025207730045296205),
            (11, 14.612189431581639),
            (13, -4.582052607422712),
            (14, -76.06939182307472),
        ];
        for (column, want) in row0_expected {
            assert!(
                (values[column] - want).abs() < 1e-9,
                "row0 col{column}: {} vs {want}",
                values[column]
            );
        }
        let row1_expected = [
            (1, -0.15213878364614944),
            (3, 1.0),
            (8, -15.249803605306584),
            (12, 15.292072168643202),
            (14, 0.0),
        ];
        for (column, want) in row1_expected {
            assert!(
                (values[15 + column] - want).abs() < 1e-9,
                "row1 col{column}: {} vs {want}",
                values[15 + column]
            );
        }
    }

    #[test]
    fn rejects_single_channel_points_and_matrix_rvec() {
        let object = f64_mat(3, 3, 1, &[0.0; 9]);
        let zero3 = f64_mat(3, 1, 1, &[0.0, 0.0, 0.0]);
        assert_eq!(
            fisheye_project_points_into(
                &object,
                &empty(),
                &zero3,
                &zero3,
                &camera(),
                &distortion(),
                0.0,
                &empty(),
                false,
            )
            .unwrap_err(),
            FisheyeError::UnsupportedPoints
        );
        let vector = f64_mat(1, 3, 3, &[0.1, 0.2, 1.0, -0.3, 0.05, 1.5, 0.0, 0.0, 2.0]);
        let r33 = f64_mat(3, 3, 1, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            fisheye_project_points_into(
                &vector, &empty(), &r33, &zero3, &camera(), &distortion(), 0.0, &empty(), false,
            )
            .unwrap_err(),
            FisheyeError::UnsupportedVector
        );
    }

    #[test]
    fn rejects_nonfour_distortion_lengths() {
        let vector = f64_mat(1, 1, 3, &[0.1, 0.2, 1.0]);
        let zero3 = f64_mat(3, 1, 1, &[0.0, 0.0, 0.0]);
        for length in [0usize, 3, 5] {
            let d = if length == 0 {
                empty()
            } else {
                f64_mat(1, length as u32, 1, &vec![0.01; length])
            };
            assert_eq!(
                fisheye_project_points_into(
                    &vector, &empty(), &zero3, &zero3, &camera(), &d, 0.0, &empty(), false,
                )
                .unwrap_err(),
                FisheyeError::UnsupportedDistortion
            );
        }
    }

    #[test]
    fn map_matches_pinned_distorted_values() {
        let r = f64_mat(3, 3, 1, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let map1 = empty();
        let map2 = empty();
        fisheye_init_undistort_rectify_map_into(
            &camera(), &distortion(), &r, &camera(), 6, 4, MAP_F32C1, &map1, &map2,
        )
        .expect("maps");
        let x = f32_values(&map1);
        let y = f32_values(&map2);
        let expected_x = [
            47.647972106933594f32,
            48.35820388793945,
            49.06949234008789,
            49.781837463378906,
        ];
        let expected_y = [
            35.73598098754883f32,
            35.62999725341797,
            35.52414321899414,
            35.418426513671875,
        ];
        for index in 0..4 {
            assert!((x[index] - expected_x[index]).abs() < 1e-4, "{x:?}");
            assert!((y[index] - expected_y[index]).abs() < 1e-4, "{y:?}");
        }
    }

    #[test]
    fn empty_projection_rectifies_against_identity() {
        let map1 = empty();
        let map2 = empty();
        fisheye_init_undistort_rectify_map_into(
            &camera(),
            &distortion(),
            &empty(),
            &empty(),
            6,
            4,
            MAP_F32C1,
            &map1,
            &map2,
        )
        .expect("maps");
        let x = f32_values(&map1);
        let y = f32_values(&map2);
        // Empty P resolves to identity: pixel (0,0) back-projects to the
        // principal point and pixel (1,0) to 718.045.
        assert!((x[0] - 320.0).abs() < 1e-4, "{x:?}");
        assert!((y[0] - 240.0).abs() < 1e-4, "{y:?}");
        assert!((x[1] - 718.0449829101562).abs() < 1e-4, "{x:?}");
        assert!((y[6] - 638.0449829101562).abs() < 1e-4, "{y:?}");
    }

    #[test]
    fn fixed_point_maps_encode_5_bit_fractions() {
        let r = f64_mat(3, 3, 1, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let map1 = empty();
        let map2 = empty();
        fisheye_init_undistort_rectify_map_into(
            &camera(), &distortion(), &r, &camera(), 6, 4, MAP_FIXED, &map1, &map2,
        )
        .expect("maps");
        assert_eq!(map1.depth(), MatDepth::I16);
        assert_eq!(map1.channels(), 2);
        assert_eq!(map2.depth(), MatDepth::U16);
        let m1 = map1.compact_bytes();
        let first_x = i16::from_ne_bytes([m1[0], m1[1]]);
        let first_y = i16::from_ne_bytes([m1[2], m1[3]]);
        assert_eq!(first_x, 47);
        assert_eq!(first_y, 35);
        let m2 = map2.compact_bytes();
        let tab = u16::from_ne_bytes([m2[0], m2[1]]);
        assert_eq!(tab & 31, 21);
        assert_eq!(tab >> 5, 24);
    }

    #[test]
    fn rejects_combined_float_map() {
        let r = f64_mat(3, 3, 1, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            fisheye_init_undistort_rectify_map_into(
                &camera(),
                &distortion(),
                &r,
                &camera(),
                6,
                4,
                13,
                &empty(),
                &empty(),
            )
            .unwrap_err(),
            FisheyeError::UnsupportedMapType
        );
    }
}

