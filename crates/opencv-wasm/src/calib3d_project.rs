use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProjectError {
    Matrix(MatError),
    UnsupportedDepth,
    UnsupportedPoints,
    UnsupportedVector,
    UnsupportedCamera,
    UnsupportedDistortion,
}

impl fmt::Display for ProjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth => formatter.write_str("points require an F32 or F64 depth"),
            Self::UnsupportedPoints => {
                formatter.write_str("object points require Nx1 3-channel or Nx3/3xN 1-channel shape")
            }
            Self::UnsupportedVector => formatter.write_str("rotation/translation require 3x1 or 1x3"),
            Self::UnsupportedCamera => formatter.write_str("camera matrix requires 3x3"),
            Self::UnsupportedDistortion => {
                formatter.write_str("distortion coefficients require 4, 5, 8, 12, or 14 elements")
            }
        }
    }
}

impl Error for ProjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for ProjectError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

fn read_elements(source: &Mat) -> Result<(Vec<f64>, MatDepth), ProjectError> {
    let depth = source.depth();
    let element_size = match depth {
        MatDepth::F64 => 8,
        MatDepth::F32 => 4,
        _ => return Err(ProjectError::UnsupportedDepth),
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
    Ok((values, depth))
}

fn read_vector3(source: &Mat) -> Result<[f64; 3], ProjectError> {
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    if source.channels() != 1 || !((rows, columns) == (3, 1) || (rows, columns) == (1, 3)) {
        return Err(ProjectError::UnsupportedVector);
    }
    let (values, _) = read_elements(source)?;
    Ok([values[0], values[1], values[2]])
}

fn rotation_from_vector(rvec: [f64; 3]) -> [f64; 9] {
    let [x, y, z] = rvec;
    let theta = (x * x + y * y + z * z).sqrt();
    let (a, b) = if theta > 1e-14 {
        (theta.sin() / theta, (1.0 - theta.cos()) / (theta * theta))
    } else {
        (1.0, 0.5)
    };
    let skew = [0.0, -z, y, z, 0.0, -x, -y, x, 0.0];
    let mut rotation = [0.0f64; 9];
    for row in 0..3 {
        for column in 0..3 {
            let mut s2 = 0.0;
            for k in 0..3 {
                s2 += skew[row * 3 + k] * skew[k * 3 + column];
            }
            rotation[row * 3 + column] = a * skew[row * 3 + column] + b * s2;
        }
    }
    rotation[0] += 1.0;
    rotation[4] += 1.0;
    rotation[8] += 1.0;
    rotation
}

/// Solves a 3x3 linear system `matrix * x = rhs` by Cramer's rule.
fn solve3(matrix: &[f64; 9], rhs: &[f64; 3]) -> [f64; 3] {
    let det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        let mut m = *matrix;
        for row in 0..3 {
            m[row * 3 + i] = rhs[row];
        }
        let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        out[i] = d / det;
    }
    out
}

/// Projects 3D object points onto the image plane using the pinhole model
/// with radial, tangential, rational, thin-prism, and tilted-sensor
/// distortion terms. Implements the standard documented model:
/// `x' = (R p + t)` normalized, `x_d = distort(x')`, `u = fx x_d + cx`.
/// Optionally writes the 2N x 15 F64 analytic Jacobian over
/// `[rvec, tvec, fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
///
/// Reference: standard pinhole camera projection with the polynomial
/// distortion model documented for `cv::projectPoints`.
pub(crate) fn project_points(
    object_points: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    image_points: &Mat,
    jacobian: &Mat,
    with_jacobian: bool,
    aspect_ratio: f64,
) -> Result<(), ProjectError> {
    let (point_values, depth) = read_elements(object_points)?;
    let rows = object_points.rows() as usize;
    let columns = object_points.columns() as usize;
    let channels = object_points.channels() as usize;

    let mut points: Vec<[f64; 3]> = Vec::new();
    if channels == 3 {
        if point_values.len() % 3 != 0 {
            return Err(ProjectError::UnsupportedPoints);
        }
        for chunk in point_values.chunks_exact(3) {
            points.push([chunk[0], chunk[1], chunk[2]]);
        }
    } else if channels == 1 && columns == 3 {
        for chunk in point_values.chunks_exact(3) {
            points.push([chunk[0], chunk[1], chunk[2]]);
        }
    } else if channels == 1 && rows == 3 {
        for column in 0..columns {
            points.push([
                point_values[column],
                point_values[columns + column],
                point_values[2 * columns + column],
            ]);
        }
    } else {
        return Err(ProjectError::UnsupportedPoints);
    }
    let count = points.len();

    let rvec_values = read_vector3(rvec)?;
    let rotation = rotation_from_vector(rvec_values);
    let dr_drvec = crate::calib3d_rodrigues::rotation_jacobian(
        rvec_values[0],
        rvec_values[1],
        rvec_values[2],
    );
    let translation = read_vector3(tvec)?;

    if camera_matrix.channels() != 1
        || camera_matrix.rows() != 3
        || camera_matrix.columns() != 3
    {
        return Err(ProjectError::UnsupportedCamera);
    }
    let (k_values, _) = read_elements(camera_matrix)?;
    let fx = k_values[0];
    let mut fy = k_values[4];
    let cx = k_values[2];
    let cy = k_values[5];
    if aspect_ratio != 0.0 {
        fy = fx * aspect_ratio;
    }

    let dist_element_count = dist_coeffs.rows() as usize
        * dist_coeffs.columns() as usize
        * dist_coeffs.channels() as usize;
    let dist_values = if dist_element_count == 0 {
        Vec::new()
    } else {
        read_elements(dist_coeffs)?.0
    };
    let dist_len = dist_values.len();
    if !(dist_len == 0 || dist_len == 4 || dist_len == 5 || dist_len == 8
        || dist_len == 12 || dist_len == 14)
    {
        return Err(ProjectError::UnsupportedDistortion);
    }
    let mut d = [0.0f64; 14];
    for (index, value) in dist_values.iter().enumerate() {
        d[index] = *value;
    }
    let (k1, k2, p1, p2, k3) = (d[0], d[1], d[2], d[3], d[4]);
    let (k4, k5, k6) = (d[5], d[6], d[7]);
    let (s1, s2, s3, s4) = (d[8], d[9], d[10], d[11]);
    let (tau_x, tau_y) = (d[12], d[13]);
    let has_rational = dist_len >= 8;
    let has_prism = dist_len >= 12;
    let has_tilt = dist_len == 14;

    // Tilted-sensor transform: project through R(tau_y, tau_x)^-1.
    let tilt_inverse: Option<[f64; 9]> = if has_tilt {
        let alpha = tau_y.atan2(tau_x);
        let (ca, sa) = (alpha.cos(), alpha.sin());
        let (cx_, sx_) = (tau_x.cos(), tau_x.sin());
        let (cy_, sy_) = (tau_y.cos(), tau_y.sin());
        // R = Rz(alpha) Rx(tau_x) Ry(tau_y) Rz(-alpha)
        let rx = [1.0, 0.0, 0.0, 0.0, cx_, -sx_, 0.0, sx_, cx_];
        let ry = [cy_, 0.0, sy_, 0.0, 1.0, 0.0, -sy_, 0.0, cy_];
        let rz_p = [ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0];
        let rz_n = [ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0];
        let mul = |a: &[f64; 9], b: &[f64; 9]| -> [f64; 9] {
            let mut out = [0.0f64; 9];
            for row in 0..3 {
                for column in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += a[row * 3 + k] * b[k * 3 + column];
                    }
                    out[row * 3 + column] = sum;
                }
            }
            out
        };
        Some(mul(&rz_p, &mul(&rx, &mul(&ry, &rz_n))))
    } else {
        None
    };

    let mut projected = Vec::with_capacity(count * 2);
    let mut jac_rows = if with_jacobian {
        Vec::with_capacity(count * 30)
    } else {
        Vec::new()
    };

    for point in &points {
        let xc = rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2]
            + translation[0];
        let yc = rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2]
            + translation[1];
        let zc = rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2]
            + translation[2];
        let x = xc / zc;
        let y = yc / zc;
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let (icdist, cdist_dr, icdist_dr) = if has_rational {
            (
                1.0 + k4 * r2 + k5 * r4 + k6 * r6,
                k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4,
                k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4,
            )
        } else {
            (1.0, k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4, 0.0)
        };
        let ratio = cdist / icdist;

        let mut xd = x * ratio + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let mut yd = y * ratio + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        if has_prism {
            xd += s1 * r2 + s2 * r4;
            yd += s3 * r2 + s4 * r4;
        }

        if let Some(inverse) = &tilt_inverse {
            let q = solve3(inverse, &[xd, yd, 1.0]);
            xd = q[0] / q[2];
            yd = q[1] / q[2];
        }

        projected.push(fx * xd + cx);
        projected.push(fy * yd + cy);

        if with_jacobian {
            // Derivatives of (xd, yd) w.r.t. normalized (x, y).
            let ratio_dx = 2.0 * x * (cdist_dr * icdist - cdist * icdist_dr) / (icdist * icdist);
            let ratio_dy = 2.0 * y * (cdist_dr * icdist - cdist * icdist_dr) / (icdist * icdist);
            let mut j11 = ratio + x * ratio_dx + 2.0 * p1 * y + 6.0 * p2 * x;
            let mut j12 = x * ratio_dy + 2.0 * p1 * x + 2.0 * p2 * y;
            let mut j21 = y * ratio_dx + 2.0 * p1 * x + 2.0 * p2 * y;
            let mut j22 = ratio + y * ratio_dy + 6.0 * p1 * y + 2.0 * p2 * x;
            if has_prism {
                j11 += 2.0 * s1 * x + 4.0 * s2 * x * r2;
                j12 += 2.0 * s1 * y + 4.0 * s2 * y * r2;
                j21 += 2.0 * s3 * x + 4.0 * s4 * x * r2;
                j22 += 2.0 * s3 * y + 4.0 * s4 * y * r2;
            }

            // d(normalized x, y)/d(rvec_p) through the full dR/dr chain:
            // dX'/dr_p = (dR/dr_p) * point, then through perspective division.
            let inv_z = 1.0 / zc;
            let nt = [
                [inv_z, 0.0, -x * inv_z],
                [0.0, inv_z, -y * inv_z],
            ];
            let mut row_u = [0.0f64; 15];
            let mut row_v = [0.0f64; 15];
            for column in 0..3 {
                let m = &dr_drvec[column * 9..column * 9 + 9];
                let dxp = [
                    m[0] * point[0] + m[1] * point[1] + m[2] * point[2],
                    m[3] * point[0] + m[4] * point[1] + m[5] * point[2],
                    m[6] * point[0] + m[7] * point[1] + m[8] * point[2],
                ];
                let dxn = dxp[0] * inv_z - x * dxp[2] * inv_z;
                let dyn_ = dxp[1] * inv_z - y * dxp[2] * inv_z;
                row_u[column] = fx * (j11 * dxn + j12 * dyn_);
                row_v[column] = fy * (j21 * dxn + j22 * dyn_);
                row_u[3 + column] = fx * (j11 * nt[0][column] + j12 * nt[1][column]);
                row_v[3 + column] = fy * (j21 * nt[0][column] + j22 * nt[1][column]);
            }
            row_u[6] = xd;
            row_u[8] = 1.0;
            row_v[7] = yd;
            row_v[9] = 1.0;
            row_u[10] = fx * x * r2;
            row_u[11] = fx * x * r4;
            row_u[12] = fx * 2.0 * x * y;
            row_u[13] = fx * (r2 + 2.0 * x * x);
            row_u[14] = fx * x * r6;
            row_v[10] = fy * y * r2;
            row_v[11] = fy * y * r4;
            row_v[12] = fy * (r2 + 2.0 * y * y);
            row_v[13] = fy * 2.0 * x * y;
            row_v[14] = fy * y * r6;
            jac_rows.extend_from_slice(&row_u);
            jac_rows.extend_from_slice(&row_v);
        }
    }

    // Output: Nx1 2-channel, depth follows the input points.
    let element_size = if depth == MatDepth::F64 { 8 } else { 4 };
    let mut out = Vec::with_capacity(count * 2 * element_size);
    for value in &projected {
        if depth == MatDepth::F64 {
            out.extend_from_slice(&value.to_ne_bytes());
        } else {
            out.extend_from_slice(&(*value as f32).to_ne_bytes());
        }
    }
    image_points.write_output(out, count as u32, 1, 2, depth)?;

    if with_jacobian {
        let mut jac_bytes = Vec::with_capacity(count * 30 * 8);
        for value in &jac_rows {
            jac_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        jacobian.write_output(jac_bytes, (count * 2) as u32, 15, 1, MatDepth::F64)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f32, mat_from_f64};

    fn f64_values(mat: &Mat) -> Vec<f64> {
        mat.compact_bytes()
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().expect("f64")))
            .collect()
    }

    fn camera() -> Mat {
        mat_from_f64(&[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0], 3, 3, 1)
            .expect("camera")
    }

    fn zero_vec() -> Mat {
        mat_from_f64(&[0.0, 0.0, 0.0], 3, 1, 1).expect("vec")
    }

    #[test]
    fn projects_identity_without_distortion() {
        let points = mat_from_f64(&[0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0, 10.0], 3, 1, 3)
            .expect("points");
        let dist = mat_from_f64(&[0.0; 5], 1, 5, 1).expect("dist");
        let out = mat_empty();
        project_points(
            &points, &zero_vec(), &zero_vec(), &camera(), &dist, &out, &mat_empty(),
            false, 0.0,
        )
        .expect("project");
        let values = f64_values(&out);
        let expected = [320.0, 240.0, 370.0, 240.0, 320.0, 290.0];
        for (a, e) in values.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "{a} != {e}");
        }
    }

    #[test]
    fn projects_with_radial_tangential_distortion() {
        let points = mat_from_f64(&[0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0, 10.0], 3, 1, 3)
            .expect("points");
        let dist = mat_from_f64(&[0.1, 0.01, 0.001, 0.002, 0.0], 1, 5, 1).expect("dist");
        let out = mat_empty();
        project_points(
            &points, &zero_vec(), &zero_vec(), &camera(), &dist, &out, &mat_empty(),
            false, 0.0,
        )
        .expect("project");
        let values = f64_values(&out);
        let expected = [
            320.0,
            240.0,
            370.080_049_999_999_97,
            240.005,
            320.01,
            290.065_050_000_000_04,
        ];
        for (a, e) in values.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "{a} != {e}");
        }
    }

    #[test]
    fn jacobian_matches_pinned_columns() {
        let points = mat_from_f64(&[1.0, 2.0, 10.0], 1, 1, 3).expect("points");
        let dist = mat_from_f64(&[0.0; 5], 1, 5, 1).expect("dist");
        let out = mat_empty();
        let jac = mat_empty();
        project_points(
            &points, &zero_vec(), &zero_vec(), &camera(), &dist, &out, &jac, true, 0.0,
        )
        .expect("project");
        let j = f64_values(&jac);
        assert_eq!(j.len(), 30);
        let expected = [
            -10.0, 505.0, -100.0, 50.0, 0.0, -5.0, 0.1, 0.0, 1.0, 0.0, 2.5, 0.125, 20.0, 35.0,
            0.00625, -520.0, 10.0, 50.0, 0.0, 50.0, -10.0, 0.0, 0.2, 0.0, 1.0, 5.0, 0.25, 65.0,
            20.0, 0.0125,
        ];
        for (index, (a, e)) in j.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-9, "jacobian[{index}] {a} != {e}");
        }
    }

    #[test]
    fn supports_row_major_and_f32_inputs() {
        let points = mat_from_f32(&[1.0, 0.0, 10.0, 0.0, 1.0, 10.0], 2, 3, 1).expect("points");
        let dist = mat_empty();
        let out = mat_empty();
        project_points(
            &points, &zero_vec(), &zero_vec(), &camera(), &dist, &out, &mat_empty(),
            false, 0.0,
        )
        .expect("project");
        assert_eq!(out.depth(), MatDepth::F32);
        assert_eq!(out.rows(), 2);
        assert_eq!(out.channels(), 2);
        let bytes = out.compact_bytes();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().expect("f32")))
            .collect();
        let expected = [370.0f32, 240.0, 320.0, 290.0];
        for (a, e) in values.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "{a} != {e}");
        }
    }

    #[test]
    fn aspect_ratio_scales_fy() {
        let points = mat_from_f64(&[0.0, 1.0, 10.0], 1, 1, 3).expect("points");
        let dist = mat_empty();
        let out = mat_empty();
        project_points(
            &points, &zero_vec(), &zero_vec(), &camera(), &dist, &out, &mat_empty(),
            false, 2.0,
        )
        .expect("project");
        let values = f64_values(&out);
        assert!((values[0] - 320.0).abs() < 1e-9);
        assert!((values[1] - (240.0 + 1000.0 * 0.1)).abs() < 1e-9, "{}", values[1]);
    }
}
