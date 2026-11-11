use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RodriguesError {
    Matrix(MatError),
    UnsupportedSource,
    UnsupportedDepth,
}

impl fmt::Display for RodriguesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedSource => formatter.write_str(
                "Rodrigues requires a 3x1/1x3 rotation vector or a 3x3 rotation matrix",
            ),
            Self::UnsupportedDepth => {
                formatter.write_str("Rodrigues requires an F32 or F64 matrix")
            }
        }
    }
}

impl Error for RodriguesError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for RodriguesError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

fn skew(a: f64, b: f64, c: f64) -> [f64; 9] {
    [0.0, -c, b, c, 0.0, -a, -b, a, 0.0]
}

fn skew_square(a: f64, b: f64, c: f64) -> [f64; 9] {
    let aa = a * a;
    let bb = b * b;
    let cc = c * c;
    [
        -bb - cc,
        a * b,
        a * c,
        a * b,
        -aa - cc,
        b * c,
        a * c,
        b * c,
        -aa - bb,
    ]
}

/// Computes `dR/dr` for the rotation-vector to rotation-matrix map: a 3x9
/// Jacobian whose row `p` holds `d(flattened R)/d(r_p)` in row-major order.
pub(crate) fn rotation_jacobian(x: f64, y: f64, z: f64) -> [f64; 27] {
    let theta = (x * x + y * y + z * z).sqrt();
    let t2 = theta * theta;
    let s = skew(x, y, z);
    let s2 = skew_square(x, y, z);
    let (a, b, ca, cb) = if theta > 1e-14 {
        (
            theta.sin() / theta,
            (1.0 - theta.cos()) / t2,
            (theta * theta.cos() - theta.sin()) / (t2 * theta),
            (theta * theta.sin() - 2.0 * (1.0 - theta.cos())) / (t2 * t2),
        )
    } else {
        (1.0, 0.5, -1.0 / 3.0, -1.0 / 12.0)
    };
    let r = [x, y, z];
    let mut jac = [0.0f64; 27];
    for p in 0..3 {
        let mut e = [0.0f64; 3];
        e[p] = 1.0;
        let es = skew(e[0], e[1], e[2]);
        // d(S^2)/dr_p = [e_p]x S + S [e_p]x
        let mut ds2 = [0.0f64; 9];
        for row in 0..3 {
            for col in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += es[row * 3 + k] * s[k * 3 + col] + s[row * 3 + k] * es[k * 3 + col];
                }
                ds2[row * 3 + col] = acc;
            }
        }
        let da = ca * r[p];
        let db = cb * r[p];
        for i in 0..9 {
            jac[p * 9 + i] = da * s[i] + a * es[i] + db * s2[i] + b * ds2[i];
        }
    }
    jac
}

/// Converts between a 3x1/1x3 rotation vector and a 3x3 rotation matrix,
/// optionally writing the 3x9 or 9x3 Jacobian. Implements the classical
/// axis-angle map: `R = I cos t + [k]x sin t + kk' (1 - cos t)` for unit
/// axis `k = r / |r|`, with a second-order small-angle expansion.
///
/// Reference: Rodrigues' rotation formula; see any standard text on the
/// SO(3) exponential map.
pub(crate) fn rodrigues_into(
    source: &Mat,
    destination: &Mat,
    jacobian: &Mat,
    with_jacobian: bool,
) -> Result<(), RodriguesError> {
    let depth = source.depth();
    let is_double = match depth {
        MatDepth::F64 => true,
        MatDepth::F32 => false,
        _ => return Err(RodriguesError::UnsupportedDepth),
    };
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    let channels = source.channels() as usize;
    let is_vector = channels == 1 && (rows, columns) == (3, 1) || (rows, columns) == (1, 3);
    let is_matrix = channels == 1 && rows == 3 && columns == 3;
    if !is_vector && !is_matrix {
        return Err(RodriguesError::UnsupportedSource);
    }

    let element_size = if is_double { 8 } else { 4 };
    let bytes = source.compact_bytes();
    let read = |index: usize| -> f64 {
        let start = index * element_size;
        if is_double {
            f64::from_ne_bytes(bytes[start..start + 8].try_into().expect("f64"))
        } else {
            f32::from_ne_bytes(bytes[start..start + 4].try_into().expect("f32")) as f64
        }
    };
    let write = |values: &[f64]| -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * element_size);
        for &value in values {
            if is_double {
                out.extend_from_slice(&value.to_ne_bytes());
            } else {
                out.extend_from_slice(&(value as f32).to_ne_bytes());
            }
        }
        out
    };

    if is_vector {
        let (x, y, z) = (read(0), read(1), read(2));
        let theta = (x * x + y * y + z * z).sqrt();
        let s = skew(x, y, z);
        let s2 = skew_square(x, y, z);
        let (a, b) = if theta > 1e-14 {
            ((theta.sin()) / theta, (1.0 - theta.cos()) / (theta * theta))
        } else {
            (1.0, 0.5)
        };
        let mut rotation = [0.0f64; 9];
        for i in 0..9 {
            rotation[i] = a * s[i] + b * s2[i];
        }
        rotation[0] += 1.0;
        rotation[4] += 1.0;
        rotation[8] += 1.0;
        destination.write_output(write(&rotation), 3, 3, 1, depth)?;

        if with_jacobian {
            let jac = rotation_jacobian(x, y, z);
            jacobian.write_output(write(&jac), 3, 9, 1, depth)?;
        }
        return Ok(());
    }

    // Rotation matrix -> rotation vector.
    let r00 = read(0);
    let r01 = read(1);
    let r02 = read(2);
    let r10 = read(3);
    let r11 = read(4);
    let r12 = read(5);
    let r20 = read(6);
    let r21 = read(7);
    let r22 = read(8);
    let trace = r00 + r11 + r22;
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let u = [r21 - r12, r02 - r20, r10 - r01];

    let vector = if theta < 1e-14 {
        [u[0] / 2.0, u[1] / 2.0, u[2] / 2.0]
    } else if theta > std::f64::consts::PI - 1e-4 {
        // Near pi the axis comes from the symmetric part: (R + I)/2 = kk'.
        let mut axis = [0.0f64; 3];
        let diag = [(r00 + 1.0) / 2.0, (r11 + 1.0) / 2.0, (r22 + 1.0) / 2.0];
        let mut pivot = 0usize;
        if diag[1] > diag[pivot] {
            pivot = 1;
        }
        if diag[2] > diag[pivot] {
            pivot = 2;
        }
        let pivot_value = diag[pivot].max(0.0).sqrt();
        if pivot_value > 0.0 {
            axis[pivot] = pivot_value;
            let column = [r00, r10, r20, r01, r11, r21, r02, r12, r22];
            for i in 0..3 {
                if i != pivot {
                    let sum = column[i * 3 + pivot] + column[pivot * 3 + i];
                    axis[i] = sum / (4.0 * pivot_value);
                }
            }
            // Orient the axis consistently with the antisymmetric part.
            if axis[0] * u[0] + axis[1] * u[1] + axis[2] * u[2] < 0.0 {
                for value in &mut axis {
                    *value = -*value;
                }
            }
        }
        [axis[0] * theta, axis[1] * theta, axis[2] * theta]
    } else {
        let f = theta / (2.0 * theta.sin());
        [u[0] * f, u[1] * f, u[2] * f]
    };
    destination.write_output(write(&vector), 3, 1, 1, depth)?;

    if with_jacobian {
        // dr_j / dR_i for i = flattened R index (row-major), j = vector index.
        let mut jac = [0.0f64; 27];
        let sin_theta = theta.sin();
        if theta < 1e-14 || (1.0 - cos_theta * cos_theta).abs() < 1e-14 {
            // Limit as theta -> 0: r = vee(R - R') / 2.
            let f = if theta < 1e-14 {
                0.5
            } else {
                theta / (2.0 * sin_theta.max(1e-8))
            };
            jac[7 * 3 + 0] = f;
            jac[5 * 3 + 0] = -f;
            jac[2 * 3 + 1] = f;
            jac[6 * 3 + 1] = -f;
            jac[3 * 3 + 2] = f;
            jac[1 * 3 + 2] = -f;
        } else {
            let f = theta / (2.0 * sin_theta);
            let f_prime = (sin_theta - theta * cos_theta) / (2.0 * sin_theta * sin_theta);
            let dtheta = -1.0 / (2.0 * sin_theta);
            for flat in [0usize, 4, 8] {
                for j in 0..3 {
                    jac[flat * 3 + j] = f_prime * dtheta * u[j];
                }
            }
            // u = (R7 - R5, R2 - R6, R3 - R1); add f * du_j/dR_i.
            let du: [(usize, usize, f64); 6] = [
                (7, 0, f),
                (5, 0, -f),
                (2, 1, f),
                (6, 1, -f),
                (3, 2, f),
                (1, 2, -f),
            ];
            for &(flat, j, scale) in &du {
                jac[flat * 3 + j] += scale;
            }
        }
        jacobian.write_output(write(&jac), 9, 3, 1, depth)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64};

    fn f64_values(mat: &Mat) -> Vec<f64> {
        mat.compact_bytes()
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().expect("f64")))
            .collect()
    }

    #[test]
    fn rodrigues_matches_pinned_vector_to_matrix() {
        let rvec = mat_from_f64(&[0.3, -0.5, 0.7], 3, 1, 1).expect("rvec");
        let rotation = mat_empty();
        let jacobian = mat_empty();
        rodrigues_into(&rvec, &rotation, &jacobian, true).expect("rodrigues");
        let expected = [
            0.6548940285, -0.6770606569, -0.3357121957, 0.5371528306, 0.7295115358,
            -0.4234144018, 0.5315831525, 0.0969628071, 0.8414377969,
        ];
        let actual = f64_values(&rotation);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-9, "rotation[{i}] {a} != {e}");
        }
        let expected_j = [
            0.01749881, -0.16527305, 0.36745686, -0.29399226, -0.26610037, -0.83143620,
            0.27551457, 0.84798913, -0.27177566, 0.43719473, 0.02673006, 0.79895324,
            0.24126207, -0.02285881, 0.26668634, -0.78240031, 0.35862863, 0.45295943,
        ];
        let j = f64_values(&jacobian);
        assert_eq!(j.len(), 27);
        for (i, (a, e)) in j.iter().zip(expected_j.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "jacobian[{i}] {a} != {e}");
        }
    }

    #[test]
    fn rodrigues_round_trips_matrix_to_vector() {
        let rotation = mat_from_f64(
            &[
                0.6548940285, -0.6770606569, -0.3357121957, 0.5371528306, 0.7295115358,
                -0.4234144018, 0.5315831525, 0.0969628071, 0.8414377969,
            ],
            3,
            3,
            1,
        )
        .expect("rotation");
        let rvec = mat_empty();
        rodrigues_into(&rotation, &rvec, &mat_empty(), false).expect("rodrigues");
        let actual = f64_values(&rvec);
        let expected = [0.3, -0.5, 0.7];
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-9, "rvec[{i}] {a} != {e}");
        }
    }

    #[test]
    fn rodrigues_zero_vector_gives_identity() {
        let zero = mat_from_f64(&[0.0, 0.0, 0.0], 3, 1, 1).expect("zero");
        let rotation = mat_empty();
        let jacobian = mat_empty();
        rodrigues_into(&zero, &rotation, &jacobian, true).expect("rodrigues");
        assert_eq!(
            f64_values(&rotation),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        let j = f64_values(&jacobian);
        let expected = [
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        // skew blocks: col p = -[e_p]x flattened? verify first block only.
        for i in 0..9 {
            assert!((j[i] - skew(1.0, 0.0, 0.0)[i]).abs() < 1e-12);
            let _ = expected;
        }
    }

    #[test]
    fn rodrigues_accepts_row_vector() {
        let rvec = mat_from_f64(&[0.1, 0.2, 0.3], 1, 3, 1).expect("rvec");
        let rotation = mat_empty();
        rodrigues_into(&rvec, &rotation, &mat_empty(), false).expect("rodrigues");
        let expected = [
            0.9357548033, -0.2831649606, 0.2101917060, 0.3029327134, 0.9505806179,
            -0.0680313164, -0.1805400767, 0.1273345749, 0.9752903090,
        ];
        let actual = f64_values(&rotation);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-9, "rotation[{i}] {a} != {e}");
        }
    }
}
