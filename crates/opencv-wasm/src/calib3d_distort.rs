//! Shared camera-distortion helpers for the calib3d module.
//!
//! Implements the documented polynomial model: radial `k1..k3`, tangential
//! `p1, p2`, rational `k4..k6` (8-coefficient form), thin-prism `s1..s4`
//! (12-coefficient form), and tilted-sensor `tau_x, tau_y` (14-coefficient
//! form). Coefficient vectors of length 4, 5, 8, 12, or 14 are accepted;
//! shorter vectors zero-pad the leading terms.

/// Precomputed tilt transform for the 14-coefficient model.
pub(crate) fn tilt_rotation(tau_x: f64, tau_y: f64) -> [f64; 9] {
    let alpha = tau_y.atan2(tau_x);
    let (ca, sa) = (alpha.cos(), alpha.sin());
    let (cx_, sx_) = (tau_x.cos(), tau_x.sin());
    let (cy_, sy_) = (tau_y.cos(), tau_y.sin());
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
    mul(&rz_p, &mul(&rx, &mul(&ry, &rz_n)))
}

/// Solves a 3x3 linear system `matrix * x = rhs` by Cramer's rule.
pub(crate) fn solve3(matrix: &[f64; 9], rhs: &[f64; 3]) -> [f64; 3] {
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

/// Applies the documented distortion model to normalized coordinates.
/// `coeffs` is the zero-padded 14-element coefficient vector; `length` is
/// the source vector length (0, 4, 5, 8, 12, or 14).
pub(crate) fn distort_normalized(x: f64, y: f64, coeffs: &[f64; 14], length: usize) -> (f64, f64) {
    let (k1, k2, p1, p2, k3) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
    let (k4, k5, k6) = (coeffs[5], coeffs[6], coeffs[7]);
    let (s1, s2, s3, s4) = (coeffs[8], coeffs[9], coeffs[10], coeffs[11]);
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let ratio = if length >= 8 {
        cdist / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    } else {
        cdist
    };
    let mut xd = x * ratio + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let mut yd = y * ratio + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    if length >= 12 {
        xd += s1 * r2 + s2 * r4;
        yd += s3 * r2 + s4 * r4;
    }
    if length == 14 {
        let q = solve3(&tilt_rotation(coeffs[12], coeffs[13]), &[xd, yd, 1.0]);
        xd = q[0] / q[2];
        yd = q[1] / q[2];
    }
    (xd, yd)
}
