//! Small image-transform matrix construction kernels.

use std::{error::Error, fmt};

/// Failures reported while constructing an image-transform matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransformMatrixError {
    UnsupportedSolveMethod(i32),
    DegenerateGeometry,
    NumericalFailure,
}

impl fmt::Display for TransformMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSolveMethod(method) => write!(
                formatter,
                "perspective solve method {method} is not implemented; use LU (0) or QR (4)"
            ),
            Self::DegenerateGeometry => formatter
                .write_str("transform matrix cannot be determined from degenerate input geometry"),
            Self::NumericalFailure => formatter
                .write_str("transform matrix calculation produced a non-finite intermediate value"),
        }
    }
}

impl Error for TransformMatrixError {}

/// Builds a 2-by-3 matrix that rotates around `center` in degrees and applies `scale`.
///
/// Every IEEE-754 input is accepted. Non-finite values and signed zero propagate through the
/// coefficient calculation exactly as they do in the pinned browser binding.
pub(crate) fn rotation_matrix_2d(center: [f64; 2], angle_degrees: f64, scale: f64) -> [f64; 6] {
    let angle_radians = angle_degrees.to_radians();
    let alpha = scale * angle_radians.cos();
    let beta = scale * angle_radians.sin();
    let [center_x, center_y] = center;
    [
        alpha,
        beta,
        (1.0 - alpha).mul_add(center_x, -beta * center_y),
        -beta,
        alpha,
        beta.mul_add(center_x, (1.0 - alpha) * center_y),
    ]
}

/// Finds the 2-by-3 affine map between three point correspondences.
pub(crate) fn affine_transform(
    source: &[[f64; 2]; 3],
    destination: &[[f64; 2]; 3],
) -> Result<[f64; 6], TransformMatrixError> {
    if source
        .iter()
        .chain(destination)
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Ok([f64::NAN; 6]);
    }
    let mut coefficients = source.map(|[x, y]| [x, y, 1.0]);
    let mut right_hand_sides = *destination;
    for column in 0..3 {
        let pivot_row = (column..3)
            .max_by(|&left, &right| {
                coefficients[left][column]
                    .abs()
                    .total_cmp(&coefficients[right][column].abs())
            })
            .expect("each affine column has at least one pivot candidate");
        if coefficients[pivot_row][column] == 0.0 {
            return Ok([0.0; 6]);
        }
        if pivot_row != column {
            coefficients.swap(column, pivot_row);
            right_hand_sides.swap(column, pivot_row);
        }
        let negative_inverse_pivot = -1.0 / coefficients[column][column];
        for row in column + 1..3 {
            let factor = coefficients[row][column] * negative_inverse_pivot;
            for target in column + 1..3 {
                coefficients[row][target] += factor * coefficients[column][target];
            }
            for target in 0..2 {
                right_hand_sides[row][target] += factor * right_hand_sides[column][target];
            }
        }
    }
    for row in (0..3).rev() {
        for target in 0..2 {
            let mut value = right_hand_sides[row][target];
            for column in row + 1..3 {
                value -= coefficients[row][column] * right_hand_sides[column][target];
            }
            right_hand_sides[row][target] = value / coefficients[row][row];
        }
    }
    Ok([
        right_hand_sides[0][0],
        right_hand_sides[1][0],
        right_hand_sides[2][0],
        right_hand_sides[0][1],
        right_hand_sides[1][1],
        right_hand_sides[2][1],
    ])
}

/// Inverts a 2-by-3 affine matrix using the observed floating-point operation order.
pub(crate) fn invert_affine_transform(
    transform: &[f64; 6],
) -> Result<[f64; 6], TransformMatrixError> {
    let [a, b, translation_x, d, e, translation_y] = *transform;
    let determinant = a * e - b * d;
    let inverse_determinant = if determinant != 0.0 {
        1.0 / determinant
    } else {
        0.0
    };
    let inverse_a = e * inverse_determinant;
    let inverse_b = -b * inverse_determinant;
    let inverse_d = -d * inverse_determinant;
    let inverse_e = a * inverse_determinant;
    Ok([
        inverse_a,
        inverse_b,
        -inverse_a * translation_x - inverse_b * translation_y,
        inverse_d,
        inverse_e,
        -inverse_d * translation_x - inverse_e * translation_y,
    ])
}

/// F32 counterpart of [`invert_affine_transform`] that preserves source-depth arithmetic.
pub(crate) fn invert_affine_transform_f32(transform: &[f32; 6]) -> [f32; 6] {
    let [a, b, translation_x, d, e, translation_y] = *transform;
    let determinant = a * e - b * d;
    let inverse_determinant = if determinant != 0.0_f32 {
        1.0_f32 / determinant
    } else {
        0.0_f32
    };
    let inverse_a = e * inverse_determinant;
    let inverse_b = -b * inverse_determinant;
    let inverse_d = -d * inverse_determinant;
    let inverse_e = a * inverse_determinant;
    [
        inverse_a,
        inverse_b,
        -inverse_a * translation_x - inverse_b * translation_y,
        inverse_d,
        inverse_e,
        -inverse_d * translation_x - inverse_e * translation_y,
    ]
}

/// Finds a 3-by-3 projective map between four point correspondences.
///
/// This partial implementation fixes the lower-right coefficient to one and uses LU or QR. It rejects correspondences that need a projective matrix with a zero lower-right
/// coefficient.
pub(crate) fn perspective_transform(
    source: &[[f64; 2]; 4],
    destination: &[[f64; 2]; 4],
    method: i32,
) -> Result<[f64; 9], TransformMatrixError> {
    if !matches!(method, 0 | 4 | 16 | 20) {
        return Err(TransformMatrixError::UnsupportedSolveMethod(method));
    }
    if source
        .iter()
        .chain(destination)
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Ok([f64::NAN; 9]);
    }
    let mut coefficients = [[0.0; 8]; 8];
    let mut right_hand_side = [0.0; 8];
    for (index, (&[source_x, source_y], &[destination_x, destination_y])) in
        source.iter().zip(destination).enumerate()
    {
        let x_row = index * 2;
        coefficients[x_row] = [
            source_x,
            source_y,
            1.0,
            0.0,
            0.0,
            0.0,
            -destination_x * source_x,
            -destination_x * source_y,
        ];
        right_hand_side[x_row] = destination_x;

        let y_row = x_row + 1;
        coefficients[y_row] = [
            0.0,
            0.0,
            0.0,
            source_x,
            source_y,
            1.0,
            -destination_y * source_x,
            -destination_y * source_y,
        ];
        right_hand_side[y_row] = destination_y;
    }
    let coefficients = coefficients.into_iter().flatten().collect::<Vec<_>>();
    let solution = if method & 15 == 4 {
        crate::core_algebra::solve_qr(&coefficients, 8, 8, &right_hand_side, 1)
    } else {
        crate::core_algebra::solve_lu(&coefficients, 8, &right_hand_side, 1)
    }
    .map_err(|_| TransformMatrixError::NumericalFailure)?
    .ok_or(TransformMatrixError::DegenerateGeometry)?;
    let result = [
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
        solution[6],
        solution[7],
        1.0,
    ];
    validate_result(&result)?;
    Ok(result)
}

fn validate_result(values: &[f64]) -> Result<(), TransformMatrixError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformMatrixError::NumericalFailure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index} was {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn rotation_keeps_the_center_fixed_while_scaling() {
        let matrix = rotation_matrix_2d([10.0, 20.0], 90.0, 2.0);
        assert_close(&matrix, &[0.0, 2.0, -30.0, -2.0, 0.0, 40.0], 1.0e-12);
    }

    #[test]
    fn rotation_matches_pinned_f64_coefficients() {
        let matrix = rotation_matrix_2d([1.25, -2.5], 33.3, 0.75);

        assert_eq!(
            matrix.map(f64::to_bits),
            [
                0.626_855_521_026_202_7,
                0.411_767_113_498_598_74,
                1.495_848_382_463_743_4,
                -0.411_767_113_498_598_74,
                0.626_855_521_026_202_7,
                -0.418_152_305_561_244_94,
            ]
            .map(f64::to_bits)
        );
    }

    #[test]
    fn rotation_preserves_signed_zero_and_non_finite_results() {
        let signed_zero = rotation_matrix_2d([-0.0, -0.0], -0.0, -0.0);
        assert_eq!(
            signed_zero.map(f64::to_bits),
            [-0.0, 0.0, 0.0, -0.0, -0.0, -0.0].map(f64::to_bits)
        );

        let non_finite = rotation_matrix_2d([1.0, 2.0], f64::INFINITY, 2.0);
        assert!(non_finite.into_iter().all(f64::is_nan));
    }

    #[test]
    fn three_point_correspondences_determine_an_affine_matrix() {
        let source = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let destination = [[2.0, 3.0], [4.0, 4.0], [1.0, 6.0]];

        let matrix = affine_transform(&source, &destination).expect("independent points");

        assert_close(&matrix, &[2.0, -1.0, 2.0, 1.0, 3.0, 3.0], 1.0e-14);
    }

    #[test]
    fn affine_transform_matches_the_pinned_lu_operation_order() {
        let source = [[0.25, -0.5], [2.5, 0.75], [-1.25, 3.5]];
        let destination = [[4.5, -2.25], [8.75, 1.5], [-3.5, 9.25]];

        let matrix = affine_transform(&source, &destination).expect("independent points");

        assert_eq!(
            matrix.map(f64::to_bits),
            [
                0x4003_DCB0_8D3D_CB08,
                0xBFF1_1A7B_9611_A7BA,
                0x400A_C234_F72C_2350,
                0x3FAD_6CDF_A1D6_CDED,
                0x4007_2C23_4F72_C235,
                0xBFEA_1D6C_DFA1_D6CD,
            ]
        );
    }

    #[test]
    fn affine_inverse_undoes_scale_and_translation() {
        let inverse = invert_affine_transform(&[2.0, 0.0, 4.0, 0.0, 3.0, -6.0])
            .expect("invertible affine matrix");

        assert_close(&inverse, &[0.5, 0.0, -2.0, 0.0, 1.0 / 3.0, 2.0], 1.0e-14);
    }

    #[test]
    fn affine_inverse_matches_pinned_source_depth_arithmetic() {
        let inverse = invert_affine_transform(&[1.25, -0.5, 3.75, 2.5, 4.25, -1.5])
            .expect("invertible affine matrix");
        assert_eq!(
            inverse.map(f64::to_bits),
            [
                0x3FE4_B94B_94B9_4B95,
                0x3FB3_8138_1381_3814,
                0xC002_83A8_3A83_A83B,
                0xBFD8_6186_1861_8619,
                0x3FC8_6186_1861_8619,
                0x3FFB_6DB6_DB6D_B6DC,
            ]
        );
        assert_eq!(
            invert_affine_transform_f32(&[2.0, 0.0, 4.0, 0.0, 3.0, -6.0]).map(f32::to_bits),
            [
                0x3F00_0000,
                0x8000_0000,
                0xC000_0000,
                0x8000_0000,
                0x3EAA_AAAB,
                0x4000_0000,
            ]
        );
    }

    #[test]
    fn four_point_correspondences_determine_a_projective_matrix() {
        let source = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let destination = [[1.0, 2.0], [2.0, 7.0 / 6.0], [2.0, 19.0 / 7.0], [1.2, 4.0]];

        let matrix = perspective_transform(&source, &destination, 0).expect("valid quadrilaterals");

        assert_close(
            &matrix,
            &[2.0, 0.5, 1.0, -0.25, 3.0, 2.0, 0.5, 0.25, 1.0],
            1.0e-13,
        );
    }

    #[test]
    fn singular_and_non_finite_affine_inverses_follow_the_runtime_contract() {
        assert_eq!(
            invert_affine_transform(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
                .expect("singular matrices return zeros")
                .map(f64::to_bits),
            [0.0, -0.0, 0.0, -0.0, 0.0, 0.0].map(f64::to_bits)
        );
        assert!(
            invert_affine_transform(&[f64::NAN, 0.0, 1.0, 0.0, 1.0, 2.0])
                .expect("NaN values propagate")
                .into_iter()
                .all(f64::is_nan)
        );
    }

    #[test]
    fn perspective_lu_qr_and_normal_flags_preserve_correspondences() {
        let source = [[0., 0.], [4., 0.], [4., 4.], [0., 4.]];
        let target = [[1., 1.], [5., 0.], [4., 5.], [0., 4.]];
        for method in [0, 4, 16, 20] {
            let m = perspective_transform(&source, &target, method).unwrap();
            for ([x, y], [u, v]) in source.into_iter().zip(target) {
                let w = m[6] * x + m[7] * y + m[8];
                assert!(((m[0] * x + m[1] * y + m[2]) / w - u).abs() < 1e-10);
                assert!(((m[3] * x + m[4] * y + m[5]) / w - v).abs() < 1e-10);
            }
        }
        assert!(perspective_transform(&source, &target, 1).is_err());
        assert!(
            perspective_transform(&[[f64::NAN, 0.]; 4], &target, 0)
                .unwrap()
                .into_iter()
                .all(f64::is_nan)
        );
    }
    #[test]
    fn invalid_perspective_geometry_is_rejected() {
        let source = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let destination = [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]];
        assert_eq!(
            perspective_transform(&source, &destination, 0),
            Err(TransformMatrixError::DegenerateGeometry)
        );
    }
}
