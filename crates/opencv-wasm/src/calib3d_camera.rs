use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CameraError {
    Matrix(MatError),
    UnsupportedDepth,
}

impl fmt::Display for CameraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth => {
                formatter.write_str("camera matrices require an F32 or F64 depth")
            }
        }
    }
}

impl Error for CameraError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for CameraError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

/// Copies a camera matrix into an F64 destination, optionally re-centering
/// the principal point at `(width - 1) / 2`, `(height - 1) / 2`. This is the
/// browser-binding contract for `cv.getDefaultNewCameraMatrix`, which takes
/// no alpha or output-size parameters.
pub(crate) fn default_new_camera_matrix(
    camera_matrix: &Mat,
    destination: &Mat,
    image_width: i32,
    image_height: i32,
    center_principal_point: bool,
) -> Result<(), CameraError> {
    let depth = camera_matrix.depth();
    let is_double = match depth {
        MatDepth::F64 => true,
        MatDepth::F32 => false,
        _ => return Err(CameraError::UnsupportedDepth),
    };
    let rows = camera_matrix.rows() as usize;
    let columns = camera_matrix.columns() as usize;
    let channels = camera_matrix.channels() as usize;
    let element_size = if is_double { 8 } else { 4 };
    let bytes = camera_matrix.compact_bytes();
    let element_count = rows * columns * channels;

    let mut values = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let start = index * element_size;
        let value = if is_double {
            f64::from_ne_bytes(bytes[start..start + 8].try_into().expect("f64"))
        } else {
            f32::from_ne_bytes(bytes[start..start + 4].try_into().expect("f32")) as f64
        };
        values.push(value);
    }

    if center_principal_point && channels == 1 && rows >= 3 && columns >= 3 {
        values[2] = (image_width as f64 - 1.0) / 2.0;
        values[5] = (image_height as f64 - 1.0) / 2.0;
    }

    let mut out = Vec::with_capacity(values.len() * 8);
    for value in &values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    destination.write_output(
        out,
        rows as u32,
        columns as u32,
        channels as u16,
        MatDepth::F64,
    )?;
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
    fn default_camera_matrix_copies_without_centering() {
        let camera = mat_from_f64(
            &[1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0],
            3,
            3,
            1,
        )
        .expect("camera");
        let destination = mat_empty();
        default_new_camera_matrix(&camera, &destination, 640, 480, false).expect("copy");
        assert_eq!(
            f64_values(&destination),
            vec![1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn default_camera_matrix_centers_principal_point() {
        let camera = mat_from_f64(
            &[1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0],
            3,
            3,
            1,
        )
        .expect("camera");
        let destination = mat_empty();
        default_new_camera_matrix(&camera, &destination, 640, 480, true).expect("copy");
        assert_eq!(
            f64_values(&destination),
            vec![1000.0, 0.0, 319.5, 0.0, 1000.0, 239.5, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn default_camera_matrix_promotes_f32() {
        let camera = crate::mat::mat_from_f32(
            &[500.0, 0.0, 100.5, 0.0, 500.0, 50.25, 0.0, 0.0, 1.0],
            3,
            3,
            1,
        )
        .expect("camera");
        let destination = mat_empty();
        default_new_camera_matrix(&camera, &destination, 200, 100, true).expect("copy");
        assert_eq!(destination.depth(), MatDepth::F64);
        assert_eq!(
            f64_values(&destination),
            vec![500.0, 0.0, 99.5, 0.0, 500.0, 49.5, 0.0, 0.0, 1.0]
        );
    }
}
