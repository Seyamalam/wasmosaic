//! `drawFrameAxes`: projects the world-frame origin and the three unit
//! axis tips through the pinhole model and draws them on an image.
//!
//! Pinned contract (OpenCV.js 4.13.0 differential probing):
//! - Object points are the origin plus `(length, 0, 0)`, `(0, length, 0)`,
//!   and `(0, 0, length)`, projected with the standard distortion model
//!   (zero depth is treated as `1/z = 1`, matching `projectPoints`).
//! - The x, y, and z axes are drawn in that order as 8-connected lines
//!   with BGR colors `(0, 0, 255)`, `(0, 255, 0)`, and `(255, 0, 0)`.
//! - `length <= 0` and `thickness <= 0` are rejected; the image must have
//!   1, 3, or 4 channels; `rvec` accepts a 3-vector or a 3x3 matrix.
use std::{error::Error, fmt};

use crate::{
    calib3d_distort::distort_normalized,
    imgproc_draw::{self, DrawError},
    mat::{Mat, MatDepth, MatError},
};

#[derive(Debug)]
pub(crate) enum FrameAxesError {
    Matrix(MatError),
    UnsupportedDepth,
    UnsupportedCamera,
    UnsupportedVector,
    UnsupportedDistortion,
    UnsupportedImage,
    InvalidLength,
    InvalidThickness,
    Draw(DrawError),
}

impl fmt::Display for FrameAxesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::UnsupportedDepth => {
                formatter.write_str("calibration inputs require an F32 or F64 depth")
            }
            Self::UnsupportedCamera => formatter.write_str("camera matrix requires 3x3"),
            Self::UnsupportedVector => formatter
                .write_str("rotation requires 3x1, 1x3, or 3x3; translation requires 3x1 or 1x3"),
            Self::UnsupportedDistortion => {
                formatter.write_str("distortion coefficients require 4, 5, 8, 12, or 14 elements")
            }
            Self::UnsupportedImage => formatter.write_str("image requires 1, 3, or 4 channels"),
            Self::InvalidLength => formatter.write_str("axis length must be positive"),
            Self::InvalidThickness => formatter.write_str("thickness must be positive"),
            Self::Draw(error) => error.fmt(formatter),
        }
    }
}

impl Error for FrameAxesError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            Self::Draw(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for FrameAxesError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

impl From<DrawError> for FrameAxesError {
    fn from(error: DrawError) -> Self {
        Self::Draw(error)
    }
}

fn read_elements(source: &Mat) -> Result<Vec<f64>, FrameAxesError> {
    let element_size = match source.depth() {
        MatDepth::F64 => 8,
        MatDepth::F32 => 4,
        _ => return Err(FrameAxesError::UnsupportedDepth),
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

fn read_rotation(source: &Mat) -> Result<[f64; 9], FrameAxesError> {
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    if source.channels() != 1 {
        return Err(FrameAxesError::UnsupportedVector);
    }
    let values = read_elements(source)?;
    if rows == 3 && columns == 3 {
        let mut rotation = [0.0f64; 9];
        rotation.copy_from_slice(&values[..9]);
        return Ok(rotation);
    }
    if (rows, columns) == (3, 1) || (rows, columns) == (1, 3) {
        return Ok(crate::calib3d_rodrigues::rotation_from_vector([
            values[0], values[1], values[2],
        ]));
    }
    Err(FrameAxesError::UnsupportedVector)
}

fn read_vector3(source: &Mat) -> Result<[f64; 3], FrameAxesError> {
    let rows = source.rows() as usize;
    let columns = source.columns() as usize;
    if source.channels() != 1 || !((rows, columns) == (3, 1) || (rows, columns) == (1, 3)) {
        return Err(FrameAxesError::UnsupportedVector);
    }
    let values = read_elements(source)?;
    Ok([values[0], values[1], values[2]])
}

fn read_camera(source: &Mat) -> Result<[f64; 9], FrameAxesError> {
    if source.channels() != 1 || source.rows() != 3 || source.columns() != 3 {
        return Err(FrameAxesError::UnsupportedCamera);
    }
    let values = read_elements(source)?;
    let mut camera = [0.0f64; 9];
    camera.copy_from_slice(&values[..9]);
    Ok(camera)
}

fn read_distortion(source: &Mat) -> Result<([f64; 14], usize), FrameAxesError> {
    let count = source.rows() as usize * source.columns() as usize * source.channels() as usize;
    if count == 0 {
        return Ok(([0.0; 14], 0));
    }
    if !(count == 4 || count == 5 || count == 8 || count == 12 || count == 14) {
        return Err(FrameAxesError::UnsupportedDistortion);
    }
    let values = read_elements(source)?;
    let mut coeffs = [0.0f64; 14];
    for (index, value) in values.iter().enumerate() {
        coeffs[index] = *value;
    }
    Ok((coeffs, count))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_frame_axes_into(
    image: &Mat,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
    rvec: &Mat,
    tvec: &Mat,
    length: f64,
    thickness: i32,
) -> Result<(), FrameAxesError> {
    if !(length > 0.0) {
        return Err(FrameAxesError::InvalidLength);
    }
    if thickness <= 0 {
        return Err(FrameAxesError::InvalidThickness);
    }
    let channels = image.channels();
    if !(channels == 1 || channels == 3 || channels == 4) {
        return Err(FrameAxesError::UnsupportedImage);
    }

    let camera = read_camera(camera_matrix)?;
    let (coeffs, dist_len) = read_distortion(dist_coeffs)?;
    let rotation = read_rotation(rvec)?;
    let translation = read_vector3(tvec)?;

    let tips = [
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ];
    let mut pixels = [[0i64; 2]; 4];
    for (index, point) in tips.iter().enumerate() {
        let xc = rotation[0] * point[0]
            + rotation[1] * point[1]
            + rotation[2] * point[2]
            + translation[0];
        let yc = rotation[3] * point[0]
            + rotation[4] * point[1]
            + rotation[5] * point[2]
            + translation[1];
        let zc = rotation[6] * point[0]
            + rotation[7] * point[1]
            + rotation[8] * point[2]
            + translation[2];
        let inv_z = if zc != 0.0 { 1.0 / zc } else { 1.0 };
        let (xd, yd) = distort_normalized(xc * inv_z, yc * inv_z, &coeffs, dist_len);
        let u = camera[0] * xd + camera[2];
        let v = camera[4] * yd + camera[5];
        // Pinned behavior: projected points are rounded half-to-even; the
        // line rasterizer applies the integer clip-to-rect itself.
        pixels[index] = [u.round_ties_even() as i64, v.round_ties_even() as i64];
    }

    let origin = pixels[0];
    for (axis, color) in [
        (1usize, [0.0, 0.0, 255.0, 0.0]),
        (2, [0.0, 255.0, 0.0, 0.0]),
        (3, [255.0, 0.0, 0.0, 0.0]),
    ] {
        imgproc_draw::line_into(
            image,
            origin[0],
            origin[1],
            pixels[axis][0],
            pixels[axis][1],
            &color,
            thickness,
            8,
            0,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_f64, mat_from_u8};

    fn f64_vec3() -> Mat {
        mat_from_f64(&[0.0, 0.0, 0.0], 3, 1, 1).expect("vec")
    }

    fn camera() -> Mat {
        mat_from_f64(&[2.0, 0.0, 4.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0], 3, 3, 1).expect("camera")
    }

    fn canvas(channels: u16) -> Mat {
        mat_from_u8(&vec![0u8; 9 * 9 * channels as usize], 9, 9, channels).expect("canvas")
    }

    fn pixel(image: &Mat, x: usize, y: usize) -> [u8; 3] {
        let bytes = image.compact_bytes();
        let base = (y * 9 + x) * 3;
        [bytes[base], bytes[base + 1], bytes[base + 2]]
    }

    #[test]
    fn draws_axes_on_color_image() {
        let image = canvas(3);
        let dist = mat_empty();
        draw_frame_axes_into(&image, &camera(), &dist, &f64_vec3(), &f64_vec3(), 1.0, 1)
            .expect("draw");
        // Origin ends blue (z axis drawn last); x runs right in red; y runs
        // down in green.
        assert_eq!(pixel(&image, 4, 4), [255, 0, 0]);
        assert_eq!(pixel(&image, 6, 4), [0, 0, 255]);
        assert_eq!(pixel(&image, 4, 6), [0, 255, 0]);
    }

    #[test]
    fn thick_axes_stamp_endpoint_disks() {
        let image = canvas(3);
        let dist = mat_empty();
        draw_frame_axes_into(&image, &camera(), &dist, &f64_vec3(), &f64_vec3(), 1.0, 3)
            .expect("draw");
        // Degenerate z axis (tip projects onto the origin) stamps a radius-2
        // disk: 13 pixels in a diamond around (4,4).
        assert_eq!(pixel(&image, 4, 2), [255, 0, 0]);
        assert_eq!(pixel(&image, 2, 4), [255, 0, 0]);
        assert_eq!(pixel(&image, 3, 3), [255, 0, 0]);
        assert_eq!(pixel(&image, 3, 2), [0, 0, 0]);
    }

    #[test]
    fn rejects_nonpositive_length() {
        let image = canvas(3);
        let dist = mat_empty();
        assert!(
            draw_frame_axes_into(&image, &camera(), &dist, &f64_vec3(), &f64_vec3(), 0.0, 1)
                .is_err()
        );
        assert!(
            draw_frame_axes_into(&image, &camera(), &dist, &f64_vec3(), &f64_vec3(), -1.0, 1)
                .is_err()
        );
    }

    #[test]
    fn rejects_two_channel_image() {
        let image = canvas(2);
        let dist = mat_empty();
        assert!(
            draw_frame_axes_into(&image, &camera(), &dist, &f64_vec3(), &f64_vec3(), 1.0, 1)
                .is_err()
        );
    }

    #[test]
    fn accepts_rotation_matrix() {
        let image = canvas(3);
        let dist = mat_empty();
        let rot =
            mat_from_f64(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3, 1).expect("rot");
        draw_frame_axes_into(&image, &camera(), &dist, &rot, &f64_vec3(), 1.0, 1).expect("draw");
    }
}
