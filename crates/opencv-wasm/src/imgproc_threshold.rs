use std::{error::Error, fmt};

use crate::mat::{Mat, MatDepth, MatError};

pub(crate) const THRESH_BINARY: i32 = 0;
pub(crate) const THRESH_BINARY_INVERSE: i32 = 1;
pub(crate) const THRESH_TRUNCATE: i32 = 2;
pub(crate) const THRESH_TO_ZERO: i32 = 3;
pub(crate) const THRESH_TO_ZERO_INVERSE: i32 = 4;
pub(crate) const THRESH_MODE_MASK: i32 = 7;
pub(crate) const THRESH_OTSU: i32 = 8;
pub(crate) const THRESH_TRIANGLE: i32 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ThresholdError {
    InvalidMode(i32),
    Matrix(MatError),
    OtsuRequiresSingleChannel,
    UnsupportedAutomaticMode(i32),
    UnsupportedDepth(MatDepth),
}

impl fmt::Display for ThresholdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMode(mode) => write!(formatter, "unsupported threshold mode {mode}"),
            Self::Matrix(error) => error.fmt(formatter),
            Self::OtsuRequiresSingleChannel => formatter
                .write_str("automatic thresholding requires a supported single-channel matrix"),
            Self::UnsupportedAutomaticMode(mode) => {
                write!(formatter, "unsupported automatic threshold flags {mode}")
            }
            Self::UnsupportedDepth(depth) => {
                write!(formatter, "threshold depth {depth:?} is not implemented")
            }
        }
    }
}

impl Error for ThresholdError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatError> for ThresholdError {
    fn from(error: MatError) -> Self {
        Self::Matrix(error)
    }
}

// Definitions: OpenCV 4.13 public threshold API; Otsu's between-class variance
// and the Triangle histogram-to-line distance construction. All loops are original.
pub(crate) fn threshold_into(
    source: &Mat,
    destination: &Mat,
    threshold: f64,
    maximum: f64,
    threshold_type: i32,
) -> Result<f64, ThresholdError> {
    let depth = source.depth();
    let mode = threshold_type & THRESH_MODE_MASK;
    let dry_run = threshold_type & 128 != 0;
    let automatic = threshold_type & !(THRESH_MODE_MASK | 128);
    let invalid = if automatic == (THRESH_OTSU | THRESH_TRIANGLE) {
        Some(ThresholdError::UnsupportedAutomaticMode(automatic))
    } else if (automatic == THRESH_OTSU
        && (source.channels() != 1 || !matches!(depth, MatDepth::U8 | MatDepth::U16)))
        || (automatic == THRESH_TRIANGLE && (source.channels() != 1 || depth != MatDepth::U8))
    {
        Some(ThresholdError::OtsuRequiresSingleChannel)
    } else if matches!(depth, MatDepth::I8 | MatDepth::I32) {
        Some(ThresholdError::UnsupportedDepth(depth))
    } else if !dry_run && mode > 4 && depth != MatDepth::U8 {
        Some(ThresholdError::InvalidMode(mode))
    } else {
        None
    };
    if let Some(error) = invalid {
        if !dry_run
            && (source.rows() != destination.rows()
                || source.columns() != destination.columns()
                || source.channels() != destination.channels()
                || depth != destination.depth())
        {
            destination.write_output(
                vec![0; source.compact_bytes().len()],
                source.rows(),
                source.columns(),
                source.channels(),
                depth,
            )?;
        }
        return Err(error);
    }
    let input = source.compact_bytes();
    let selected = match automatic {
        THRESH_OTSU | THRESH_TRIANGLE => {
            let mut histogram = vec![0_u64; if depth == MatDepth::U16 { 65536 } else { 256 }];
            for bytes in input.chunks_exact(depth.byte_width()) {
                histogram[crate::imgproc_sample::decode(bytes, depth) as usize] += 1;
            }
            if automatic == THRESH_OTSU {
                otsu_threshold(&histogram)
            } else {
                triangle_threshold(&histogram)
            }
        }
        _ => threshold,
    };
    let integer = !matches!(depth, MatDepth::F32 | MatDepth::F64);
    let used = if integer {
        f64::from(floor_i32(selected))
    } else {
        selected
    };
    if dry_run {
        return Ok(used);
    }
    let cutoff = if depth == MatDepth::F32 {
        f64::from(used as f32)
    } else {
        used
    };
    let maximum = if integer {
        f64::from(round_i32(maximum))
    } else {
        maximum
    };
    let trunc_floor = match depth {
        MatDepth::I16 => f64::from(i16::MIN),
        _ => 0.0,
    };
    let truncate = if integer && cutoff < trunc_floor {
        0.0
    } else {
        cutoff
    };
    let operation = |bytes: &[u8], output: &mut [u8]| {
        let value = crate::imgproc_sample::decode(bytes, depth);
        let result = match mode {
            THRESH_BINARY => {
                if value > cutoff {
                    maximum
                } else {
                    0.0
                }
            }
            THRESH_BINARY_INVERSE => {
                if value <= cutoff {
                    maximum
                } else {
                    0.0
                }
            }
            THRESH_TRUNCATE => {
                if value > cutoff {
                    truncate
                } else {
                    value
                }
            }
            THRESH_TO_ZERO => {
                if value > cutoff {
                    value
                } else {
                    0.0
                }
            }
            THRESH_TO_ZERO_INVERSE => {
                if value <= cutoff {
                    value
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        crate::imgproc_sample::encode(result, depth, output);
    };
    if !destination.try_write_shared_unary_scalars(source, depth.byte_width(), operation)? {
        let mut output = vec![0; input.len()];
        for (bytes, result) in input
            .chunks_exact(depth.byte_width())
            .zip(output.chunks_exact_mut(depth.byte_width()))
        {
            operation(bytes, result);
        }
        destination.write_output(
            output,
            source.rows(),
            source.columns(),
            source.channels(),
            depth,
        )?;
    }
    Ok(used)
}

fn round_i32(value: f64) -> i32 {
    let rounded = value.round_ties_even();
    if !rounded.is_finite() || rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        i32::MIN
    } else {
        rounded as i32
    }
}

fn floor_i32(value: f64) -> i32 {
    let rounded = round_i32(value);
    rounded.wrapping_sub(i32::from(value < f64::from(rounded)))
}

fn otsu_threshold(histogram: &[u64]) -> f64 {
    let total = histogram.iter().sum::<u64>() as f64;
    let total_sum = histogram
        .iter()
        .enumerate()
        .map(|(value, &count)| value as f64 * count as f64)
        .sum::<f64>();
    let mut background_count = 0_u64;
    let mut background_sum = 0.0;
    let mut best_variance = -1.0;
    let mut best_threshold = 0;
    for (threshold, &count) in histogram.iter().enumerate() {
        background_count += count;
        background_sum += threshold as f64 * count as f64;
        if background_count == 0 {
            continue;
        }
        let foreground_count = total - background_count as f64;
        if foreground_count == 0.0 {
            break;
        }
        let difference = background_sum / background_count as f64
            - (total_sum - background_sum) / foreground_count;
        let variance = background_count as f64 * foreground_count * difference * difference;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = threshold;
        }
    }
    best_threshold as f64
}

fn triangle_threshold(histogram: &[u64]) -> f64 {
    let Some(first) = histogram.iter().position(|&count| count != 0) else {
        return 2.0;
    };
    let last = histogram
        .iter()
        .rposition(|&count| count != 0)
        .expect("nonempty");
    let peak = histogram
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("bins")
        .0;
    let low = first.saturating_sub(1);
    let high = (last + 1).min(255);
    let reverse = high - peak > peak - low;
    let (tail, top) = if reverse {
        (255 - high, 255 - peak)
    } else {
        (low, peak)
    };
    let height = histogram[peak] as f64;
    let width = (top - tail) as f64;
    let mut best = 0.0;
    let mut selected = tail;
    for bin in tail + 1..=top {
        let count = histogram[if reverse { 255 - bin } else { bin }] as f64;
        let distance = height * bin as f64 - width * count;
        if distance > best {
            best = distance;
            selected = bin;
        }
    }
    let selected = selected as f64 - 1.0;
    if reverse { 255.0 - selected } else { selected }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{mat_empty, mat_from_u8};

    #[test]
    fn binary_threshold_uses_strict_greater_than() {
        let source = mat_from_u8(&[0, 99, 100, 101, 255], 1, 5, 1).expect("source");
        let destination = mat_empty();
        let used =
            threshold_into(&source, &destination, 100.0, 200.0, THRESH_BINARY).expect("threshold");
        assert_eq!(used, 100.0);
        assert_eq!(destination.compact_bytes(), [0, 0, 0, 200, 200]);
    }

    #[test]
    fn fixed_modes_cover_truncate_and_both_to_zero_directions() {
        let source = mat_from_u8(&[99, 100, 101], 1, 3, 1).expect("source");
        for (mode, expected) in [
            (THRESH_TRUNCATE, [99, 100, 100]),
            (THRESH_TO_ZERO, [0, 0, 101]),
            (THRESH_TO_ZERO_INVERSE, [99, 100, 0]),
        ] {
            let destination = mat_empty();
            threshold_into(&source, &destination, 100.0, 255.0, mode).expect("threshold");
            assert_eq!(destination.compact_bytes(), expected);
        }
    }

    #[test]
    fn otsu_selects_the_first_best_bimodal_split() {
        let source = mat_from_u8(&[10, 10, 10, 200, 200, 200], 1, 6, 1).expect("source");
        let destination = mat_empty();
        let used = threshold_into(
            &source,
            &destination,
            0.0,
            255.0,
            THRESH_BINARY | THRESH_OTSU,
        )
        .expect("threshold");
        assert_eq!(used, 10.0);
        assert_eq!(destination.compact_bytes(), [0, 0, 0, 255, 255, 255]);
    }
}
