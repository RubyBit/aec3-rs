//! LPC coefficients and LP residual computation for pitch estimation.

/// Linear predictive coding (LPC) inverse filter length.
pub const NUM_LPC_COEFFICIENTS: usize = 5;

fn compute_auto_correlation(x: &[f32], auto_corr: &mut [f32; NUM_LPC_COEFFICIENTS]) {
    let max_lag = auto_corr.len();
    assert!(max_lag < x.len());

    for lag in 0..max_lag {
        let mut acc = 0.0f32;
        for i in 0..(x.len() - lag) {
            acc += x[i] * x[i + lag];
        }
        auto_corr[lag] = acc;
    }
}

fn denoise_auto_correlation(auto_corr: &mut [f32; NUM_LPC_COEFFICIENTS]) {
    // Assume -40 dB white noise floor.
    auto_corr[0] *= 1.0001;
    // Hard-coded values obtained as: (0.008 * 0.008 * i * i), i=1..4.
    auto_corr[1] -= auto_corr[1] * 0.000064;
    auto_corr[2] -= auto_corr[2] * 0.000256;
    auto_corr[3] -= auto_corr[3] * 0.000576;
    auto_corr[4] -= auto_corr[4] * 0.001024;
}

fn compute_initial_inverse_filter_coefficients(
    auto_corr: &[f32; NUM_LPC_COEFFICIENTS],
    lpc_coeffs: &mut [f32; NUM_LPC_COEFFICIENTS - 1],
) {
    let mut error = auto_corr[0];
    for i in 0..(NUM_LPC_COEFFICIENTS - 1) {
        let mut reflection_coeff = 0.0f32;
        for j in 0..i {
            reflection_coeff += lpc_coeffs[j] * auto_corr[i - j];
        }
        reflection_coeff += auto_corr[i + 1];

        // Avoid division by numbers close to zero.
        const MIN_ERROR_MAGNITUDE: f32 = 1e-6;
        if error.abs() < MIN_ERROR_MAGNITUDE {
            error = if error.is_sign_negative() {
                -MIN_ERROR_MAGNITUDE
            } else {
                MIN_ERROR_MAGNITUDE
            };
        }

        reflection_coeff /= -error;
        // Update LPC coefficients and total error.
        lpc_coeffs[i] = reflection_coeff;
        for j in 0..((i + 1) >> 1) {
            let tmp1 = lpc_coeffs[j];
            let tmp2 = lpc_coeffs[i - 1 - j];
            lpc_coeffs[j] = tmp1 + reflection_coeff * tmp2;
            lpc_coeffs[i - 1 - j] = tmp2 + reflection_coeff * tmp1;
        }

        error -= reflection_coeff * reflection_coeff * error;
        if error < 0.001 * auto_corr[0] {
            break;
        }
    }
}

/// Given frame `x`, computes a post-processed version of LPC coefficients
/// tailored for pitch estimation.
pub fn compute_and_post_process_lpc_coefficients(
    x: &[f32],
    lpc_coeffs: &mut [f32; NUM_LPC_COEFFICIENTS],
) {
    let mut auto_corr = [0.0f32; NUM_LPC_COEFFICIENTS];
    compute_auto_correlation(x, &mut auto_corr);

    if auto_corr[0] == 0.0 {
        lpc_coeffs.fill(0.0);
        return;
    }

    denoise_auto_correlation(&mut auto_corr);

    let mut lpc_coeffs_pre = [0.0f32; NUM_LPC_COEFFICIENTS - 1];
    compute_initial_inverse_filter_coefficients(&auto_corr, &mut lpc_coeffs_pre);

    // LPC coefficients post-processing.
    lpc_coeffs_pre[0] *= 0.9;
    lpc_coeffs_pre[1] *= 0.9 * 0.9;
    lpc_coeffs_pre[2] *= 0.9 * 0.9 * 0.9;
    lpc_coeffs_pre[3] *= 0.9 * 0.9 * 0.9 * 0.9;

    const C: f32 = 0.8;
    lpc_coeffs[0] = lpc_coeffs_pre[0] + C;
    lpc_coeffs[1] = lpc_coeffs_pre[1] + C * lpc_coeffs_pre[0];
    lpc_coeffs[2] = lpc_coeffs_pre[2] + C * lpc_coeffs_pre[1];
    lpc_coeffs[3] = lpc_coeffs_pre[3] + C * lpc_coeffs_pre[2];
    lpc_coeffs[4] = C * lpc_coeffs_pre[3];
}

/// Computes the LP residual for input frame `x` and LPC coefficients
/// `lpc_coeffs`. Supports in-place operation (`x` and `y` may share storage).
pub fn compute_lp_residual(lpc_coeffs: &[f32; NUM_LPC_COEFFICIENTS], x: &[f32], y: &mut [f32]) {
    assert!(x.len() > NUM_LPC_COEFFICIENTS);
    assert_eq!(x.len(), y.len());

    y[0] = x[0];

    // Edge case: i < NUM_LPC_COEFFICIENTS.
    for i in 1..NUM_LPC_COEFFICIENTS {
        let mut acc = x[i];
        for k in 0..i {
            acc += x[i - 1 - k] * lpc_coeffs[k];
        }
        y[i] = acc;
    }

    // Regular case.
    for i in NUM_LPC_COEFFICIENTS..y.len() {
        let mut acc = x[i];
        for k in 0..NUM_LPC_COEFFICIENTS {
            acc += x[i - 1 - k] * lpc_coeffs[k];
        }
        y[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::rnn_vad::common::BUF_SIZE_24_KHZ;
    use crate::audio_processing::agc2::rnn_vad::common::FRAME_SIZE_10_MS_24_KHZ;
    use crate::audio_processing::agc2::rnn_vad::test_data::{
        FLOAT_MIN, create_lp_residual_and_pitch_info_reader, create_pitch_buffer_24khz_reader,
        expect_near_absolute,
    };

    #[test]
    fn lp_residual_of_empty_frame() {
        let empty_frame = [0.0f32; FRAME_SIZE_10_MS_24_KHZ];
        let mut lpc = [0.0f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(&empty_frame, &mut lpc);

        let mut lp_residual = [0.0f32; FRAME_SIZE_10_MS_24_KHZ];
        compute_lp_residual(&lpc, &empty_frame, &mut lp_residual);

        assert!(lpc.iter().all(|v| *v == 0.0));
        assert!(lp_residual.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn lp_residual_in_place_matches_out_of_place() {
        let mut x = [0.0f32; 64];
        for (i, s) in x.iter_mut().enumerate() {
            *s = ((i as f32) * 0.07).sin();
        }

        let mut lpc = [0.0f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(&x, &mut lpc);

        let mut out = [0.0f32; 64];
        compute_lp_residual(&lpc, &x, &mut out);

        let mut in_place = x;
        let snapshot = in_place;
        compute_lp_residual(&lpc, &snapshot, &mut in_place);

        for (a, b) in out.iter().zip(in_place.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a}, b={b}");
        }
    }

    #[test]
    fn lp_residual_pipeline_bit_exactness_with_fixture() {
        let mut pitch_buffer_reader = create_pitch_buffer_24khz_reader();
        let mut lp_pitch_reader = create_lp_residual_and_pitch_info_reader();

        let num_frames = pitch_buffer_reader.num_chunks().min(300);
        assert!(lp_pitch_reader.num_chunks() >= num_frames);

        let mut pitch_buffer_24khz = vec![0.0f32; BUF_SIZE_24_KHZ];
        let mut lpc = [0.0f32; NUM_LPC_COEFFICIENTS];
        let mut computed_lp_residual = vec![0.0f32; BUF_SIZE_24_KHZ];
        let mut expected_lp_and_pitch = vec![0.0f32; BUF_SIZE_24_KHZ + 2];

        for i in 0..num_frames {
            assert!(
                pitch_buffer_reader.read_chunk(&mut pitch_buffer_24khz),
                "frame={i}"
            );
            assert!(
                lp_pitch_reader.read_chunk(&mut expected_lp_and_pitch),
                "frame={i}"
            );

            if i % 20 == 0 {
                compute_and_post_process_lpc_coefficients(&pitch_buffer_24khz, &mut lpc);
                compute_lp_residual(&lpc, &pitch_buffer_24khz, &mut computed_lp_residual);
                expect_near_absolute(
                    &expected_lp_and_pitch[..BUF_SIZE_24_KHZ],
                    &computed_lp_residual,
                    FLOAT_MIN,
                );
            }
        }
    }
}
