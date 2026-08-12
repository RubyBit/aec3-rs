use crate::api::config::{EchoCanceller3Config, EchoModel, EpStrength};
use crate::audio_processing::aec3::aec_state::AecState;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::reverb_model::ReverbModel;
use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;

/// Estimates the residual echo power seen on the capture channels.
pub struct ResidualEchoEstimator {
    config: EchoCanceller3Config,
    num_render_channels: usize,
    x2_noise_floor: [f32; FFT_LENGTH_BY_2_PLUS_1],
    x2_noise_floor_counter: [usize; FFT_LENGTH_BY_2_PLUS_1],
    echo_reverb: ReverbModel,
}

impl ResidualEchoEstimator {
    pub fn new(config: EchoCanceller3Config, num_render_channels: usize) -> Self {
        assert!(num_render_channels > 0);
        let mut instance = Self {
            config,
            num_render_channels,
            x2_noise_floor: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            x2_noise_floor_counter: [0; FFT_LENGTH_BY_2_PLUS_1],
            echo_reverb: ReverbModel::new(),
        };
        instance.reset();
        instance
    }

    pub fn estimate(
        &mut self,
        aec_state: &AecState,
        render_buffer: &RenderBuffer<'_>,
        s2_linear: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        assert_eq!(r2.len(), y2.len());
        assert_eq!(r2.len(), s2_linear.len());
        let num_capture_channels = r2.len();

        self.update_render_noise_power(render_buffer);

        if aec_state.usable_linear_estimate() {
            if aec_state.saturated_echo() {
                copy_power_spectra(y2, r2);
            } else if let Some(erle_uncertainty) = aec_state.erle_uncertainty() {
                linear_estimate_with_uncertainty(s2_linear, erle_uncertainty, r2);
            } else {
                linear_estimate_with_erle(s2_linear, aec_state.erle(), r2);
            }

            self.add_reverb(ReverbType::Linear, aec_state, render_buffer, r2);
        } else {
            let echo_path_gain = get_echo_path_gain(aec_state, &self.config.ep_strength);

            if aec_state.saturated_echo() {
                copy_power_spectra(y2, r2);
            } else {
                let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                echo_generating_power(
                    self.num_render_channels,
                    render_buffer.spectrum_buffer(),
                    &self.config.echo_model,
                    aec_state.min_direct_path_filter_delay(),
                    &mut x2,
                );
                if !aec_state.use_stationarity_properties() {
                    apply_noise_gate(&self.config.echo_model, &mut x2);
                }
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    x2[k] -= self.config.echo_model.stationary_gate_slope * self.x2_noise_floor[k];
                    if x2[k] < 0.0 {
                        x2[k] = 0.0;
                    }
                }
                non_linear_estimate(echo_path_gain, &x2, r2);
            }

            if self.config.echo_model.model_reverb_in_nonlinear_mode
                && !aec_state.transparent_mode()
            {
                self.add_reverb(ReverbType::NonLinear, aec_state, render_buffer, r2);
            }
        }

        if aec_state.use_stationarity_properties() {
            let mut residual_scaling = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            aec_state.get_residual_echo_scaling(&mut residual_scaling);
            for ch in 0..num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    r2[ch][k] *= residual_scaling[k];
                }
            }
        }
    }

    fn reset(&mut self) {
        self.echo_reverb.reset();
        self.x2_noise_floor
            .fill(self.config.echo_model.min_noise_floor_power);
        self.x2_noise_floor_counter
            .fill(self.config.echo_model.noise_floor_hold);
    }

    fn update_render_noise_power(&mut self, render_buffer: &RenderBuffer<'_>) {
        let spectra = render_buffer.spectrum(0);
        debug_assert_eq!(self.num_render_channels, spectra.len());
        let mut render_power_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1] = if self.num_render_channels > 1 {
            render_power_data.fill(0.0);
            for ch in 0..self.num_render_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    render_power_data[k] += spectra[ch][k];
                }
            }
            &render_power_data
        } else {
            &spectra[0]
        };

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if render_power[k] < self.x2_noise_floor[k] {
                self.x2_noise_floor[k] = render_power[k];
                self.x2_noise_floor_counter[k] = 0;
            } else if self.x2_noise_floor_counter[k] >= self.config.echo_model.noise_floor_hold {
                self.x2_noise_floor[k] = (self.x2_noise_floor[k] * 1.1)
                    .max(self.config.echo_model.min_noise_floor_power);
            } else {
                self.x2_noise_floor_counter[k] += 1;
            }
        }
    }

    fn add_reverb(
        &mut self,
        reverb_type: ReverbType,
        aec_state: &AecState,
        render_buffer: &RenderBuffer<'_>,
        r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        let num_capture_channels = r2.len();
        let first_reverb_partition = match reverb_type {
            ReverbType::Linear => (aec_state.filter_length_blocks() + 1) as isize,
            ReverbType::NonLinear => (aec_state.min_direct_path_filter_delay() + 1) as isize,
        };
        let spectra = render_buffer.spectrum(first_reverb_partition);
        debug_assert_eq!(self.num_render_channels, spectra.len());

        let mut render_power_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1] = if self.num_render_channels > 1 {
            render_power_data.fill(0.0);
            for ch in 0..self.num_render_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    render_power_data[k] += spectra[ch][k];
                }
            }
            &render_power_data
        } else {
            &spectra[0]
        };

        if matches!(reverb_type, ReverbType::Linear) {
            self.echo_reverb.update_reverb(
                render_power,
                aec_state.get_reverb_frequency_response(),
                aec_state.reverb_decay(),
            );
        } else {
            let echo_path_gain = get_echo_path_gain(aec_state, &self.config.ep_strength);
            self.echo_reverb.update_reverb_no_freq_shaping(
                render_power,
                echo_path_gain,
                aec_state.reverb_decay(),
            );
        }

        let reverb_power = self.echo_reverb.reverb();
        for ch in 0..num_capture_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                r2[ch][k] += reverb_power[k];
            }
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ReverbType {
    Linear,
    NonLinear,
}

fn copy_power_spectra(
    src: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    dst: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.copy_from_slice(s);
    }
}

fn linear_estimate_with_erle(
    s2_linear: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erle: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    for ch in 0..r2.len() {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            debug_assert!(erle[ch][k] > 0.0);
            r2[ch][k] = if erle[ch][k] > 0.0 {
                s2_linear[ch][k] / erle[ch][k]
            } else {
                0.0
            };
        }
    }
}

fn linear_estimate_with_uncertainty(
    s2_linear: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erle_uncertainty: f32,
    r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    for ch in 0..r2.len() {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            r2[ch][k] = s2_linear[ch][k] * erle_uncertainty;
        }
    }
}

fn non_linear_estimate(
    echo_path_gain: f32,
    x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    for ch in 0..r2.len() {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            r2[ch][k] = x2[k] * echo_path_gain;
        }
    }
}

fn apply_noise_gate(echo_model: &EchoModel, x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        if echo_model.noise_gate_power > x2[k] {
            let delta = echo_model.noise_gate_power - x2[k];
            x2[k] = (x2[k] - echo_model.noise_gate_slope * delta).max(0.0);
        }
    }
}

fn echo_generating_power(
    num_render_channels: usize,
    spectrum_buffer: &SpectrumBuffer,
    echo_model: &EchoModel,
    filter_delay_blocks: i32,
    x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let (mut idx_start, idx_stop) =
        get_render_indexes_to_analyze(spectrum_buffer, echo_model, filter_delay_blocks);
    x2.fill(0.0);
    if num_render_channels == 1 {
        while idx_start != idx_stop {
            let channel_power = &spectrum_buffer.buffer[idx_start][0];
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[k] = x2[k].max(channel_power[k]);
            }
            idx_start = spectrum_buffer.inc_index(idx_start);
        }
    } else {
        while idx_start != idx_stop {
            let mut render_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for ch in 0..num_render_channels {
                let channel_power = &spectrum_buffer.buffer[idx_start][ch];
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    render_power[k] += channel_power[k];
                }
            }
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[k] = x2[k].max(render_power[k]);
            }
            idx_start = spectrum_buffer.inc_index(idx_start);
        }
    }
}

fn get_render_indexes_to_analyze(
    spectrum_buffer: &SpectrumBuffer,
    echo_model: &EchoModel,
    filter_delay_blocks: i32,
) -> (usize, usize) {
    let pre = echo_model.render_pre_window_size as i32;
    let post = echo_model.render_post_window_size as i32;
    let window_start = (filter_delay_blocks - pre).max(0);
    let window_end = filter_delay_blocks + post;
    let idx_start = spectrum_buffer.offset_index(spectrum_buffer.read, window_start as isize);
    let idx_stop = spectrum_buffer.offset_index(spectrum_buffer.read, (window_end + 1) as isize);
    (idx_start, idx_stop)
}

fn get_echo_path_gain(aec_state: &AecState, config: &EpStrength) -> f32 {
    let gain_amplitude = if aec_state.transparent_mode() {
        0.01
    } else {
        config.default_gain
    };
    gain_amplitude * gain_amplitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, get_time_domain_length, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
    use crate::test_support::echo_canceller_test_tools::randomize_sample_vector;
    use crate::test_support::random::Random;

    #[test]
    fn basic_test() {
        const SAMPLE_RATE_HZ: i32 = 48_000;
        for &num_render_channels in &[1usize, 2, 4] {
            for &num_capture_channels in &[1usize, 2, 4] {
                let config = EchoCanceller3Config::default();
                let mut estimator = ResidualEchoEstimator::new(config.clone(), num_render_channels);
                let mut aec_state = AecState::new(config.clone(), num_capture_channels);
                let mut render_delay_buffer =
                    RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);

                let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
                assert!(num_bands > 0);
                let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; num_bands];

                let mut h2 = vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 10]; num_capture_channels];
                for channel in &mut h2 {
                    for partition in channel.iter_mut() {
                        partition.fill(0.01);
                    }
                    channel[2].fill(10.0);
                    channel[2][0] = 0.1;
                }

                let impulse_len = get_time_domain_length(config.filter.main.length_blocks);
                let h = vec![vec![0.0f32; impulse_len]; num_capture_channels];

                let mut random_generator = Random::new(42);
                let mut outputs = vec![SubtractorOutput::default(); num_capture_channels];
                for output in &mut outputs {
                    output.reset();
                    output.s_main.fill(100.0);
                }
                let mut e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut s2_linear = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut r2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let delay_estimate: Option<DelayEstimate> = None;

                const LEVEL: f32 = 10.0;
                for ch in 0..num_capture_channels {
                    e2_main[ch].fill(LEVEL);
                    y2[ch].fill(LEVEL);
                }
                s2_linear[0].fill(LEVEL);

                for iteration in 0..1993 {
                    randomize_sample_vector(&mut random_generator, &mut x[0][0]);
                    render_delay_buffer.insert(&x);
                    if iteration == 0 {
                        render_delay_buffer.reset();
                    }
                    let _ = render_delay_buffer.prepare_capture_processing();
                    let render_buffer = render_delay_buffer.render_buffer();

                    aec_state.update(
                        delay_estimate,
                        &h2,
                        &h,
                        &render_buffer,
                        &e2_main,
                        &y2,
                        &outputs,
                    );

                    estimator.estimate(&aec_state, &render_buffer, &s2_linear, &y2, &mut r2);
                }
            }
        }
    }
}
