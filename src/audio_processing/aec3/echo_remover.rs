use std::ptr;

use crate::api::config::EchoCanceller3Config;
use crate::api::control::Metrics;
use crate::audio_processing::aec3::aec_state::AecState;
use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1, detect_optimization, log2_to_db,
    num_bands_for_rate, valid_full_band_rate,
};
use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
use crate::audio_processing::aec3::comfort_noise_generator::ComfortNoiseGenerator;
use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
use crate::audio_processing::aec3::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::audio_processing::aec3::echo_remover_metrics::{
    EchoRemoverMetrics, EchoRemoverMetricsReport,
};
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
use crate::audio_processing::aec3::residual_echo_estimator::ResidualEchoEstimator;
use crate::audio_processing::aec3::subtractor::Subtractor;
use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
use crate::audio_processing::aec3::suppression_filter::SuppressionFilter;
use crate::audio_processing::aec3::suppression_gain::SuppressionGain;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

const GAIN_CHANGE_HANGOVER_BLOCKS: i32 = 3;

/// Top-level component that removes echo from the capture signal by
/// orchestrating the linear echo canceller and suppression stages.
pub struct EchoRemover {
    #[allow(dead_code)]
    config: EchoCanceller3Config,
    fft: Aec3Fft,
    #[allow(dead_code)]
    data_dumper: ApmDataDumper,
    optimization: Aec3Optimization,
    sample_rate_hz: i32,
    num_render_channels: usize,
    num_capture_channels: usize,
    use_shadow_filter_output: bool,
    subtractor: Subtractor,
    suppression_gain: SuppressionGain,
    cng: ComfortNoiseGenerator,
    suppression_filter: SuppressionFilter,
    render_signal_analyzer: RenderSignalAnalyzer,
    residual_echo_estimator: ResidualEchoEstimator,
    echo_leakage_detected: bool,
    aec_state: AecState,
    metrics: EchoRemoverMetrics,
    block_counter: usize,
    gain_change_hangover: i32,
    main_filter_output_last_selected: bool,
    e_blocks: Vec<[f32; BLOCK_SIZE]>,
    e_old: Vec<[f32; BLOCK_SIZE]>,
    y_old: Vec<[f32; BLOCK_SIZE]>,
    y_fft: Vec<FftData>,
    e_fft: Vec<FftData>,
    y2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    e2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    r2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    s2_linear: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    comfort_noise: Vec<FftData>,
    high_band_comfort_noise: Vec<FftData>,
    subtractor_output: Vec<SubtractorOutput>,
}

impl EchoRemover {
    pub fn new(
        config: EchoCanceller3Config,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        assert!(valid_full_band_rate(sample_rate_hz));
        assert!(num_capture_channels > 0);
        assert!(num_render_channels > 0);

        let fft = Aec3Fft::new();
        let data_dumper = ApmDataDumper::new_unique();
        let optimization = detect_optimization();
        let subtractor = Subtractor::new(
            config.clone(),
            num_render_channels,
            num_capture_channels,
            data_dumper.clone(),
            optimization,
        );
        let suppression_gain = SuppressionGain::new(
            config.clone(),
            optimization,
            sample_rate_hz,
            num_capture_channels,
        );
        let cng = ComfortNoiseGenerator::new(optimization, num_capture_channels);
        let suppression_filter =
            SuppressionFilter::new(optimization, sample_rate_hz, num_capture_channels);
        let render_signal_analyzer = RenderSignalAnalyzer::new(&config);
        let residual_echo_estimator =
            ResidualEchoEstimator::new(config.clone(), num_render_channels);
        let aec_state = AecState::new(config.clone(), num_capture_channels);
        let metrics = EchoRemoverMetrics::new();
        let use_shadow_filter_output = config.filter.enable_shadow_filter_output_usage;

        Self {
            config,
            fft,
            data_dumper,
            optimization,
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
            use_shadow_filter_output,
            subtractor,
            suppression_gain,
            cng,
            suppression_filter,
            render_signal_analyzer,
            residual_echo_estimator,
            echo_leakage_detected: false,
            aec_state,
            metrics,
            block_counter: 0,
            gain_change_hangover: 0,
            main_filter_output_last_selected: true,
            e_blocks: vec![[0.0; BLOCK_SIZE]; num_capture_channels],
            e_old: vec![[0.0; BLOCK_SIZE]; num_capture_channels],
            y_old: vec![[0.0; BLOCK_SIZE]; num_capture_channels],
            y_fft: vec![FftData::default(); num_capture_channels],
            e_fft: vec![FftData::default(); num_capture_channels],
            y2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            e2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            r2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            s2_linear: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            comfort_noise: vec![FftData::default(); num_capture_channels],
            high_band_comfort_noise: vec![FftData::default(); num_capture_channels],
            subtractor_output: vec![SubtractorOutput::default(); num_capture_channels],
        }
    }

    pub fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        self.echo_leakage_detected = leakage_detected;
    }

    pub fn metrics(&self) -> Metrics {
        Metrics {
            echo_return_loss: (-10.0f64) * f64::from(self.aec_state.erl_time_domain()).log10(),
            echo_return_loss_enhancement: f64::from(log2_to_db(
                self.aec_state.fullband_erle_log2(),
            )),
            delay_ms: 0,
            render_jitter_min: 0,
            render_jitter_max: 0,
            capture_jitter_min: 0,
            capture_jitter_max: 0,
        }
    }

    pub fn metrics_report(&self) -> Option<&EchoRemoverMetricsReport> {
        if self.metrics.metrics_reported() {
            Some(self.metrics.report())
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn process_capture(
        &mut self,
        mut echo_path_variability: EchoPathVariability,
        capture_signal_saturation: bool,
        external_delay: Option<DelayEstimate>,
        render_buffer: &RenderBuffer<'_>,
        mut linear_output: Option<&mut Vec<Vec<Vec<f32>>>>,
        capture: &mut Vec<Vec<Vec<f32>>>,
    ) {
        self.block_counter += 1;

        let render_block = render_buffer.block(0);
        let num_bands = num_bands_for_rate(self.sample_rate_hz);
        assert_eq!(render_block.len(), num_bands);
        assert_eq!(capture.len(), num_bands);
        assert_eq!(render_block[0].len(), self.num_render_channels);
        assert_eq!(capture[0].len(), self.num_capture_channels);
        assert_eq!(render_block[0][0].len(), BLOCK_SIZE);
        assert_eq!(capture[0][0].len(), BLOCK_SIZE);

        self.aec_state
            .update_capture_saturation(capture_signal_saturation);

        if echo_path_variability.audio_path_changed() {
            if echo_path_variability.gain_change {
                if self.gain_change_hangover == 0 {
                    self.gain_change_hangover = GAIN_CHANGE_HANGOVER_BLOCKS;
                } else {
                    echo_path_variability.gain_change = false;
                }
            }

            self.subtractor
                .handle_echo_path_change(&echo_path_variability);
            self.aec_state
                .handle_echo_path_change(&echo_path_variability);

            if !matches!(echo_path_variability.delay_change, DelayAdjustment::None) {
                self.suppression_gain.set_initial_state(true);
            }
        }

        if self.gain_change_hangover > 0 {
            self.gain_change_hangover -= 1;
        }

        let min_delay = self.aec_state.min_direct_path_filter_delay();
        let delay_partitions = (min_delay >= 0).then_some(min_delay as usize);
        self.render_signal_analyzer
            .update(render_buffer, delay_partitions);

        if self.aec_state.transition_triggered() {
            self.subtractor.exit_initial_state();
            self.suppression_gain.set_initial_state(false);
        }

        {
            let capture_band0 = &capture[0];
            self.subtractor.process(
                render_buffer,
                capture_band0,
                &self.render_signal_analyzer,
                &self.aec_state,
                &mut self.subtractor_output,
            );
        }

        {
            let capture_band0 = &capture[0];
            for ch in 0..self.num_capture_channels {
                form_linear_filter_output(
                    self.use_shadow_filter_output,
                    &mut self.main_filter_output_last_selected,
                    &self.subtractor_output[ch],
                    &mut self.e_blocks[ch],
                );
                windowed_padded_fft(
                    &self.fft,
                    &capture_band0[ch],
                    &mut self.y_old[ch],
                    &mut self.y_fft[ch],
                );
                windowed_padded_fft(
                    &self.fft,
                    &self.e_blocks[ch],
                    &mut self.e_old[ch],
                    &mut self.e_fft[ch],
                );
                linear_echo_power(&self.e_fft[ch], &self.y_fft[ch], &mut self.s2_linear[ch]);
                self.y_fft[ch].spectrum(self.optimization, &mut self.y2[ch]);
                self.e_fft[ch].spectrum(self.optimization, &mut self.e2[ch]);
            }
        }

        if let Some(linear) = linear_output.as_deref_mut() {
            assert_eq!(1, linear.len());
            assert_eq!(self.num_capture_channels, linear[0].len());
            for ch in 0..self.num_capture_channels {
                assert_eq!(BLOCK_SIZE, linear[0][ch].len());
                linear[0][ch].copy_from_slice(&self.e_blocks[ch]);
            }
        }

        self.aec_state.update(
            external_delay,
            self.subtractor.filter_frequency_responses(),
            self.subtractor.filter_impulse_responses(),
            render_buffer,
            &self.e2,
            &self.y2,
            &self.subtractor_output,
        );

        let y_fft = if self.aec_state.use_linear_filter_output() {
            &self.e_fft
        } else {
            &self.y_fft
        };

        self.residual_echo_estimator.estimate(
            &self.aec_state,
            render_buffer,
            &self.s2_linear,
            &self.y2,
            &mut self.r2,
        );

        self.cng.compute(
            self.aec_state.saturated_capture(),
            &self.y2,
            &mut self.comfort_noise,
            &mut self.high_band_comfort_noise,
        );

        if self.aec_state.usable_linear_estimate() {
            for ch in 0..self.num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    if self.e2[ch][k] > self.y2[ch][k] {
                        self.e2[ch][k] = self.y2[ch][k];
                    }
                }
            }
        }

        let nearend_spectrum = if self.aec_state.usable_linear_estimate() {
            &self.e2
        } else {
            &self.y2
        };

        let echo_spectrum = if self.aec_state.usable_linear_estimate() {
            &self.s2_linear
        } else {
            &self.r2
        };

        let mut high_bands_gain = 1.0f32;
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let clock_drift =
            self.config.echo_removal_control.has_clock_drift || echo_path_variability.clock_drift;
        self.suppression_gain.get_gain(
            nearend_spectrum,
            echo_spectrum,
            &self.r2,
            self.cng.noise_spectrum(),
            &self.render_signal_analyzer,
            &self.aec_state,
            render_block,
            clock_drift,
            Some(&mut high_bands_gain),
            Some(&mut gain),
        );

        self.suppression_filter.apply_gain(
            &self.comfort_noise,
            &self.high_band_comfort_noise,
            &gain,
            high_bands_gain,
            y_fft,
            Some(capture),
        );

        let noise_spectrum = self.cng.noise_spectrum();
        self.metrics
            .update(&self.aec_state, &noise_spectrum[0], &gain);
    }
}

fn windowed_padded_fft(
    fft: &Aec3Fft,
    v: &[f32],
    v_old: &mut [f32; BLOCK_SIZE],
    spectrum: &mut FftData,
) {
    assert_eq!(v.len(), BLOCK_SIZE);
    fft.padded_fft_with_window(v, v_old, Window::SqrtHanning, spectrum);
    v_old.copy_from_slice(v);
}

fn linear_echo_power(e: &FftData, y: &FftData, dst: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        let dr = y.re[k] - e.re[k];
        let di = y.im[k] - e.im[k];
        dst[k] = dr * dr + di * di;
    }
}

fn signal_transition(from: &[f32], to: &[f32], out: &mut [f32]) {
    assert_eq!(from.len(), to.len());
    assert_eq!(from.len(), out.len());
    if ptr::eq(from.as_ptr(), to.as_ptr()) {
        out.copy_from_slice(to);
        return;
    }

    const TRANSITION_SIZE: usize = 30;
    assert!(TRANSITION_SIZE <= out.len());
    let inv = 1.0f32 / (TRANSITION_SIZE as f32 + 1.0);
    for k in 0..TRANSITION_SIZE {
        let a = (k as f32 + 1.0) * inv;
        out[k] = a * to[k] + (1.0 - a) * from[k];
    }
    out[TRANSITION_SIZE..].copy_from_slice(&to[TRANSITION_SIZE..]);
}

fn form_linear_filter_output(
    use_shadow_filter_output: bool,
    main_filter_output_last_selected: &mut bool,
    subtractor_output: &SubtractorOutput,
    output: &mut [f32; BLOCK_SIZE],
) {
    let mut use_main = true;
    if use_shadow_filter_output {
        if subtractor_output.e2_shadow < 0.9f32 * subtractor_output.e2_main
            && subtractor_output.y2 > 30.0f32 * 30.0f32 * BLOCK_SIZE as f32
            && (subtractor_output.s2_main > 60.0f32 * 60.0f32 * BLOCK_SIZE as f32
                || subtractor_output.s2_shadow > 60.0f32 * 60.0f32 * BLOCK_SIZE as f32)
        {
            use_main = false;
        } else if subtractor_output.e2_shadow < subtractor_output.e2_main
            && subtractor_output.y2 < subtractor_output.e2_main
        {
            use_main = false;
        }
    }

    let from = if *main_filter_output_last_selected {
        &subtractor_output.e_main
    } else {
        &subtractor_output.e_shadow
    };
    let to = if use_main {
        &subtractor_output.e_main
    } else {
        &subtractor_output.e_shadow
    };
    signal_transition(from, to, output);
    *main_filter_output_last_selected = use_main;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, num_bands_for_rate};
    use crate::audio_processing::aec3::echo_path_variability::{
        DelayAdjustment, EchoPathVariability,
    };
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    #[test]
    fn basic_api_calls() {
        let delay_estimate: Option<DelayEstimate> = None;
        for rate in [16_000, 32_000, 48_000] {
            for num_render_channels in [1usize, 2, 8] {
                for num_capture_channels in [1usize, 2, 8] {
                    let config = EchoCanceller3Config::default();
                    let mut remover = EchoRemover::new(
                        config.clone(),
                        rate,
                        num_render_channels,
                        num_capture_channels,
                    );
                    let mut render_buffer =
                        RenderDelayBuffer::new(config.clone(), rate, num_render_channels);
                    let num_bands = num_bands_for_rate(rate);
                    let render =
                        vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; num_bands];
                    let mut capture =
                        vec![vec![vec![0.0f32; BLOCK_SIZE]; num_capture_channels]; num_bands];

                    for k in 0..100usize {
                        let variability = EchoPathVariability::new(
                            k % 3 == 0,
                            if k % 5 == 0 {
                                DelayAdjustment::NewDetectedDelay
                            } else {
                                DelayAdjustment::None
                            },
                            false,
                        );
                        render_buffer.insert(&render);
                        render_buffer.prepare_capture_processing();
                        let render_view = render_buffer.render_buffer();
                        remover.process_capture(
                            variability,
                            k % 2 == 0,
                            delay_estimate,
                            &render_view,
                            None,
                            &mut capture,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn basic_echo_removal() {
        const NUM_BLOCKS_TO_PROCESS: usize = 500;
        let mut random_generator = Random::new(42);
        let delay_estimate: Option<DelayEstimate> = None;
        for num_channels in [1usize, 2, 4] {
            for rate in [16_000, 32_000, 48_000] {
                let num_bands = num_bands_for_rate(rate);
                let config = EchoCanceller3Config::default();
                let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_channels]; num_bands];
                let mut y = x.clone();
                let variability = EchoPathVariability::new(false, DelayAdjustment::None, false);

                for delay_samples in [0usize, 64, 150, 200, 301] {
                    let mut remover =
                        EchoRemover::new(config.clone(), rate, num_channels, num_channels);
                    let mut render_buffer =
                        RenderDelayBuffer::new(config.clone(), rate, num_channels);
                    render_buffer.align_from_delay(delay_samples / BLOCK_SIZE);

                    let mut delay_buffers =
                        vec![vec![DelayBuffer::<f32>::new(delay_samples); num_channels]; num_bands];

                    let mut input_energy = 0.0f32;
                    let mut output_energy = 0.0f32;

                    for block_index in 0..NUM_BLOCKS_TO_PROCESS {
                        let silence = block_index < 100 || (block_index % 100 >= 10);
                        for band in 0..num_bands {
                            for channel in 0..num_channels {
                                if silence {
                                    x[band][channel].fill(0.0);
                                } else {
                                    randomize_sample_vector(
                                        &mut random_generator,
                                        &mut x[band][channel],
                                    );
                                }
                                delay_buffers[band][channel]
                                    .delay(&x[band][channel], &mut y[band][channel]);
                            }
                        }

                        if block_index > NUM_BLOCKS_TO_PROCESS / 2 {
                            input_energy +=
                                y[0][0].iter().map(|&sample| sample * sample).sum::<f32>();
                        }

                        render_buffer.insert(&x);
                        render_buffer.prepare_capture_processing();
                        let render_view = render_buffer.render_buffer();
                        remover.process_capture(
                            variability,
                            false,
                            delay_estimate,
                            &render_view,
                            None,
                            &mut y,
                        );

                        if block_index > NUM_BLOCKS_TO_PROCESS / 2 {
                            output_energy +=
                                y[0][0].iter().map(|&sample| sample * sample).sum::<f32>();
                        }
                    }

                    assert!(
                        input_energy > 10.0 * output_energy,
                        "rate {} channels {} delay {} input {} output {}",
                        rate,
                        num_channels,
                        delay_samples,
                        input_energy,
                        output_energy,
                    );
                }
            }
        }
    }
}
