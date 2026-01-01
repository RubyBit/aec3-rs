use crate::api::config::MainConfiguration;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

const H_ERROR_INITIAL: f32 = 10_000.0;
const POOR_EXCITATION_COUNTER_INITIAL: usize = 1000;

/// Provides the adaptive gain computation for the main filter.
pub struct MainFilterUpdateGain {
    data_dumper: ApmDataDumper,
    config_change_duration_blocks: i32,
    one_by_config_change_duration_blocks: f32,
    current_config: MainConfiguration,
    target_config: MainConfiguration,
    old_target_config: MainConfiguration,
    h_error: [f32; FFT_LENGTH_BY_2_PLUS_1],
    poor_signal_excitation_counter: usize,
    call_counter: usize,
    config_change_counter: i32,
}

impl MainFilterUpdateGain {
    pub fn new(config: MainConfiguration, config_change_duration_blocks: usize) -> Self {
        assert!(config_change_duration_blocks > 0);
        let duration = config_change_duration_blocks as i32;
        let current_config = config.clone();
        Self {
            data_dumper: ApmDataDumper::new_unique(),
            config_change_duration_blocks: duration,
            one_by_config_change_duration_blocks: 1.0 / duration as f32,
            current_config: current_config.clone(),
            target_config: current_config.clone(),
            old_target_config: config,
            h_error: [H_ERROR_INITIAL; FFT_LENGTH_BY_2_PLUS_1],
            poor_signal_excitation_counter: POOR_EXCITATION_COUNTER_INITIAL,
            call_counter: 0,
            config_change_counter: 0,
        }
    }

    pub fn set_config(&mut self, config: MainConfiguration, immediate_effect: bool) {
        if immediate_effect {
            self.current_config = config.clone();
            self.target_config = config.clone();
            self.old_target_config = config;
            self.config_change_counter = 0;
        } else {
            self.old_target_config = self.current_config.clone();
            self.target_config = config;
            self.config_change_counter = self.config_change_duration_blocks;
        }
    }

    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        if !matches!(variability.delay_change, DelayAdjustment::None) {
            self.h_error.fill(H_ERROR_INITIAL);
        }

        if !variability.gain_change {
            self.poor_signal_excitation_counter = POOR_EXCITATION_COUNTER_INITIAL;
            self.call_counter = 0;
        } else {
            // TODO(bugs.webrtc.org/9526): handle gain changes explicitly.
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        subtractor_output: &SubtractorOutput,
        erl: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        size_partitions: usize,
        saturated_capture_signal: bool,
        disallow_leakage_diverged: bool,
        gain_fft: &mut FftData,
    ) {
        let e_main_fft = &subtractor_output.e_main_fft;
        let e2_main = &subtractor_output.e2_main_spectrum;
        let e2_shadow = &subtractor_output.e2_shadow_spectrum;
        self.call_counter = self.call_counter.saturating_add(1);

        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_signal_excitation_counter = 0;
        }

        self.poor_signal_excitation_counter = self.poor_signal_excitation_counter.saturating_add(1);

        if self.poor_signal_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            gain_fft.re.fill(0.0);
            gain_fft.im.fill(0.0);
        } else {
            let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let size_partitions_f32 = size_partitions as f32;
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                if render_power[k] >= self.current_config.noise_gate {
                    mu[k] = self.h_error[k]
                        / (0.5 * self.h_error[k] * render_power[k]
                            + size_partitions_f32 * e2_main[k]);
                } else {
                    mu[k] = 0.0;
                }
            }
            render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);

            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.h_error[k] -= 0.5 * mu[k] * render_power[k] * self.h_error[k];
            }

            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                gain_fft.re[k] = mu[k] * e_main_fft.re[k];
                gain_fft.im[k] = mu[k] * e_main_fft.im[k];
            }
        }

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if e2_shadow[k] >= e2_main[k] || disallow_leakage_diverged {
                self.h_error[k] += self.current_config.leakage_converged * erl[k];
            } else {
                self.h_error[k] += self.current_config.leakage_diverged * erl[k];
            }
            self.h_error[k] = self.h_error[k].max(self.current_config.error_floor);
            self.h_error[k] = self.h_error[k].min(self.current_config.error_ceil);
        }

        self.data_dumper
            .dump_raw_f32_slice("aec3_main_gain_H_error", &self.h_error);
    }

    fn update_current_config(&mut self) {
        if self.config_change_counter > 0 {
            self.config_change_counter -= 1;
            if self.config_change_counter > 0 {
                let change_factor =
                    self.config_change_counter as f32 * self.one_by_config_change_duration_blocks;
                let average =
                    |from: f32, to: f32| from * change_factor + to * (1.0 - change_factor);
                self.current_config.leakage_converged = average(
                    self.old_target_config.leakage_converged,
                    self.target_config.leakage_converged,
                );
                self.current_config.leakage_diverged = average(
                    self.old_target_config.leakage_diverged,
                    self.target_config.leakage_diverged,
                );
                self.current_config.error_floor = average(
                    self.old_target_config.error_floor,
                    self.target_config.error_floor,
                );
                self.current_config.error_ceil = average(
                    self.old_target_config.error_ceil,
                    self.target_config.error_ceil,
                );
                self.current_config.noise_gate = average(
                    self.old_target_config.noise_gate,
                    self.target_config.noise_gate,
                );
            } else {
                self.current_config = self.target_config.clone();
                self.old_target_config = self.target_config.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::adaptive_fir_filter::AdaptiveFirFilter;
    use crate::audio_processing::aec3::adaptive_fir_filter_erl::compute_erl;
    use crate::audio_processing::aec3::aec_state::AecState;
    use crate::audio_processing::aec3::aec3_common::{
        Aec3Optimization, BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
        detect_optimization, get_time_domain_length, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
    use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
    use crate::audio_processing::aec3::echo_path_variability::{
        DelayAdjustment, EchoPathVariability,
    };
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
    use crate::audio_processing::aec3::shadow_filter_update_gain::ShadowFilterUpdateGain;
    use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
    use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    struct FilterUpdateResult {
        e_last_block: [f32; BLOCK_SIZE],
        y_last_block: [f32; BLOCK_SIZE],
        g_last_block: FftData,
    }

    fn make_render_block(num_bands: usize, num_channels: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![0.0f32; BLOCK_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn randomize_block(rng: &mut Random, block: &mut Vec<Vec<Vec<f32>>>) {
        for band in block.iter_mut() {
            for channel in band.iter_mut() {
                randomize_sample_vector(rng, channel);
            }
        }
    }

    fn zero_block(block: &mut Vec<Vec<Vec<f32>>>) {
        for band in block.iter_mut() {
            for channel in band.iter_mut() {
                for sample in channel.iter_mut() {
                    *sample = 0.0;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_filter_update_test(
        num_blocks_to_process: usize,
        delay_samples: usize,
        filter_length_blocks: usize,
        blocks_with_echo_path_changes: &[usize],
        blocks_with_saturation: &[usize],
        use_silent_render_in_second_half: bool,
    ) -> FilterUpdateResult {
        const NUM_RENDER_CHANNELS: usize = 1;
        const NUM_CAPTURE_CHANNELS: usize = 1;
        const SAMPLE_RATE_HZ: i32 = 48_000;
        let mut config = EchoCanceller3Config::default();
        config.filter.main.length_blocks = filter_length_blocks;
        config.filter.shadow.length_blocks = filter_length_blocks;
        config.delay.default_delay = 1;

        let optimization = detect_optimization();
        let data_dumper = ApmDataDumper::new(42);
        let mut main_filter = AdaptiveFirFilter::new(
            config.filter.main.length_blocks,
            config.filter.main.length_blocks,
            config.filter.config_change_duration_blocks,
            NUM_RENDER_CHANNELS,
            optimization,
            data_dumper.clone(),
        );
        let mut shadow_filter = AdaptiveFirFilter::new(
            config.filter.shadow.length_blocks,
            config.filter.shadow.length_blocks,
            config.filter.config_change_duration_blocks,
            NUM_RENDER_CHANNELS,
            optimization,
            data_dumper,
        );

        let mut shadow_gain = ShadowFilterUpdateGain::new(
            config.filter.shadow.clone(),
            config.filter.config_change_duration_blocks,
        );
        let mut main_gain = MainFilterUpdateGain::new(
            config.filter.main.clone(),
            config.filter.config_change_duration_blocks,
        );

        let mut render_delay_buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, NUM_RENDER_CHANNELS);
        let mut aec_state = AecState::new(config.clone(), NUM_CAPTURE_CHANNELS);
        let mut render_signal_analyzer = RenderSignalAnalyzer::new(&config);
        let mut delay_buffer = DelayBuffer::<f32>::new(delay_samples);
        let mut rng = Random::new(42);
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut x = make_render_block(num_bands, NUM_RENDER_CHANNELS);
        let mut y = [0.0f32; BLOCK_SIZE];
        let mut scratch = [0.0f32; FFT_LENGTH];
        let mut e_shadow = [0.0f32; BLOCK_SIZE];
        let mut e_shadow_fft = FftData::default();
        let mut s_freq = FftData::default();
        let mut g = FftData::default();
        let mut render_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut output = vec![SubtractorOutput::new(); NUM_CAPTURE_CHANNELS];
        for subtractor_output in output.iter_mut() {
            subtractor_output.reset();
        }
        let mut h =
            vec![
                vec![0.0f32; get_time_domain_length(main_filter.max_filter_size_partitions())];
                NUM_CAPTURE_CHANNELS
            ];
        let mut h2 =
            vec![
                vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; main_filter.max_filter_size_partitions()];
                NUM_CAPTURE_CHANNELS
            ];
        let mut e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let fft = Aec3Fft::new();
        let no_change = EchoPathVariability::new(false, DelayAdjustment::None, false);
        let scale = 1.0f32 / FFT_LENGTH_BY_2 as f32;

        for block_idx in 0..num_blocks_to_process {
            if blocks_with_echo_path_changes.contains(&block_idx) {
                main_filter.handle_echo_path_change();
            }

            let saturation = blocks_with_saturation.contains(&block_idx);

            if use_silent_render_in_second_half && block_idx > num_blocks_to_process / 2 {
                zero_block(&mut x);
            } else {
                randomize_block(&mut rng, &mut x);
            }

            delay_buffer.delay(&x[0][0], &mut y);
            render_delay_buffer.insert(&x);
            if block_idx == 0 {
                render_delay_buffer.reset();
            }
            render_delay_buffer.prepare_capture_processing();

            let render_buffer_view = render_delay_buffer.render_buffer();
            let min_delay = aec_state.min_direct_path_filter_delay();
            let delay_partitions = (min_delay >= 0).then_some(min_delay as usize);
            render_signal_analyzer.update(&render_buffer_view, delay_partitions);

            let subtractor_output = &mut output[0];
            main_filter.filter(&render_buffer_view, &mut s_freq);
            fft.ifft(&s_freq, &mut scratch);
            for i in 0..BLOCK_SIZE {
                let value = y[i] - scratch[i + FFT_LENGTH_BY_2] * scale;
                subtractor_output.e_main[i] = value.clamp(-32_768.0, 32_767.0);
            }
            fft.zero_padded_fft(
                &subtractor_output.e_main,
                Window::Rectangular,
                &mut subtractor_output.e_main_fft,
            );

            shadow_filter.filter(&render_buffer_view, &mut s_freq);
            fft.ifft(&s_freq, &mut scratch);
            for i in 0..BLOCK_SIZE {
                let value = y[i] - scratch[i + FFT_LENGTH_BY_2] * scale;
                e_shadow[i] = value.clamp(-32_768.0, 32_767.0);
            }
            fft.zero_padded_fft(&e_shadow, Window::Rectangular, &mut e_shadow_fft);

            subtractor_output.e_main_fft.spectrum(
                Aec3Optimization::None,
                &mut subtractor_output.e2_main_spectrum,
            );
            e_shadow_fft.spectrum(
                Aec3Optimization::None,
                &mut subtractor_output.e2_shadow_spectrum,
            );

            render_buffer_view.spectral_sum(shadow_filter.size_partitions(), &mut render_power);
            shadow_gain.compute(
                &render_power,
                &render_signal_analyzer,
                &e_shadow_fft,
                shadow_filter.size_partitions(),
                saturation,
                &mut g,
            );
            shadow_filter.adapt(&render_buffer_view, &g);

            render_buffer_view.spectral_sum(main_filter.size_partitions(), &mut render_power);
            compute_erl(optimization, &h2[0], &mut erl);
            main_gain.compute(
                &render_power,
                &render_signal_analyzer,
                subtractor_output,
                &erl,
                main_filter.size_partitions(),
                saturation,
                false,
                &mut g,
            );
            main_filter.adapt_with_impulse_response(&render_buffer_view, &g, &mut h[0]);

            aec_state.handle_echo_path_change(&no_change);
            main_filter.compute_frequency_response(&mut h2[0]);
            e2_main[0].copy_from_slice(&subtractor_output.e2_main_spectrum);
            aec_state.update(
                Option::<DelayEstimate>::None,
                &h2,
                &h,
                &render_buffer_view,
                &e2_main,
                &y2,
                &output,
            );
        }

        let mut e_last_block = [0.0f32; BLOCK_SIZE];
        e_last_block.copy_from_slice(&output[0].e_main);
        let y_last_block = y;
        let g_last_block = g;

        FilterUpdateResult {
            e_last_block,
            y_last_block,
            g_last_block,
        }
    }

    fn block_energy(block: &[f32; BLOCK_SIZE]) -> f32 {
        block.iter().map(|sample| sample * sample).sum()
    }

    #[test]
    fn gain_causes_filter_to_converge() {
        let blocks_with_echo_path_changes = Vec::new();
        let blocks_with_saturation = Vec::new();
        for &filter_length_blocks in &[12usize, 20, 30] {
            for &delay_samples in &[0usize, 64, 150, 200, 301] {
                let result = run_filter_update_test(
                    600,
                    delay_samples,
                    filter_length_blocks,
                    &blocks_with_echo_path_changes,
                    &blocks_with_saturation,
                    false,
                );
                let e_energy = block_energy(&result.e_last_block);
                let y_energy = block_energy(&result.y_last_block);
                if filter_length_blocks == 12 {
                    assert!(
                        1000.0 * e_energy < y_energy,
                        "delay={}, length={}",
                        delay_samples,
                        filter_length_blocks
                    );
                } else {
                    assert!(
                        e_energy < y_energy,
                        "delay={}, length={}",
                        delay_samples,
                        filter_length_blocks
                    );
                }
            }
        }
    }

    #[test]
    fn decreasing_gain() {
        let blocks_with_echo_path_changes = Vec::new();
        let blocks_with_saturation = Vec::new();
        let a = run_filter_update_test(
            250,
            65,
            12,
            &blocks_with_echo_path_changes,
            &blocks_with_saturation,
            false,
        );
        let b = run_filter_update_test(
            500,
            65,
            12,
            &blocks_with_echo_path_changes,
            &blocks_with_saturation,
            false,
        );
        let c = run_filter_update_test(
            750,
            65,
            12,
            &blocks_with_echo_path_changes,
            &blocks_with_saturation,
            false,
        );

        let mut a_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut b_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut c_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        a.g_last_block
            .spectrum(Aec3Optimization::None, &mut a_power);
        b.g_last_block
            .spectrum(Aec3Optimization::None, &mut b_power);
        c.g_last_block
            .spectrum(Aec3Optimization::None, &mut c_power);

        let sum_a: f32 = a_power.iter().sum();
        let sum_b: f32 = b_power.iter().sum();
        let sum_c: f32 = c_power.iter().sum();
        assert!(sum_a > sum_b);
        assert!(sum_b > sum_c);
    }

    #[test]
    fn saturation_behavior() {
        let blocks_with_echo_path_changes = Vec::new();
        let mut blocks_with_saturation = Vec::new();
        for idx in 99..200 {
            blocks_with_saturation.push(idx);
        }

        for &filter_length_blocks in &[12usize, 20, 30] {
            let result = run_filter_update_test(
                100,
                65,
                filter_length_blocks,
                &blocks_with_echo_path_changes,
                &blocks_with_saturation,
                false,
            );
            let zero_gain = FftData::default();
            assert_eq!(zero_gain.re, result.g_last_block.re);
            assert_eq!(zero_gain.im, result.g_last_block.im);

            let a = run_filter_update_test(
                99,
                65,
                filter_length_blocks,
                &blocks_with_echo_path_changes,
                &blocks_with_saturation,
                false,
            );
            let b = run_filter_update_test(
                201,
                65,
                filter_length_blocks,
                &blocks_with_echo_path_changes,
                &blocks_with_saturation,
                false,
            );
            let mut a_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut b_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            a.g_last_block
                .spectrum(Aec3Optimization::None, &mut a_power);
            b.g_last_block
                .spectrum(Aec3Optimization::None, &mut b_power);
            let sum_a: f32 = a_power.iter().sum();
            let sum_b: f32 = b_power.iter().sum();
            assert!(sum_a < sum_b);
        }
    }

    #[test]
    #[ignore = "Parity with reference - disabled upstream"]
    fn echo_path_change_behavior() {
        let mut blocks_with_echo_path_changes = Vec::new();
        blocks_with_echo_path_changes.push(99);
        let blocks_with_saturation = Vec::new();

        for &filter_length_blocks in &[12usize, 20, 30] {
            let a = run_filter_update_test(
                100,
                65,
                filter_length_blocks,
                &blocks_with_echo_path_changes,
                &blocks_with_saturation,
                false,
            );
            let b = run_filter_update_test(
                101,
                65,
                filter_length_blocks,
                &blocks_with_echo_path_changes,
                &blocks_with_saturation,
                false,
            );
            let mut a_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut b_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            a.g_last_block
                .spectrum(Aec3Optimization::None, &mut a_power);
            b.g_last_block
                .spectrum(Aec3Optimization::None, &mut b_power);
            let sum_a: f32 = a_power.iter().sum();
            let sum_b: f32 = b_power.iter().sum();
            assert!(sum_a < sum_b);
        }
    }
}
