use std::cmp::max;

use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::adaptive_fir_filter::AdaptiveFirFilter;
use crate::audio_processing::aec3::adaptive_fir_filter_erl::compute_erl;
use crate::audio_processing::aec3::aec_state::AecState;
use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
    get_time_domain_length,
};
use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
use crate::audio_processing::aec3::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::main_filter_update_gain::MainFilterUpdateGain;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
use crate::audio_processing::aec3::shadow_filter_update_gain::ShadowFilterUpdateGain;
use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

/// Provides the linear echo subtraction functionality for the AEC3 pipeline.
pub struct Subtractor {
    fft: Aec3Fft,
    data_dumper: ApmDataDumper,
    optimization: Aec3Optimization,
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    main_filters: Vec<AdaptiveFirFilter>,
    shadow_filters: Vec<AdaptiveFirFilter>,
    main_gains: Vec<MainFilterUpdateGain>,
    shadow_gains: Vec<ShadowFilterUpdateGain>,
    filter_misadjustment_estimators: Vec<FilterMisadjustmentEstimator>,
    poor_shadow_filter_counters: Vec<usize>,
    main_frequency_responses: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    main_impulse_responses: Vec<Vec<f32>>,
}

impl Subtractor {
    pub fn new(
        config: EchoCanceller3Config,
        num_render_channels: usize,
        num_capture_channels: usize,
        data_dumper: ApmDataDumper,
        optimization: Aec3Optimization,
    ) -> Self {
        assert!(num_capture_channels > 0);
        let mut main_filters = Vec::with_capacity(num_capture_channels);
        let mut shadow_filters = Vec::with_capacity(num_capture_channels);
        let mut main_gains = Vec::with_capacity(num_capture_channels);
        let mut shadow_gains = Vec::with_capacity(num_capture_channels);
        let mut filter_misadjustment_estimators = Vec::with_capacity(num_capture_channels);

        for _ in 0..num_capture_channels {
            main_filters.push(AdaptiveFirFilter::new(
                config.filter.main.length_blocks,
                config.filter.main_initial.length_blocks,
                config.filter.config_change_duration_blocks,
                num_render_channels,
                optimization,
                data_dumper.clone(),
            ));
            shadow_filters.push(AdaptiveFirFilter::new(
                config.filter.shadow.length_blocks,
                config.filter.shadow_initial.length_blocks,
                config.filter.config_change_duration_blocks,
                num_render_channels,
                optimization,
                data_dumper.clone(),
            ));
            main_gains.push(MainFilterUpdateGain::new(
                config.filter.main_initial.clone(),
                config.filter.config_change_duration_blocks,
            ));
            shadow_gains.push(ShadowFilterUpdateGain::new(
                config.filter.shadow_initial.clone(),
                config.filter.config_change_duration_blocks,
            ));
            filter_misadjustment_estimators.push(FilterMisadjustmentEstimator::default());
        }

        let max_main_partitions = max(
            config.filter.main_initial.length_blocks,
            config.filter.main.length_blocks,
        );
        let impulse_len = get_time_domain_length(max_main_partitions);
        let mut main_frequency_responses = Vec::with_capacity(num_capture_channels);
        let mut main_impulse_responses = Vec::with_capacity(num_capture_channels);
        for _ in 0..num_capture_channels {
            main_frequency_responses
                .push(vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; max_main_partitions]);
            main_impulse_responses.push(vec![0.0f32; impulse_len]);
        }

        Self {
            fft: Aec3Fft::new(),
            data_dumper,
            optimization,
            config,
            num_capture_channels,
            main_filters,
            shadow_filters,
            main_gains,
            shadow_gains,
            filter_misadjustment_estimators,
            poor_shadow_filter_counters: vec![0; num_capture_channels],
            main_frequency_responses,
            main_impulse_responses,
        }
    }

    pub fn process(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        capture: &[Vec<f32>],
        render_signal_analyzer: &RenderSignalAnalyzer,
        aec_state: &AecState,
        outputs: &mut [SubtractorOutput],
    ) {
        assert_eq!(self.num_capture_channels, capture.len());
        assert_eq!(self.num_capture_channels, outputs.len());

        let same_filter_sizes =
            self.main_filters[0].size_partitions() == self.shadow_filters[0].size_partitions();
        let mut x2_main = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut x2_shadow = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        if same_filter_sizes {
            render_buffer.spectral_sum(self.main_filters[0].size_partitions(), &mut x2_main);
            x2_shadow.copy_from_slice(&x2_main);
        } else if self.main_filters[0].size_partitions() > self.shadow_filters[0].size_partitions()
        {
            render_buffer.spectral_sums(
                self.shadow_filters[0].size_partitions(),
                self.main_filters[0].size_partitions(),
                &mut x2_shadow,
                &mut x2_main,
            );
        } else {
            render_buffer.spectral_sums(
                self.main_filters[0].size_partitions(),
                self.shadow_filters[0].size_partitions(),
                &mut x2_main,
                &mut x2_shadow,
            );
        }

        for ch in 0..self.num_capture_channels {
            let y = &capture[ch];
            assert_eq!(BLOCK_SIZE, y.len());
            let output = &mut outputs[ch];
            let mut s_freq = FftData::default();

            self.main_filters[ch].filter(render_buffer, &mut s_freq);
            prediction_error(
                &self.fft,
                &s_freq,
                y,
                &mut output.e_main,
                Some(&mut output.s_main),
            );

            self.shadow_filters[ch].filter(render_buffer, &mut s_freq);
            prediction_error(
                &self.fft,
                &s_freq,
                y,
                &mut output.e_shadow,
                Some(&mut output.s_shadow),
            );

            output.compute_metrics(y);

            let misadjustment = &mut self.filter_misadjustment_estimators[ch];
            misadjustment.update(output);
            let mut main_filters_adjusted = false;
            if misadjustment.is_adjustment_needed() {
                let scale = misadjustment.misadjustment();
                self.main_filters[ch].scale_filter(scale);
                for sample in &mut self.main_impulse_responses[ch] {
                    *sample *= scale;
                }
                scale_filter_output(y, scale, &mut output.e_main, &mut output.s_main);
                misadjustment.reset();
                main_filters_adjusted = true;
            }

            self.fft
                .zero_padded_fft(&output.e_main, Window::Hanning, &mut output.e_main_fft);
            let mut e_shadow_fft = FftData::default();
            self.fft
                .zero_padded_fft(&output.e_shadow, Window::Hanning, &mut e_shadow_fft);
            output
                .e_main_fft
                .spectrum(self.optimization, &mut output.e2_main_spectrum);
            e_shadow_fft.spectrum(self.optimization, &mut output.e2_shadow_spectrum);

            let mut g = FftData::default();
            if !main_filters_adjusted {
                let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                compute_erl(
                    self.optimization,
                    &self.main_frequency_responses[ch],
                    &mut erl,
                );
                self.main_gains[ch].compute(
                    &x2_main,
                    render_signal_analyzer,
                    output,
                    &erl,
                    self.main_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    &mut g,
                );
            }

            if main_filters_adjusted {
                g.re.fill(0.0);
                g.im.fill(0.0);
            }

            self.main_filters[ch].adapt_with_impulse_response(
                render_buffer,
                &g,
                &mut self.main_impulse_responses[ch],
            );
            self.main_filters[ch]
                .compute_frequency_response(&mut self.main_frequency_responses[ch]);

            if ch == 0 {
                self.data_dumper
                    .dump_raw_f32_slice("aec3_subtractor_G_main", &g.re);
                self.data_dumper
                    .dump_raw_f32_slice("aec3_subtractor_G_main", &g.im);
            }

            if output.e2_main < output.e2_shadow {
                self.poor_shadow_filter_counters[ch] += 1;
            } else {
                self.poor_shadow_filter_counters[ch] = 0;
            }

            if self.poor_shadow_filter_counters[ch] < 5 {
                self.shadow_gains[ch].compute(
                    &x2_shadow,
                    render_signal_analyzer,
                    &e_shadow_fft,
                    self.shadow_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    &mut g,
                );
            } else {
                self.poor_shadow_filter_counters[ch] = 0;
                self.shadow_filters[ch].set_filter(
                    self.main_filters[ch].size_partitions(),
                    self.main_filters[ch].get_filter(),
                );
                self.shadow_gains[ch].compute(
                    &x2_shadow,
                    render_signal_analyzer,
                    &output.e_main_fft,
                    self.shadow_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    &mut g,
                );
            }

            self.shadow_filters[ch].adapt(render_buffer, &g);

            if ch == 0 {
                self.data_dumper
                    .dump_raw_f32_slice("aec3_subtractor_G_shadow", &g.re);
                self.data_dumper
                    .dump_raw_f32_slice("aec3_subtractor_G_shadow", &g.im);
                self.filter_misadjustment_estimators[ch].dump(&self.data_dumper);
                self.dump_filters();
            }

            for sample in &mut output.e_main {
                *sample = sample.clamp(-32_768.0, 32_767.0);
            }

            self.data_dumper.dump_wav(
                "aec3_main_filters_output",
                BLOCK_SIZE,
                &output.e_main,
                16_000,
                1,
            );
            self.data_dumper.dump_wav(
                "aec3_shadow_filter_output",
                BLOCK_SIZE,
                &output.e_shadow,
                16_000,
                1,
            );
        }
    }

    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        let mut perform_full_reset = false;
        if !matches!(variability.delay_change, DelayAdjustment::None) {
            perform_full_reset = true;
        }

        if perform_full_reset {
            for ch in 0..self.num_capture_channels {
                self.main_filters[ch].handle_echo_path_change();
                self.shadow_filters[ch].handle_echo_path_change();
                self.main_gains[ch].handle_echo_path_change(variability);
                self.shadow_gains[ch].handle_echo_path_change();
                self.main_gains[ch].set_config(self.config.filter.main_initial.clone(), true);
                self.shadow_gains[ch].set_config(self.config.filter.shadow_initial.clone(), true);
                self.main_filters[ch]
                    .set_size_partitions(self.config.filter.main_initial.length_blocks, true);
                self.shadow_filters[ch]
                    .set_size_partitions(self.config.filter.shadow_initial.length_blocks, true);
            }
        } else if variability.gain_change {
            for gain in &mut self.main_gains {
                gain.handle_echo_path_change(variability);
            }
        }
    }

    pub fn exit_initial_state(&mut self) {
        for ch in 0..self.num_capture_channels {
            self.main_gains[ch].set_config(self.config.filter.main.clone(), false);
            self.shadow_gains[ch].set_config(self.config.filter.shadow.clone(), false);
            self.main_filters[ch].set_size_partitions(self.config.filter.main.length_blocks, false);
            self.shadow_filters[ch]
                .set_size_partitions(self.config.filter.shadow.length_blocks, false);
        }
    }

    pub fn filter_frequency_responses(&self) -> &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>] {
        &self.main_frequency_responses
    }

    pub fn filter_impulse_responses(&self) -> &[Vec<f32>] {
        &self.main_impulse_responses
    }

    fn dump_filters(&self) {
        if self.main_filters.is_empty() {
            return;
        }
        if let Some(response) = self.main_impulse_responses.first() {
            self.data_dumper
                .dump_raw_f32_slice("aec3_subtractor_h_main", response);
        }
        let filters = self.main_filters[0].get_filter();
        for partition in filters {
            if partition.is_empty() {
                continue;
            }
            self.data_dumper
                .dump_raw_f32_slice("aec3_subtractor_H_main", &partition[0].re);
            self.data_dumper
                .dump_raw_f32_slice("aec3_subtractor_H_main", &partition[0].im);
        }
    }
}

fn prediction_error(
    fft: &Aec3Fft,
    spectrum: &FftData,
    capture: &[f32],
    error: &mut [f32; BLOCK_SIZE],
    signal: Option<&mut [f32; BLOCK_SIZE]>,
) {
    let mut tmp = [0.0f32; FFT_LENGTH];
    fft.ifft(spectrum, &mut tmp);
    let scale = 1.0f32 / FFT_LENGTH_BY_2 as f32;
    for (idx, dst) in error.iter_mut().enumerate() {
        *dst = capture[idx] - tmp[idx + FFT_LENGTH_BY_2] * scale;
    }
    if let Some(s) = signal {
        for (idx, dst) in s.iter_mut().enumerate() {
            *dst = scale * tmp[idx + FFT_LENGTH_BY_2];
        }
    }
}

fn scale_filter_output(
    capture: &[f32],
    factor: f32,
    error: &mut [f32; BLOCK_SIZE],
    signal: &mut [f32; BLOCK_SIZE],
) {
    for idx in 0..BLOCK_SIZE {
        signal[idx] *= factor;
        error[idx] = capture[idx] - signal[idx];
    }
}

struct FilterMisadjustmentEstimator {
    n_blocks: usize,
    n_blocks_acum: usize,
    e2_acum: f32,
    y2_acum: f32,
    inv_misadjustment: f32,
    overhang: i32,
}

impl Default for FilterMisadjustmentEstimator {
    fn default() -> Self {
        Self {
            n_blocks: 4,
            n_blocks_acum: 0,
            e2_acum: 0.0,
            y2_acum: 0.0,
            inv_misadjustment: 0.0,
            overhang: 0,
        }
    }
}

impl FilterMisadjustmentEstimator {
    fn update(&mut self, output: &SubtractorOutput) {
        self.e2_acum += output.e2_main;
        self.y2_acum += output.y2;
        self.n_blocks_acum += 1;

        if self.n_blocks_acum == self.n_blocks {
            let high_energy_threshold = self.n_blocks as f32 * 200.0 * 200.0 * BLOCK_SIZE as f32;
            if self.y2_acum > high_energy_threshold {
                let update = self.e2_acum / self.y2_acum;
                let saturation_threshold =
                    self.n_blocks as f32 * 7_500.0 * 7_500.0 * BLOCK_SIZE as f32;
                if self.e2_acum > saturation_threshold {
                    self.overhang = 4;
                } else {
                    self.overhang = (self.overhang - 1).max(0);
                }

                if update < self.inv_misadjustment || self.overhang > 0 {
                    self.inv_misadjustment += 0.1 * (update - self.inv_misadjustment);
                }
            }

            self.e2_acum = 0.0;
            self.y2_acum = 0.0;
            self.n_blocks_acum = 0;
        }
    }

    fn misadjustment(&self) -> f32 {
        debug_assert!(self.inv_misadjustment > 0.0);
        2.0 / self.inv_misadjustment.sqrt()
    }

    fn is_adjustment_needed(&self) -> bool {
        self.inv_misadjustment > 10.0
    }

    fn reset(&mut self) {
        self.e2_acum = 0.0;
        self.y2_acum = 0.0;
        self.n_blocks_acum = 0;
        self.inv_misadjustment = 0.0;
        self.overhang = 0;
    }

    fn dump(&self, data_dumper: &ApmDataDumper) {
        data_dumper.dump_raw_f32("aec3_inv_misadjustment_factor", self.inv_misadjustment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{detect_optimization, num_bands_for_rate};
    use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::utility::cascaded_biquad_filter::{
        BiQuadCoefficients, CascadedBiQuadFilter,
    };
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    const SAMPLE_RATE_HZ: i32 = 48_000;

    fn run_subtractor_test(
        num_render_channels: usize,
        num_capture_channels: usize,
        num_blocks_to_process: usize,
        delay_samples: usize,
        main_filter_length_blocks: usize,
        shadow_filter_length_blocks: usize,
        uncorrelated_inputs: bool,
        blocks_with_echo_path_changes: &[usize],
    ) -> Vec<f32> {
        let mut config = EchoCanceller3Config::default();
        config.filter.main.length_blocks = main_filter_length_blocks;
        config.filter.shadow.length_blocks = shadow_filter_length_blocks;
        config.delay.default_delay = 1;

        let optimization = detect_optimization();
        let data_dumper = ApmDataDumper::new(42);
        let mut subtractor = Subtractor::new(
            config.clone(),
            num_render_channels,
            num_capture_channels,
            data_dumper,
            optimization,
        );

        let mut render_delay_buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);
        let mut render_signal_analyzer = RenderSignalAnalyzer::new(&config);
        let mut aec_state = AecState::new(config.clone(), num_capture_channels);
        let delay_estimate: Option<DelayEstimate> = None;

        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; num_bands];
        let mut y = vec![vec![0.0f32; BLOCK_SIZE]; num_capture_channels];
        let mut output = vec![SubtractorOutput::new(); num_capture_channels];
        let mut rng = Random::new(42);

        let mut delay_buffers = vec![vec![]; num_capture_channels];
        for capture_ch in 0..num_capture_channels {
            for _ in 0..num_render_channels {
                delay_buffers[capture_ch].push(DelayBuffer::new(delay_samples));
            }
        }

        const HIGH_PASS: BiQuadCoefficients = BiQuadCoefficients {
            b: [0.97261, -1.94523, 0.97261],
            a: [-1.94448, 0.94598],
        };
        let mut x_hp_filters = Vec::with_capacity(num_render_channels);
        for _ in 0..num_render_channels {
            x_hp_filters.push(CascadedBiQuadFilter::with_coefficients(HIGH_PASS, 1));
        }
        let mut y_hp_filters = Vec::with_capacity(num_capture_channels);
        for _ in 0..num_capture_channels {
            y_hp_filters.push(CascadedBiQuadFilter::with_coefficients(HIGH_PASS, 1));
        }

        let y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];

        for block_idx in 0..num_blocks_to_process {
            for render_ch in 0..num_render_channels {
                randomize_sample_vector(&mut rng, &mut x[0][render_ch]);
            }
            if uncorrelated_inputs {
                for capture_ch in 0..num_capture_channels {
                    randomize_sample_vector(&mut rng, &mut y[capture_ch]);
                }
            } else {
                for capture_ch in 0..num_capture_channels {
                    y[capture_ch].fill(0.0);
                    for render_ch in 0..num_render_channels {
                        let mut y_channel = [0.0f32; BLOCK_SIZE];
                        delay_buffers[capture_ch][render_ch]
                            .delay(&x[0][render_ch], &mut y_channel);
                        for (dst, src) in y[capture_ch].iter_mut().zip(y_channel.iter()) {
                            *dst += *src / num_render_channels as f32;
                        }
                    }
                }
            }

            for ch in 0..num_render_channels {
                x_hp_filters[ch].process_in_place(&mut x[0][ch]);
            }
            for ch in 0..num_capture_channels {
                y_hp_filters[ch].process_in_place(&mut y[ch]);
            }

            render_delay_buffer.insert(&x);
            if block_idx == 0 {
                render_delay_buffer.reset();
            }
            render_delay_buffer.prepare_capture_processing();
            let render_buffer_view = render_delay_buffer.render_buffer();
            let min_delay = aec_state.min_direct_path_filter_delay();
            let delay_partitions = if min_delay >= 0 {
                Some(min_delay as usize)
            } else {
                None
            };
            render_signal_analyzer.update(&render_buffer_view, delay_partitions);

            if blocks_with_echo_path_changes.contains(&block_idx) {
                let variability =
                    EchoPathVariability::new(true, DelayAdjustment::NewDetectedDelay, false);
                subtractor.handle_echo_path_change(&variability);
            }

            subtractor.process(
                &render_buffer_view,
                &y,
                &render_signal_analyzer,
                &aec_state,
                output.as_mut_slice(),
            );

            let variability = EchoPathVariability::new(false, DelayAdjustment::None, false);
            aec_state.handle_echo_path_change(&variability);
            aec_state.update(
                delay_estimate,
                subtractor.filter_frequency_responses(),
                subtractor.filter_impulse_responses(),
                &render_buffer_view,
                &e2_main,
                &y2,
                &output,
            );
        }

        let mut results = Vec::with_capacity(num_capture_channels);
        for ch in 0..num_capture_channels {
            let output_power: f32 = output[ch]
                .e_main
                .iter()
                .zip(output[ch].e_main.iter())
                .map(|(a, b)| a * b)
                .sum();
            let y_power: f32 = y[ch].iter().map(|sample| sample * sample).sum();
            if y_power == 0.0 {
                results.push(-1.0);
            } else {
                results.push(output_power / y_power);
            }
        }
        results
    }

    fn produce_debug_text(
        num_render_channels: usize,
        num_capture_channels: usize,
        delay: usize,
        filter_length_blocks: usize,
    ) -> String {
        format!(
            "delay: {delay}, filter_length_blocks: {filter_length_blocks}, num_render_channels: {num_render_channels}, num_capture_channels: {num_capture_channels}"
        )
    }

    #[test]
    fn convergence() {
        let blocks_with_echo_path_changes = Vec::new();
        for &filter_length_blocks in &[12usize, 20, 30] {
            for &delay_samples in &[0usize, 64, 150, 200, 301] {
                let context = produce_debug_text(1, 1, delay_samples, filter_length_blocks);
                let ratios = run_subtractor_test(
                    1,
                    1,
                    2500,
                    delay_samples,
                    filter_length_blocks,
                    filter_length_blocks,
                    false,
                    &blocks_with_echo_path_changes,
                );
                for ratio in ratios {
                    assert!(ratio < 0.1, "{context}");
                }
            }
        }
    }

    #[test]
    fn convergence_multi_channel() {
        let blocks_with_echo_path_changes = Vec::new();
        for &num_render_channels in &[1usize, 2, 4, 8] {
            for &num_capture_channels in &[1usize, 2, 4] {
                let context = produce_debug_text(num_render_channels, num_capture_channels, 64, 20);
                let ratios = run_subtractor_test(
                    num_render_channels,
                    num_capture_channels,
                    2500 * num_render_channels,
                    64,
                    20,
                    20,
                    false,
                    &blocks_with_echo_path_changes,
                );
                for ratio in ratios {
                    assert!(ratio < 0.1, "{context}");
                }
            }
        }
    }

    #[test]
    fn main_filter_longer_than_shadow_filter() {
        let blocks_with_echo_path_changes = Vec::new();
        let ratios =
            run_subtractor_test(1, 1, 400, 64, 20, 15, false, &blocks_with_echo_path_changes);
        for ratio in ratios {
            assert!(ratio < 0.5);
        }
    }

    #[test]
    fn shadow_filter_longer_than_main_filter() {
        let blocks_with_echo_path_changes = Vec::new();
        let ratios =
            run_subtractor_test(1, 1, 400, 64, 15, 20, false, &blocks_with_echo_path_changes);
        for ratio in ratios {
            assert!(ratio < 0.5);
        }
    }

    #[test]
    fn non_convergence_on_uncorrelated_signals() {
        let blocks_with_echo_path_changes = Vec::new();
        for &filter_length_blocks in &[12usize, 20, 30] {
            for &delay_samples in &[0usize, 64, 150, 200, 301] {
                let context = produce_debug_text(1, 1, delay_samples, filter_length_blocks);
                let ratios = run_subtractor_test(
                    1,
                    1,
                    3000,
                    delay_samples,
                    filter_length_blocks,
                    filter_length_blocks,
                    true,
                    &blocks_with_echo_path_changes,
                );
                for ratio in ratios {
                    assert!((ratio - 1.0).abs() < 0.1, "{context}");
                }
            }
        }
    }

    #[test]
    fn non_convergence_on_uncorrelated_signals_multi_channel() {
        let blocks_with_echo_path_changes = Vec::new();
        for &num_render_channels in &[1usize, 2, 4] {
            for &num_capture_channels in &[1usize, 2, 4] {
                let context = produce_debug_text(num_render_channels, num_capture_channels, 64, 20);
                let ratios = run_subtractor_test(
                    num_render_channels,
                    num_capture_channels,
                    5000 * num_render_channels,
                    64,
                    20,
                    20,
                    true,
                    &blocks_with_echo_path_changes,
                );
                for ratio in ratios {
                    assert!(ratio > 0.8, "{context}");
                    assert!((ratio - 1.0).abs() < 0.25, "{context}");
                }
            }
        }
    }
}
