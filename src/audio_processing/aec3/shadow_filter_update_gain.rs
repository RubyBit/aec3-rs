use crate::api::config::ShadowConfiguration;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;

/// Implements the fixed gain computation for the shadow adaptive filter.
pub struct ShadowFilterUpdateGain {
    current_config: ShadowConfiguration,
    target_config: ShadowConfiguration,
    old_target_config: ShadowConfiguration,
    config_change_duration_blocks: i32,
    one_by_config_change_duration_blocks: f32,
    poor_signal_excitation_counter: usize,
    call_counter: usize,
    config_change_counter: i32,
}

impl ShadowFilterUpdateGain {
    pub fn new(config: ShadowConfiguration, config_change_duration_blocks: usize) -> Self {
        assert!(config_change_duration_blocks > 0);
        let duration = config_change_duration_blocks as i32;
        Self {
            current_config: config.clone(),
            target_config: config.clone(),
            old_target_config: config,
            config_change_duration_blocks: duration,
            one_by_config_change_duration_blocks: 1.0 / duration as f32,
            poor_signal_excitation_counter: 0,
            call_counter: 0,
            config_change_counter: 0,
        }
    }

    pub fn handle_echo_path_change(&mut self) {
        self.poor_signal_excitation_counter = 0;
        self.call_counter = 0;
    }

    pub fn set_config(&mut self, config: ShadowConfiguration, immediate_effect: bool) {
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

    pub fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        e_shadow: &FftData,
        size_partitions: usize,
        saturated_capture_signal: bool,
        g: &mut FftData,
    ) {
        self.call_counter += 1;
        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_signal_excitation_counter = 0;
        }

        self.poor_signal_excitation_counter += 1;
        if self.poor_signal_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            g.re.fill(0.0);
            g.im.fill(0.0);
            return;
        }

        let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if render_power[k] > self.current_config.noise_gate {
                mu[k] = self.current_config.rate / render_power[k];
            } else {
                mu[k] = 0.0;
            }
        }
        render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            g.re[k] = mu[k] * e_shadow.re[k];
            g.im[k] = mu[k] * e_shadow.im[k];
        }
    }

    fn update_current_config(&mut self) {
        if self.config_change_counter > 0 {
            self.config_change_counter -= 1;
            if self.config_change_counter > 0 {
                let change_factor =
                    self.config_change_counter as f32 * self.one_by_config_change_duration_blocks;
                let average =
                    |from: f32, to: f32| from * change_factor + to * (1.0 - change_factor);
                self.current_config.rate =
                    average(self.old_target_config.rate, self.target_config.rate);
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
    use crate::audio_processing::aec3::aec3_common::{
        Aec3Optimization, BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
        detect_optimization, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
    use crate::audio_processing::aec3::fft_data::FftData;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
    use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    struct ShadowFilterUpdateResult {
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

    fn run_filter_update_test(
        num_blocks_to_process: usize,
        delay_samples: usize,
        num_render_channels: usize,
        filter_length_blocks: usize,
        blocks_with_saturation: &[usize],
    ) -> ShadowFilterUpdateResult {
        let mut config = EchoCanceller3Config::default();
        config.filter.main.length_blocks = filter_length_blocks;
        config.filter.shadow.length_blocks = filter_length_blocks;
        config.delay.default_delay = 1;

        let optimization = detect_optimization();
        let data_dumper = ApmDataDumper::new(42);
        let mut shadow_filter = AdaptiveFirFilter::new(
            config.filter.shadow.length_blocks,
            config.filter.shadow.length_blocks,
            config.filter.config_change_duration_blocks,
            num_render_channels,
            optimization,
            data_dumper,
        );
        let mut shadow_gain = ShadowFilterUpdateGain::new(
            config.filter.shadow.clone(),
            config.filter.config_change_duration_blocks,
        );
        let fft = Aec3Fft::new();

        const SAMPLE_RATE_HZ: i32 = 48_000;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut render_delay_buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);
        let mut render_signal_analyzer = RenderSignalAnalyzer::new(&config);
        let mut rng = Random::new(42);
        let mut x = make_render_block(num_bands, num_render_channels);
        let mut y = [0.0f32; BLOCK_SIZE];
        let mut s = [0.0f32; FFT_LENGTH];
        let mut e_shadow = [0.0f32; BLOCK_SIZE];
        let mut e_shadow_fft = FftData::default();
        let mut g = FftData::default();
        let mut s_freq = FftData::default();
        let mut render_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut delay_buffer = DelayBuffer::<f32>::new(delay_samples);
        let k_scale = 1.0f32 / FFT_LENGTH_BY_2 as f32;

        for block_idx in 0..num_blocks_to_process {
            let saturation = blocks_with_saturation.contains(&block_idx);

            for band in 0..num_bands {
                for channel in 0..num_render_channels {
                    randomize_sample_vector(&mut rng, &mut x[band][channel]);
                }
            }

            delay_buffer.delay(&x[0][0], &mut y);
            render_delay_buffer.insert(&x);
            if block_idx == 0 {
                render_delay_buffer.reset();
            }
            render_delay_buffer.prepare_capture_processing();

            let render_buffer_view = render_delay_buffer.render_buffer();
            let delay_partitions = Some(delay_samples / BLOCK_SIZE);
            render_signal_analyzer.update(&render_buffer_view, delay_partitions);

            shadow_filter.filter(&render_buffer_view, &mut s_freq);
            fft.ifft(&s_freq, &mut s);
            for i in 0..BLOCK_SIZE {
                let value = y[i] - s[i + FFT_LENGTH_BY_2] * k_scale;
                e_shadow[i] = value.clamp(-32_768.0, 32_767.0);
            }
            fft.zero_padded_fft(&e_shadow, Window::Rectangular, &mut e_shadow_fft);

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
        }

        ShadowFilterUpdateResult {
            e_last_block: e_shadow,
            y_last_block: y,
            g_last_block: g,
        }
    }

    fn block_energy(block: &[f32; BLOCK_SIZE]) -> f32 {
        block.iter().map(|sample| sample * sample).sum()
    }

    #[test]
    fn gain_causes_filter_to_converge() {
        let blocks_with_saturation = Vec::new();
        for &num_render_channels in &[1usize, 2, 8] {
            for &filter_length_blocks in &[12usize, 20, 30] {
                for &delay_samples in &[0usize, 64, 150, 200, 301] {
                    let result = run_filter_update_test(
                        5000,
                        delay_samples,
                        num_render_channels,
                        filter_length_blocks,
                        &blocks_with_saturation,
                    );
                    let e_energy = block_energy(&result.e_last_block);
                    let y_energy = block_energy(&result.y_last_block);
                    if filter_length_blocks == 12 {
                        assert!(
                            1000.0 * e_energy < y_energy,
                            "delay={}, length={}, channels={}",
                            delay_samples,
                            filter_length_blocks,
                            num_render_channels
                        );
                    } else {
                        assert!(
                            e_energy < y_energy,
                            "delay={}, length={}, channels={}",
                            delay_samples,
                            filter_length_blocks,
                            num_render_channels
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn decreasing_gain() {
        for &num_render_channels in &[1usize, 2, 4] {
            for &filter_length_blocks in &[12usize, 20, 30] {
                let blocks_with_saturation = Vec::new();
                let a = run_filter_update_test(
                    100,
                    65,
                    num_render_channels,
                    filter_length_blocks,
                    &blocks_with_saturation,
                );
                let b = run_filter_update_test(
                    200,
                    65,
                    num_render_channels,
                    filter_length_blocks,
                    &blocks_with_saturation,
                );
                let c = run_filter_update_test(
                    300,
                    65,
                    num_render_channels,
                    filter_length_blocks,
                    &blocks_with_saturation,
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
                assert!(
                    sum_a > sum_b && sum_b > sum_c,
                    "channels={}, length={}",
                    num_render_channels,
                    filter_length_blocks
                );
            }
        }
    }

    #[test]
    fn saturation_behavior() {
        let mut blocks_with_saturation = Vec::new();
        for idx in 99..200 {
            blocks_with_saturation.push(idx);
        }

        for &num_render_channels in &[1usize, 2, 8] {
            for &filter_length_blocks in &[12usize, 20, 30] {
                let result = run_filter_update_test(
                    100,
                    65,
                    num_render_channels,
                    filter_length_blocks,
                    &blocks_with_saturation,
                );
                assert!(result.g_last_block.re.iter().all(|&v| v == 0.0));
                assert!(result.g_last_block.im.iter().all(|&v| v == 0.0));
            }
        }
    }
}
