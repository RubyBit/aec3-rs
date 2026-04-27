use crate::api::config::EchoCanceller3Config;
#[cfg(test)]
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::fullband_erle_estimator::FullBandErleEstimator;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::signal_dependent_erle_estimator::SignalDependentErleEstimator;
use crate::audio_processing::aec3::subband_erle_estimator::SubbandErleEstimator;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

pub struct ErleEstimator {
    startup_phase_length_blocks: usize,
    fullband_erle_estimator: FullBandErleEstimator,
    subband_erle_estimator: SubbandErleEstimator,
    signal_dependent_erle_estimator: Option<SignalDependentErleEstimator>,
    blocks_since_reset: usize,
}

impl ErleEstimator {
    pub fn new(
        startup_phase_length_blocks: usize,
        config: &EchoCanceller3Config,
        num_capture_channels: usize,
    ) -> Self {
        let fullband_erle_estimator =
            FullBandErleEstimator::new(&config.erle, num_capture_channels);
        let subband_erle_estimator = SubbandErleEstimator::new(config, num_capture_channels);
        let signal_dependent_erle_estimator = if config.erle.num_sections > 1 {
            Some(SignalDependentErleEstimator::new(
                config,
                num_capture_channels,
            ))
        } else {
            None
        };
        let mut instance = Self {
            startup_phase_length_blocks,
            fullband_erle_estimator,
            subband_erle_estimator,
            signal_dependent_erle_estimator,
            blocks_since_reset: 0,
        };
        instance.reset(true);
        instance
    }

    pub fn reset(&mut self, delay_change: bool) {
        self.fullband_erle_estimator.reset();
        self.subband_erle_estimator.reset();
        if let Some(signal_dependent) = &mut self.signal_dependent_erle_estimator {
            signal_dependent.reset();
        }
        if delay_change {
            self.blocks_since_reset = 0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        avg_render_spectrum_with_reverb: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        capture_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        subtractor_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        self.blocks_since_reset += 1;
        if self.blocks_since_reset < self.startup_phase_length_blocks {
            return;
        }

        self.subband_erle_estimator.update(
            avg_render_spectrum_with_reverb,
            capture_spectra,
            subtractor_spectra,
            converged_filters,
        );

        if let Some(signal_dependent) = &mut self.signal_dependent_erle_estimator {
            signal_dependent.update(
                render_buffer,
                filter_frequency_responses,
                avg_render_spectrum_with_reverb,
                capture_spectra,
                subtractor_spectra,
                self.subband_erle_estimator.erle(),
                converged_filters,
            );
        }

        self.fullband_erle_estimator.update(
            avg_render_spectrum_with_reverb,
            capture_spectra,
            subtractor_spectra,
            converged_filters,
        );
    }

    pub fn erle(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if let Some(signal_dependent) = &self.signal_dependent_erle_estimator {
            signal_dependent.erle()
        } else {
            self.subband_erle_estimator.erle()
        }
    }

    pub fn erle_onsets(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        self.subband_erle_estimator.erle_onsets()
    }

    pub fn fullband_erle_log2(&self) -> f32 {
        self.fullband_erle_estimator.fullband_erle_log2()
    }

    pub fn get_inst_linear_quality_estimates(&self) -> &[Option<f32>] {
        self.fullband_erle_estimator.get_linear_quality_estimates()
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        self.fullband_erle_estimator.dump(dumper);
        self.subband_erle_estimator.dump(dumper);
        if let Some(signal_dependent) = &self.signal_dependent_erle_estimator {
            signal_dependent.dump(dumper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, num_bands_for_rate};
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;

    const K_SAMPLE_RATE_HZ: i32 = 48_000;
    const K_TRUE_ERLE: f32 = 10.0;
    const K_TRUE_ERLE_ONSETS: f32 = 1.0;
    const K_ECHO_PATH_GAIN: f32 = 3.0;

    fn form_farend_time_frame(x: &mut [Vec<Vec<f32>>]) {
        const FRAME: [f32; BLOCK_SIZE] = [
            7459.88, 17209.6, 17383.0, 20768.9, 16816.7, 18386.3, 4492.83, 9675.85, 6665.52,
            14808.6, 9342.3, 7483.28, 19261.7, 4145.98, 1622.18, 13475.2, 7166.32, 6856.61,
            21937.0, 7263.14, 9569.07, 14919.0, 8413.32, 7551.89, 7848.65, 6011.27, 13080.6,
            15865.2, 12656.0, 17459.6, 4263.93, 4503.03, 9311.79, 21095.8, 12657.9, 13906.6,
            19267.2, 11338.1, 16828.9, 11501.6, 11405.0, 15031.4, 14541.6, 19765.5, 18346.3,
            19350.2, 3157.47, 18095.8, 1743.68, 21328.2, 19727.5, 7295.16, 10332.4, 11055.5,
            20107.4, 14708.4, 12416.2, 16434.0, 2454.69, 9840.8, 6867.23, 1615.75, 6059.9, 8394.19,
        ];
        for band in x.iter_mut() {
            for channel in band.iter_mut() {
                channel.copy_from_slice(&FRAME);
            }
        }
    }

    fn form_farend_frame(
        render_buffer: &RenderBuffer<'_>,
        desired_erle: f32,
        x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        e2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        let spectrum_buffer = render_buffer.spectrum_buffer();
        let idx = spectrum_buffer.write;
        x2.fill(0.0);
        let num_render_channels = spectrum_buffer.buffer[idx].len();
        for channel in 0..num_render_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[k] += spectrum_buffer.buffer[idx][channel][k] / num_render_channels as f32;
            }
        }
        for ch in 0..y2.len() {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                y2[ch][k] = x2[k] * K_ECHO_PATH_GAIN * K_ECHO_PATH_GAIN;
                e2[ch][k] = y2[ch][k] / desired_erle;
            }
        }
    }

    fn form_nearend_frame(
        x: &mut [Vec<Vec<f32>>],
        x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        e2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        for band in x.iter_mut() {
            for channel in band.iter_mut() {
                channel.fill(0.0);
            }
        }
        x2.fill(0.0);
        for ch in 0..y2.len() {
            y2[ch].fill(500.0 * 1_000_000.0);
            e2[ch].fill(y2[ch][0]);
        }
    }

    fn get_filter_freq(
        delay_headroom_samples: usize,
        filter_frequency_response: &mut [Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
    ) {
        let delay_headroom_blocks = delay_headroom_samples / BLOCK_SIZE;
        for capture_ch in 0..filter_frequency_response.len() {
            for block in &mut filter_frequency_response[capture_ch] {
                block.fill(0.0);
            }
            if delay_headroom_blocks < filter_frequency_response[capture_ch].len() {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    filter_frequency_response[capture_ch][delay_headroom_blocks][k] =
                        K_ECHO_PATH_GAIN;
                }
            }
        }
    }

    fn verify_erle_bands(
        erle: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        reference_lf: f32,
        reference_hf: f32,
    ) {
        let low_frequency_limit = FFT_LENGTH_BY_2 / 2;
        for channel in erle {
            channel[..low_frequency_limit]
                .iter()
                .for_each(|&value| assert!((value - reference_lf).abs() < 0.001));
            channel[low_frequency_limit..]
                .iter()
                .for_each(|&value| assert!((value - reference_hf).abs() < 0.001));
        }
    }

    fn fullband_erle(fullband_log2: f32) -> f32 {
        2.0_f32.powf(fullband_log2)
    }

    #[test]
    fn verify_erle_increase_and_hold() {
        let k_num_bands = num_bands_for_rate(K_SAMPLE_RATE_HZ);
        for &num_render_channels in &[1, 2, 4, 8] {
            for &num_capture_channels in &[1, 2, 4] {
                let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let converged_filters = vec![true; num_capture_channels];

                let mut config = EchoCanceller3Config::default();
                config.erle.onset_detection = true;

                let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; k_num_bands];

                let mut filter_frequency_response =
                    vec![
                        vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; config.filter.main.length_blocks];
                        num_capture_channels
                    ];

                let mut render_delay_buffer =
                    RenderDelayBuffer::new(config.clone(), K_SAMPLE_RATE_HZ, num_render_channels);

                get_filter_freq(
                    config.delay.delay_headroom_samples,
                    &mut filter_frequency_response,
                );

                let mut estimator = ErleEstimator::new(0, &config, num_capture_channels);

                form_farend_time_frame(&mut x);
                render_delay_buffer.insert(&x);
                render_delay_buffer.prepare_capture_processing();
                let render_buffer = render_delay_buffer.render_buffer();
                form_farend_frame(&render_buffer, K_TRUE_ERLE, &mut x2, &mut e2, &mut y2);

                for _ in 0..200 {
                    render_delay_buffer.insert(&x);
                    render_delay_buffer.prepare_capture_processing();
                    let render_buffer = render_delay_buffer.render_buffer();
                    estimator.update(
                        &render_buffer,
                        &filter_frequency_response,
                        &x2,
                        &y2,
                        &e2,
                        &converged_filters,
                    );
                }

                verify_erle_bands(estimator.erle(), config.erle.max_l, config.erle.max_h);
                assert!(fullband_erle(estimator.fullband_erle_log2()) > config.erle.max_l);

                form_nearend_frame(&mut x, &mut x2, &mut e2, &mut y2);
                for _ in 0..50 {
                    render_delay_buffer.insert(&x);
                    render_delay_buffer.prepare_capture_processing();
                    let render_buffer = render_delay_buffer.render_buffer();
                    estimator.update(
                        &render_buffer,
                        &filter_frequency_response,
                        &x2,
                        &y2,
                        &e2,
                        &converged_filters,
                    );
                }
                verify_erle_bands(estimator.erle(), config.erle.max_l, config.erle.max_h);
                assert!(fullband_erle(estimator.fullband_erle_log2()) > config.erle.max_l);
            }
        }
    }

    #[test]
    fn verify_erle_tracking_on_onsets() {
        let k_num_bands = num_bands_for_rate(K_SAMPLE_RATE_HZ);
        for &num_render_channels in &[1, 2, 4, 8] {
            for &num_capture_channels in &[1, 2, 4] {
                let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let converged_filters = vec![true; num_capture_channels];

                let mut config = EchoCanceller3Config::default();
                config.erle.onset_detection = true;

                let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; k_num_bands];

                let mut filter_frequency_response =
                    vec![
                        vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; config.filter.main.length_blocks];
                        num_capture_channels
                    ];

                let mut render_delay_buffer =
                    RenderDelayBuffer::new(config.clone(), K_SAMPLE_RATE_HZ, num_render_channels);

                get_filter_freq(
                    config.delay.delay_headroom_samples,
                    &mut filter_frequency_response,
                );

                let mut estimator = ErleEstimator::new(0, &config, num_capture_channels);

                form_farend_time_frame(&mut x);
                render_delay_buffer.insert(&x);
                render_delay_buffer.prepare_capture_processing();

                for _ in 0..20 {
                    let render_buffer = render_delay_buffer.render_buffer();
                    form_farend_frame(
                        &render_buffer,
                        K_TRUE_ERLE_ONSETS,
                        &mut x2,
                        &mut e2,
                        &mut y2,
                    );
                    for _ in 0..10 {
                        render_delay_buffer.insert(&x);
                        render_delay_buffer.prepare_capture_processing();
                        let render_buffer = render_delay_buffer.render_buffer();
                        estimator.update(
                            &render_buffer,
                            &filter_frequency_response,
                            &x2,
                            &y2,
                            &e2,
                            &converged_filters,
                        );
                    }
                    let render_buffer = render_delay_buffer.render_buffer();
                    form_farend_frame(&render_buffer, K_TRUE_ERLE, &mut x2, &mut e2, &mut y2);
                    for _ in 0..200 {
                        render_delay_buffer.insert(&x);
                        render_delay_buffer.prepare_capture_processing();
                        let render_buffer = render_delay_buffer.render_buffer();
                        estimator.update(
                            &render_buffer,
                            &filter_frequency_response,
                            &x2,
                            &y2,
                            &e2,
                            &converged_filters,
                        );
                    }
                    form_nearend_frame(&mut x, &mut x2, &mut e2, &mut y2);
                    for _ in 0..300 {
                        render_delay_buffer.insert(&x);
                        render_delay_buffer.prepare_capture_processing();
                        let render_buffer = render_delay_buffer.render_buffer();
                        estimator.update(
                            &render_buffer,
                            &filter_frequency_response,
                            &x2,
                            &y2,
                            &e2,
                            &converged_filters,
                        );
                    }
                }

                verify_erle_bands(estimator.erle_onsets(), config.erle.min, config.erle.min);

                form_nearend_frame(&mut x, &mut x2, &mut e2, &mut y2);
                let render_buffer = render_delay_buffer.render_buffer();
                for _ in 0..1000 {
                    estimator.update(
                        &render_buffer,
                        &filter_frequency_response,
                        &x2,
                        &y2,
                        &e2,
                        &converged_filters,
                    );
                }
                verify_erle_bands(estimator.erle(), config.erle.min, config.erle.min);
                assert!(fullband_erle(estimator.fullband_erle_log2()) >= config.erle.min);
            }
        }
    }
}
