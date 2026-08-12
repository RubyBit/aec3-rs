use crate::api::config::{EchoCanceller3Config, HighFrequencySuppression};
use crate::audio_processing::aec3::aec_state::AecState;
use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_MINUS_1, FFT_LENGTH_BY_2_PLUS_1,
};
use crate::audio_processing::aec3::dominant_nearend_detector::DominantNearendDetector;
use crate::audio_processing::aec3::moving_average::MovingAverage;
use crate::audio_processing::aec3::nearend_detector::NearendDetector;
use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
use crate::audio_processing::aec3::subband_nearend_detector::SubbandNearendDetector;
use crate::audio_processing::aec3::vector_math::VectorMath;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

pub struct SuppressionGain {
    #[allow(dead_code)]
    data_dumper: ApmDataDumper,
    #[allow(dead_code)]
    optimization: Aec3Optimization,
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    state_change_duration_blocks: i32,
    last_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    last_nearend: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    last_echo: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    low_render_detector: LowNoiseRenderDetector,
    initial_state: bool,
    initial_state_change_counter: i32,
    nearend_smoothers: Vec<MovingAverage>,
    nearend_params: GainParameters,
    normal_params: GainParameters,
    nearend_detector: Box<dyn NearendDetector>,
}

impl SuppressionGain {
    pub fn new(
        config: EchoCanceller3Config,
        optimization: Aec3Optimization,
        _sample_rate_hz: i32,
        num_capture_channels: usize,
    ) -> Self {
        assert!(num_capture_channels > 0);
        let data_dumper = ApmDataDumper::new_unique();
        let state_change_duration_blocks = config.filter.config_change_duration_blocks as i32;
        assert!(state_change_duration_blocks > 0);
        let nearend_smoothers = (0..num_capture_channels)
            .map(|_| {
                MovingAverage::new(
                    FFT_LENGTH_BY_2_PLUS_1,
                    config.suppressor.nearend_average_blocks,
                )
            })
            .collect();
        let nearend_params = GainParameters::new(&config.suppressor.nearend_tuning);
        let normal_params = GainParameters::new(&config.suppressor.normal_tuning);
        let detector: Box<dyn NearendDetector> = if config.suppressor.use_subband_nearend_detection
        {
            Box::new(SubbandNearendDetector::new(
                &config.suppressor.subband_nearend_detection,
                num_capture_channels,
            ))
        } else {
            Box::new(DominantNearendDetector::new(
                &config.suppressor.dominant_nearend_detection,
                num_capture_channels,
            ))
        };

        Self {
            data_dumper,
            optimization,
            config,
            num_capture_channels,
            state_change_duration_blocks,
            last_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            last_nearend: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            last_echo: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            low_render_detector: LowNoiseRenderDetector::default(),
            initial_state: true,
            initial_state_change_counter: 0,
            nearend_smoothers,
            nearend_params,
            normal_params,
            nearend_detector: detector,
        }
    }

    pub fn set_initial_state(&mut self, state: bool) {
        self.initial_state = state;
        if state {
            self.initial_state_change_counter = self.state_change_duration_blocks;
        } else {
            self.initial_state_change_counter = 0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_gain(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        render_signal_analyzer: &RenderSignalAnalyzer,
        aec_state: &AecState,
        render: &[Vec<Vec<f32>>],
        clock_drift: bool,
        high_bands_gain: Option<&mut f32>,
        low_band_gain: Option<&mut [f32; FFT_LENGTH_BY_2_PLUS_1]>,
    ) {
        let high_bands_gain = high_bands_gain.expect("high_bands_gain must be provided");
        let low_band_gain = low_band_gain.expect("low_band_gain must be provided");
        assert_eq!(nearend_spectrum.len(), self.num_capture_channels);
        assert_eq!(echo_spectrum.len(), self.num_capture_channels);
        assert_eq!(residual_echo_spectrum.len(), self.num_capture_channels);
        assert_eq!(comfort_noise_spectrum.len(), self.num_capture_channels);
        assert!(!render.is_empty());

        self.nearend_detector.update(
            nearend_spectrum,
            residual_echo_spectrum,
            comfort_noise_spectrum,
            self.initial_state,
        );

        let low_noise_render = self.low_render_detector.detect(render);
        self.lower_band_gain(
            low_noise_render,
            aec_state,
            nearend_spectrum,
            residual_echo_spectrum,
            comfort_noise_spectrum,
            clock_drift,
            low_band_gain,
        );

        let narrow_peak_band = render_signal_analyzer.narrow_peak_band();
        *high_bands_gain = self.upper_bands_gain(
            echo_spectrum,
            comfort_noise_spectrum,
            narrow_peak_band,
            aec_state.saturated_echo(),
            render,
            low_band_gain,
        );
    }

    fn upper_bands_gain(
        &self,
        echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        narrow_peak_band: Option<usize>,
        saturated_echo: bool,
        render: &[Vec<Vec<f32>>],
        low_band_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) -> f32 {
        if render.len() == 1 {
            return 1.0;
        }

        if let Some(band) = narrow_peak_band {
            if band > FFT_LENGTH_BY_2_PLUS_1 - 10 {
                return 0.001;
            }
        }

        let low_band_gain_limit = FFT_LENGTH_BY_2 / 2;
        let mut gain_below_8khz = low_band_gain[low_band_gain_limit..]
            .iter()
            .fold(f32::MAX, |acc, &g| acc.min(g));
        if !gain_below_8khz.is_finite() {
            gain_below_8khz = 1.0;
        }
        gain_below_8khz = gain_below_8khz.min(1.0);

        if saturated_echo {
            return gain_below_8khz.min(0.001);
        }

        let mut low_band_energy = 0.0f32;
        let mut high_band_energy = 0.0f32;
        for (band_index, band) in render.iter().enumerate() {
            for channel in band {
                let energy: f32 = channel.iter().map(|s| s * s).sum();
                if band_index == 0 {
                    low_band_energy = low_band_energy.max(energy);
                } else {
                    high_band_energy = high_band_energy.max(energy);
                }
            }
        }

        let activation_threshold = BLOCK_SIZE as f32
            * self
                .config
                .suppressor
                .high_bands_suppression
                .anti_howling_activation_threshold;
        let anti_howling_gain = if high_band_energy < low_band_energy.max(activation_threshold) {
            1.0
        } else {
            (self
                .config
                .suppressor
                .high_bands_suppression
                .anti_howling_gain
                * (low_band_energy / high_band_energy).sqrt())
            .min(1.0)
        };

        let mut gain_bound = 1.0f32;
        if !self.nearend_detector.is_nearend_state() {
            let cfg = &self.config.suppressor.high_bands_suppression;
            let low_frequency_energy =
                |spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1]| -> f32 { spectrum[1..16].iter().sum() };
            for ch in 0..self.num_capture_channels {
                let echo_sum = low_frequency_energy(&echo_spectrum[ch]);
                let noise_sum = low_frequency_energy(&comfort_noise_spectrum[ch]);
                if echo_sum > cfg.enr_threshold * noise_sum {
                    gain_bound = cfg.max_gain_during_echo;
                    break;
                }
            }
        }

        gain_below_8khz
            .min(anti_howling_gain)
            .min(gain_bound)
            .clamp(0.0, 1.0)
    }

    fn lower_band_gain(
        &mut self,
        low_noise_render: bool,
        aec_state: &AecState,
        suppressor_input: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        clock_drift: bool,
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        gain.fill(1.0);
        let saturated_echo = aec_state.saturated_echo();
        let mut max_gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        self.get_max_gain(&mut max_gain);

        for ch in 0..self.num_capture_channels {
            let mut nearend = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.nearend_smoothers[ch].average(&suppressor_input[ch], &mut nearend);

            let mut weighted_residual_echo = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            weight_echo_for_audibility(
                &self.config,
                &residual_echo[ch],
                &mut weighted_residual_echo,
            );

            let mut min_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.get_min_gain(
                &weighted_residual_echo,
                &self.last_nearend[ch],
                &self.last_echo[ch],
                low_noise_render,
                saturated_echo,
                &mut min_gain,
            );

            let mut channel_gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.gain_to_no_audible_echo(
                &nearend,
                &weighted_residual_echo,
                &comfort_noise[0],
                &mut channel_gain,
            );

            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                let clamped = channel_gain[k].min(max_gain[k]).max(min_gain[k]);
                gain[k] = gain[k].min(clamped);
            }

            self.last_nearend[ch] = nearend;
            self.last_echo[ch] = weighted_residual_echo;
        }

        limit_low_frequency_gains(gain);
        // Use conservative high-frequency gains during clock-drift or when not
        // in dominant nearend.
        let conservative_hf_suppression = self.config.suppressor.conservative_hf_suppression;
        if !self.nearend_detector.is_nearend_state() || clock_drift || conservative_hf_suppression {
            limit_high_frequency_gains(
                &self.config.suppressor.high_frequency_suppression,
                conservative_hf_suppression,
                gain,
            );
        }

        self.last_gain.copy_from_slice(gain);
        sqrt_in_place(self.optimization, gain);
    }

    fn gain_to_no_audible_echo(
        &self,
        nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        masker: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        let params = if self.nearend_detector.is_nearend_state() {
            &self.nearend_params
        } else {
            &self.normal_params
        };
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let enr = echo[k] / (nearend[k] + 1.0);
            let emr = echo[k] / (masker[k] + 1.0);
            let mut g = 1.0f32;
            if enr > params.enr_transparent[k] && emr > params.emr_transparent[k] {
                g = (params.enr_suppress[k] - enr)
                    / (params.enr_suppress[k] - params.enr_transparent[k]);
                g = g.max(params.emr_transparent[k] / emr);
            }
            gain[k] = g.clamp(0.0, 1.0);
        }
    }

    fn get_min_gain(
        &self,
        weighted_residual_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        low_noise_render: bool,
        saturated_echo: bool,
        min_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        if saturated_echo {
            min_gain.fill(0.0);
            return;
        }

        let min_echo_power = if low_noise_render {
            self.config.echo_audibility.low_render_limit
        } else {
            self.config.echo_audibility.normal_render_limit
        };

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let value = if weighted_residual_echo[k] > 0.0 {
                min_echo_power / weighted_residual_echo[k]
            } else {
                1.0
            };
            min_gain[k] = value.min(1.0);
        }

        let dec = if self.nearend_detector.is_nearend_state() {
            self.nearend_params.max_dec_factor_lf
        } else {
            self.normal_params.max_dec_factor_lf
        };
        for k in 0..6 {
            if last_nearend[k] > last_echo[k] {
                min_gain[k] = min_gain[k].max(self.last_gain[k] * dec).min(1.0);
            }
        }
    }

    fn get_max_gain(&self, max_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        let inc = if self.nearend_detector.is_nearend_state() {
            self.nearend_params.max_inc_factor
        } else {
            self.normal_params.max_inc_factor
        };
        let floor = self.config.suppressor.floor_first_increase;
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            max_gain[k] = (self.last_gain[k] * inc).max(floor).min(1.0);
        }
    }
}

fn sqrt_in_place(optimization: Aec3Optimization, values: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    for value in values.iter_mut() {
        *value = value.max(0.0);
    }
    VectorMath::new(optimization).sqrt(values);
}

fn weight_echo_for_audibility(
    config: &EchoCanceller3Config,
    echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    weighted_echo: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let floor_power = config.echo_audibility.floor_power;
    weighted_echo.copy_from_slice(echo);

    let mut apply = |range: std::ops::Range<usize>, threshold_multiplier: f32| {
        let threshold = floor_power * threshold_multiplier;
        let denom = (threshold - floor_power).max(1e-12);
        let normalizer = 1.0 / denom;
        for k in range.clone() {
            if echo[k] < threshold {
                let tmp = (threshold - echo[k]) * normalizer;
                let factor = (1.0 - tmp * tmp).max(0.0);
                weighted_echo[k] = echo[k] * factor;
            } else {
                weighted_echo[k] = echo[k];
            }
        }
    };

    apply(0..3, config.echo_audibility.audibility_threshold_lf);
    apply(3..7, config.echo_audibility.audibility_threshold_mf);
    apply(
        7..FFT_LENGTH_BY_2_PLUS_1,
        config.echo_audibility.audibility_threshold_hf,
    );
}

/// Limits the low frequency gains to avoid the impact of the high-pass filter
/// on the lower-frequency gain influencing the overall achieved gain.
fn limit_low_frequency_gains(gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    gain[0] = gain[1].min(gain[2]);
    gain[1] = gain[0];
}

/// Limits the high frequency gains to avoid echo leakage due to an imperfect
/// filter.
fn limit_high_frequency_gains(
    high_frequency_suppression: &HighFrequencySuppression,
    conservative_hf_suppression: bool,
    gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let limiting_gain_band = high_frequency_suppression.limiting_gain_band;
    let bands_in_limiting_gain = high_frequency_suppression.bands_in_limiting_gain;
    if bands_in_limiting_gain > 0 {
        debug_assert!(limiting_gain_band + bands_in_limiting_gain <= FFT_LENGTH_BY_2_PLUS_1);
        let mut min_upper_gain = 1.0f32;
        for &g in &gain[limiting_gain_band..limiting_gain_band + bands_in_limiting_gain] {
            min_upper_gain = min_upper_gain.min(g);
        }
        for g in gain[limiting_gain_band + 1..].iter_mut() {
            *g = (*g).min(min_upper_gain);
        }
    }
    gain[FFT_LENGTH_BY_2] = gain[FFT_LENGTH_BY_2_MINUS_1];

    if conservative_hf_suppression {
        // Limits the gain in the frequencies for which the adaptive filter has
        // not converged.
        const UPPER_ACCURATE_BAND_PLUS1: usize = 29;
        let one_by_bands = 1.0 / (UPPER_ACCURATE_BAND_PLUS1 - 20) as f32;
        let hf_gain_bound: f32 =
            gain[20..UPPER_ACCURATE_BAND_PLUS1].iter().sum::<f32>() * one_by_bands;
        for g in gain[UPPER_ACCURATE_BAND_PLUS1..].iter_mut() {
            *g = (*g).min(hf_gain_bound);
        }
    }
}

#[derive(Clone)]
struct GainParameters {
    max_inc_factor: f32,
    max_dec_factor_lf: f32,
    enr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
    enr_suppress: [f32; FFT_LENGTH_BY_2_PLUS_1],
    emr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl GainParameters {
    fn new(tuning: &crate::api::config::Tuning) -> Self {
        const LAST_LF_BAND: usize = 5;
        const FIRST_HF_BAND: usize = 8;
        let mut enr_transparent = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut enr_suppress = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut emr_transparent = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let alpha = if k <= LAST_LF_BAND {
                0.0
            } else if k < FIRST_HF_BAND {
                (k - LAST_LF_BAND) as f32 / (FIRST_HF_BAND - LAST_LF_BAND) as f32
            } else {
                1.0
            };
            enr_transparent[k] = (1.0 - alpha) * tuning.mask_lf.enr_transparent
                + alpha * tuning.mask_hf.enr_transparent;
            enr_suppress[k] =
                (1.0 - alpha) * tuning.mask_lf.enr_suppress + alpha * tuning.mask_hf.enr_suppress;
            emr_transparent[k] = (1.0 - alpha) * tuning.mask_lf.emr_transparent
                + alpha * tuning.mask_hf.emr_transparent;
        }
        Self {
            max_inc_factor: tuning.max_inc_factor,
            max_dec_factor_lf: tuning.max_dec_factor_lf,
            enr_transparent,
            enr_suppress,
            emr_transparent,
        }
    }
}

struct LowNoiseRenderDetector {
    average_power: f32,
}

impl Default for LowNoiseRenderDetector {
    fn default() -> Self {
        Self {
            average_power: 32_768.0 * 32_768.0,
        }
    }
}

impl LowNoiseRenderDetector {
    fn detect(&mut self, render: &[Vec<Vec<f32>>]) -> bool {
        if render.is_empty() || render[0].is_empty() {
            return false;
        }
        let mut x2_sum = 0.0f32;
        let mut x2_max = 0.0f32;
        for channel in &render[0] {
            for &sample in channel {
                let x2 = sample * sample;
                x2_sum += x2;
                x2_max = x2_max.max(x2);
            }
        }
        let num_render_channels = render[0].len() as f32;
        if num_render_channels == 0.0 {
            return false;
        }
        x2_sum /= num_render_channels;
        const THRESHOLD: f32 = 50.0 * 50.0 * 64.0;
        let low_noise_render = self.average_power < THRESHOLD && x2_max < 3.0 * self.average_power;
        self.average_power = self.average_power * 0.9 + x2_sum * 0.1;
        low_noise_render
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec_state::AecState;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, NUM_BLOCKS_PER_SECOND, detect_optimization, get_time_domain_length,
        num_bands_for_rate,
    };
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
    use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;

    /// The band-29 bound is part of conservative HF suppression, which is off
    /// by default upstream. Enabling it must not be a no-op, and leaving it off
    /// must leave the accurate bands untouched.
    #[test]
    fn conservative_hf_suppression_gates_the_upper_band_bound() {
        let hf = HighFrequencySuppression {
            limiting_gain_band: 16,
            bands_in_limiting_gain: 1,
        };
        // Bands 20..29 average 0.1; band 40 sits well above that bound.
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for g in gain[20..29].iter_mut() {
            *g = 0.1;
        }

        let mut unbounded = gain;
        limit_high_frequency_gains(&hf, false, &mut unbounded);
        assert!(
            (unbounded[40] - 1.0).abs() < 1e-6,
            "band 40 was bounded with conservative_hf_suppression disabled: {}",
            unbounded[40]
        );

        let mut bounded = gain;
        limit_high_frequency_gains(&hf, true, &mut bounded);
        assert!(
            (bounded[40] - 0.1).abs() < 1e-6,
            "band 40 was not bounded with conservative_hf_suppression enabled: {}",
            bounded[40]
        );
    }

    /// Mirrors upstream `LimitHighFrequencyGains`: bands above
    /// `limiting_gain_band` are capped by the minimum gain within the limiting
    /// window, and `bands_in_limiting_gain == 0` disables the limiting.
    #[test]
    fn high_frequency_limiting_follows_the_configured_window() {
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        gain[16] = 0.25;

        let mut limited = gain;
        limit_high_frequency_gains(
            &HighFrequencySuppression {
                limiting_gain_band: 16,
                bands_in_limiting_gain: 1,
            },
            false,
            &mut limited,
        );
        assert!((limited[16] - 0.25).abs() < 1e-6);
        assert!((limited[17] - 0.25).abs() < 1e-6);
        assert!((limited[15] - 1.0).abs() < 1e-6, "bands below were modified");

        let mut disabled = gain;
        limit_high_frequency_gains(
            &HighFrequencySuppression {
                limiting_gain_band: 16,
                bands_in_limiting_gain: 0,
            },
            false,
            &mut disabled,
        );
        assert!((disabled[17] - 1.0).abs() < 1e-6);
    }

    /// Mirrors upstream `LimitLowFrequencyGains`.
    #[test]
    fn low_frequency_limiting_matches_reference() {
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        gain[1] = 0.5;
        gain[2] = 0.3;
        limit_low_frequency_gains(&mut gain);
        assert!((gain[0] - 0.3).abs() < 1e-6);
        assert!((gain[1] - 0.3).abs() < 1e-6);
        assert!((gain[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn null_output_gains_panics() {
        let config = EchoCanceller3Config::default();
        let mut gain = SuppressionGain::new(config.clone(), detect_optimization(), 16_000, 1);
        let spectrum = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        let render: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; BLOCK_SIZE]]];
        let analyzer = RenderSignalAnalyzer::new(&config);
        let aec_state = AecState::new(config, 1);
        gain.get_gain(
            &spectrum, &spectrum, &spectrum, &spectrum, &analyzer, &aec_state, &render, false,
            None, None,
        );
    }

    #[test]
    fn basic_gain_computation_matches_reference() {
        const NUM_RENDER_CHANNELS: usize = 1;
        const NUM_CAPTURE_CHANNELS: usize = 2;
        const SAMPLE_RATE_HZ: i32 = 16_000;
        let config = EchoCanceller3Config::default();
        let mut suppression_gain = SuppressionGain::new(
            config.clone(),
            detect_optimization(),
            SAMPLE_RATE_HZ,
            NUM_CAPTURE_CHANNELS,
        );
        let analyzer = RenderSignalAnalyzer::new(&config);
        let mut high_bands_gain = 0.0f32;
        let mut low_band_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let render = vec![vec![vec![0.0f32; BLOCK_SIZE]; NUM_RENDER_CHANNELS]; num_bands];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut s2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut r2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut n2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut subtractor_output = vec![SubtractorOutput::new(); NUM_CAPTURE_CHANNELS];
        let mut aec_state = AecState::new(config.clone(), NUM_CAPTURE_CHANNELS);
        let mut render_delay_buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, NUM_RENDER_CHANNELS);
        render_delay_buffer.insert(&render);
        render_delay_buffer.prepare_capture_processing();
        let filter_length_blocks = config.filter.main.length_blocks;
        let mut filter_frequency_response =
            vec![
                vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; filter_length_blocks];
                NUM_CAPTURE_CHANNELS
            ];
        for channel in &mut filter_frequency_response {
            for block in channel {
                block.fill(0.01);
            }
        }
        let impulse_response_len = get_time_domain_length(filter_length_blocks);
        let impulse_response = vec![vec![0.0f32; impulse_response_len]; NUM_CAPTURE_CHANNELS];
        let delay_estimate: Option<crate::audio_processing::aec3::delay_estimate::DelayEstimate> =
            None;

        for ch in 0..NUM_CAPTURE_CHANNELS {
            e2[ch].fill(10.0);
            y2[ch].fill(10.0);
            r2[ch].fill(0.1);
            n2[ch].fill(100.0);
            subtractor_output[ch].reset();
        }

        for _ in 0..=NUM_BLOCKS_PER_SECOND / 5 + 1 {
            let render_buffer = render_delay_buffer.render_buffer();
            aec_state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2,
                &y2,
                &subtractor_output,
            );
        }

        for _ in 0..100 {
            let render_buffer = render_delay_buffer.render_buffer();
            aec_state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2,
                &y2,
                &subtractor_output,
            );
            suppression_gain.get_gain(
                &e2,
                &s2,
                &r2,
                &n2,
                &analyzer,
                &aec_state,
                &render,
                false,
                Some(&mut high_bands_gain),
                Some(&mut low_band_gain),
            );
        }
        for &g in &low_band_gain {
            assert!((g - 1.0).abs() < 1e-3);
        }

        for ch in 0..NUM_CAPTURE_CHANNELS {
            e2[ch].fill(100.0);
            y2[ch].fill(100.0);
            r2[ch].fill(0.1);
            s2[ch].fill(0.1);
            n2[ch].fill(0.0);
        }

        for _ in 0..100 {
            let render_buffer = render_delay_buffer.render_buffer();
            aec_state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2,
                &y2,
                &subtractor_output,
            );
            suppression_gain.get_gain(
                &e2,
                &s2,
                &r2,
                &n2,
                &analyzer,
                &aec_state,
                &render,
                false,
                Some(&mut high_bands_gain),
                Some(&mut low_band_gain),
            );
        }
        for &g in &low_band_gain {
            assert!((g - 1.0).abs() < 1e-3);
        }

        e2[1].fill(1_000_000_000.0);
        r2[1].fill(10_000_000_000_000.0);

        for _ in 0..10 {
            suppression_gain.get_gain(
                &e2,
                &s2,
                &r2,
                &n2,
                &analyzer,
                &aec_state,
                &render,
                false,
                Some(&mut high_bands_gain),
                Some(&mut low_band_gain),
            );
        }

        for &g in &low_band_gain {
            assert!(g < 0.01);
        }
    }
}
