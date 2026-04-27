//! Noise suppressor top-level pipeline.

use super::fast_math::sqrt_fast_approximation;
use super::noise_estimator::NoiseEstimator;
use super::ns_common::{FFT_SIZE, FFT_SIZE_BY_2_PLUS_1, NS_FRAME_SIZE, OVERLAP_SIZE};
use super::ns_config::NsConfig;
use super::ns_fft::NrFft;
use super::speech_probability_estimator::SpeechProbabilityEstimator;
use super::suppression_params::SuppressionParams;
use super::wiener_filter::WienerFilter;
use crate::audio_processing::audio_buffer::AudioBuffer;

fn num_bands_for_rate(sample_rate_hz: usize) -> usize {
    debug_assert!(matches!(sample_rate_hz, 16_000 | 32_000 | 48_000));
    sample_rate_hz / 16_000
}

const BLOCKS_160W_256_FIRST_HALF: [f32; 96] = [
    0.00000000, 0.01636173, 0.03271908, 0.04906767, 0.06540313, 0.08172107, 0.09801714, 0.11428696,
    0.13052619, 0.14673047, 0.16289547, 0.17901686, 0.19509032, 0.21111155, 0.22707626, 0.24298018,
    0.25881905, 0.27458862, 0.29028468, 0.30590302, 0.32143947, 0.33688985, 0.35225005, 0.36751594,
    0.38268343, 0.39774847, 0.41270703, 0.42755509, 0.44228869, 0.45690388, 0.47139674, 0.48576339,
    0.50000000, 0.51410274, 0.52806785, 0.54189158, 0.55557023, 0.56910015, 0.58247770, 0.59569930,
    0.60876143, 0.62166057, 0.63439328, 0.64695615, 0.65934582, 0.67155895, 0.68359230, 0.69544264,
    0.70710678, 0.71858162, 0.72986407, 0.74095113, 0.75183981, 0.76252720, 0.77301045, 0.78328675,
    0.79335334, 0.80320753, 0.81284668, 0.82226822, 0.83146961, 0.84044840, 0.84920218, 0.85772861,
    0.86602540, 0.87409034, 0.88192126, 0.88951608, 0.89687274, 0.90398929, 0.91086382, 0.91749450,
    0.92387953, 0.93001722, 0.93590593, 0.94154407, 0.94693013, 0.95206268, 0.95694034, 0.96156180,
    0.96592583, 0.97003125, 0.97387698, 0.97746197, 0.98078528, 0.98384601, 0.98664333, 0.98917651,
    0.99144486, 0.99344778, 0.99518473, 0.99665524, 0.99785892, 0.99879546, 0.99946459, 0.99986614,
];

fn apply_filter_bank_window(x: &mut [f32; FFT_SIZE]) {
    for (i, &w) in BLOCKS_160W_256_FIRST_HALF.iter().enumerate() {
        x[i] *= w;
    }
    for (i, k) in (161..FFT_SIZE).zip((1..=95).rev()) {
        x[i] *= BLOCKS_160W_256_FIRST_HALF[k];
    }
}

fn form_extended_frame(
    frame: &[f32; NS_FRAME_SIZE],
    old_data: &mut [f32; OVERLAP_SIZE],
    extended_frame: &mut [f32; FFT_SIZE],
) {
    let old_len = old_data.len();
    extended_frame[..old_len].copy_from_slice(old_data);
    extended_frame[old_len..old_len + frame.len()].copy_from_slice(frame);
    old_data.copy_from_slice(&extended_frame[FFT_SIZE - old_len..]);
}

fn overlap_and_add(
    extended_frame: &[f32; FFT_SIZE],
    overlap_memory: &mut [f32; OVERLAP_SIZE],
    output_frame: &mut [f32; NS_FRAME_SIZE],
) {
    for i in 0..OVERLAP_SIZE {
        output_frame[i] = overlap_memory[i] + extended_frame[i];
    }
    output_frame[OVERLAP_SIZE..].copy_from_slice(&extended_frame[OVERLAP_SIZE..NS_FRAME_SIZE]);
    overlap_memory.copy_from_slice(&extended_frame[NS_FRAME_SIZE..]);
}

fn delay_signal(
    frame: &[f32; NS_FRAME_SIZE],
    delay_buffer: &mut [f32; OVERLAP_SIZE],
    delayed_frame: &mut [f32; NS_FRAME_SIZE],
) {
    const SAMPLES_FROM_FRAME: usize = NS_FRAME_SIZE - OVERLAP_SIZE;
    delayed_frame[..OVERLAP_SIZE].copy_from_slice(delay_buffer);
    delayed_frame[OVERLAP_SIZE..].copy_from_slice(&frame[..SAMPLES_FROM_FRAME]);
    delay_buffer.copy_from_slice(&frame[SAMPLES_FROM_FRAME..]);
}

fn compute_energy_of_extended_frame(x: &[f32; FFT_SIZE]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

fn compute_energy_of_extended_frame_components(
    frame: &[f32; NS_FRAME_SIZE],
    old_data: &[f32; OVERLAP_SIZE],
) -> f32 {
    old_data.iter().map(|v| v * v).sum::<f32>() + frame.iter().map(|v| v * v).sum::<f32>()
}

fn compute_magnitude_spectrum(
    real: &[f32; FFT_SIZE],
    imag: &[f32; FFT_SIZE],
    signal_spectrum: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
) {
    signal_spectrum[0] = real[0].abs() + 1.0;
    signal_spectrum[FFT_SIZE_BY_2_PLUS_1 - 1] = real[FFT_SIZE_BY_2_PLUS_1 - 1].abs() + 1.0;
    for i in 1..(FFT_SIZE_BY_2_PLUS_1 - 1) {
        signal_spectrum[i] = sqrt_fast_approximation(real[i] * real[i] + imag[i] * imag[i]) + 1.0;
    }
}

fn compute_snr(
    filter: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prev_signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prev_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prior_snr: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    post_snr: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
) {
    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        let prev_estimate = prev_signal_spectrum[i] / (prev_noise_spectrum[i] + 0.0001) * filter[i];
        post_snr[i] = if signal_spectrum[i] > noise_spectrum[i] {
            signal_spectrum[i] / (noise_spectrum[i] + 0.0001) - 1.0
        } else {
            0.0
        };
        prior_snr[i] = 0.98 * prev_estimate + 0.02 * post_snr[i];
    }
}

fn compute_upper_bands_gain(
    minimum_attenuating_gain: f32,
    filter: &[f32; FFT_SIZE_BY_2_PLUS_1],
    speech_probability: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prev_analysis_signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
) -> f32 {
    const NUM_AVG_BINS: usize = 32;
    const ONE_BY_NUM_AVG_BINS: f32 = 1.0 / NUM_AVG_BINS as f32;

    let mut avg_prob_speech = 0.0;
    let mut avg_filter_gain = 0.0;
    for i in (FFT_SIZE_BY_2_PLUS_1 - NUM_AVG_BINS - 1)..(FFT_SIZE_BY_2_PLUS_1 - 1) {
        avg_prob_speech += speech_probability[i];
        avg_filter_gain += filter[i];
    }
    avg_prob_speech *= ONE_BY_NUM_AVG_BINS;
    avg_filter_gain *= ONE_BY_NUM_AVG_BINS;

    let sum_analysis_spectrum = prev_analysis_signal_spectrum.iter().sum::<f32>();
    let sum_processing_spectrum = signal_spectrum.iter().sum::<f32>();
    if sum_analysis_spectrum > 0.0 {
        avg_prob_speech *= sum_processing_spectrum / sum_analysis_spectrum;
    }

    let mut gain = 0.5 * (1.0 + (2.0 * avg_prob_speech - 1.0).tanh());
    if avg_prob_speech >= 0.5 {
        gain = 0.25 * gain + 0.75 * avg_filter_gain;
    } else {
        gain = 0.5 * gain + 0.5 * avg_filter_gain;
    }
    gain.clamp(minimum_attenuating_gain, 1.0)
}

struct ChannelState {
    speech_probability_estimator: SpeechProbabilityEstimator,
    wiener_filter: WienerFilter,
    noise_estimator: NoiseEstimator,
    prev_analysis_signal_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    analyze_analysis_memory: [f32; OVERLAP_SIZE],
    process_analysis_memory: [f32; OVERLAP_SIZE],
    process_synthesis_memory: [f32; OVERLAP_SIZE],
    process_delay_memory: Vec<[f32; OVERLAP_SIZE]>,
}

impl ChannelState {
    fn new(suppression_params: SuppressionParams, num_bands: usize) -> Self {
        Self {
            speech_probability_estimator: SpeechProbabilityEstimator::default(),
            wiener_filter: WienerFilter::new(suppression_params),
            noise_estimator: NoiseEstimator::new(suppression_params),
            prev_analysis_signal_spectrum: [1.0; FFT_SIZE_BY_2_PLUS_1],
            analyze_analysis_memory: [0.0; OVERLAP_SIZE],
            process_analysis_memory: [0.0; OVERLAP_SIZE],
            process_synthesis_memory: [0.0; OVERLAP_SIZE],
            process_delay_memory: vec![[0.0; OVERLAP_SIZE]; num_bands.saturating_sub(1)],
        }
    }
}

/// Standalone noise suppressor module.
pub struct NoiseSuppressor {
    num_bands: usize,
    num_channels: usize,
    suppression_params: SuppressionParams,
    num_analyzed_frames: i32,
    fft: NrFft,
    capture_output_used: bool,
    channels: Vec<ChannelState>,
}

impl NoiseSuppressor {
    pub fn new(config: NsConfig, sample_rate_hz: usize, num_channels: usize) -> Self {
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let suppression_params = SuppressionParams::new(config.target_level);
        let channels = (0..num_channels)
            .map(|_| ChannelState::new(suppression_params, num_bands))
            .collect();

        Self {
            num_bands,
            num_channels,
            suppression_params,
            num_analyzed_frames: -1,
            fft: NrFft::default(),
            capture_output_used: true,
            channels,
        }
    }

    pub fn set_capture_output_usage(&mut self, used: bool) {
        self.capture_output_used = used;
    }

    /// Analyze the current capture frame without modifying it.
    pub fn analyze(&mut self, capture: &AudioBuffer) {
        for ch in 0..self.num_channels {
            self.channels[ch].noise_estimator.prepare_analysis();
        }

        let mut zero_frame = true;
        for ch in 0..self.num_channels {
            let src = capture.split_band(ch, 0);
            let mut frame = [0.0f32; NS_FRAME_SIZE];
            frame.copy_from_slice(&src[..NS_FRAME_SIZE]);
            let energy = compute_energy_of_extended_frame_components(
                &frame,
                &self.channels[ch].analyze_analysis_memory,
            );
            if energy > 0.0 {
                zero_frame = false;
                break;
            }
        }
        if zero_frame {
            return;
        }

        self.num_analyzed_frames += 1;
        if self.num_analyzed_frames < 0 {
            self.num_analyzed_frames = 0;
        }

        for ch in 0..self.num_channels {
            let chs = &mut self.channels[ch];
            let src = capture.split_band(ch, 0);

            let mut frame = [0.0f32; NS_FRAME_SIZE];
            frame.copy_from_slice(&src[..NS_FRAME_SIZE]);

            let mut extended_frame = [0.0f32; FFT_SIZE];
            form_extended_frame(
                &frame,
                &mut chs.analyze_analysis_memory,
                &mut extended_frame,
            );
            apply_filter_bank_window(&mut extended_frame);

            let mut real = [0.0f32; FFT_SIZE];
            let mut imag = [0.0f32; FFT_SIZE];
            self.fft.fft(&mut extended_frame, &mut real, &mut imag);

            let mut signal_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
            compute_magnitude_spectrum(&real, &imag, &mut signal_spectrum);

            let signal_energy = (0..FFT_SIZE_BY_2_PLUS_1)
                .map(|i| real[i] * real[i] + imag[i] * imag[i])
                .sum::<f32>()
                / FFT_SIZE_BY_2_PLUS_1 as f32;
            let signal_spectral_sum = signal_spectrum.iter().sum::<f32>();

            chs.noise_estimator.pre_update(
                self.num_analyzed_frames,
                &signal_spectrum,
                signal_spectral_sum,
            );

            let mut prior_snr = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
            let mut post_snr = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
            compute_snr(
                chs.wiener_filter.filter(),
                &chs.prev_analysis_signal_spectrum,
                &signal_spectrum,
                chs.noise_estimator.prev_noise_spectrum(),
                chs.noise_estimator.noise_spectrum(),
                &mut prior_snr,
                &mut post_snr,
            );

            chs.speech_probability_estimator.update(
                self.num_analyzed_frames,
                &prior_snr,
                &post_snr,
                chs.noise_estimator.conservative_noise_spectrum(),
                &signal_spectrum,
                signal_spectral_sum,
                signal_energy,
            );

            chs.noise_estimator.post_update(
                chs.speech_probability_estimator.probability(),
                &signal_spectrum,
            );

            chs.prev_analysis_signal_spectrum
                .copy_from_slice(&signal_spectrum);
        }
    }

    fn aggregate_wiener_filters(&self, filter: &mut [f32; FFT_SIZE_BY_2_PLUS_1]) {
        filter.copy_from_slice(self.channels[0].wiener_filter.filter());
        for ch in 1..self.num_channels {
            let fch = self.channels[ch].wiener_filter.filter();
            for k in 0..FFT_SIZE_BY_2_PLUS_1 {
                filter[k] = filter[k].min(fch[k]);
            }
        }
    }

    /// Apply suppression to the current capture frame.
    pub fn process(&mut self, capture: &mut AudioBuffer) {
        let mut reals = vec![[0.0f32; FFT_SIZE]; self.num_channels];
        let mut imags = vec![[0.0f32; FFT_SIZE]; self.num_channels];
        let mut extended_frames = vec![[0.0f32; FFT_SIZE]; self.num_channels];
        let mut upper_band_gains = vec![0.0f32; self.num_channels];
        let mut energies_before_filtering = vec![0.0f32; self.num_channels];
        let mut gain_adjustments = vec![0.0f32; self.num_channels];

        for ch in 0..self.num_channels {
            let chs = &mut self.channels[ch];

            let src = capture.split_band(ch, 0);
            let mut frame = [0.0f32; NS_FRAME_SIZE];
            frame.copy_from_slice(&src[..NS_FRAME_SIZE]);

            form_extended_frame(
                &frame,
                &mut chs.process_analysis_memory,
                &mut extended_frames[ch],
            );
            apply_filter_bank_window(&mut extended_frames[ch]);

            energies_before_filtering[ch] = compute_energy_of_extended_frame(&extended_frames[ch]);

            self.fft
                .fft(&mut extended_frames[ch], &mut reals[ch], &mut imags[ch]);

            let mut signal_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
            compute_magnitude_spectrum(&reals[ch], &imags[ch], &mut signal_spectrum);

            chs.wiener_filter.update(
                self.num_analyzed_frames,
                chs.noise_estimator.noise_spectrum(),
                chs.noise_estimator.prev_noise_spectrum(),
                chs.noise_estimator.parametric_noise_spectrum(),
                &signal_spectrum,
            );

            if self.num_bands > 1 {
                upper_band_gains[ch] = compute_upper_bands_gain(
                    self.suppression_params.minimum_attenuating_gain,
                    chs.wiener_filter.filter(),
                    chs.speech_probability_estimator.probability(),
                    &chs.prev_analysis_signal_spectrum,
                    &signal_spectrum,
                );
            }
        }

        if !self.capture_output_used {
            return;
        }

        let mut filter = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        if self.num_channels == 1 {
            filter.copy_from_slice(self.channels[0].wiener_filter.filter());
        } else {
            self.aggregate_wiener_filters(&mut filter);
        }

        for ch in 0..self.num_channels {
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                reals[ch][i] *= filter[i];
                imags[ch][i] *= filter[i];
            }
            self.fft
                .ifft(&reals[ch], &imags[ch], &mut extended_frames[ch]);
        }

        for ch in 0..self.num_channels {
            let energy_after_filtering = compute_energy_of_extended_frame(&extended_frames[ch]);
            apply_filter_bank_window(&mut extended_frames[ch]);
            gain_adjustments[ch] = self.channels[ch]
                .wiener_filter
                .compute_overall_scaling_factor(
                    self.num_analyzed_frames,
                    self.channels[ch]
                        .speech_probability_estimator
                        .prior_probability(),
                    energies_before_filtering[ch],
                    energy_after_filtering,
                );
        }

        let gain_adjustment = gain_adjustments
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        for frame in &mut extended_frames {
            for x in frame.iter_mut() {
                *x *= gain_adjustment;
            }
        }

        for ch in 0..self.num_channels {
            let mut out = [0.0f32; NS_FRAME_SIZE];
            overlap_and_add(
                &extended_frames[ch],
                &mut self.channels[ch].process_synthesis_memory,
                &mut out,
            );
            capture.split_band_mut(ch, 0)[..NS_FRAME_SIZE].copy_from_slice(&out);
        }

        if self.num_bands > 1 {
            let upper_band_gain = upper_band_gains
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);

            for ch in 0..self.num_channels {
                for b in 1..self.num_bands {
                    let src = capture.split_band(ch, b);
                    let mut frame = [0.0f32; NS_FRAME_SIZE];
                    frame.copy_from_slice(&src[..NS_FRAME_SIZE]);

                    let mut delayed = [0.0f32; NS_FRAME_SIZE];
                    delay_signal(
                        &frame,
                        &mut self.channels[ch].process_delay_memory[b - 1],
                        &mut delayed,
                    );

                    let dst = capture.split_band_mut(ch, b);
                    for (y, d) in dst.iter_mut().zip(delayed.iter()) {
                        *y = upper_band_gain * *d;
                    }
                }
            }
        }

        for ch in 0..self.num_channels {
            for b in 0..self.num_bands {
                let y = capture.split_band_mut(ch, b);
                for v in y.iter_mut().take(NS_FRAME_SIZE) {
                    *v = v.clamp(-32768.0, 32767.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::ns::ns_config::SuppressionLevel;

    fn fill_test_signal(channel: &mut [f32], sample_rate_hz: f32) {
        for (i, x) in channel.iter_mut().enumerate() {
            let t = i as f32 / sample_rate_hz;
            let tone_a = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let tone_b = (2.0 * std::f32::consts::PI * 1200.0 * t).cos();
            *x = 2000.0 * tone_a + 400.0 * tone_b;
        }
    }

    fn populate_input_frame_with_identical_channels(
        num_channels: usize,
        num_bands: usize,
        frame_index: usize,
        audio: &mut AudioBuffer,
    ) {
        for ch in 0..num_channels {
            for b in 0..num_bands {
                for i in 0..NS_FRAME_SIZE {
                    let value = (frame_index * NS_FRAME_SIZE + i) as i32;
                    audio.split_band_mut(ch, b)[i] = if value > 0 {
                        5000.0 * b as f32 + value as f32
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    fn verify_identical_channels(
        num_channels: usize,
        num_bands: usize,
        audio: &AudioBuffer,
        debug_text: &str,
    ) {
        assert!(num_channels > 1, "{debug_text}");
        for ch in 1..num_channels {
            for b in 0..num_bands {
                for i in 0..NS_FRAME_SIZE {
                    assert_eq!(
                        audio.split_band(ch, b)[i],
                        audio.split_band(0, b)[i],
                        "{debug_text}"
                    );
                }
            }
        }
    }

    #[test]
    fn ns_processes_single_band_without_invalid_values() {
        let mut capture = AudioBuffer::from_sample_rates(16_000, 1, 16_000, 1, 16_000);
        fill_test_signal(capture.channel_mut(0), 16_000.0);

        let mut ns = NoiseSuppressor::new(
            NsConfig {
                target_level: SuppressionLevel::K12dB,
                analyze_linear_aec_output_when_available: false,
            },
            16_000,
            1,
        );

        ns.analyze(&capture);
        ns.process(&mut capture);

        for &x in capture.channel(0) {
            assert!(x.is_finite());
            assert!((-32_768.0..=32_767.0).contains(&x));
        }
    }

    #[test]
    fn ns_processes_multiband_and_updates_upper_bands() {
        let mut capture = AudioBuffer::from_sample_rates(48_000, 1, 48_000, 1, 48_000);
        fill_test_signal(capture.channel_mut(0), 48_000.0);
        capture.split_into_frequency_bands();

        let upper_before = capture.split_band(0, 1).to_vec();

        let mut ns = NoiseSuppressor::new(
            NsConfig {
                target_level: SuppressionLevel::K18dB,
                analyze_linear_aec_output_when_available: false,
            },
            48_000,
            1,
        );

        ns.analyze(&capture);
        ns.process(&mut capture);

        let upper_after = capture.split_band(0, 1);
        assert_ne!(upper_before, upper_after);

        for band in 0..capture.num_bands() {
            for &x in capture.split_band(0, band) {
                assert!(x.is_finite());
                assert!((-32_768.0..=32_767.0).contains(&x));
            }
        }
    }

    #[test]
    fn identical_channel_effects() {
        for &rate in &[16_000usize, 32_000, 48_000] {
            for &num_channels in &[1usize, 4, 8] {
                for &level in &[
                    SuppressionLevel::K6dB,
                    SuppressionLevel::K12dB,
                    SuppressionLevel::K18dB,
                    SuppressionLevel::K21dB,
                ] {
                    let debug_text = format!(
                        "sample rate: {rate}, num_channels: {num_channels}, level: {:?}",
                        level
                    );

                    let num_bands = rate / 16_000;
                    let mut audio = AudioBuffer::from_sample_rates(
                        rate,
                        num_channels,
                        rate,
                        num_channels,
                        rate,
                    );
                    let mut ns = NoiseSuppressor::new(
                        NsConfig {
                            target_level: level,
                            analyze_linear_aec_output_when_available: false,
                        },
                        rate,
                        num_channels,
                    );

                    for frame_index in 0..1000 {
                        if rate > 16_000 {
                            audio.split_into_frequency_bands();
                        }

                        populate_input_frame_with_identical_channels(
                            num_channels,
                            num_bands,
                            frame_index,
                            &mut audio,
                        );

                        ns.analyze(&audio);
                        ns.process(&mut audio);

                        if num_channels > 1 {
                            verify_identical_channels(num_channels, num_bands, &audio, &debug_text);
                        }
                    }
                }
            }
        }
    }
}
