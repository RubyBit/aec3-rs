use std::cmp::min;

use crate::api::config::EchoCanceller3Config;

use super::aec3_common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use super::render_buffer::RenderBuffer;

const COUNTER_THRESHOLD: usize = 5;

pub struct RenderSignalAnalyzer {
    strong_peak_freeze_duration: usize,
    narrow_band_counters: [usize; FFT_LENGTH_BY_2 - 1],
    narrow_peak_band: Option<usize>,
    narrow_peak_counter: usize,
}

impl RenderSignalAnalyzer {
    pub fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            strong_peak_freeze_duration: config.filter.main.length_blocks,
            narrow_band_counters: [0; FFT_LENGTH_BY_2 - 1],
            narrow_peak_band: None,
            narrow_peak_counter: 0,
        }
    }

    pub fn update(&mut self, render_buffer: &RenderBuffer, delay_partitions: Option<usize>) {
        identify_small_narrow_band_regions(
            render_buffer,
            delay_partitions,
            &mut self.narrow_band_counters,
        );
        identify_strong_narrow_band_component(
            render_buffer,
            self.strong_peak_freeze_duration,
            &mut self.narrow_peak_band,
            &mut self.narrow_peak_counter,
        );
    }

    pub fn poor_signal_excitation(&self) -> bool {
        self.narrow_band_counters
            .iter()
            .any(|&counter| counter > 10)
    }

    pub fn mask_regions_around_narrow_bands(&self, v: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        if self.narrow_band_counters[0] > COUNTER_THRESHOLD {
            v[0] = 0.0;
            v[1] = 0.0;
        }
        for k in 2..FFT_LENGTH_BY_2 - 1 {
            if self.narrow_band_counters[k - 1] > COUNTER_THRESHOLD {
                v[k - 2] = 0.0;
                v[k - 1] = 0.0;
                v[k] = 0.0;
                v[k + 1] = 0.0;
                v[k + 2] = 0.0;
            }
        }
        if self.narrow_band_counters[FFT_LENGTH_BY_2 - 2] > COUNTER_THRESHOLD {
            v[FFT_LENGTH_BY_2 - 1] = 0.0;
            v[FFT_LENGTH_BY_2] = 0.0;
        }
    }

    pub fn narrow_peak_band(&self) -> Option<usize> {
        self.narrow_peak_band
    }
}

fn identify_small_narrow_band_regions(
    render_buffer: &RenderBuffer,
    delay_partitions: Option<usize>,
    counters: &mut [usize; FFT_LENGTH_BY_2 - 1],
) {
    if delay_partitions.is_none() {
        counters.fill(0);
        return;
    }
    let mut channel_counters = [0usize; FFT_LENGTH_BY_2 - 1];
    let spectra = render_buffer.spectrum(delay_partitions.unwrap() as isize);
    for spectrum in spectra {
        for k in 1..FFT_LENGTH_BY_2 {
            if spectrum[k] > 3.0 * spectrum[k - 1].max(spectrum[k + 1]) {
                channel_counters[k - 1] += 1;
            }
        }
    }
    for k in 1..FFT_LENGTH_BY_2 {
        counters[k - 1] = if channel_counters[k - 1] > 0 {
            counters[k - 1] + 1
        } else {
            0
        };
    }
}

fn identify_strong_narrow_band_component(
    render_buffer: &RenderBuffer,
    strong_peak_freeze_duration: usize,
    narrow_peak_band: &mut Option<usize>,
    narrow_peak_counter: &mut usize,
) {
    if narrow_peak_band.is_some() {
        *narrow_peak_counter += 1;
        if *narrow_peak_counter > strong_peak_freeze_duration {
            *narrow_peak_band = None;
        }
    }

    let x_latest = render_buffer.block(0);
    let mut max_peak_level = 0.0f32;
    let spectra = render_buffer.spectrum(0);
    for channel in 0..spectra.len() {
        let spectrum = &spectra[channel];
        let mut peak_bin = 0usize;
        let mut peak_value = spectrum[0];
        for (k, &value) in spectrum.iter().enumerate() {
            if value > peak_value {
                peak_value = value;
                peak_bin = k;
            }
        }

        let mut non_peak_power = 0.0f32;
        let start_left = peak_bin.saturating_sub(14);
        let end_left = peak_bin.saturating_sub(4);
        if start_left < end_left {
            for k in start_left..end_left {
                non_peak_power = non_peak_power.max(spectrum[k]);
            }
        }
        let start_right = peak_bin + 5;
        let end_right = min(peak_bin + 15, FFT_LENGTH_BY_2_PLUS_1);
        for k in start_right..end_right {
            non_peak_power = non_peak_power.max(spectrum[k]);
        }

        let mut max_abs = x_latest[0][channel]
            .iter()
            .copied()
            .fold(0.0f32, |acc, sample| acc.max(sample.abs()));
        if x_latest.len() > 1 {
            let band = &x_latest[1][channel];
            max_abs = max_abs.max(
                band.iter()
                    .copied()
                    .fold(0.0, |acc, sample| acc.max(sample.abs())),
            );
        }

        let peak_level = spectrum[peak_bin];
        if peak_bin > 0 && max_abs > 100.0 && peak_level > 100.0 * non_peak_power {
            if peak_level > max_peak_level {
                max_peak_level = peak_level;
                *narrow_peak_band = Some(peak_bin);
                *narrow_peak_counter = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::block_buffer::BlockBuffer;
    use crate::audio_processing::aec3::fft_buffer::FftBuffer;
    use crate::audio_processing::aec3::render_buffer::RenderBuffer;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;
    use crate::test_support::echo_canceller_test_tools::{
        randomize_sample_vector, randomize_sample_vector_with_amplitude,
    };
    use crate::test_support::random::Random;

    #[test]
    fn detects_narrow_band_when_delay_known() {
        let config = EchoCanceller3Config::default();
        let mut block_buffer = BlockBuffer::new(8, 1, 1, BLOCK_SIZE);
        let mut spectrum_buffer = SpectrumBuffer::new(8, 1);
        let fft_buffer = FftBuffer::new(8, 1);

        block_buffer.buffer[block_buffer.read][0][0].fill(150.0);
        let peak_bin = 10;
        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            spectrum_buffer.buffer[spectrum_buffer.read][0][bin] =
                if bin == peak_bin { 100_000.0 } else { 1.0 };
        }

        let render_buffer = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer, false);
        let mut analyzer = RenderSignalAnalyzer::new(&config);
        for _ in 0..15 {
            analyzer.update(&render_buffer, Some(0));
        }

        let mut mask = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        analyzer.mask_regions_around_narrow_bands(&mut mask);
        assert_eq!(0.0, mask[peak_bin]);
        assert!(analyzer.poor_signal_excitation());
        assert_eq!(Some(peak_bin), analyzer.narrow_peak_band());

        analyzer.update(&render_buffer, None);
        mask.fill(1.0);
        analyzer.mask_regions_around_narrow_bands(&mut mask);
        assert!(mask.iter().all(|&v| v == 1.0));
        assert!(!analyzer.poor_signal_excitation());
    }

    fn produce_sinusoid_in_noise(
        sample_rate_hz: i32,
        sinusoid_channel: usize,
        sinusoidal_frequency_hz: f32,
        rng: &mut Random,
        sample_counter: &mut usize,
        block: &mut [Vec<Vec<f32>>],
    ) {
        for band in block.iter_mut() {
            for channel in band.iter_mut() {
                randomize_sample_vector_with_amplitude(rng, channel, 500.0);
            }
        }

        for (j, sample_index) in (*sample_counter..*sample_counter + BLOCK_SIZE).enumerate() {
            let angle =
                2.0f32 * std::f32::consts::PI * sinusoidal_frequency_hz * (sample_index as f32)
                    / sample_rate_hz as f32;
            block[0][sinusoid_channel][j] += 32_000.0 * angle.sin();
        }
        *sample_counter += BLOCK_SIZE;
    }

    fn make_block(num_bands: usize, num_channels: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![0.0f32; BLOCK_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn run_narrow_band_detection_sequence(
        analyzer: &mut RenderSignalAnalyzer,
        render_delay_buffer: &mut RenderDelayBuffer,
        block: &mut Vec<Vec<Vec<f32>>>,
        rng: &mut Random,
        num_channels: usize,
        known_delay: bool,
    ) {
        const SINUS_BIN: usize = 32;
        let mut sample_counter = 0usize;
        let sinus_frequency_hz = 8_000.0 * SINUS_BIN as f32 / FFT_LENGTH_BY_2 as f32;
        for iteration in 0..100 {
            produce_sinusoid_in_noise(
                16_000,
                num_channels - 1,
                sinus_frequency_hz,
                rng,
                &mut sample_counter,
                block,
            );
            render_delay_buffer.insert(&block[..]);
            if iteration == 0 {
                render_delay_buffer.reset();
            }
            render_delay_buffer.prepare_capture_processing();
            let render_view = render_delay_buffer.render_buffer();
            analyzer.update(&render_view, known_delay.then_some(0));
        }
    }

    #[test]
    fn render_signal_analyzer_no_false_detection_of_narrow_bands() {
        for &num_channels in &[1usize, 2, 8] {
            let mut analyzer = RenderSignalAnalyzer::new(&EchoCanceller3Config::default());
            let mut rng = Random::new(42);
            let num_bands = num_bands_for_rate(48_000);
            let mut block = make_block(num_bands, num_channels);
            let mut render_delay_buffer =
                RenderDelayBuffer::new(EchoCanceller3Config::default(), 48_000, num_channels);

            for iteration in 0..100 {
                for band in 0..num_bands {
                    for channel in 0..num_channels {
                        randomize_sample_vector(&mut rng, &mut block[band][channel]);
                    }
                }
                render_delay_buffer.insert(&block);
                if iteration == 0 {
                    render_delay_buffer.reset();
                }
                render_delay_buffer.prepare_capture_processing();
                let render_view = render_delay_buffer.render_buffer();
                analyzer.update(&render_view, Some(0));
            }

            let mut mask = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
            analyzer.mask_regions_around_narrow_bands(&mut mask);
            assert!(mask.iter().all(|&value| (value - 1.0).abs() < f32::EPSILON));
            assert!(!analyzer.poor_signal_excitation());
            assert!(analyzer.narrow_peak_band().is_none());
        }
    }

    #[test]
    fn render_signal_analyzer_narrow_band_detection_matches_reference() {
        for &num_channels in &[1usize, 2, 8] {
            let config = EchoCanceller3Config::default();
            let mut analyzer = RenderSignalAnalyzer::new(&config);
            let mut rng = Random::new(42);
            let num_bands = num_bands_for_rate(48_000);
            let mut block = make_block(num_bands, num_channels);
            let mut render_delay_buffer =
                RenderDelayBuffer::new(config.clone(), 48_000, num_channels);

            run_narrow_band_detection_sequence(
                &mut analyzer,
                &mut render_delay_buffer,
                &mut block,
                &mut rng,
                num_channels,
                true,
            );

            let mut mask = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
            analyzer.mask_regions_around_narrow_bands(&mut mask);
            println!(
                "channels {} mask around peak: {:?}",
                num_channels,
                &mask[28..37]
            );
            for (k, &value) in mask.iter().enumerate() {
                if (k as isize - 32).abs() <= 2 {
                    assert_eq!(0.0, value, "expected notch at bin {}", k);
                } else {
                    assert_eq!(1.0, value, "unexpected notch at bin {}", k);
                }
            }
            assert!(analyzer.poor_signal_excitation());
            assert_eq!(Some(32), analyzer.narrow_peak_band());

            run_narrow_band_detection_sequence(
                &mut analyzer,
                &mut render_delay_buffer,
                &mut block,
                &mut rng,
                num_channels,
                false,
            );

            mask.fill(1.0);
            analyzer.mask_regions_around_narrow_bands(&mut mask);
            assert!(mask.iter().all(|&value| (value - 1.0).abs() < f32::EPSILON));
            assert!(!analyzer.poor_signal_excitation());
        }
    }
}
