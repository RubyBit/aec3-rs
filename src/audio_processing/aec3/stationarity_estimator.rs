use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

use super::aec3_common::{FFT_LENGTH_BY_2_PLUS_1, NUM_BLOCKS_PER_SECOND};
use super::spectrum_buffer::SpectrumBuffer;

const WINDOW_LENGTH: usize = 13;
const MIN_NOISE_POWER: f32 = 10.0;
const HANGOVER_BLOCKS: i32 = (NUM_BLOCKS_PER_SECOND / 20) as i32;
const BLOCKS_AVERAGE_INIT_PHASE: usize = 20;
const BLOCKS_INITIAL_PHASE: usize = 2 * NUM_BLOCKS_PER_SECOND;

/// Estimates render stationarity for each frequency bin.
pub struct StationarityEstimator {
    data_dumper: ApmDataDumper,
    noise: NoiseSpectrum,
    hangovers: [i32; FFT_LENGTH_BY_2_PLUS_1],
    stationarity_flags: [bool; FFT_LENGTH_BY_2_PLUS_1],
}

impl StationarityEstimator {
    pub fn new() -> Self {
        let mut estimator = Self {
            data_dumper: ApmDataDumper::new_unique(),
            noise: NoiseSpectrum::new(),
            hangovers: [0; FFT_LENGTH_BY_2_PLUS_1],
            stationarity_flags: [false; FFT_LENGTH_BY_2_PLUS_1],
        };
        estimator.reset();
        estimator
    }

    pub fn reset(&mut self) {
        self.noise.reset();
        self.hangovers.fill(0);
        self.stationarity_flags.fill(false);
    }

    pub fn update_noise_estimator(&mut self, spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]]) {
        if spectrum.is_empty() {
            return;
        }
        self.noise.update(spectrum);
        self.data_dumper
            .dump_raw_f32_slice("aec3_stationarity_noise_spectrum", self.noise.spectrum());
        self.data_dumper.dump_raw_f32(
            "aec3_stationarity_is_block_stationary",
            if self.is_block_stationary() { 1.0 } else { 0.0 },
        );
    }

    pub fn update_stationarity_flags(
        &mut self,
        spectrum_buffer: &SpectrumBuffer,
        average_reverb: &[f32],
        idx_current: usize,
        num_lookahead: i32,
    ) {
        assert_eq!(average_reverb.len(), FFT_LENGTH_BY_2_PLUS_1);

        let mut indexes = [0usize; WINDOW_LENGTH];
        let num_lookahead_bounded = num_lookahead.clamp(0, (WINDOW_LENGTH - 1) as i32);
        let mut idx = idx_current;
        if num_lookahead_bounded < (WINDOW_LENGTH as i32 - 1) {
            let num_lookback = (WINDOW_LENGTH as i32 - 1) - num_lookahead_bounded;
            idx = spectrum_buffer.offset_index(idx_current, num_lookback as isize);
        }
        indexes[0] = idx;
        for k in 1..WINDOW_LENGTH {
            indexes[k] = spectrum_buffer.dec_index(indexes[k - 1]);
        }

        for band in 0..FFT_LENGTH_BY_2_PLUS_1 {
            self.stationarity_flags[band] =
                self.estimate_band_stationarity(spectrum_buffer, average_reverb, &indexes, band);
        }
        self.update_hangover();
        self.smooth_stationary_per_freq();
    }

    pub fn is_band_stationary(&self, band: usize) -> bool {
        self.stationarity_flags[band] && self.hangovers[band] == 0
    }

    pub fn is_block_stationary(&self) -> bool {
        let stationary_bins = self
            .stationarity_flags
            .iter()
            .enumerate()
            .filter(|(band, _)| self.is_band_stationary(*band))
            .count();
        (stationary_bins as f32 / FFT_LENGTH_BY_2_PLUS_1 as f32) > 0.75
    }

    fn estimate_band_stationarity(
        &self,
        spectrum_buffer: &SpectrumBuffer,
        average_reverb: &[f32],
        indexes: &[usize; WINDOW_LENGTH],
        band: usize,
    ) -> bool {
        const STATIONARITY_THRESHOLD: f32 = 10.0;
        let num_render_channels = spectrum_buffer.buffer[0].len();
        let inv_channels = 1.0 / num_render_channels as f32;
        let mut accumulated_power = 0.0f32;
        for &idx in indexes {
            for ch in 0..num_render_channels {
                accumulated_power += spectrum_buffer.buffer[idx][ch][band] * inv_channels;
            }
        }
        accumulated_power += average_reverb[band];
        let noise = WINDOW_LENGTH as f32 * self.noise.power(band);
        assert!(noise > 0.0);
        let ratio = accumulated_power / noise;
        self.data_dumper
            .dump_raw_f32("aec3_stationarity_long_ratio", ratio);
        accumulated_power < STATIONARITY_THRESHOLD * noise
    }

    fn update_hangover(&mut self) {
        let all_stationary = self.stationarity_flags.iter().all(|flag| *flag);
        for (k, flag) in self.stationarity_flags.iter().enumerate() {
            if !flag {
                self.hangovers[k] = HANGOVER_BLOCKS;
            } else if all_stationary {
                self.hangovers[k] = (self.hangovers[k] - 1).max(0);
            }
        }
    }

    fn smooth_stationary_per_freq(&mut self) {
        if FFT_LENGTH_BY_2_PLUS_1 <= 2 {
            return;
        }
        let mut smoothed = [false; FFT_LENGTH_BY_2_PLUS_1];
        for k in 1..(FFT_LENGTH_BY_2_PLUS_1 - 1) {
            smoothed[k] = self.stationarity_flags[k - 1]
                && self.stationarity_flags[k]
                && self.stationarity_flags[k + 1];
        }
        smoothed[0] = smoothed[1];
        smoothed[FFT_LENGTH_BY_2_PLUS_1 - 1] = smoothed[FFT_LENGTH_BY_2_PLUS_1 - 2];
        self.stationarity_flags = smoothed;
    }
}

struct NoiseSpectrum {
    noise_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    block_counter: usize,
}

impl NoiseSpectrum {
    fn new() -> Self {
        let mut spectrum = Self {
            noise_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            block_counter: 0,
        };
        spectrum.reset();
        spectrum
    }

    fn reset(&mut self) {
        self.block_counter = 0;
        self.noise_spectrum.fill(MIN_NOISE_POWER);
    }

    fn spectrum(&self) -> &[f32] {
        &self.noise_spectrum
    }

    fn power(&self, band: usize) -> f32 {
        self.noise_spectrum[band]
    }

    fn update(&mut self, spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]]) {
        let num_channels = spectrum.len();
        let mut avg_storage = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let avg: &[f32; FFT_LENGTH_BY_2_PLUS_1] = if num_channels == 1 {
            &spectrum[0]
        } else {
            avg_storage.fill(0.0);
            for ch in 0..num_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    avg_storage[k] += spectrum[ch][k];
                }
            }
            let inv = 1.0 / num_channels as f32;
            for value in &mut avg_storage {
                *value *= inv;
            }
            &avg_storage
        };

        self.block_counter += 1;
        let alpha = self.get_alpha();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if self.block_counter <= BLOCKS_AVERAGE_INIT_PHASE {
                self.noise_spectrum[k] += avg[k] / BLOCKS_AVERAGE_INIT_PHASE as f32;
            } else {
                self.noise_spectrum[k] =
                    self.update_band_by_smoothing(avg[k], self.noise_spectrum[k], alpha);
            }
        }
    }

    fn get_alpha(&self) -> f32 {
        const ALPHA: f32 = 0.004;
        const ALPHA_INIT: f32 = 0.04;
        const TILT: f32 = (ALPHA_INIT - ALPHA) / BLOCKS_INITIAL_PHASE as f32;
        if self.block_counter > BLOCKS_INITIAL_PHASE + BLOCKS_AVERAGE_INIT_PHASE {
            ALPHA
        } else {
            ALPHA_INIT - TILT * (self.block_counter as f32 - BLOCKS_AVERAGE_INIT_PHASE as f32)
        }
    }

    fn update_band_by_smoothing(&self, power_band: f32, mut noise_band: f32, alpha: f32) -> f32 {
        if noise_band < power_band {
            if power_band > 0.0 {
                let mut alpha_inc = alpha * (noise_band / power_band);
                if self.block_counter > BLOCKS_INITIAL_PHASE && 10.0 * noise_band < power_band {
                    alpha_inc *= 0.1;
                }
                noise_band += alpha_inc * (power_band - noise_band);
            }
        } else {
            noise_band += alpha * (power_band - noise_band);
            noise_band = noise_band.max(MIN_NOISE_POWER);
        }
        noise_band
    }
}

impl Default for StationarityEstimator {
    fn default() -> Self {
        Self::new()
    }
}
