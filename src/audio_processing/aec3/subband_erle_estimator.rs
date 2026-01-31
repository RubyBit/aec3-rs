use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};

const X2_BAND_ENERGY_THRESHOLD: f32 = 44_015_068.0;
const BLOCKS_TO_HOLD_ERLE: i32 = 100;
const BLOCKS_FOR_ONSET_DETECTION: i32 = BLOCKS_TO_HOLD_ERLE + 150;
const POINTS_TO_ACCUMULATE: usize = 6;

fn set_max_erle_bands(max_erle_l: f32, max_erle_h: f32) -> [f32; FFT_LENGTH_BY_2_PLUS_1] {
    let mut max_erle = [max_erle_h; FFT_LENGTH_BY_2_PLUS_1];
    for value in max_erle.iter_mut().take(FFT_LENGTH_BY_2 / 2) {
        *value = max_erle_l;
    }
    max_erle
}

fn use_min_erle_during_onsets() -> bool {
    true
}

pub struct SubbandErleEstimator {
    use_onset_detection: bool,
    min_erle: f32,
    max_erle: [f32; FFT_LENGTH_BY_2_PLUS_1],
    use_min_erle_during_onsets: bool,
    accum_spectra: AccumulatedSpectra,
    erle: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_onsets: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    coming_onset: Vec<[bool; FFT_LENGTH_BY_2_PLUS_1]>,
    hold_counters: Vec<[i32; FFT_LENGTH_BY_2_PLUS_1]>,
}

impl SubbandErleEstimator {
    pub fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let mut instance = Self {
            use_onset_detection: config.erle.onset_detection,
            min_erle: config.erle.min,
            max_erle: set_max_erle_bands(config.erle.max_l, config.erle.max_h),
            use_min_erle_during_onsets: use_min_erle_during_onsets(),
            accum_spectra: AccumulatedSpectra::new(num_capture_channels),
            erle: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            erle_onsets: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            coming_onset: vec![[true; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            hold_counters: vec![[0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
        };
        instance.reset();
        instance
    }

    pub fn reset(&mut self) {
        for erle in &mut self.erle {
            erle.fill(self.min_erle);
        }
        for ch in 0..self.erle_onsets.len() {
            self.erle_onsets[ch].fill(self.min_erle);
            self.coming_onset[ch].fill(true);
            self.hold_counters[ch].fill(0);
        }
        self.accum_spectra.reset();
    }

    pub fn update(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        assert_eq!(y2.len(), e2.len());
        assert_eq!(y2.len(), converged_filters.len());
        self.update_accumulated_spectra(x2, y2, e2, converged_filters);
        self.update_bands(converged_filters);
        if self.use_onset_detection {
            self.decrease_erle_per_band_for_low_render_signals();
        }
        for erle in &mut self.erle {
            erle[0] = erle[1];
            erle[FFT_LENGTH_BY_2] = erle[FFT_LENGTH_BY_2 - 1];
        }
    }

    pub fn erle(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.erle
    }

    pub fn erle_onsets(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.erle_onsets
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        if let Some(erle) = self.erle_onsets.first() {
            dumper.dump_raw_f32_slice(DiagnosticLevel::DeepDebug, "aec3_erle_onset", erle);
        }
    }

    fn update_accumulated_spectra(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        let st = &mut self.accum_spectra;
        for ch in 0..y2.len() {
            if !converged_filters[ch] {
                continue;
            }
            if st.num_points[ch] == POINTS_TO_ACCUMULATE {
                st.num_points[ch] = 0;
                st.y2[ch].fill(0.0);
                st.e2[ch].fill(0.0);
                st.low_render_energy[ch].fill(false);
            }
            for (dst, &value) in st.y2[ch].iter_mut().zip(y2[ch].iter()) {
                *dst += value;
            }
            for (dst, &value) in st.e2[ch].iter_mut().zip(e2[ch].iter()) {
                *dst += value;
            }
            for (flag, &value) in st.low_render_energy[ch].iter_mut().zip(x2.iter()) {
                *flag = *flag || value < X2_BAND_ENERGY_THRESHOLD;
            }
            st.num_points[ch] += 1;
        }
    }

    fn update_bands(&mut self, converged_filters: &[bool]) {
        for ch in 0..self.erle.len() {
            if !converged_filters[ch] {
                continue;
            }
            if self.accum_spectra.num_points[ch] != POINTS_TO_ACCUMULATE {
                continue;
            }

            let mut new_erle = [0.0f32; FFT_LENGTH_BY_2];
            let mut updated = [false; FFT_LENGTH_BY_2];
            for k in 1..FFT_LENGTH_BY_2 {
                if self.accum_spectra.e2[ch][k] > 0.0 {
                    new_erle[k] = self.accum_spectra.y2[ch][k] / self.accum_spectra.e2[ch][k];
                    updated[k] = true;
                }
            }

            if self.use_onset_detection {
                for k in 1..FFT_LENGTH_BY_2 {
                    if updated[k] && !self.accum_spectra.low_render_energy[ch][k] {
                        if self.coming_onset[ch][k] {
                            self.coming_onset[ch][k] = false;
                            if !self.use_min_erle_during_onsets {
                                let alpha = if new_erle[k] < self.erle_onsets[ch][k] {
                                    0.3
                                } else {
                                    0.15
                                };
                                self.erle_onsets[ch][k] = (self.erle_onsets[ch][k]
                                    + alpha * (new_erle[k] - self.erle_onsets[ch][k]))
                                    .clamp(self.min_erle, self.max_erle[k]);
                            }
                        }
                        self.hold_counters[ch][k] = BLOCKS_FOR_ONSET_DETECTION;
                    }
                }
            }

            for k in 1..FFT_LENGTH_BY_2 {
                if updated[k] {
                    let mut alpha = 0.05;
                    if new_erle[k] < self.erle[ch][k] {
                        alpha = if self.accum_spectra.low_render_energy[ch][k] {
                            0.0
                        } else {
                            0.1
                        };
                    }
                    self.erle[ch][k] = (self.erle[ch][k]
                        + alpha * (new_erle[k] - self.erle[ch][k]))
                        .clamp(self.min_erle, self.max_erle[k]);
                }
            }
        }
    }

    fn decrease_erle_per_band_for_low_render_signals(&mut self) {
        for ch in 0..self.erle.len() {
            for k in 1..FFT_LENGTH_BY_2 {
                self.hold_counters[ch][k] -= 1;
                if self.hold_counters[ch][k] <= (BLOCKS_FOR_ONSET_DETECTION - BLOCKS_TO_HOLD_ERLE) {
                    if self.erle[ch][k] > self.erle_onsets[ch][k] {
                        self.erle[ch][k] = self.erle_onsets[ch][k]
                            .max(0.97 * self.erle[ch][k])
                            .max(self.min_erle);
                    }
                    if self.hold_counters[ch][k] <= 0 {
                        self.coming_onset[ch][k] = true;
                        self.hold_counters[ch][k] = 0;
                    }
                }
            }
        }
    }
}

struct AccumulatedSpectra {
    y2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    e2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    low_render_energy: Vec<[bool; FFT_LENGTH_BY_2_PLUS_1]>,
    num_points: Vec<usize>,
}

impl AccumulatedSpectra {
    fn new(num_capture_channels: usize) -> Self {
        Self {
            y2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            e2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            low_render_energy: vec![[false; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            num_points: vec![0; num_capture_channels],
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.y2 {
            buf.fill(0.0);
        }
        for buf in &mut self.e2 {
            buf.fill(0.0);
        }
        for buf in &mut self.low_render_energy {
            buf.fill(false);
        }
        for value in &mut self.num_points {
            *value = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increases_erle_for_strong_echo() {
        let mut config = EchoCanceller3Config::default();
        config.erle.max_l = 20.0;
        config.erle.max_h = 20.0;
        config.erle.onset_detection = false;
        let mut estimator = SubbandErleEstimator::new(&config, 1);
        let x2 = [100_000_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        y2[0].fill(1_000_000_000.0);
        e2[0].fill(y2[0][0] / 10.0);
        let converged = vec![true];

        for _ in 0..(POINTS_TO_ACCUMULATE * 60) {
            estimator.update(&x2, &y2, &e2, &converged);
        }

        for k in 1..FFT_LENGTH_BY_2 {
            assert!(
                (estimator.erle()[0][k] - 10.0).abs() < 1.0,
                "bin {k} deviated: {}",
                estimator.erle()[0][k]
            );
        }
    }
}
