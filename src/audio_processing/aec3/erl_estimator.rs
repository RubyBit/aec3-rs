use crate::audio_processing::aec3::aec3_common::{
    FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_MINUS_1, FFT_LENGTH_BY_2_PLUS_1,
};

const MIN_ERL: f32 = 0.01;
const MAX_ERL: f32 = 1000.0;
const HOLD_BLOCKS: i32 = 1000;
const X2_BAND_ENERGY_THRESHOLD: f32 = 44_015_068.0;

pub struct ErlEstimator {
    startup_phase_length_blocks: usize,
    erl: [f32; FFT_LENGTH_BY_2_PLUS_1],
    hold_counters: [i32; FFT_LENGTH_BY_2_MINUS_1],
    erl_time_domain: f32,
    hold_counter_time_domain: i32,
    blocks_since_reset: usize,
}

impl ErlEstimator {
    pub fn new(startup_phase_length_blocks: usize) -> Self {
        Self {
            startup_phase_length_blocks,
            erl: [MAX_ERL; FFT_LENGTH_BY_2_PLUS_1],
            hold_counters: [0; FFT_LENGTH_BY_2_MINUS_1],
            erl_time_domain: MAX_ERL,
            hold_counter_time_domain: 0,
            blocks_since_reset: 0,
        }
    }

    pub fn reset(&mut self) {
        self.blocks_since_reset = 0;
    }

    pub fn update(
        &mut self,
        converged_filters: &[bool],
        render_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        capture_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        assert_eq!(capture_spectra.len(), converged_filters.len());
        assert!(!render_spectra.is_empty());

        let first_converged = match converged_filters.iter().position(|&x| x) {
            Some(idx) => idx,
            None => {
                self.blocks_since_reset = self.blocks_since_reset.saturating_add(1);
                return;
            }
        };

        self.blocks_since_reset = self.blocks_since_reset.saturating_add(1);
        if self.blocks_since_reset < self.startup_phase_length_blocks {
            return;
        }

        // Aggregate capture spectra across converged filters only.
        let mut max_capture = capture_spectra[first_converged].clone();
        for (ch, spectrum) in capture_spectra.iter().enumerate() {
            if ch == first_converged || !converged_filters[ch] {
                continue;
            }
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                if spectrum[k] > max_capture[k] {
                    max_capture[k] = spectrum[k];
                }
            }
        }

        // Aggregate render spectra across all channels.
        let mut max_render = render_spectra[0].clone();
        for spectrum in render_spectra.iter().skip(1) {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                if spectrum[k] > max_render[k] {
                    max_render[k] = spectrum[k];
                }
            }
        }

        // Update per-band ERL estimates.
        for k in 1..FFT_LENGTH_BY_2 {
            if max_render[k] > X2_BAND_ENERGY_THRESHOLD {
                let new_erl = max_capture[k] / max_render[k];
                if new_erl < self.erl[k] {
                    self.hold_counters[k - 1] = HOLD_BLOCKS;
                    self.erl[k] += 0.1 * (new_erl - self.erl[k]);
                    if self.erl[k] < MIN_ERL {
                        self.erl[k] = MIN_ERL;
                    }
                }
            }
        }

        for counter in &mut self.hold_counters {
            *counter -= 1;
        }

        for (bin, counter) in self.hold_counters.iter().enumerate() {
            if *counter <= 0 {
                let idx = bin + 1;
                self.erl[idx] = (2.0 * self.erl[idx]).min(MAX_ERL);
            }
        }

        self.erl[0] = self.erl[1];
        self.erl[FFT_LENGTH_BY_2] = self.erl[FFT_LENGTH_BY_2 - 1];

        // Update the time-domain ERL.
        let x2_sum: f32 = max_render.iter().sum();
        if x2_sum > X2_BAND_ENERGY_THRESHOLD * FFT_LENGTH_BY_2_PLUS_1 as f32 {
            let y2_sum: f32 = max_capture.iter().sum();
            let new_erl = y2_sum / x2_sum;
            if new_erl < self.erl_time_domain {
                self.hold_counter_time_domain = HOLD_BLOCKS;
                self.erl_time_domain += 0.1 * (new_erl - self.erl_time_domain);
                if self.erl_time_domain < MIN_ERL {
                    self.erl_time_domain = MIN_ERL;
                }
            }
        }

        self.hold_counter_time_domain -= 1;
        if self.hold_counter_time_domain <= 0 {
            self.erl_time_domain = (2.0 * self.erl_time_domain).min(MAX_ERL);
        }
    }

    pub fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.erl
    }

    pub fn erl_time_domain(&self) -> f32 {
        self.erl_time_domain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_erl(erl: &[f32; FFT_LENGTH_BY_2_PLUS_1], erl_time_domain: f32, reference: f32) {
        for &value in erl.iter() {
            assert!(
                (value - reference).abs() < 1e-3,
                "value={value}, reference={reference}"
            );
        }
        assert!(
            (erl_time_domain - reference).abs() < 1e-3,
            "time_domain={erl_time_domain}, reference={reference}"
        );
    }

    #[test]
    fn estimates_track_reference_behavior() {
        let render_channel_options = [1, 2, 8];
        let capture_channel_options = [1, 2, 8];
        for &num_render_channels in &render_channel_options {
            for &num_capture_channels in &capture_channel_options {
                let mut x2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_render_channels];
                let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut converged = vec![false; num_capture_channels];
                let converged_idx = num_capture_channels - 1;
                converged[converged_idx] = true;

                let mut estimator = ErlEstimator::new(0);

                for spectrum in &mut x2 {
                    spectrum.fill(500.0 * 1_000_000.0);
                }
                y2[converged_idx].fill(10.0 * x2[0][0]);

                for _ in 0..200 {
                    estimator.update(&converged, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 10.0);

                y2[converged_idx].fill(10_000.0 * x2[0][0]);
                for _ in 0..998 {
                    estimator.update(&converged, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 10.0);

                estimator.update(&converged, &x2, &y2);
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 20.0);

                for _ in 0..1000 {
                    estimator.update(&converged, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 1000.0);

                for spectrum in &mut x2 {
                    spectrum.fill(1_000_000.0);
                }
                y2[converged_idx].fill(10.0 * x2[0][0]);
                for _ in 0..200 {
                    estimator.update(&converged, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 1000.0);
            }
        }
    }
}
