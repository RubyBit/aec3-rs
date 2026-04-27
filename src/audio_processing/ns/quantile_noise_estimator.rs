//! Quantile-based noise floor estimator.

use super::fast_math::{exp_approximation_slice, log_approximation_slice};
use super::ns_common::{FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS};

pub const SIMULT: usize = 3;

#[derive(Debug, Clone)]
pub struct QuantileNoiseEstimator {
    density: [f32; SIMULT * FFT_SIZE_BY_2_PLUS_1],
    log_quantile: [f32; SIMULT * FFT_SIZE_BY_2_PLUS_1],
    quantile: [f32; FFT_SIZE_BY_2_PLUS_1],
    counter: [i32; SIMULT],
    num_updates: i32,
}

impl Default for QuantileNoiseEstimator {
    fn default() -> Self {
        let mut counter = [0i32; SIMULT];
        let one_by_simult = 1.0 / SIMULT as f32;
        for (i, c) in counter.iter_mut().enumerate() {
            *c = (LONG_STARTUP_PHASE_BLOCKS as f32 * (i as f32 + 1.0) * one_by_simult).floor()
                as i32;
        }

        Self {
            density: [0.3; SIMULT * FFT_SIZE_BY_2_PLUS_1],
            log_quantile: [8.0; SIMULT * FFT_SIZE_BY_2_PLUS_1],
            quantile: [0.0; FFT_SIZE_BY_2_PLUS_1],
            counter,
            num_updates: 1,
        }
    }
}

impl QuantileNoiseEstimator {
    pub fn estimate(
        &mut self,
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        noise_spectrum: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        let mut log_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        log_approximation_slice(signal_spectrum, &mut log_spectrum);

        let mut quantile_index_to_return: Option<usize> = None;

        for s in 0..SIMULT {
            let k = s * FFT_SIZE_BY_2_PLUS_1;
            let one_by_counter_plus_1 = 1.0 / (self.counter[s] as f32 + 1.0);

            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                let j = k + i;
                let delta = if self.density[j] > 1.0 {
                    40.0 / self.density[j]
                } else {
                    40.0
                };

                let multiplier = delta * one_by_counter_plus_1;
                if log_spectrum[i] > self.log_quantile[j] {
                    self.log_quantile[j] += 0.25 * multiplier;
                } else {
                    self.log_quantile[j] -= 0.75 * multiplier;
                }

                const WIDTH: f32 = 0.01;
                const ONE_BY_WIDTH_PLUS_2: f32 = 1.0 / (2.0 * WIDTH);
                if (log_spectrum[i] - self.log_quantile[j]).abs() < WIDTH {
                    self.density[j] = (self.counter[s] as f32 * self.density[j]
                        + ONE_BY_WIDTH_PLUS_2)
                        * one_by_counter_plus_1;
                }
            }

            if self.counter[s] >= LONG_STARTUP_PHASE_BLOCKS {
                self.counter[s] = 0;
                if self.num_updates >= LONG_STARTUP_PHASE_BLOCKS {
                    quantile_index_to_return = Some(k);
                }
            }

            self.counter[s] += 1;
        }

        if self.num_updates < LONG_STARTUP_PHASE_BLOCKS {
            quantile_index_to_return = Some(FFT_SIZE_BY_2_PLUS_1 * (SIMULT - 1));
            self.num_updates += 1;
        }

        if let Some(k) = quantile_index_to_return {
            exp_approximation_slice(
                &self.log_quantile[k..k + FFT_SIZE_BY_2_PLUS_1],
                &mut self.quantile,
            );
        }

        noise_spectrum.copy_from_slice(&self.quantile);
    }
}
