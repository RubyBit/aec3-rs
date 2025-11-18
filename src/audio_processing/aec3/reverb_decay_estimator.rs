use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

use super::aec3_common::{
    FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_LOG2, fast_approx_log2f, get_time_domain_length,
};

const EARLY_REVERB_MIN_SIZE_BLOCKS: usize = 3;
const BLOCKS_PER_SECTION: usize = 6;
const EARLY_REVERB_FIRST_POINT_AT_LINEAR_REGRESSORS: f32 =
    -0.5 * BLOCKS_PER_SECTION as f32 * FFT_LENGTH_BY_2 as f32 + 0.5;

pub struct ReverbDecayEstimator {
    filter_length_blocks: usize,
    filter_length_coefficients: usize,
    use_adaptive_echo_decay: bool,
    late_reverb_decay_estimator: LateReverbLinearRegressor,
    early_reverb_estimator: EarlyReverbLengthEstimator,
    late_reverb_start: usize,
    late_reverb_end: usize,
    block_to_analyze: usize,
    estimation_region_candidate_size: usize,
    estimation_region_identified: bool,
    previous_gains: Vec<f32>,
    decay: f32,
    tail_gain: f32,
    smoothing_constant: f32,
}

impl ReverbDecayEstimator {
    pub fn new(config: &EchoCanceller3Config) -> Self {
        assert!(config.filter.main.length_blocks > EARLY_REVERB_MIN_SIZE_BLOCKS);
        let filter_length_blocks = config.filter.main.length_blocks;
        Self {
            filter_length_blocks,
            filter_length_coefficients: get_time_domain_length(filter_length_blocks),
            use_adaptive_echo_decay: config.ep_strength.default_len < 0.0,
            late_reverb_decay_estimator: LateReverbLinearRegressor::new(),
            early_reverb_estimator: EarlyReverbLengthEstimator::new(
                filter_length_blocks - EARLY_REVERB_MIN_SIZE_BLOCKS,
            ),
            late_reverb_start: EARLY_REVERB_MIN_SIZE_BLOCKS,
            late_reverb_end: EARLY_REVERB_MIN_SIZE_BLOCKS,
            block_to_analyze: 0,
            estimation_region_candidate_size: 0,
            estimation_region_identified: false,
            previous_gains: vec![0.0; filter_length_blocks],
            decay: config.ep_strength.default_len.abs(),
            tail_gain: 0.0,
            smoothing_constant: 0.0,
        }
    }

    pub fn update(
        &mut self,
        filter: &[f32],
        filter_quality: Option<f32>,
        filter_delay_blocks: i32,
        usable_linear_filter: bool,
        stationary_signal: bool,
    ) {
        if stationary_signal {
            return;
        }

        let filter_size = filter.len();
        let estimation_feasible = filter_delay_blocks > 0
            && (filter_delay_blocks as usize)
                <= self.filter_length_blocks - EARLY_REVERB_MIN_SIZE_BLOCKS - 1
            && filter_size == self.filter_length_coefficients
            && usable_linear_filter;

        if !estimation_feasible {
            self.reset_decay_estimation();
            return;
        }

        if !self.use_adaptive_echo_decay {
            return;
        }

        let new_smoothing = filter_quality.unwrap_or(0.0) * 0.2;
        if new_smoothing > self.smoothing_constant {
            self.smoothing_constant = new_smoothing;
        }
        if self.smoothing_constant == 0.0 {
            return;
        }

        if self.block_to_analyze < self.filter_length_blocks {
            self.analyze_filter(filter);
            self.block_to_analyze += 1;
        } else {
            self.estimate_decay(filter, filter_delay_blocks as usize);
        }
    }

    pub fn decay(&self) -> f32 {
        self.decay
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        dumper.dump_raw_f32("aec3_reverb_decay", self.decay);
        dumper.dump_raw_f32("aec3_reverb_tail_energy", self.tail_gain);
        dumper.dump_raw_f32("aec3_reverb_alpha", self.smoothing_constant);
        let num_blocks = if self.late_reverb_end >= self.late_reverb_start {
            (self.late_reverb_end - self.late_reverb_start) as f32
        } else {
            0.0
        };
        dumper.dump_raw_f32("aec3_num_reverb_decay_blocks", num_blocks);
        dumper.dump_raw_i32("aec3_late_reverb_start", self.late_reverb_start as i32);
        dumper.dump_raw_i32("aec3_late_reverb_end", self.late_reverb_end as i32);
        self.early_reverb_estimator.dump(dumper);
    }

    fn reset_decay_estimation(&mut self) {
        self.early_reverb_estimator.reset();
        self.late_reverb_decay_estimator.reset(0);
        self.block_to_analyze = 0;
        self.estimation_region_candidate_size = 0;
        self.estimation_region_identified = false;
        self.smoothing_constant = 0.0;
        self.late_reverb_start = 0;
        self.late_reverb_end = 0;
    }

    fn estimate_decay(&mut self, filter: &[f32], peak_block: usize) {
        self.block_to_analyze = (peak_block + EARLY_REVERB_MIN_SIZE_BLOCKS)
            .min(self.filter_length_blocks.saturating_sub(1));

        let first_reverb_gain = block_energy_average(filter, self.block_to_analyze);
        let h_size_blocks = filter.len() >> FFT_LENGTH_BY_2_LOG2;
        self.tail_gain = block_energy_average(filter, h_size_blocks - 1);
        let peak_energy = block_energy_peak(filter, peak_block);
        let sufficient_reverb_decay = first_reverb_gain > 4.0 * self.tail_gain;
        let valid_filter = first_reverb_gain > 2.0 * self.tail_gain && peak_energy < 100.0;

        let size_early_reverb = self.early_reverb_estimator.estimate();
        let size_late_reverb = self
            .estimation_region_candidate_size
            .saturating_sub(size_early_reverb);

        if size_late_reverb >= 5 {
            if valid_filter && self.late_reverb_decay_estimator.estimate_available() {
                let mut decay = (2.0f32)
                    .powf(self.late_reverb_decay_estimator.estimate() * FFT_LENGTH_BY_2 as f32);
                const MAX_DECAY: f32 = 0.95;
                const MIN_DECAY: f32 = 0.02;
                decay = decay.max(0.97 * self.decay);
                decay = decay.min(MAX_DECAY);
                decay = decay.max(MIN_DECAY);
                self.decay += self.smoothing_constant * (decay - self.decay);
            }

            self.late_reverb_decay_estimator
                .reset(size_late_reverb * FFT_LENGTH_BY_2);
            self.late_reverb_start =
                (peak_block + EARLY_REVERB_MIN_SIZE_BLOCKS + size_early_reverb)
                    .min(self.filter_length_blocks.saturating_sub(1));
            self.late_reverb_end = if self.estimation_region_candidate_size > 0 {
                self.block_to_analyze + self.estimation_region_candidate_size - 1
            } else {
                self.block_to_analyze
            }
            .min(self.filter_length_blocks.saturating_sub(1));
        } else {
            self.late_reverb_decay_estimator.reset(0);
            self.late_reverb_start = 0;
            self.late_reverb_end = 0;
        }

        self.estimation_region_identified = !(valid_filter && sufficient_reverb_decay);
        self.estimation_region_candidate_size = 0;
        self.smoothing_constant = 0.0;
        self.early_reverb_estimator.reset();
    }

    fn analyze_filter(&mut self, filter: &[f32]) {
        if self.block_to_analyze >= self.filter_length_blocks {
            return;
        }
        let start = self.block_to_analyze * FFT_LENGTH_BY_2;
        if start + FFT_LENGTH_BY_2 > filter.len() {
            return;
        }
        let h_block = &filter[start..start + FFT_LENGTH_BY_2];
        let mut h2 = [0.0f32; FFT_LENGTH_BY_2];
        for (dst, &value) in h2.iter_mut().zip(h_block.iter()) {
            *dst = value * value;
        }

        let (adapting, above_noise_floor) = analyze_block_gain(
            &h2,
            self.tail_gain,
            &mut self.previous_gains[self.block_to_analyze],
        );
        if adapting || !above_noise_floor {
            self.estimation_region_identified = true;
        } else if !self.estimation_region_identified {
            self.estimation_region_candidate_size += 1;
        }

        if self.block_to_analyze <= self.late_reverb_end {
            if self.block_to_analyze >= self.late_reverb_start {
                for &h2_k in &h2 {
                    let h2_log2 = fast_approx_log2f(h2_k + 1e-10);
                    self.late_reverb_decay_estimator.accumulate(h2_log2);
                    self.early_reverb_estimator
                        .accumulate(h2_log2, self.smoothing_constant);
                }
            } else {
                for &h2_k in &h2 {
                    let h2_log2 = fast_approx_log2f(h2_k + 1e-10);
                    self.early_reverb_estimator
                        .accumulate(h2_log2, self.smoothing_constant);
                }
            }
        }
    }
}

struct LateReverbLinearRegressor {
    nz: f32,
    nn: f32,
    count: f32,
    total_points: usize,
    accumulated_points: usize,
}

impl LateReverbLinearRegressor {
    fn new() -> Self {
        Self {
            nz: 0.0,
            nn: 0.0,
            count: 0.0,
            total_points: 0,
            accumulated_points: 0,
        }
    }

    fn reset(&mut self, num_data_points: usize) {
        self.total_points = num_data_points;
        self.accumulated_points = 0;
        self.nz = 0.0;
        if num_data_points == 0 {
            self.nn = 0.0;
            self.count = 0.0;
        } else {
            self.nn = symmetric_arithmetic_sum(num_data_points);
            self.count = -(num_data_points as f32) * 0.5 + 0.5;
        }
    }

    fn accumulate(&mut self, z: f32) {
        self.nz += self.count * z;
        self.count += 1.0;
        self.accumulated_points += 1;
    }

    fn estimate_available(&self) -> bool {
        self.total_points != 0 && self.accumulated_points == self.total_points
    }

    fn estimate(&self) -> f32 {
        if self.nn == 0.0 {
            0.0
        } else {
            self.nz / self.nn
        }
    }
}

struct EarlyReverbLengthEstimator {
    numerators_smooth: Vec<f32>,
    numerators: Vec<f32>,
    coefficients_counter: usize,
    block_counter: usize,
    n_sections: usize,
}

impl EarlyReverbLengthEstimator {
    fn new(max_blocks: usize) -> Self {
        let size = max_blocks.saturating_sub(BLOCKS_PER_SECTION);
        Self {
            numerators_smooth: vec![0.0; size],
            numerators: vec![0.0; size],
            coefficients_counter: 0,
            block_counter: 0,
            n_sections: 0,
        }
    }

    fn reset(&mut self) {
        self.coefficients_counter = 0;
        self.block_counter = 0;
        self.n_sections = 0;
        self.numerators.fill(0.0);
    }

    fn accumulate(&mut self, value: f32, smoothing: f32) {
        if self.numerators.is_empty() {
            self.advance_counters();
            return;
        }
        let first_section_index = self.block_counter.saturating_sub(BLOCKS_PER_SECTION - 1);
        let last_section_index = self
            .block_counter
            .min(self.numerators.len().saturating_sub(1));
        let x_value =
            self.coefficients_counter as f32 + EARLY_REVERB_FIRST_POINT_AT_LINEAR_REGRESSORS;
        let value_to_inc = FFT_LENGTH_BY_2 as f32 * value;
        let mut value_to_add =
            x_value * value + ((self.block_counter - last_section_index) as f32) * value_to_inc;
        for section in (first_section_index..=last_section_index).rev() {
            self.numerators[section] += value_to_add;
            value_to_add += value_to_inc;
        }

        if self.coefficients_counter + 1 == FFT_LENGTH_BY_2 {
            if self.block_counter + 1 >= BLOCKS_PER_SECTION {
                let section = self.block_counter + 1 - BLOCKS_PER_SECTION;
                if section < self.numerators.len() {
                    self.numerators_smooth[section] +=
                        smoothing * (self.numerators[section] - self.numerators_smooth[section]);
                    self.n_sections = section + 1;
                }
            }
            self.block_counter += 1;
            self.coefficients_counter = 0;
        } else {
            self.coefficients_counter += 1;
        }
    }

    fn estimate(&self) -> usize {
        const NUM_SECTIONS_TO_ANALYZE: usize = 9;
        if self.n_sections < NUM_SECTIONS_TO_ANALYZE
            || self.numerators_smooth.len() < NUM_SECTIONS_TO_ANALYZE
        {
            return 0;
        }
        let n = BLOCKS_PER_SECTION * FFT_LENGTH_BY_2;
        let nn = symmetric_arithmetic_sum(n);
        const NUMERATOR_11: f32 = 0.13750352374993502;
        const NUMERATOR_08: f32 = -0.32192809488736229;
        let numerator_11 = NUMERATOR_11 * nn / FFT_LENGTH_BY_2 as f32;
        let numerator_08 = NUMERATOR_08 * nn / FFT_LENGTH_BY_2 as f32;
        let min_tail = self.numerators_smooth[NUM_SECTIONS_TO_ANALYZE..self.n_sections]
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v));
        let mut early_reverb_size_minus1 = 0usize;
        for k in 0..NUM_SECTIONS_TO_ANALYZE {
            let value = self.numerators_smooth[k];
            if value > numerator_11 || (value < numerator_08 && value < 0.9 * min_tail) {
                early_reverb_size_minus1 = k;
            }
        }
        if early_reverb_size_minus1 == 0 {
            0
        } else {
            early_reverb_size_minus1 + 1
        }
    }

    fn dump(&self, dumper: &ApmDataDumper) {
        dumper.dump_raw_f32_slice("aec3_er_acum_numerator", &self.numerators_smooth);
    }

    fn advance_counters(&mut self) {
        if self.coefficients_counter + 1 == FFT_LENGTH_BY_2 {
            self.block_counter += 1;
            self.coefficients_counter = 0;
        } else {
            self.coefficients_counter += 1;
        }
    }
}

fn symmetric_arithmetic_sum(n: usize) -> f32 {
    let n_f = n as f32;
    n_f * (n_f * n_f - 1.0) / 12.0
}

fn block_energy_peak(filter: &[f32], block_index: usize) -> f32 {
    let start = block_index * FFT_LENGTH_BY_2;
    let end = start + FFT_LENGTH_BY_2;
    filter[start..end].iter().map(|v| v * v).fold(0.0, f32::max)
}

fn block_energy_average(filter: &[f32], block_index: usize) -> f32 {
    let start = block_index * FFT_LENGTH_BY_2;
    let end = start + FFT_LENGTH_BY_2;
    let sum: f32 = filter[start..end].iter().map(|v| v * v).sum();
    sum / FFT_LENGTH_BY_2 as f32
}

fn analyze_block_gain(
    block: &[f32; FFT_LENGTH_BY_2],
    floor_gain: f32,
    previous_gain: &mut f32,
) -> (bool, bool) {
    let gain = block.iter().copied().sum::<f32>() / FFT_LENGTH_BY_2 as f32;
    let gain = gain.max(1e-32);
    let block_adapting = *previous_gain > 1.1 * gain || *previous_gain < 0.9 * gain;
    let decaying_gain = gain > floor_gain;
    *previous_gain = gain;
    (block_adapting, decaying_gain)
}
