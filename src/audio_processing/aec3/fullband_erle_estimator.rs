use crate::api::config::Erle as ErleConfig;
use crate::audio_processing::aec3::aec3_common::fast_approx_log2f;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;

const EPSILON: f32 = 1e-3;
const X2_BAND_ENERGY_THRESHOLD: f32 = 44_015_068.0;
const BLOCKS_TO_HOLD_ERLE: i32 = 100;
const POINTS_TO_ACCUMULATE: i32 = 6;

pub struct FullBandErleEstimator {
    min_erle_log2: f32,
    max_erle_lf_log2: f32,
    hold_counters_time_domain: Vec<i32>,
    erle_time_domain_log2: Vec<f32>,
    instantaneous_erle: Vec<ErleInstantaneous>,
    linear_filters_qualities: Vec<Option<f32>>,
}

impl FullBandErleEstimator {
    pub fn new(config: &ErleConfig, num_capture_channels: usize) -> Self {
        let min_erle_log2 = fast_approx_log2f(config.min + EPSILON);
        let max_erle_lf_log2 = fast_approx_log2f(config.max_l + EPSILON);
        let mut estimator = Self {
            min_erle_log2,
            max_erle_lf_log2,
            hold_counters_time_domain: vec![0; num_capture_channels],
            erle_time_domain_log2: vec![min_erle_log2; num_capture_channels],
            instantaneous_erle: vec![ErleInstantaneous::new(config); num_capture_channels],
            linear_filters_qualities: vec![None; num_capture_channels],
        };
        estimator.reset();
        estimator
    }

    pub fn reset(&mut self) {
        for inst in &mut self.instantaneous_erle {
            inst.reset();
        }
        self.update_quality_estimates();
        for value in &mut self.erle_time_domain_log2 {
            *value = self.min_erle_log2;
        }
        for counter in &mut self.hold_counters_time_domain {
            *counter = 0;
        }
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

        let x2_sum: f32 = x2.iter().sum();

        for (ch, converged) in converged_filters.iter().enumerate() {
            if *converged {
                if x2_sum > X2_BAND_ENERGY_THRESHOLD * x2.len() as f32 {
                    let y2_sum: f32 = y2[ch].iter().sum();
                    let e2_sum: f32 = e2[ch].iter().sum();
                    if self.instantaneous_erle[ch].update(y2_sum, e2_sum) {
                        self.hold_counters_time_domain[ch] = BLOCKS_TO_HOLD_ERLE;
                        if let Some(inst_erle) = self.instantaneous_erle[ch].inst_erle_log2() {
                            self.erle_time_domain_log2[ch] +=
                                0.1 * (inst_erle - self.erle_time_domain_log2[ch]);
                            self.erle_time_domain_log2[ch] = self.erle_time_domain_log2[ch]
                                .clamp(self.min_erle_log2, self.max_erle_lf_log2);
                        }
                    }
                }
            }

            self.hold_counters_time_domain[ch] -= 1;
            if self.hold_counters_time_domain[ch] <= 0 {
                self.erle_time_domain_log2[ch] =
                    (self.erle_time_domain_log2[ch] - 0.044).max(self.min_erle_log2);
            }
            if self.hold_counters_time_domain[ch] == 0 {
                self.instantaneous_erle[ch].reset_accumulators();
            }
        }

        self.update_quality_estimates();
    }

    pub fn fullband_erle_log2(&self) -> f32 {
        self.erle_time_domain_log2
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(self.min_erle_log2)
    }

    pub fn get_linear_quality_estimates(&self) -> &[Option<f32>] {
        &self.linear_filters_qualities
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        dumper.dump_raw_f32("aec3_fullband_erle_log2", self.fullband_erle_log2());
        if let Some(inst) = self.instantaneous_erle.first() {
            inst.dump(dumper);
        }
    }

    fn update_quality_estimates(&mut self) {
        for (quality, inst) in self
            .linear_filters_qualities
            .iter_mut()
            .zip(self.instantaneous_erle.iter())
        {
            *quality = inst.quality_estimate();
        }
    }
}

#[derive(Clone)]
struct ErleInstantaneous {
    clamp_quality_to_zero: bool,
    clamp_quality_to_one: bool,
    erle_log2: Option<f32>,
    inst_quality_estimate: f32,
    max_erle_log2: f32,
    min_erle_log2: f32,
    y2_accum: f32,
    e2_accum: f32,
    num_points: i32,
}

impl ErleInstantaneous {
    fn new(config: &ErleConfig) -> Self {
        let mut instance = Self {
            clamp_quality_to_zero: config.clamp_quality_estimate_to_zero,
            clamp_quality_to_one: config.clamp_quality_estimate_to_one,
            erle_log2: None,
            inst_quality_estimate: 0.0,
            max_erle_log2: -10.0,
            min_erle_log2: 33.0,
            y2_accum: 0.0,
            e2_accum: 0.0,
            num_points: 0,
        };
        instance.reset();
        instance
    }

    fn reset(&mut self) {
        self.reset_accumulators();
        self.max_erle_log2 = -10.0;
        self.min_erle_log2 = 33.0;
        self.inst_quality_estimate = 0.0;
    }

    fn reset_accumulators(&mut self) {
        self.erle_log2 = None;
        self.inst_quality_estimate = 0.0;
        self.num_points = 0;
        self.e2_accum = 0.0;
        self.y2_accum = 0.0;
    }

    fn update(&mut self, y2_sum: f32, e2_sum: f32) -> bool {
        self.e2_accum += e2_sum;
        self.y2_accum += y2_sum;
        self.num_points += 1;
        let mut updated = false;
        if self.num_points == POINTS_TO_ACCUMULATE {
            if self.e2_accum > 0.0 {
                self.erle_log2 = Some(fast_approx_log2f(self.y2_accum / self.e2_accum + EPSILON));
                updated = true;
            }
            self.num_points = 0;
            self.e2_accum = 0.0;
            self.y2_accum = 0.0;
        }
        if updated {
            self.update_max_min();
            self.update_quality_estimate();
        }
        updated
    }

    fn inst_erle_log2(&self) -> Option<f32> {
        self.erle_log2
    }

    fn quality_estimate(&self) -> Option<f32> {
        self.erle_log2.map(|_| {
            let mut value = self.inst_quality_estimate;
            if self.clamp_quality_to_zero {
                value = value.max(0.0);
            }
            if self.clamp_quality_to_one {
                value = value.min(1.0);
            }
            value
        })
    }

    fn dump(&self, dumper: &ApmDataDumper) {
        dumper.dump_raw_f32(
            "aec3_fullband_erle_inst_log2",
            self.erle_log2.unwrap_or(-10.0),
        );
        dumper.dump_raw_f32(
            "aec3_erle_instantaneous_quality",
            self.quality_estimate().unwrap_or(0.0),
        );
        dumper.dump_raw_f32("aec3_fullband_erle_max_log2", self.max_erle_log2);
        dumper.dump_raw_f32("aec3_fullband_erle_min_log2", self.min_erle_log2);
    }

    fn update_max_min(&mut self) {
        if let Some(value) = self.erle_log2 {
            if value > self.max_erle_log2 {
                self.max_erle_log2 = value;
            } else {
                self.max_erle_log2 -= 0.0004;
            }
            if value < self.min_erle_log2 {
                self.min_erle_log2 = value;
            } else {
                self.min_erle_log2 += 0.0004;
            }
        }
    }

    fn update_quality_estimate(&mut self) {
        if let Some(value) = self.erle_log2 {
            let mut quality = 0.0;
            if self.max_erle_log2 > self.min_erle_log2 {
                quality = (value - self.min_erle_log2) / (self.max_erle_log2 - self.min_erle_log2);
            }
            if quality > self.inst_quality_estimate {
                self.inst_quality_estimate = quality;
            } else {
                self.inst_quality_estimate += 0.07 * (quality - self.inst_quality_estimate);
            }
        }
    }
}
