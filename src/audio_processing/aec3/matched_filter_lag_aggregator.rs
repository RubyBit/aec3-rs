use crate::api::config::DelaySelectionThresholds;
use crate::audio_processing::aec3::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::audio_processing::aec3::matched_filter::LagEstimate;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

const HISTOGRAM_DATA_SIZE: usize = 250;

pub struct MatchedFilterLagAggregator {
    data_dumper: ApmDataDumper,
    histogram: Vec<i32>,
    histogram_data: [usize; HISTOGRAM_DATA_SIZE],
    histogram_data_index: usize,
    significant_candidate_found: bool,
    thresholds: DelaySelectionThresholds,
}

impl MatchedFilterLagAggregator {
    pub fn new(
        data_dumper: ApmDataDumper,
        max_filter_lag: usize,
        thresholds: DelaySelectionThresholds,
    ) -> Self {
        Self {
            data_dumper,
            histogram: vec![0; max_filter_lag + 1],
            histogram_data: [0; HISTOGRAM_DATA_SIZE],
            histogram_data_index: 0,
            significant_candidate_found: false,
            thresholds,
        }
    }

    pub fn reset(&mut self, hard_reset: bool) {
        self.histogram.fill(0);
        self.histogram_data.fill(0);
        self.histogram_data_index = 0;
        if hard_reset {
            self.significant_candidate_found = false;
        }
    }

    pub fn aggregate(&mut self, lag_estimates: &[LagEstimate]) -> Option<DelayEstimate> {
        let mut best_accuracy = 0.0f32;
        let mut best_lag_index: Option<usize> = None;
        for (idx, estimate) in lag_estimates.iter().enumerate() {
            if estimate.updated && estimate.reliable && estimate.accuracy > best_accuracy {
                best_accuracy = estimate.accuracy;
                best_lag_index = Some(idx);
            }
        }

        self.data_dumper.dump_raw_i32(
            "aec3_echo_path_delay_estimator_best_index",
            best_lag_index.map(|i| i as i32).unwrap_or(-1),
        );
        self.data_dumper
            .dump_raw_i32_slice("aec3_echo_path_delay_estimator_histogram", &self.histogram);

        let best_idx = match best_lag_index {
            Some(idx) => idx,
            None => return None,
        };

        // Remove the oldest histogram contribution.
        let old_lag = self.histogram_data[self.histogram_data_index];
        if old_lag < self.histogram.len() {
            self.histogram[old_lag] -= 1;
        }

        // Insert the new candidate lag.
        let mut lag = lag_estimates[best_idx].lag;
        if lag >= self.histogram.len() {
            lag = self.histogram.len() - 1;
        }
        self.histogram_data[self.histogram_data_index] = lag;
        self.histogram[lag] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % HISTOGRAM_DATA_SIZE;

        let (candidate, &count) = self
            .histogram
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .unwrap();

        if count > self.thresholds.converged {
            self.significant_candidate_found = true;
        }

        if count > self.thresholds.converged
            || (count > self.thresholds.initial && !self.significant_candidate_found)
        {
            let quality = if self.significant_candidate_found {
                DelayEstimateQuality::Refined
            } else {
                DelayEstimateQuality::Coarse
            };
            return Some(DelayEstimate::new(quality, candidate));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;

    const NUM_LAGS_BEFORE_DETECTION: usize = 26;

    #[test]
    fn most_accurate_lag_chosen() {
        let config = EchoCanceller3Config::default();
        let thresholds = config.delay.delay_selection_thresholds.clone();
        let mut aggregator =
            MatchedFilterLagAggregator::new(ApmDataDumper::new_unique(), 10, thresholds);
        let mut lag_estimates = vec![
            LagEstimate::new(1.0, true, 5, true),
            LagEstimate::new(0.5, true, 10, true),
        ];

        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            aggregator.aggregate(&lag_estimates);
        }

        let mut aggregated = aggregator.aggregate(&lag_estimates);
        assert!(aggregated.is_some());
        assert_eq!(5, aggregated.unwrap().delay);

        lag_estimates[0] = LagEstimate::new(0.5, true, 5, true);
        lag_estimates[1] = LagEstimate::new(1.0, true, 10, true);

        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            let result = aggregator.aggregate(&lag_estimates);
            assert!(result.is_some());
            assert_eq!(5, result.unwrap().delay);
        }

        aggregator.aggregate(&lag_estimates);
        aggregated = aggregator.aggregate(&lag_estimates);
        assert!(aggregated.is_some());
        assert_eq!(10, aggregated.unwrap().delay);
    }

    #[test]
    fn lag_estimate_invariance_required() {
        let config = EchoCanceller3Config::default();
        let thresholds = config.delay.delay_selection_thresholds.clone();
        let mut aggregator =
            MatchedFilterLagAggregator::new(ApmDataDumper::new_unique(), 100, thresholds);

        let mut lag_estimates = vec![LagEstimate::new(1.0, true, 10, true)];
        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            aggregator.aggregate(&lag_estimates);
        }
        let mut aggregated = aggregator.aggregate(&lag_estimates);
        assert!(aggregated.is_some());

        for k in 0..(NUM_LAGS_BEFORE_DETECTION * 100) {
            lag_estimates[0] = LagEstimate::new(1.0, true, k % 100, true);
            aggregated = aggregator.aggregate(&lag_estimates);
        }
        assert!(aggregated.is_none());

        for k in 0..(NUM_LAGS_BEFORE_DETECTION * 100) {
            lag_estimates[0] = LagEstimate::new(1.0, true, k % 100, true);
            let result = aggregator.aggregate(&lag_estimates);
            assert!(result.is_none());
        }
    }
}
