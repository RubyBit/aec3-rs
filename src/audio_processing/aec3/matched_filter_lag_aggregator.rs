use crate::api::config::{Delay, DelaySelectionThresholds};
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE_LOG2, MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS, NUM_BLOCKS_PER_SECOND,
};
use crate::audio_processing::aec3::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::audio_processing::aec3::matched_filter::LagEstimate;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};

const HISTOGRAM_DATA_SIZE: usize = 250;
const PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED: i32 = -1;

/// Number of bits to shift a downsampled lag by to get a block index.
fn down_sampling_block_size_log2(down_sampling_factor: usize) -> u32 {
    let down_sampling_factor_log2 = down_sampling_factor.max(1).ilog2();
    (BLOCK_SIZE_LOG2 as u32).saturating_sub(down_sampling_factor_log2)
}

/// Tracks the most frequent lag over the recent history.
struct HighestPeakAggregator {
    histogram: Vec<i32>,
    histogram_data: [usize; HISTOGRAM_DATA_SIZE],
    histogram_data_index: usize,
    candidate: i32,
}

impl HighestPeakAggregator {
    fn new(max_filter_lag: usize) -> Self {
        Self {
            histogram: vec![0; max_filter_lag + 1],
            histogram_data: [0; HISTOGRAM_DATA_SIZE],
            histogram_data_index: 0,
            candidate: -1,
        }
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data.fill(0);
        self.histogram_data_index = 0;
    }

    fn aggregate(&mut self, lag: usize) {
        let lag = lag.min(self.histogram.len() - 1);
        self.histogram[self.histogram_data[self.histogram_data_index]] -= 1;
        self.histogram_data[self.histogram_data_index] = lag;
        self.histogram[lag] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % HISTOGRAM_DATA_SIZE;
        self.candidate = self
            .histogram
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .map(|(idx, _)| idx as i32)
            .unwrap_or(-1);
    }

    fn candidate(&self) -> i32 {
        self.candidate
    }

    fn histogram(&self) -> &[i32] {
        &self.histogram
    }
}

/// Tracks the most frequent pre-echo lag, in blocks, over the recent history.
struct PreEchoLagAggregator {
    block_size_log2: u32,
    histogram_data: [i32; HISTOGRAM_DATA_SIZE],
    histogram: Vec<i32>,
    histogram_data_index: usize,
    pre_echo_candidate: i32,
    number_updates: usize,
}

impl PreEchoLagAggregator {
    fn new(max_filter_lag: usize, down_sampling_factor: usize) -> Self {
        let histogram_len = ((max_filter_lag + 1) * down_sampling_factor) >> BLOCK_SIZE_LOG2;
        let mut instance = Self {
            block_size_log2: down_sampling_block_size_log2(down_sampling_factor),
            histogram_data: [PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED; HISTOGRAM_DATA_SIZE],
            histogram: vec![0; histogram_len.max(1)],
            histogram_data_index: 0,
            pre_echo_candidate: 0,
            number_updates: 0,
        };
        instance.reset();
        instance
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data
            .fill(PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED);
        self.histogram_data_index = 0;
        self.pre_echo_candidate = 0;
    }

    fn aggregate(&mut self, pre_echo_lag: usize) {
        let pre_echo_block_size = ((pre_echo_lag >> self.block_size_log2) as i32)
            .clamp(0, self.histogram.len() as i32 - 1);

        // Drop the oldest point, skipping the initial unfilled entries.
        let oldest = self.histogram_data[self.histogram_data_index];
        if oldest != PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED {
            self.histogram[oldest as usize] -= 1;
        }
        self.histogram_data[self.histogram_data_index] = pre_echo_block_size;
        self.histogram[pre_echo_block_size as usize] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % HISTOGRAM_DATA_SIZE;

        let mut pre_echo_candidate_block_size = 0usize;
        if self.number_updates < NUM_BLOCKS_PER_SECOND * 2 {
            self.number_updates += 1;
            // Favour earlier windows while the histogram is still filling.
            let mut penalization_per_delay = 1.0f32;
            let mut max_histogram_value = -1.0f32;
            let mut window_start = 0usize;
            while window_start + MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS <= self.histogram.len() {
                let window = &self.histogram
                    [window_start..window_start + MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS];
                let (offset, &value) = window
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, value)| *value)
                    .expect("non-empty window");
                let weighted_max_value = value as f32 * penalization_per_delay;
                if weighted_max_value > max_histogram_value {
                    max_histogram_value = weighted_max_value;
                    pre_echo_candidate_block_size = window_start + offset;
                }
                penalization_per_delay *= 0.7;
                window_start += MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS;
            }
        } else {
            pre_echo_candidate_block_size = self
                .histogram
                .iter()
                .enumerate()
                .max_by_key(|(_, value)| *value)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }
        self.pre_echo_candidate = (pre_echo_candidate_block_size << self.block_size_log2) as i32;
    }

    fn pre_echo_candidate(&self) -> i32 {
        self.pre_echo_candidate
    }

    fn dump(&self, dumper: &ApmDataDumper) {
        dumper.dump_raw_i32(
            DiagnosticLevel::DeepDebug,
            "aec3_pre_echo_delay_candidate",
            self.pre_echo_candidate,
        );
    }
}

pub struct MatchedFilterLagAggregator {
    data_dumper: ApmDataDumper,
    significant_candidate_found: bool,
    thresholds: DelaySelectionThresholds,
    headroom: usize,
    highest_peak_aggregator: HighestPeakAggregator,
    pre_echo_lag_aggregator: Option<PreEchoLagAggregator>,
}

impl MatchedFilterLagAggregator {
    pub fn new(data_dumper: ApmDataDumper, max_filter_lag: usize, delay_config: &Delay) -> Self {
        assert!(delay_config.down_sampling_factor > 0);
        assert!(
            delay_config.delay_selection_thresholds.initial
                <= delay_config.delay_selection_thresholds.converged
        );
        Self {
            data_dumper,
            significant_candidate_found: false,
            thresholds: delay_config.delay_selection_thresholds.clone(),
            headroom: delay_config.delay_headroom_samples / delay_config.down_sampling_factor,
            highest_peak_aggregator: HighestPeakAggregator::new(max_filter_lag),
            pre_echo_lag_aggregator: delay_config.detect_pre_echo.then(|| {
                PreEchoLagAggregator::new(max_filter_lag, delay_config.down_sampling_factor)
            }),
        }
    }

    pub fn reset(&mut self, hard_reset: bool) {
        self.highest_peak_aggregator.reset();
        if let Some(aggregator) = self.pre_echo_lag_aggregator.as_mut() {
            aggregator.reset();
        }
        if hard_reset {
            self.significant_candidate_found = false;
        }
    }

    /// Whether a delay candidate has been seen often enough to be trusted.
    pub fn reliable_delay_found(&self) -> bool {
        self.significant_candidate_found
    }

    /// The delay taken from the highest peak of the matched filters, ignoring
    /// any pre-echo.
    pub fn delay_at_highest_peak(&self) -> i32 {
        self.highest_peak_aggregator.candidate()
    }

    pub fn aggregate(&mut self, lag_estimate: Option<LagEstimate>) -> Option<DelayEstimate> {
        let lag_estimate = lag_estimate?;

        if let Some(aggregator) = self.pre_echo_lag_aggregator.as_mut() {
            aggregator.dump(&self.data_dumper);
            aggregator.aggregate(lag_estimate.pre_echo_lag.saturating_sub(self.headroom));
        }

        self.highest_peak_aggregator
            .aggregate(lag_estimate.lag.saturating_sub(self.headroom));

        self.data_dumper.dump_raw_i32_slice(
            DiagnosticLevel::DeepDebug,
            "aec3_echo_path_delay_estimator_histogram",
            self.highest_peak_aggregator.histogram(),
        );

        let candidate = self.highest_peak_aggregator.candidate();
        let count = self.highest_peak_aggregator.histogram()[candidate as usize];
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
            let reported_delay = match self.pre_echo_lag_aggregator.as_ref() {
                Some(aggregator) => aggregator.pre_echo_candidate() as usize,
                None => candidate as usize,
            };
            return Some(DelayEstimate::new(quality, reported_delay));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;

    const NUM_LAGS_BEFORE_DETECTION: usize = 26;

    /// A realistic `max_filter_lag`, so the pre-echo histogram is long enough
    /// for the windowed search to run.
    const MAX_FILTER_LAG: usize = 2432;

    #[test]
    fn lag_estimate_invariance_required() {
        let config = EchoCanceller3Config::default();
        let mut aggregator =
            MatchedFilterLagAggregator::new(ApmDataDumper::new_unique(), 100, &config.delay);

        let mut aggregated = None;
        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            aggregated = aggregator.aggregate(Some(LagEstimate::new(10, 10)));
        }
        assert!(aggregated.is_some());

        for k in 0..(NUM_LAGS_BEFORE_DETECTION * 100) {
            aggregated = aggregator.aggregate(Some(LagEstimate::new(k % 100, k % 100)));
        }
        assert!(aggregated.is_none());

        for k in 0..(NUM_LAGS_BEFORE_DETECTION * 100) {
            assert!(
                aggregator
                    .aggregate(Some(LagEstimate::new(k % 100, k % 100)))
                    .is_none()
            );
        }
    }

    #[test]
    fn no_estimate_without_a_lag() {
        let config = EchoCanceller3Config::default();
        let mut aggregator =
            MatchedFilterLagAggregator::new(ApmDataDumper::new_unique(), 100, &config.delay);
        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            assert!(aggregator.aggregate(None).is_none());
        }
    }

    /// With pre-echo detection the reported delay follows the pre-echo lag;
    /// without it, the highest matched filter peak.
    #[test]
    fn pre_echo_detection_selects_the_reported_delay() {
        const LAG: usize = 1000;
        const PRE_ECHO_LAG: usize = 200;

        let report = |detect_pre_echo: bool| -> usize {
            let mut config = EchoCanceller3Config::default();
            config.delay.detect_pre_echo = detect_pre_echo;
            let headroom = config.delay.delay_headroom_samples / config.delay.down_sampling_factor;
            let mut aggregator = MatchedFilterLagAggregator::new(
                ApmDataDumper::new_unique(),
                MAX_FILTER_LAG,
                &config.delay,
            );
            let mut aggregated = None;
            for _ in 0..NUM_LAGS_BEFORE_DETECTION {
                aggregated = aggregator.aggregate(Some(LagEstimate::new(LAG, PRE_ECHO_LAG)));
            }
            let _ = headroom;
            aggregated.expect("a delay should be reported").delay
        };

        let headroom = EchoCanceller3Config::default().delay.delay_headroom_samples
            / EchoCanceller3Config::default().delay.down_sampling_factor;

        // Without pre-echo detection the highest peak is reported directly.
        assert_eq!(LAG - headroom, report(false));

        // With it, the pre-echo lag is reported, quantised to the block grid
        // the pre-echo histogram uses.
        let reported = report(true);
        assert!(
            reported < LAG - headroom,
            "pre-echo delay {reported} should precede the main peak"
        );
        assert!(
            reported.abs_diff(PRE_ECHO_LAG - headroom) <= 16,
            "pre-echo delay {reported} should be near {}",
            PRE_ECHO_LAG - headroom
        );
    }
}
