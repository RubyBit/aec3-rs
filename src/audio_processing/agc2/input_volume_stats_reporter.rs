//! Input volume stats reporter mirrored from WebRTC AGC2.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

const FRAMES_IN_60_SECONDS: i32 = 6000;
const MIN_INPUT_VOLUME: i32 = 0;
const MAX_INPUT_VOLUME: i32 = 255;
const MAX_UPDATE: i32 = MAX_INPUT_VOLUME - MIN_INPUT_VOLUME;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputVolumeType {
    Applied,
    Recommended,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VolumeUpdateStats {
    pub num_decreases: i32,
    pub num_increases: i32,
    pub sum_decreases: i32,
    pub sum_increases: i32,
}

#[derive(Debug, Clone, Default)]
pub struct HistogramSamples {
    samples: BTreeMap<i32, i32>,
}

impl HistogramSamples {
    fn add(&mut self, value: i32) {
        *self.samples.entry(value).or_insert(0) += 1;
    }

    #[cfg(test)]
    fn as_pairs(&self) -> Vec<(i32, i32)> {
        self.samples.iter().map(|(k, v)| (*k, *v)).collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Histograms {
    pub on_volume_change: HistogramSamples,
    pub decrease_rate: HistogramSamples,
    pub decrease_average: HistogramSamples,
    pub increase_rate: HistogramSamples,
    pub increase_average: HistogramSamples,
    pub update_rate: HistogramSamples,
    pub update_average: HistogramSamples,
}

pub struct InputVolumeStatsReporter {
    volume_update_stats: VolumeUpdateStats,
    histograms: Histograms,
    cannot_log_stats: bool,
    log_volume_update_stats_counter: i32,
    previous_input_volume: Option<i32>,
    _input_volume_type: InputVolumeType,
}

impl InputVolumeStatsReporter {
    pub fn new(input_volume_type: InputVolumeType) -> Self {
        Self {
            volume_update_stats: VolumeUpdateStats::default(),
            histograms: Histograms::default(),
            cannot_log_stats: false,
            log_volume_update_stats_counter: 0,
            previous_input_volume: None,
            _input_volume_type: input_volume_type,
        }
    }

    pub fn update_statistics(&mut self, input_volume: i32) {
        if self.cannot_log_stats {
            return;
        }

        assert!((MIN_INPUT_VOLUME..=MAX_INPUT_VOLUME).contains(&input_volume));

        if let Some(previous) = self.previous_input_volume
            && input_volume != previous
        {
            self.histograms.on_volume_change.add(input_volume);

            let volume_change = input_volume - previous;
            if volume_change < 0 {
                self.volume_update_stats.num_decreases += 1;
                self.volume_update_stats.sum_decreases -= volume_change;
            } else {
                self.volume_update_stats.num_increases += 1;
                self.volume_update_stats.sum_increases += volume_change;
            }
        }

        self.log_volume_update_stats_counter += 1;
        if self.log_volume_update_stats_counter >= FRAMES_IN_60_SECONDS {
            self.log_volume_update_stats();
            self.volume_update_stats = VolumeUpdateStats::default();
            self.log_volume_update_stats_counter = 0;
        }

        self.previous_input_volume = Some(input_volume);
    }

    fn log_volume_update_stats(&mut self) {
        self.histograms
            .decrease_rate
            .add(self.volume_update_stats.num_decreases);
        if self.volume_update_stats.num_decreases > 0 {
            let average_decrease = compute_average_update(
                self.volume_update_stats.sum_decreases,
                self.volume_update_stats.num_decreases,
            );
            self.histograms.decrease_average.add(average_decrease);
        }

        self.histograms
            .increase_rate
            .add(self.volume_update_stats.num_increases);
        if self.volume_update_stats.num_increases > 0 {
            let average_increase = compute_average_update(
                self.volume_update_stats.sum_increases,
                self.volume_update_stats.num_increases,
            );
            self.histograms.increase_average.add(average_increase);
        }

        let num_updates =
            self.volume_update_stats.num_decreases + self.volume_update_stats.num_increases;
        self.histograms.update_rate.add(num_updates);
        if num_updates > 0 {
            let average_update = compute_average_update(
                self.volume_update_stats.sum_decreases + self.volume_update_stats.sum_increases,
                num_updates,
            );
            self.histograms.update_average.add(average_update);
        }
    }

    pub fn volume_update_stats(&self) -> VolumeUpdateStats {
        self.volume_update_stats
    }

    #[cfg(test)]
    fn histograms(&self) -> &Histograms {
        &self.histograms
    }
}

fn compute_average_update(sum_updates: i32, num_updates: i32) -> i32 {
    assert!(sum_updates >= 0);
    assert!(sum_updates <= MAX_UPDATE * FRAMES_IN_60_SECONDS);
    assert!(num_updates >= 0);
    assert!(num_updates <= FRAMES_IN_60_SECONDS);
    if num_updates == 0 {
        return 0;
    }
    (sum_updates as f32 / num_updates as f32).round() as i32
}

static ON_CHANGE_TO_MATCH_TARGET: OnceLock<Mutex<BTreeMap<i32, i32>>> = OnceLock::new();

pub fn update_histogram_on_recommended_input_volume_change_to_match_target(volume: i32) {
    assert!((1..=MAX_INPUT_VOLUME).contains(&volume));
    let store = ON_CHANGE_TO_MATCH_TARGET.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(mut map) = store.lock() {
        *map.entry(volume).or_insert(0) += 1;
    }
}

#[cfg(test)]
pub(crate) fn on_change_to_match_target_samples() -> Vec<(i32, i32)> {
    ON_CHANGE_TO_MATCH_TARGET
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock")
        .iter()
        .map(|(k, v)| (*k, *v))
        .collect()
}

#[cfg(test)]
pub(crate) fn reset_on_change_to_match_target_samples() {
    ON_CHANGE_TO_MATCH_TARGET
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock")
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_for_both_types(test_fn: fn(InputVolumeType)) {
        test_fn(InputVolumeType::Applied);
        test_fn(InputVolumeType::Recommended);
    }

    #[test]
    fn check_volume_on_change_is_empty() {
        run_for_both_types(|input_volume_type| {
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            stats_reporter.update_statistics(10);
            assert_eq!(stats_reporter.histograms().on_volume_change.as_pairs(), vec![]);
        });
    }

    #[test]
    fn check_rate_average_stats_empty() {
        run_for_both_types(|input_volume_type| {
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            const INPUT_VOLUME: i32 = 10;
            stats_reporter.update_statistics(INPUT_VOLUME);
            for _ in (0..FRAMES_IN_60_SECONDS - 2).step_by(2) {
                stats_reporter.update_statistics(INPUT_VOLUME + 2);
                stats_reporter.update_statistics(INPUT_VOLUME);
            }

            assert_eq!(stats_reporter.histograms().update_rate.as_pairs(), vec![]);
            assert_eq!(stats_reporter.histograms().decrease_rate.as_pairs(), vec![]);
            assert_eq!(stats_reporter.histograms().increase_rate.as_pairs(), vec![]);
            assert_eq!(stats_reporter.histograms().update_average.as_pairs(), vec![]);
            assert_eq!(stats_reporter.histograms().decrease_average.as_pairs(), vec![]);
            assert_eq!(stats_reporter.histograms().increase_average.as_pairs(), vec![]);
        });
    }

    #[test]
    fn check_samples() {
        run_for_both_types(|input_volume_type| {
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);

            const INPUT_VOLUME_1: i32 = 10;
            stats_reporter.update_statistics(INPUT_VOLUME_1);
            const INPUT_VOLUME_2: i32 = 12;
            for _ in (0..FRAMES_IN_60_SECONDS).step_by(2) {
                stats_reporter.update_statistics(INPUT_VOLUME_2);
                stats_reporter.update_statistics(INPUT_VOLUME_1);
            }

            const INPUT_VOLUME_3: i32 = 13;
            for _ in (0..FRAMES_IN_60_SECONDS).step_by(2) {
                stats_reporter.update_statistics(INPUT_VOLUME_3);
                stats_reporter.update_statistics(INPUT_VOLUME_1);
            }

            assert_eq!(
                stats_reporter.histograms().on_volume_change.as_pairs(),
                vec![
                    (INPUT_VOLUME_1, FRAMES_IN_60_SECONDS),
                    (INPUT_VOLUME_2, FRAMES_IN_60_SECONDS / 2),
                    (INPUT_VOLUME_3, FRAMES_IN_60_SECONDS / 2)
                ]
            );

            assert_eq!(
                stats_reporter.histograms().update_rate.as_pairs(),
                vec![(FRAMES_IN_60_SECONDS - 1, 1), (FRAMES_IN_60_SECONDS, 1)]
            );
            assert_eq!(
                stats_reporter.histograms().decrease_rate.as_pairs(),
                vec![(FRAMES_IN_60_SECONDS / 2 - 1, 1), (FRAMES_IN_60_SECONDS / 2, 1)]
            );
            assert_eq!(
                stats_reporter.histograms().increase_rate.as_pairs(),
                vec![(FRAMES_IN_60_SECONDS / 2, 2)]
            );

            assert_eq!(
                stats_reporter.histograms().update_average.as_pairs(),
                vec![(2, 1), (3, 1)]
            );
            assert_eq!(
                stats_reporter.histograms().decrease_average.as_pairs(),
                vec![(2, 1), (3, 1)]
            );
            assert_eq!(
                stats_reporter.histograms().increase_average.as_pairs(),
                vec![(2, 1), (3, 1)]
            );
        });
    }

    #[test]
    fn check_volume_update_stats_for_empty_stats() {
        run_for_both_types(|input_volume_type| {
            let stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            let update_stats = stats_reporter.volume_update_stats();
            assert_eq!(update_stats.num_decreases, 0);
            assert_eq!(update_stats.sum_decreases, 0);
            assert_eq!(update_stats.num_increases, 0);
            assert_eq!(update_stats.sum_increases, 0);
        });
    }

    #[test]
    fn check_volume_update_stats_after_no_volume_change() {
        run_for_both_types(|input_volume_type| {
            const INPUT_VOLUME: i32 = 10;
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            stats_reporter.update_statistics(INPUT_VOLUME);
            stats_reporter.update_statistics(INPUT_VOLUME);
            stats_reporter.update_statistics(INPUT_VOLUME);
            let update_stats = stats_reporter.volume_update_stats();
            assert_eq!(update_stats.num_decreases, 0);
            assert_eq!(update_stats.sum_decreases, 0);
            assert_eq!(update_stats.num_increases, 0);
            assert_eq!(update_stats.sum_increases, 0);
        });
    }

    #[test]
    fn check_volume_update_stats_after_volume_increase() {
        run_for_both_types(|input_volume_type| {
            const INPUT_VOLUME: i32 = 10;
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            stats_reporter.update_statistics(INPUT_VOLUME);
            stats_reporter.update_statistics(INPUT_VOLUME + 4);
            stats_reporter.update_statistics(INPUT_VOLUME + 5);
            let update_stats = stats_reporter.volume_update_stats();
            assert_eq!(update_stats.num_decreases, 0);
            assert_eq!(update_stats.sum_decreases, 0);
            assert_eq!(update_stats.num_increases, 2);
            assert_eq!(update_stats.sum_increases, 5);
        });
    }

    #[test]
    fn check_volume_update_stats_after_volume_decrease() {
        run_for_both_types(|input_volume_type| {
            const INPUT_VOLUME: i32 = 10;
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            stats_reporter.update_statistics(INPUT_VOLUME);
            stats_reporter.update_statistics(INPUT_VOLUME - 4);
            stats_reporter.update_statistics(INPUT_VOLUME - 5);
            let update_stats = stats_reporter.volume_update_stats();
            assert_eq!(update_stats.num_decreases, 2);
            assert_eq!(update_stats.sum_decreases, 5);
            assert_eq!(update_stats.num_increases, 0);
            assert_eq!(update_stats.sum_increases, 0);
        });
    }

    #[test]
    fn check_volume_update_stats_after_reset() {
        run_for_both_types(|input_volume_type| {
            let mut stats_reporter = InputVolumeStatsReporter::new(input_volume_type);
            const INPUT_VOLUME: i32 = 10;
            stats_reporter.update_statistics(INPUT_VOLUME);
            for _ in (0..FRAMES_IN_60_SECONDS - 2).step_by(2) {
                stats_reporter.update_statistics(INPUT_VOLUME + 2);
                stats_reporter.update_statistics(INPUT_VOLUME);
            }

            let stats_before_reset = stats_reporter.volume_update_stats();
            assert_eq!(stats_before_reset.num_decreases, FRAMES_IN_60_SECONDS / 2 - 1);
            assert_eq!(stats_before_reset.sum_decreases, FRAMES_IN_60_SECONDS - 2);
            assert_eq!(stats_before_reset.num_increases, FRAMES_IN_60_SECONDS / 2 - 1);
            assert_eq!(stats_before_reset.sum_increases, FRAMES_IN_60_SECONDS - 2);

            stats_reporter.update_statistics(INPUT_VOLUME + 2);
            let stats_during_reset = stats_reporter.volume_update_stats();
            assert_eq!(stats_during_reset.num_decreases, 0);
            assert_eq!(stats_during_reset.sum_decreases, 0);
            assert_eq!(stats_during_reset.num_increases, 0);
            assert_eq!(stats_during_reset.sum_increases, 0);

            stats_reporter.update_statistics(INPUT_VOLUME);
            stats_reporter.update_statistics(INPUT_VOLUME + 3);
            let stats_after_reset = stats_reporter.volume_update_stats();
            assert_eq!(stats_after_reset.num_decreases, 1);
            assert_eq!(stats_after_reset.sum_decreases, 2);
            assert_eq!(stats_after_reset.num_increases, 1);
            assert_eq!(stats_after_reset.sum_increases, 3);
        });
    }

    #[test]
    fn update_histogram_on_recommended_input_volume_change_to_match_target_works() {
        reset_on_change_to_match_target_samples();
        update_histogram_on_recommended_input_volume_change_to_match_target(12);
        update_histogram_on_recommended_input_volume_change_to_match_target(12);
        update_histogram_on_recommended_input_volume_change_to_match_target(13);
        assert_eq!(on_change_to_match_target_samples(), vec![(12, 2), (13, 1)]);
    }
}
