use std::array;

use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1, METRICS_COLLECTION_BLOCKS,
    METRICS_REPORTING_INTERVAL_BLOCKS,
};

const ONE_BY_METRICS_COLLECTION_BLOCKS: f32 = 1.0 / METRICS_COLLECTION_BLOCKS as f32;
const COMFORT_NOISE_SCALING: f32 = 1.0 / ((BLOCK_SIZE * BLOCK_SIZE) as f32);
const NUM_BANDS: usize = 2;
const BAND_WIDTH: usize = FFT_LENGTH_BY_2_PLUS_1 / NUM_BANDS;

/// Minimal view over the `AecState` fields required by `EchoRemoverMetrics`.
pub trait EchoRemoverMetricsAecState {
    fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1];
    fn erl_time_domain(&self) -> f32;
    fn erle(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1];
    fn fullband_erle_log2(&self) -> f32;
    fn active_render(&self) -> bool;
    fn saturated_capture(&self) -> bool;
    fn usable_linear_estimate(&self) -> bool;
    fn min_direct_path_filter_delay(&self) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DbMetric {
    pub sum_value: f32,
    pub floor_value: f32,
    pub ceil_value: f32,
}

impl Default for DbMetric {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl DbMetric {
    pub const fn new(sum_value: f32, floor_value: f32, ceil_value: f32) -> Self {
        Self {
            sum_value,
            floor_value,
            ceil_value,
        }
    }

    pub fn update(&mut self, value: f32) {
        self.sum_value += value;
        self.floor_value = self.floor_value.min(value);
        self.ceil_value = self.ceil_value.max(value);
    }

    pub fn update_instant(&mut self, value: f32) {
        self.sum_value = value;
        self.floor_value = self.floor_value.min(value);
        self.ceil_value = self.ceil_value.max(value);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandMetricReport {
    pub average: i32,
    pub max: i32,
    pub min: i32,
}

impl Default for BandMetricReport {
    fn default() -> Self {
        Self {
            average: 0,
            max: 0,
            min: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimeDomainMetricReport {
    pub value: i32,
    pub max: i32,
    pub min: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoRemoverMetricsReport {
    pub erle_bands: [BandMetricReport; NUM_BANDS],
    pub erl_bands: [BandMetricReport; NUM_BANDS],
    pub comfort_noise_bands: [BandMetricReport; NUM_BANDS],
    pub suppressor_gain_bands: [BandMetricReport; NUM_BANDS],
    pub usable_linear_estimate: Option<bool>,
    pub active_render: Option<bool>,
    pub filter_delay: Option<i32>,
    pub capture_saturation: Option<bool>,
    pub erl_time_domain: TimeDomainMetricReport,
    pub erle_time_domain: TimeDomainMetricReport,
}

impl Default for EchoRemoverMetricsReport {
    fn default() -> Self {
        Self {
            erle_bands: [BandMetricReport::default(); NUM_BANDS],
            erl_bands: [BandMetricReport::default(); NUM_BANDS],
            comfort_noise_bands: [BandMetricReport::default(); NUM_BANDS],
            suppressor_gain_bands: [BandMetricReport::default(); NUM_BANDS],
            usable_linear_estimate: None,
            active_render: None,
            filter_delay: None,
            capture_saturation: None,
            erl_time_domain: TimeDomainMetricReport::default(),
            erle_time_domain: TimeDomainMetricReport::default(),
        }
    }
}

#[derive(Debug)]
pub struct EchoRemoverMetrics {
    block_counter: usize,
    erl: [DbMetric; NUM_BANDS],
    erl_time_domain: DbMetric,
    erle: [DbMetric; NUM_BANDS],
    erle_time_domain: DbMetric,
    comfort_noise: [DbMetric; NUM_BANDS],
    suppressor_gain: [DbMetric; NUM_BANDS],
    active_render_count: usize,
    saturated_capture: bool,
    metrics_reported: bool,
    report: EchoRemoverMetricsReport,
}

impl Default for EchoRemoverMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoRemoverMetrics {
    pub fn new() -> Self {
        let mut metrics = Self {
            block_counter: 0,
            erl: array::from_fn(|_| DbMetric::default()),
            erl_time_domain: DbMetric::default(),
            erle: array::from_fn(|_| DbMetric::default()),
            erle_time_domain: DbMetric::default(),
            comfort_noise: array::from_fn(|_| DbMetric::default()),
            suppressor_gain: array::from_fn(|_| DbMetric::default()),
            active_render_count: 0,
            saturated_capture: false,
            metrics_reported: false,
            report: EchoRemoverMetricsReport::default(),
        };
        metrics.reset_metrics();
        metrics
    }

    pub fn update<State: EchoRemoverMetricsAecState>(
        &mut self,
        aec_state: &State,
        comfort_noise_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        suppressor_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        self.metrics_reported = false;
        self.block_counter += 1;

        if self.block_counter <= METRICS_COLLECTION_BLOCKS {
            update_db_metric(aec_state.erl(), &mut self.erl);
            self.erl_time_domain
                .update_instant(aec_state.erl_time_domain());
            update_db_metric(aec_state.erle(), &mut self.erle);
            self.erle_time_domain
                .update_instant(aec_state.fullband_erle_log2());
            update_db_metric(comfort_noise_spectrum, &mut self.comfort_noise);
            update_db_metric(suppressor_gain, &mut self.suppressor_gain);

            if aec_state.active_render() {
                self.active_render_count += 1;
            }
            self.saturated_capture |= aec_state.saturated_capture();
            return;
        }

        const METRICS_COLLECTION_BLOCKS_BY2: usize = METRICS_COLLECTION_BLOCKS / 2;

        match self.block_counter {
            x if x == METRICS_COLLECTION_BLOCKS + 1 => {
                self.report.erle_bands[0] = self.report_erle_band(0);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 2 => {
                self.report.erle_bands[1] = self.report_erle_band(1);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 3 => {
                self.report.erl_bands[0] = self.report_erl_band(0);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 4 => {
                self.report.erl_bands[1] = self.report_erl_band(1);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 5 => {
                self.report.comfort_noise_bands[0] =
                    self.report_comfort_noise_band(0, ONE_BY_METRICS_COLLECTION_BLOCKS);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 6 => {
                self.report.comfort_noise_bands[1] =
                    self.report_comfort_noise_band(1, ONE_BY_METRICS_COLLECTION_BLOCKS);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 7 => {
                self.report.suppressor_gain_bands[0] = self.report_suppressor_gain_band(0);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 8 => {
                self.report.suppressor_gain_bands[1] = self.report_suppressor_gain_band(1);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 9 => {
                self.report.usable_linear_estimate = Some(aec_state.usable_linear_estimate());
                self.report.active_render =
                    Some(self.active_render_count > METRICS_COLLECTION_BLOCKS_BY2);
                self.report.filter_delay = Some(aec_state.min_direct_path_filter_delay());
                self.report.capture_saturation = Some(self.saturated_capture);
            }
            x if x == METRICS_COLLECTION_BLOCKS + 10 => {
                self.report.erl_time_domain = self.report_erl_time_domain();
            }
            x if x == METRICS_COLLECTION_BLOCKS + 11 => {
                self.report.erle_time_domain = self.report_erle_time_domain();
                self.metrics_reported = true;
                debug_assert_eq!(METRICS_REPORTING_INTERVAL_BLOCKS, self.block_counter);
                self.block_counter = 0;
                self.reset_metrics();
            }
            _ => {
                panic!("unexpected block counter {}", self.block_counter);
            }
        }
    }

    pub fn metrics_reported(&self) -> bool {
        self.metrics_reported
    }

    pub fn report(&self) -> &EchoRemoverMetricsReport {
        &self.report
    }

    fn reset_metrics(&mut self) {
        self.erl.fill(DbMetric::new(0.0, 10_000.0, 0.0));
        self.erl_time_domain = DbMetric::new(0.0, 10_000.0, 0.0);
        self.erle.fill(DbMetric::new(0.0, 0.0, 1_000.0));
        self.erle_time_domain = DbMetric::new(0.0, 0.0, 1_000.0);
        self.comfort_noise
            .fill(DbMetric::new(0.0, 100_000_000.0, 0.0));
        self.suppressor_gain.fill(DbMetric::new(0.0, 1.0, 0.0));
        self.active_render_count = 0;
        self.saturated_capture = false;
    }

    fn report_erle_band(&self, band: usize) -> BandMetricReport {
        BandMetricReport {
            average: transform_db_metric_for_reporting(
                true,
                0.0,
                19.0,
                0.0,
                ONE_BY_METRICS_COLLECTION_BLOCKS,
                self.erle[band].sum_value,
            ),
            max: transform_db_metric_for_reporting(
                true,
                0.0,
                19.0,
                0.0,
                1.0,
                self.erle[band].ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                true,
                0.0,
                19.0,
                0.0,
                1.0,
                self.erle[band].floor_value,
            ),
        }
    }

    fn report_erl_band(&self, band: usize) -> BandMetricReport {
        BandMetricReport {
            average: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                ONE_BY_METRICS_COLLECTION_BLOCKS,
                self.erl[band].sum_value,
            ),
            max: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                1.0,
                self.erl[band].ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                1.0,
                self.erl[band].floor_value,
            ),
        }
    }

    fn report_comfort_noise_band(&self, band: usize, avg_scaling: f32) -> BandMetricReport {
        BandMetricReport {
            average: transform_db_metric_for_reporting(
                true,
                0.0,
                89.0,
                -90.3,
                COMFORT_NOISE_SCALING * avg_scaling,
                self.comfort_noise[band].sum_value,
            ),
            max: transform_db_metric_for_reporting(
                true,
                0.0,
                89.0,
                -90.3,
                COMFORT_NOISE_SCALING,
                self.comfort_noise[band].ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                true,
                0.0,
                89.0,
                -90.3,
                COMFORT_NOISE_SCALING,
                self.comfort_noise[band].floor_value,
            ),
        }
    }

    fn report_suppressor_gain_band(&self, band: usize) -> BandMetricReport {
        BandMetricReport {
            average: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                0.0,
                ONE_BY_METRICS_COLLECTION_BLOCKS,
                self.suppressor_gain[band].sum_value,
            ),
            max: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                0.0,
                1.0,
                self.suppressor_gain[band].ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                0.0,
                1.0,
                self.suppressor_gain[band].floor_value,
            ),
        }
    }

    fn report_erl_time_domain(&self) -> TimeDomainMetricReport {
        TimeDomainMetricReport {
            value: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                1.0,
                self.erl_time_domain.sum_value,
            ),
            max: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                1.0,
                self.erl_time_domain.ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                true,
                0.0,
                59.0,
                30.0,
                1.0,
                self.erl_time_domain.floor_value,
            ),
        }
    }

    fn report_erle_time_domain(&self) -> TimeDomainMetricReport {
        TimeDomainMetricReport {
            value: transform_db_metric_for_reporting(
                false,
                0.0,
                19.0,
                0.0,
                1.0,
                self.erle_time_domain.sum_value,
            ),
            max: transform_db_metric_for_reporting(
                false,
                0.0,
                19.0,
                0.0,
                1.0,
                self.erle_time_domain.ceil_value,
            ),
            min: transform_db_metric_for_reporting(
                false,
                0.0,
                19.0,
                0.0,
                1.0,
                self.erle_time_domain.floor_value,
            ),
        }
    }
}

pub fn update_db_metric(
    value: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    statistic: &mut [DbMetric; NUM_BANDS],
) {
    for (band, stat) in statistic.iter_mut().enumerate() {
        let start = BAND_WIDTH * band;
        let end = start + BAND_WIDTH;
        let sum: f32 = value[start..end].iter().copied().sum();
        stat.update(sum / BAND_WIDTH as f32);
    }
}

pub fn transform_db_metric_for_reporting(
    negate: bool,
    min_value: f32,
    max_value: f32,
    offset: f32,
    scaling: f32,
    value: f32,
) -> i32 {
    let mut new_value = 10.0 * (value * scaling + 1e-10).log10() + offset;
    if negate {
        new_value = -new_value;
    }
    let clamped = new_value.clamp(min_value, max_value);
    clamped as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{Aec3Optimization, BLOCK_SIZE};
    use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
    use crate::audio_processing::aec3::fft_data::FftData;

    struct TestAecState {
        erl: [f32; FFT_LENGTH_BY_2_PLUS_1],
        erle: [f32; FFT_LENGTH_BY_2_PLUS_1],
        erl_td: f32,
        erle_log2: f32,
        active_render: bool,
        saturated_capture: bool,
        usable_linear_estimate: bool,
        filter_delay: i32,
    }

    impl TestAecState {
        fn new() -> Self {
            Self {
                erl: [1.0; FFT_LENGTH_BY_2_PLUS_1],
                erle: [1.0; FFT_LENGTH_BY_2_PLUS_1],
                erl_td: 1.0,
                erle_log2: 1.0,
                active_render: true,
                saturated_capture: false,
                usable_linear_estimate: true,
                filter_delay: 0,
            }
        }
    }

    impl EchoRemoverMetricsAecState for TestAecState {
        fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
            &self.erl
        }

        fn erl_time_domain(&self) -> f32 {
            self.erl_td
        }

        fn erle(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
            &self.erle
        }

        fn fullband_erle_log2(&self) -> f32 {
            self.erle_log2
        }

        fn active_render(&self) -> bool {
            self.active_render
        }

        fn saturated_capture(&self) -> bool {
            self.saturated_capture
        }

        fn usable_linear_estimate(&self) -> bool {
            self.usable_linear_estimate
        }

        fn min_direct_path_filter_delay(&self) -> i32 {
            self.filter_delay
        }
    }

    #[test]
    fn update_db_metric_updates_statistic() {
        let mut value = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut statistic = [DbMetric::new(0.0, 100.0, -100.0); NUM_BANDS];
        value[..32].fill(10.0);
        value[32..64].fill(20.0);

        update_db_metric(&value, &mut statistic);
        assert_eq!(10.0, statistic[0].sum_value);
        assert_eq!(10.0, statistic[0].ceil_value);
        assert_eq!(10.0, statistic[0].floor_value);
        assert_eq!(20.0, statistic[1].sum_value);
        assert_eq!(20.0, statistic[1].ceil_value);
        assert_eq!(20.0, statistic[1].floor_value);

        update_db_metric(&value, &mut statistic);
        assert_eq!(20.0, statistic[0].sum_value);
        assert_eq!(10.0, statistic[0].ceil_value);
        assert_eq!(10.0, statistic[0].floor_value);
        assert_eq!(40.0, statistic[1].sum_value);
    }

    #[test]
    fn transform_db_metric_handles_dbfs_scaling() {
        let x = [1000.0f32; BLOCK_SIZE];
        let fft = Aec3Fft::new();
        let mut fft_data = FftData::default();
        let mut spectrum = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        fft.zero_padded_fft(&x, Window::Rectangular, &mut fft_data);
        fft_data.spectrum(Aec3Optimization::None, &mut spectrum);

        let offset = -10.0 * (32768.0_f32 * 32768.0_f32).log10();
        assert!((offset - -90.3).abs() < 0.1);
        let result = transform_db_metric_for_reporting(
            true,
            0.0,
            90.0,
            offset,
            1.0 / ((BLOCK_SIZE * BLOCK_SIZE) as f32),
            spectrum[0],
        );
        assert_eq!(30, result);
    }

    #[test]
    fn transform_db_metric_limits_output() {
        assert_eq!(
            0,
            transform_db_metric_for_reporting(false, 0.0, 10.0, 0.0, 1.0, 0.001)
        );
        assert_eq!(
            10,
            transform_db_metric_for_reporting(false, 0.0, 10.0, 0.0, 1.0, 100.0)
        );
    }

    #[test]
    fn transform_db_metric_negates_output() {
        assert_eq!(
            10,
            transform_db_metric_for_reporting(true, -20.0, 20.0, 0.0, 1.0, 0.1)
        );
        assert_eq!(
            -10,
            transform_db_metric_for_reporting(true, -20.0, 20.0, 0.0, 1.0, 10.0)
        );
    }

    #[test]
    fn db_metric_update_accumulates_values() {
        let mut metric = DbMetric::new(0.0, 20.0, -20.0);
        for _ in 0..100 {
            metric.update(10.0);
        }
        assert_eq!(1000.0, metric.sum_value);
        assert_eq!(10.0, metric.ceil_value);
        assert_eq!(10.0, metric.floor_value);
    }

    #[test]
    fn db_metric_update_instant_tracks_bounds() {
        let mut metric = DbMetric::new(0.0, 20.0, -20.0);
        for value in -77..=33 {
            metric.update_instant(value as f32);
        }
        let last_value = (-77 + 33) as f32 / 2.0;
        metric.update_instant(last_value);
        assert_eq!(last_value, metric.sum_value);
        assert_eq!(33.0, metric.ceil_value);
        assert_eq!(-77.0, metric.floor_value);
    }

    #[test]
    fn db_metric_constructor_sets_fields() {
        let metric = DbMetric::default();
        assert_eq!(0.0, metric.sum_value);
        assert_eq!(0.0, metric.floor_value);
        assert_eq!(0.0, metric.ceil_value);

        let metric = DbMetric::new(1.0, 2.0, 3.0);
        assert_eq!(1.0, metric.sum_value);
        assert_eq!(2.0, metric.floor_value);
        assert_eq!(3.0, metric.ceil_value);
    }

    #[test]
    fn echo_remover_metrics_cycles_reporting() {
        let mut metrics = EchoRemoverMetrics::new();
        let aec_state = TestAecState::new();
        let comfort_noise = [10.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let suppressor_gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];

        for j in 0..3 {
            for _ in 0..(METRICS_REPORTING_INTERVAL_BLOCKS - 1) {
                metrics.update(&aec_state, &comfort_noise, &suppressor_gain);
                assert!(!metrics.metrics_reported());
            }
            metrics.update(&aec_state, &comfort_noise, &suppressor_gain);
            assert!(metrics.metrics_reported(), "iteration {j}");
        }
    }
}
