//! Prior model estimator using feature histograms.

use super::histograms::{HISTOGRAM_SIZE, Histograms};
use super::ns_common::{
    BIN_SIZE_LRT, BIN_SIZE_SPEC_DIFF, BIN_SIZE_SPEC_FLAT, FEATURE_UPDATE_WINDOW_SIZE,
};
use super::prior_signal_model::PriorSignalModel;

fn find_first_of_two_largest_peaks(bin_size: f32, histogram: &[i32; HISTOGRAM_SIZE]) -> (f32, i32) {
    let mut peak_value = 0;
    let mut secondary_peak_value = 0;
    let mut peak_position = 0.0f32;
    let mut secondary_peak_position = 0.0f32;
    let mut peak_weight = 0;
    let mut secondary_peak_weight = 0;

    for (i, &value) in histogram.iter().enumerate() {
        let bin_mid = (i as f32 + 0.5) * bin_size;
        if value > peak_value {
            secondary_peak_value = peak_value;
            secondary_peak_weight = peak_weight;
            secondary_peak_position = peak_position;

            peak_value = value;
            peak_weight = value;
            peak_position = bin_mid;
        } else if value > secondary_peak_value {
            secondary_peak_value = value;
            secondary_peak_weight = value;
            secondary_peak_position = bin_mid;
        }
    }

    if (secondary_peak_position - peak_position).abs() < 2.0 * bin_size
        && (secondary_peak_weight as f32) > 0.5 * (peak_weight as f32)
    {
        peak_weight += secondary_peak_weight;
        peak_position = 0.5 * (peak_position + secondary_peak_position);
    }

    (peak_position, peak_weight)
}

fn update_lrt(
    lrt_histogram: &[i32; HISTOGRAM_SIZE],
    prior_model_lrt: &mut f32,
    low_lrt_fluctuations: &mut bool,
) {
    let mut average = 0.0f32;
    let mut average_compl = 0.0f32;
    let mut average_squared = 0.0f32;
    let mut count = 0i32;

    for (i, &value) in lrt_histogram.iter().take(10).enumerate() {
        let bin_mid = (i as f32 + 0.5) * BIN_SIZE_LRT;
        average += value as f32 * bin_mid;
        count += value;
    }
    if count > 0 {
        average /= count as f32;
    }

    for (i, &value) in lrt_histogram.iter().enumerate() {
        let bin_mid = (i as f32 + 0.5) * BIN_SIZE_LRT;
        average_squared += value as f32 * bin_mid * bin_mid;
        average_compl += value as f32 * bin_mid;
    }

    let one_by_window = 1.0 / FEATURE_UPDATE_WINDOW_SIZE as f32;
    average_squared *= one_by_window;
    average_compl *= one_by_window;

    *low_lrt_fluctuations = average_squared - average * average_compl < 0.05;

    const MAX_LRT: f32 = 1.0;
    const MIN_LRT: f32 = 0.2;
    if *low_lrt_fluctuations {
        *prior_model_lrt = MAX_LRT;
    } else {
        *prior_model_lrt = (1.2 * average).clamp(MIN_LRT, MAX_LRT);
    }
}

#[derive(Debug, Clone)]
pub struct PriorSignalModelEstimator {
    prior_model: PriorSignalModel,
}

impl PriorSignalModelEstimator {
    pub fn new(lrt_initial_value: f32) -> Self {
        Self {
            prior_model: PriorSignalModel::new(lrt_initial_value),
        }
    }

    pub fn update(&mut self, histograms: &Histograms) {
        let mut low_lrt_fluctuations = false;
        update_lrt(
            histograms.lrt(),
            &mut self.prior_model.lrt,
            &mut low_lrt_fluctuations,
        );

        let (spectral_flatness_peak_position, spectral_flatness_peak_weight) =
            find_first_of_two_largest_peaks(BIN_SIZE_SPEC_FLAT, histograms.spectral_flatness());

        let (spectral_diff_peak_position, spectral_diff_peak_weight) =
            find_first_of_two_largest_peaks(BIN_SIZE_SPEC_DIFF, histograms.spectral_diff());

        let use_spec_flat = if (spectral_flatness_peak_weight as f32) < 0.3 * 500.0
            || spectral_flatness_peak_position < 0.6
        {
            0
        } else {
            1
        };

        let use_spec_diff =
            if (spectral_diff_peak_weight as f32) < 0.3 * 500.0 || low_lrt_fluctuations {
                0
            } else {
                1
            };

        self.prior_model.template_diff_threshold =
            (1.2 * spectral_diff_peak_position).clamp(0.16, 1.0);

        let one_by_feature_sum = 1.0 / (1.0 + use_spec_flat as f32 + use_spec_diff as f32);
        self.prior_model.lrt_weighting = one_by_feature_sum;

        if use_spec_flat == 1 {
            self.prior_model.flatness_threshold =
                (0.9 * spectral_flatness_peak_position).clamp(0.1, 0.95);
            self.prior_model.flatness_weighting = one_by_feature_sum;
        } else {
            self.prior_model.flatness_weighting = 0.0;
        }

        self.prior_model.difference_weighting = if use_spec_diff == 1 {
            one_by_feature_sum
        } else {
            0.0
        };
    }

    pub fn prior_model(&self) -> &PriorSignalModel {
        &self.prior_model
    }
}
