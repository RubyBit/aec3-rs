//! Signal model estimator used for speech probability.

use super::fast_math::{exp_approximation, log_approximation};
use super::histograms::Histograms;
use super::ns_common::{FEATURE_UPDATE_WINDOW_SIZE, FFT_SIZE_BY_2_PLUS_1, LRT_FEATURE_THR};
use super::prior_signal_model::PriorSignalModel;
use super::prior_signal_model_estimator::PriorSignalModelEstimator;
use super::signal_model::SignalModel;

const ONE_BY_FFT_SIZE_BY_2_PLUS_1: f32 = 1.0 / FFT_SIZE_BY_2_PLUS_1 as f32;

fn compute_spectral_diff(
    conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectral_sum: f32,
    diff_normalization: f32,
) -> f32 {
    let noise_average =
        conservative_noise_spectrum.iter().sum::<f32>() * ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    let signal_average = signal_spectral_sum * ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    let mut covariance = 0.0;
    let mut noise_variance = 0.0;
    let mut signal_variance = 0.0;

    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        let signal_diff = signal_spectrum[i] - signal_average;
        let noise_diff = conservative_noise_spectrum[i] - noise_average;
        covariance += signal_diff * noise_diff;
        noise_variance += noise_diff * noise_diff;
        signal_variance += signal_diff * signal_diff;
    }

    covariance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    noise_variance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    signal_variance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    let spectral_diff = signal_variance - (covariance * covariance) / (noise_variance + 0.0001);
    spectral_diff / (diff_normalization + 0.0001)
}

fn update_spectral_flatness(
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectral_sum: f32,
    spectral_flatness: &mut f32,
) {
    const AVERAGING: f32 = 0.3;

    if signal_spectrum.iter().skip(1).any(|&x| x == 0.0) {
        *spectral_flatness -= AVERAGING * *spectral_flatness;
        return;
    }

    let avg_num = signal_spectrum
        .iter()
        .skip(1)
        .map(|&x| log_approximation(x))
        .sum::<f32>()
        * ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    let avg_denom = (signal_spectral_sum - signal_spectrum[0]) * ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    let spectral_tmp = exp_approximation(avg_num) / avg_denom;
    *spectral_flatness += AVERAGING * (spectral_tmp - *spectral_flatness);
}

fn update_spectral_lrt(
    prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
    post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
    avg_log_lrt: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    lrt: &mut f32,
) {
    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        let tmp1 = 1.0 + 2.0 * prior_snr[i];
        let tmp2 = 2.0 * prior_snr[i] / (tmp1 + 0.0001);
        let bessel_tmp = (post_snr[i] + 1.0) * tmp2;
        avg_log_lrt[i] += 0.5 * (bessel_tmp - log_approximation(tmp1) - avg_log_lrt[i]);
    }

    *lrt = avg_log_lrt.iter().sum::<f32>() * ONE_BY_FFT_SIZE_BY_2_PLUS_1;
}

#[derive(Debug, Clone)]
pub struct SignalModelEstimator {
    diff_normalization: f32,
    signal_energy_sum: f32,
    histograms: Histograms,
    histogram_analysis_counter: i32,
    prior_model_estimator: PriorSignalModelEstimator,
    features: SignalModel,
}

impl Default for SignalModelEstimator {
    fn default() -> Self {
        Self {
            diff_normalization: 0.0,
            signal_energy_sum: 0.0,
            histograms: Histograms::default(),
            histogram_analysis_counter: FEATURE_UPDATE_WINDOW_SIZE,
            prior_model_estimator: PriorSignalModelEstimator::new(LRT_FEATURE_THR),
            features: SignalModel::default(),
        }
    }
}

impl SignalModelEstimator {
    pub fn adjust_normalization(&mut self, num_analyzed_frames: i32, signal_energy: f32) {
        self.diff_normalization *= num_analyzed_frames as f32;
        self.diff_normalization += signal_energy;
        self.diff_normalization /= num_analyzed_frames as f32 + 1.0;
    }

    pub fn update(
        &mut self,
        prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
        signal_energy: f32,
    ) {
        update_spectral_flatness(
            signal_spectrum,
            signal_spectral_sum,
            &mut self.features.spectral_flatness,
        );

        let spectral_diff = compute_spectral_diff(
            conservative_noise_spectrum,
            signal_spectrum,
            signal_spectral_sum,
            self.diff_normalization,
        );
        self.features.spectral_diff += 0.3 * (spectral_diff - self.features.spectral_diff);

        self.signal_energy_sum += signal_energy;

        self.histogram_analysis_counter -= 1;
        if self.histogram_analysis_counter > 0 {
            self.histograms.update(&self.features);
        } else {
            self.prior_model_estimator.update(&self.histograms);
            self.histograms.clear();
            self.histogram_analysis_counter = FEATURE_UPDATE_WINDOW_SIZE;

            self.signal_energy_sum /= FEATURE_UPDATE_WINDOW_SIZE as f32;
            self.diff_normalization = 0.5 * (self.signal_energy_sum + self.diff_normalization);
            self.signal_energy_sum = 0.0;
        }

        update_spectral_lrt(
            prior_snr,
            post_snr,
            &mut self.features.avg_log_lrt,
            &mut self.features.lrt,
        );
    }

    pub fn prior_model(&self) -> &PriorSignalModel {
        self.prior_model_estimator.prior_model()
    }

    pub fn model(&self) -> &SignalModel {
        &self.features
    }
}
