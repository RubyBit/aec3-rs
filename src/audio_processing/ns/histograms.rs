//! Histograms for NS feature thresholds adaptation.

use super::ns_common::{BIN_SIZE_LRT, BIN_SIZE_SPEC_DIFF, BIN_SIZE_SPEC_FLAT};
use super::signal_model::SignalModel;

pub const HISTOGRAM_SIZE: usize = 1000;

#[derive(Debug, Clone)]
pub struct Histograms {
    lrt: [i32; HISTOGRAM_SIZE],
    spectral_flatness: [i32; HISTOGRAM_SIZE],
    spectral_diff: [i32; HISTOGRAM_SIZE],
}

impl Default for Histograms {
    fn default() -> Self {
        let h = Self {
            lrt: [0; HISTOGRAM_SIZE],
            spectral_flatness: [0; HISTOGRAM_SIZE],
            spectral_diff: [0; HISTOGRAM_SIZE],
        };
        h
    }
}

impl Histograms {
    pub fn clear(&mut self) {
        self.lrt.fill(0);
        self.spectral_flatness.fill(0);
        self.spectral_diff.fill(0);
    }

    pub fn update(&mut self, features: &SignalModel) {
        // I don't know why the reference has such a complicated calculation, I simplified it to be clearer..
        let lrt_idx = (features.lrt / BIN_SIZE_LRT) as usize;
        if features.lrt >= 0.0 && lrt_idx < HISTOGRAM_SIZE {
            self.lrt[lrt_idx] += 1;
        }

        let flat_idx = (features.spectral_flatness / BIN_SIZE_SPEC_FLAT) as usize;
        if features.spectral_flatness >= 0.0 && flat_idx < HISTOGRAM_SIZE {
            self.spectral_flatness[flat_idx] += 1;
        }

        let diff_idx = (features.spectral_diff / BIN_SIZE_SPEC_DIFF) as usize;
        if features.spectral_diff >= 0.0 && diff_idx < HISTOGRAM_SIZE {
            self.spectral_diff[diff_idx] += 1;
        }
    }

    pub fn lrt(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.lrt
    }

    pub fn spectral_flatness(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.spectral_flatness
    }

    pub fn spectral_diff(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.spectral_diff
    }
}
