//! Recurrent network for voice activity detection.

use crate::audio_processing::agc2::cpu_features::{no_available_cpu_features, AvailableCpuFeatures};
use crate::audio_processing::agc2::rnn_vad::common::FEATURE_VECTOR_SIZE;
use crate::audio_processing::agc2::rnn_vad::rnn_fc::{ActivationFunction, FullyConnectedLayer};
use crate::audio_processing::agc2::rnn_vad::rnn_gru::GatedRecurrentLayer;
use crate::audio_processing::agc2::rnn_vad::weights::rnn_vad_weights::{
    hidden_gru_bias, hidden_gru_recurrent_weights, hidden_gru_weights, input_dense_bias,
    input_dense_weights, output_dense_bias, output_dense_weights, HIDDEN_LAYER_OUTPUT_SIZE,
    INPUT_LAYER_INPUT_SIZE, INPUT_LAYER_OUTPUT_SIZE, OUTPUT_LAYER_OUTPUT_SIZE,
};

/// Recurrent network with hard-coded architecture and weights for voice activity detection.
#[derive(Debug, Clone)]
pub struct RnnVad {
    input: FullyConnectedLayer,
    hidden: GatedRecurrentLayer,
    output: FullyConnectedLayer,
}

impl RnnVad {
    pub fn new(cpu_features: AvailableCpuFeatures) -> Self {
        const _: () = assert!(FEATURE_VECTOR_SIZE == INPUT_LAYER_INPUT_SIZE);

        let input = FullyConnectedLayer::new(
            INPUT_LAYER_INPUT_SIZE,
            INPUT_LAYER_OUTPUT_SIZE,
            input_dense_bias(),
            input_dense_weights(),
            ActivationFunction::TansigApproximated,
            cpu_features,
            "FC1",
        );

        let hidden = GatedRecurrentLayer::new(
            INPUT_LAYER_OUTPUT_SIZE,
            HIDDEN_LAYER_OUTPUT_SIZE,
            hidden_gru_bias(),
            hidden_gru_weights(),
            hidden_gru_recurrent_weights(),
            cpu_features,
            "GRU1",
        );

        let output = FullyConnectedLayer::new(
            HIDDEN_LAYER_OUTPUT_SIZE,
            OUTPUT_LAYER_OUTPUT_SIZE,
            output_dense_bias(),
            output_dense_weights(),
            ActivationFunction::SigmoidApproximated,
            // The output layer is just 24x1. The unoptimized code is faster.
            no_available_cpu_features(),
            "FC2",
        );

        assert_eq!(
            input.size(),
            hidden.input_size(),
            "The input and the hidden layers sizes do not match."
        );
        assert_eq!(
            hidden.size(),
            output.input_size(),
            "The hidden and the output layers sizes do not match."
        );

        Self {
            input,
            hidden,
            output,
        }
    }

    pub fn reset(&mut self) {
        self.hidden.reset();
    }

    /// Observes `feature_vector` and `is_silence`, updates the RNN and returns
    /// the current voice probability.
    pub fn compute_vad_probability(
        &mut self,
        feature_vector: &[f32; FEATURE_VECTOR_SIZE],
        is_silence: bool,
    ) -> f32 {
        if is_silence {
            self.reset();
            return 0.0;
        }

        self.input.compute_output(feature_vector);
        self.hidden.compute_output(self.input.output());
        self.output.compute_output(self.hidden.output());
        assert_eq!(1, self.output.size());
        self.output.data()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::get_available_cpu_features;

    const FEATURES: [f32; FEATURE_VECTOR_SIZE] = [
        -1.00131,
        -0.627069,
        -7.81097,
        7.86285,
        -2.87145,
        3.32365,
        -0.653161,
        0.529839,
        -0.425307,
        0.25583,
        0.235094,
        0.230527,
        -0.144687,
        0.182785,
        0.57102,
        0.125039,
        0.479482,
        -0.0255439,
        -0.0073141,
        -0.147346,
        -0.217106,
        -0.0846906,
        -8.34943,
        3.09065,
        1.42628,
        -0.85235,
        -0.220207,
        -0.811163,
        2.09032,
        -2.01425,
        -0.690268,
        -0.925327,
        -0.541354,
        0.58455,
        -0.606726,
        -0.0372358,
        0.565991,
        0.435854,
        0.420812,
        0.162198,
        -2.13,
        10.0089,
    ];

    fn warm_up_rnn_vad(rnn_vad: &mut RnnVad) {
        for _ in 0..10 {
            rnn_vad.compute_vad_probability(&FEATURES, false);
        }
    }

    #[test]
    fn check_zero_probability_with_silence() {
        let mut rnn_vad = RnnVad::new(get_available_cpu_features());
        warm_up_rnn_vad(&mut rnn_vad);
        assert_eq!(0.0, rnn_vad.compute_vad_probability(&FEATURES, true));
    }

    #[test]
    fn check_rnn_vad_reset() {
        let mut rnn_vad = RnnVad::new(get_available_cpu_features());
        warm_up_rnn_vad(&mut rnn_vad);
        let pre = rnn_vad.compute_vad_probability(&FEATURES, false);
        rnn_vad.reset();
        warm_up_rnn_vad(&mut rnn_vad);
        let post = rnn_vad.compute_vad_probability(&FEATURES, false);
        assert_eq!(pre, post);
    }

    #[test]
    fn check_rnn_vad_silence() {
        let mut rnn_vad = RnnVad::new(get_available_cpu_features());
        warm_up_rnn_vad(&mut rnn_vad);
        let pre = rnn_vad.compute_vad_probability(&FEATURES, false);
        rnn_vad.compute_vad_probability(&FEATURES, true);
        warm_up_rnn_vad(&mut rnn_vad);
        let post = rnn_vad.compute_vad_probability(&FEATURES, false);
        assert_eq!(pre, post);
    }
}
