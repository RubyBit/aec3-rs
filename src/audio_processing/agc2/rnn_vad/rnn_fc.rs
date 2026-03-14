//! Fully connected (dense) layer used by RNN-VAD.

use crate::audio_processing::agc2::cpu_features::{
    AvailableCpuFeatures,
};
use crate::audio_processing::agc2::rnn_vad::vector_math::VectorMath;
use crate::audio_processing::agc2::rnn_vad::weights::rnn_activations::{
    sigmoid_approximated, tansig_approximated,
};
use crate::audio_processing::agc2::rnn_vad::weights::rnn_vad_weights::WEIGHTS_SCALE;

/// Activation function for a neural network cell.
#[derive(Debug, Copy, Clone)]
pub enum ActivationFunction {
    TansigApproximated,
    SigmoidApproximated,
}

/// Maximum number of units for a fully connected layer.
pub const FULLY_CONNECTED_LAYER_MAX_UNITS: usize = 24;

/// Fully-connected layer with an owned output buffer.
#[derive(Debug, Clone)]
pub struct FullyConnectedLayer {
    input_size: usize,
    output_size: usize,
    bias: Vec<f32>,
    weights: Vec<f32>,
    vector_math: VectorMath,
    activation_function: ActivationFunction,
    // Over-allocated array with size equal to `output_size`.
    output: [f32; FULLY_CONNECTED_LAYER_MAX_UNITS],
}

impl FullyConnectedLayer {
    /// Creates a fully-connected layer.
    ///
    /// # Panics
    ///
    /// Panics if sizes are inconsistent with `input_size` and `output_size`.
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: &[i8],
        weights: &[i8],
        activation_function: ActivationFunction,
        cpu_features: AvailableCpuFeatures,
        layer_name: &str,
    ) -> Self {
        let bias = get_scaled_params(bias);
        let weights = preprocess_weights(weights, output_size);

        assert!(
            output_size <= FULLY_CONNECTED_LAYER_MAX_UNITS,
            "Insufficient FC layer over-allocation ({layer_name})."
        );
        assert_eq!(
            output_size,
            bias.len(),
            "Mismatching output size and bias terms array size ({layer_name})."
        );
        assert_eq!(
            input_size * output_size,
            weights.len(),
            "Mismatching input-output size and weight coefficients array size ({layer_name})."
        );

        Self {
            input_size,
            output_size,
            bias,
            weights,
            vector_math: VectorMath::new(cpu_features),
            activation_function,
            output: [0.0; FULLY_CONNECTED_LAYER_MAX_UNITS],
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn size(&self) -> usize {
        self.output_size
    }

    pub fn data(&self) -> &[f32] {
        &self.output[..self.output_size]
    }

    pub fn output(&self) -> &[f32] {
        &self.output[..self.output_size]
    }

    /// Computes the fully-connected layer output.
    pub fn compute_output(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.input_size);

        for o in 0..self.output_size {
            let row_start = o * self.input_size;
            let row_end = row_start + self.input_size;
            let weighted_sum = self.vector_math.dot_product(input, &self.weights[row_start..row_end]);
            self.output[o] = self.apply_activation(self.bias[o] + weighted_sum);
        }
    }

    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation_function {
            ActivationFunction::TansigApproximated => tansig_approximated(x),
            ActivationFunction::SigmoidApproximated => sigmoid_approximated(x),
        }
    }
}

fn get_scaled_params(params: &[i8]) -> Vec<f32> {
    params
        .iter()
        .map(|&x| WEIGHTS_SCALE * x as f32)
        .collect::<Vec<_>>()
}

// Casts, scales and re-arranges `weights` to output-major layout.
fn preprocess_weights(weights: &[i8], output_size: usize) -> Vec<f32> {
    if output_size == 1 {
        return get_scaled_params(weights);
    }

    assert_eq!(weights.len() % output_size, 0);
    let input_size = weights.len() / output_size;

    // Transpose, scale and cast.
    let mut w = vec![0.0; weights.len()];
    for o in 0..output_size {
        for i in 0..input_size {
            w[o * input_size + i] = WEIGHTS_SCALE * weights[i * output_size + o] as f32;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::no_available_cpu_features;
    use crate::audio_processing::agc2::rnn_vad::test_data::expect_near_absolute;
    use crate::audio_processing::agc2::rnn_vad::weights::rnn_vad_weights::{
        input_dense_bias, input_dense_weights, INPUT_LAYER_INPUT_SIZE, INPUT_LAYER_OUTPUT_SIZE,
    };
    use crate::audio_processing::agc2::cpu_features::{
        get_available_cpu_features
    };

    const FULLY_CONNECTED_INPUT_VECTOR: [f32; 42] = [
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

    const FULLY_CONNECTED_EXPECTED_OUTPUT: [f32; 24] = [
        -0.623293,
        -0.988299,
        0.999378,
        0.967168,
        0.103087,
        -0.978545,
        -0.856347,
        0.346675,
        1.0,
        -0.717442,
        -0.544176,
        0.960363,
        0.983443,
        0.999991,
        -0.824335,
        0.984742,
        0.990208,
        0.938179,
        0.875092,
        0.999846,
        0.997707,
        -0.999382,
        0.973153,
        -0.966605,
    ];

    fn get_cpu_features_to_test() -> Vec<AvailableCpuFeatures> {
        let mut v = vec![no_available_cpu_features()];
        let available = get_available_cpu_features();
        if available.sse2 {
            v.push(AvailableCpuFeatures::new(true, false, false));
        }
        if available.avx2 {
            v.push(AvailableCpuFeatures::new(false, true, false));
        }
        if available.neon {
            v.push(AvailableCpuFeatures::new(false, false, true));
        }
        v
    }

    #[test]
    fn check_fully_connected_layer_output() {
        for cpu_features in get_cpu_features_to_test() {
            let mut fc = FullyConnectedLayer::new(
                INPUT_LAYER_INPUT_SIZE,
                INPUT_LAYER_OUTPUT_SIZE,
                input_dense_bias(),
                input_dense_weights(),
                ActivationFunction::TansigApproximated,
                cpu_features,
                "FC",
            );

            fc.compute_output(&FULLY_CONNECTED_INPUT_VECTOR);
            expect_near_absolute(&FULLY_CONNECTED_EXPECTED_OUTPUT, fc.output(), 1e-5);
        }
    }

    #[test]
    fn preprocess_weights_output_size_one_keeps_layout() {
        let in_weights = [10i8, -20, 30, -40];
        let w = preprocess_weights(&in_weights, 1);
        assert_eq!(w.len(), in_weights.len());
        for (i, &x) in in_weights.iter().enumerate() {
            assert!((w[i] - (x as f32 * WEIGHTS_SCALE)).abs() < 1e-6);
        }
    }

    #[test]
    fn layer_size_accessors_match_ctor() {
        let fc = FullyConnectedLayer::new(
            2,
            2,
            &[1, 2],
            &[1, 2, 3, 4],
            ActivationFunction::SigmoidApproximated,
            no_available_cpu_features(),
            "tiny_fc",
        );

        assert_eq!(2, fc.input_size());
        assert_eq!(2, fc.size());
        assert_eq!(2, fc.output().len());
        assert_eq!(2, fc.data().len());
    }
}
