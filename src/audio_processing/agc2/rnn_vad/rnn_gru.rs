//! Gated recurrent unit (GRU) layer used by RNN-VAD.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::rnn_vad::vector_math::VectorMath;
use crate::audio_processing::agc2::rnn_vad::weights::rnn_activations::sigmoid_approximated;
use crate::audio_processing::agc2::rnn_vad::weights::rnn_vad_weights::WEIGHTS_SCALE;

const NUM_GRU_GATES: usize = 3; // Update, reset, output.

/// Maximum number of units for a GRU layer.
pub const GRU_LAYER_MAX_UNITS: usize = 24;

/// Recurrent layer with gated recurrent units (GRUs).
#[derive(Debug, Clone)]
pub struct GatedRecurrentLayer {
    input_size: usize,
    output_size: usize,
    bias: Vec<f32>,
    weights: Vec<f32>,
    recurrent_weights: Vec<f32>,
    vector_math: VectorMath,
    // Over-allocated array with size equal to `output_size`.
    state: [f32; GRU_LAYER_MAX_UNITS],
}

impl GatedRecurrentLayer {
    /// Creates a GRU layer.
    ///
    /// # Panics
    ///
    /// Panics when input dimensions are inconsistent.
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: &[i8],
        weights: &[i8],
        recurrent_weights: &[i8],
        cpu_features: AvailableCpuFeatures,
        layer_name: &str,
    ) -> Self {
        let bias = preprocess_gru_tensor(bias, output_size);
        let weights = preprocess_gru_tensor(weights, output_size);
        let recurrent_weights = preprocess_gru_tensor(recurrent_weights, output_size);

        assert!(
            output_size <= GRU_LAYER_MAX_UNITS,
            "Insufficient GRU layer over-allocation ({layer_name})."
        );
        assert_eq!(
            NUM_GRU_GATES * output_size,
            bias.len(),
            "Mismatching output size and bias terms array size ({layer_name})."
        );
        assert_eq!(
            NUM_GRU_GATES * input_size * output_size,
            weights.len(),
            "Mismatching input-output size and weight coefficients array size ({layer_name})."
        );
        assert_eq!(
            NUM_GRU_GATES * output_size * output_size,
            recurrent_weights.len(),
            "Mismatching input-output size and recurrent weight coefficients array size ({layer_name})."
        );

        let mut s = Self {
            input_size,
            output_size,
            bias,
            weights,
            recurrent_weights,
            vector_math: VectorMath::new(cpu_features),
            state: [0.0; GRU_LAYER_MAX_UNITS],
        };
        s.reset();
        s
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn size(&self) -> usize {
        self.output_size
    }

    pub fn data(&self) -> &[f32] {
        &self.state[..self.output_size]
    }

    pub fn output(&self) -> &[f32] {
        &self.state[..self.output_size]
    }

    /// Resets the GRU state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Computes recurrent layer output and updates internal state.
    pub fn compute_output(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.input_size);

        let stride_weights = self.input_size * self.output_size;
        let stride_recurrent_weights = self.output_size * self.output_size;

        let mut update = [0.0f32; GRU_LAYER_MAX_UNITS];
        compute_update_reset_gate(
            self.input_size,
            self.output_size,
            &self.vector_math,
            input,
            &self.state[..self.output_size],
            &self.bias[..self.output_size],
            &self.weights[..stride_weights],
            &self.recurrent_weights[..stride_recurrent_weights],
            &mut update,
        );

        let mut reset = [0.0f32; GRU_LAYER_MAX_UNITS];
        compute_update_reset_gate(
            self.input_size,
            self.output_size,
            &self.vector_math,
            input,
            &self.state[..self.output_size],
            &self.bias[self.output_size..2 * self.output_size],
            &self.weights[stride_weights..2 * stride_weights],
            &self.recurrent_weights[stride_recurrent_weights..2 * stride_recurrent_weights],
            &mut reset,
        );

        compute_state_gate(
            self.input_size,
            self.output_size,
            &self.vector_math,
            input,
            &update,
            &reset,
            &self.bias[2 * self.output_size..3 * self.output_size],
            &self.weights[2 * stride_weights..3 * stride_weights],
            &self.recurrent_weights[2 * stride_recurrent_weights..3 * stride_recurrent_weights],
            &mut self.state,
        );
    }
}

fn preprocess_gru_tensor(tensor_src: &[i8], output_size: usize) -> Vec<f32> {
    assert_eq!(tensor_src.len() % (output_size * NUM_GRU_GATES), 0);

    // `n` is the size of the first dimension of the 3-dim tensor.
    let n = tensor_src.len() / (output_size * NUM_GRU_GATES);
    let stride_src = NUM_GRU_GATES * output_size;
    let stride_dst = n * output_size;

    let mut tensor_dst = vec![0.0f32; tensor_src.len()];
    for g in 0..NUM_GRU_GATES {
        for o in 0..output_size {
            for i in 0..n {
                tensor_dst[g * stride_dst + o * n + i] =
                    WEIGHTS_SCALE * tensor_src[i * stride_src + g * output_size + o] as f32;
            }
        }
    }
    tensor_dst
}

// Computes the output for update/reset gate:
// g = sigmoid(W^T∙i + R^T∙s + b)
fn compute_update_reset_gate(
    input_size: usize,
    output_size: usize,
    vector_math: &VectorMath,
    input: &[f32],
    state: &[f32],
    bias: &[f32],
    weights: &[f32],
    recurrent_weights: &[f32],
    gate: &mut [f32],
) {
    assert_eq!(input.len(), input_size);
    assert_eq!(state.len(), output_size);
    assert_eq!(bias.len(), output_size);
    assert_eq!(weights.len(), input_size * output_size);
    assert_eq!(recurrent_weights.len(), output_size * output_size);
    assert!(gate.len() >= output_size);

    for o in 0..output_size {
        let mut x = bias[o];
        x += vector_math.dot_product(input, &weights[o * input_size..(o + 1) * input_size]);
        x += vector_math.dot_product(
            state,
            &recurrent_weights[o * output_size..(o + 1) * output_size],
        );
        gate[o] = sigmoid_approximated(x);
    }
}

// Computes state gate:
// s' = u .* s + (1 - u) .* ReLU(W^T∙i + R^T∙(s .* r) + b)
fn compute_state_gate(
    input_size: usize,
    output_size: usize,
    vector_math: &VectorMath,
    input: &[f32],
    update: &[f32],
    reset: &[f32],
    bias: &[f32],
    weights: &[f32],
    recurrent_weights: &[f32],
    state: &mut [f32],
) {
    assert_eq!(input.len(), input_size);
    assert!(update.len() >= output_size);
    assert!(reset.len() >= output_size);
    assert_eq!(bias.len(), output_size);
    assert_eq!(weights.len(), input_size * output_size);
    assert_eq!(recurrent_weights.len(), output_size * output_size);
    assert!(state.len() >= output_size);

    let mut reset_x_state = [0.0f32; GRU_LAYER_MAX_UNITS];
    for o in 0..output_size {
        reset_x_state[o] = state[o] * reset[o];
    }

    for o in 0..output_size {
        let mut x = bias[o];
        x += vector_math.dot_product(input, &weights[o * input_size..(o + 1) * input_size]);
        x += vector_math.dot_product(
            &reset_x_state[..output_size],
            &recurrent_weights[o * output_size..(o + 1) * output_size],
        );
        state[o] = update[o] * state[o] + (1.0 - update[o]) * x.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::{
        get_available_cpu_features, no_available_cpu_features,
    };
    use crate::audio_processing::agc2::rnn_vad::test_data::expect_near_absolute;

    const GRU_INPUT_SIZE: usize = 5;
    const GRU_OUTPUT_SIZE: usize = 4;

    const GRU_BIAS: [i8; 12] = [96, -99, -81, -114, 49, 119, -118, 68, -76, 91, 121, 125];

    const GRU_WEIGHTS: [i8; 60] = [
        // Input 0.
        124, 9, 1, 116, -66, -21, -118, -110, 104, 75, -23, -51,
        // Input 1.
        -72, -111, 47, 93, 77, -98, 41, -8, 40, -23, -43, -107,
        // Input 2.
        9, -73, 30, -32, -2, 64, -26, 91, -48, -24, -28, -104,
        // Input 3.
        74, -46, 116, 15, 32, 52, -126, -38, -121, 12, -16, 110,
        // Input 4.
        -95, 66, -103, -35, -38, 3, -126, -61, 28, 98, -117, -43,
    ];

    const GRU_RECURRENT_WEIGHTS: [i8; 48] = [
        // Output 0.
        -3, 87, 50, 51, -22, 27, -39, 62, 31, -83, -52, -48,
        // Output 1.
        -6, 83, -19, 104, 105, 48, 23, 68, 23, 40, 7, -120,
        // Output 2.
        64, -62, 117, 85, 51, -43, 54, -105, 120, 56, -128, -107,
        // Output 3.
        39, 50, -17, -47, -117, 14, 108, 12, -7, -72, 103, -87,
    ];

    const GRU_INPUT_SEQUENCE: [f32; 20] = [
        0.89395463,
        0.93224651,
        0.55788344,
        0.32341808,
        0.93355054,
        0.13475326,
        0.97370994,
        0.14253306,
        0.93710381,
        0.76093364,
        0.65780413,
        0.41657975,
        0.49403164,
        0.46843281,
        0.75138855,
        0.24517593,
        0.47657707,
        0.57064998,
        0.435184,
        0.19319285,
    ];

    const GRU_EXPECTED_OUTPUT_SEQUENCE: [f32; 16] = [
        0.0239123,
        0.5773077,
        0.0,
        0.0,
        0.01282811,
        0.64330572,
        0.0,
        0.04863098,
        0.00781069,
        0.75267816,
        0.0,
        0.02579715,
        0.00471378,
        0.59162533,
        0.11087593,
        0.01334511,
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

    fn test_gated_recurrent_layer(
        gru: &mut GatedRecurrentLayer,
        input_sequence: &[f32],
        expected_output_sequence: &[f32],
    ) {
        assert_eq!(0, input_sequence.len() % gru.input_size());
        assert_eq!(0, expected_output_sequence.len() % gru.size());

        let input_sequence_length = input_sequence.len() / gru.input_size();
        let output_sequence_length = expected_output_sequence.len() / gru.size();
        assert_eq!(input_sequence_length, output_sequence_length);

        gru.reset();
        for i in 0..input_sequence_length {
            gru.compute_output(&input_sequence[i * gru.input_size()..(i + 1) * gru.input_size()]);
            let expected_output = &expected_output_sequence[i * gru.size()..(i + 1) * gru.size()];
            expect_near_absolute(expected_output, gru.output(), 3e-6);
        }
    }

    #[test]
    fn check_gated_recurrent_layer() {
        for cpu_features in get_cpu_features_to_test() {
            let mut gru = GatedRecurrentLayer::new(
                GRU_INPUT_SIZE,
                GRU_OUTPUT_SIZE,
                &GRU_BIAS,
                &GRU_WEIGHTS,
                &GRU_RECURRENT_WEIGHTS,
                cpu_features,
                "GRU",
            );
            test_gated_recurrent_layer(
                &mut gru,
                &GRU_INPUT_SEQUENCE,
                &GRU_EXPECTED_OUTPUT_SEQUENCE,
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut gru = GatedRecurrentLayer::new(
            GRU_INPUT_SIZE,
            GRU_OUTPUT_SIZE,
            &GRU_BIAS,
            &GRU_WEIGHTS,
            &GRU_RECURRENT_WEIGHTS,
            no_available_cpu_features(),
            "GRU",
        );
        gru.compute_output(&GRU_INPUT_SEQUENCE[..GRU_INPUT_SIZE]);
        assert!(gru.output().iter().any(|&x| x != 0.0));
        gru.reset();
        assert!(gru.output().iter().all(|&x| x == 0.0));
    }
}
