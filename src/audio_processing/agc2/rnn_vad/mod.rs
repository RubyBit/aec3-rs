//! Rust implementation of WebRTC AGC2 RNN-VAD components.

pub mod auto_correlation;
pub mod common;
pub mod features_extraction;
pub mod lp_residual;
pub mod pitch_search;
pub mod pitch_search_internal;
pub mod ring_buffer;
pub mod rnn_fc;
pub mod rnn_gru;
pub mod rnn;
pub mod sequence_buffer;
pub mod spectral_features;
pub mod spectral_features_internal;
pub mod symmetric_matrix_buffer;
pub mod vector_math;
pub mod weights;

#[cfg(test)]
pub mod test_data;
