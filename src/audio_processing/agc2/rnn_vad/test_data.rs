//! Test fixtures and readers for RNN-VAD parity tests.

use crate::audio_processing::agc2::rnn_vad::common::{
    BUF_SIZE_24_KHZ, NUM_LAGS_12_KHZ, REFINE_NUM_LAGS_24_KHZ,
};

pub const FLOAT_MIN: f32 = f32::MIN_POSITIVE;

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bytes_i16_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
        .collect()
}

pub fn expect_near_absolute(expected: &[f32], computed: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), computed.len());
    for (i, (e, c)) in expected.iter().zip(computed.iter()).enumerate() {
        assert!(
            (e - c).abs() <= tolerance,
            "index={i}, expected={e}, computed={c}, tolerance={tolerance}"
        );
    }
}

#[derive(Debug, Clone)]
pub struct FloatChunkReader {
    chunk_size: usize,
    data: Vec<f32>,
    cursor: usize,
}

impl FloatChunkReader {
    pub fn new(data: Vec<f32>, chunk_size: usize) -> Self {
        assert!(chunk_size > 0);
        Self {
            chunk_size,
            data,
            cursor: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn num_chunks(&self) -> usize {
        self.data.len() / self.chunk_size
    }

    pub fn read_chunk(&mut self, dst: &mut [f32]) -> bool {
        if dst.len() != self.chunk_size {
            return false;
        }
        if self.cursor + self.chunk_size > self.data.len() {
            return false;
        }
        dst.copy_from_slice(&self.data[self.cursor..self.cursor + self.chunk_size]);
        self.cursor += self.chunk_size;
        true
    }

    pub fn read_value(&mut self) -> Option<f32> {
        if self.cursor >= self.data.len() {
            return None;
        }
        let v = self.data[self.cursor];
        self.cursor += 1;
        Some(v)
    }

    pub fn seek_forward(&mut self, hop: usize) {
        self.cursor = (self.cursor + hop).min(self.data.len());
    }

    pub fn seek_beginning(&mut self) {
        self.cursor = 0;
    }
}

pub fn create_pcm_samples_reader() -> FloatChunkReader {
    let data = bytes_i16_to_f32_vec(include_bytes!(
        "../../../../tests/test_data/audio_processing/agc2/rnn_vad/samples.pcm"
    ));
    FloatChunkReader::new(data, 1)
}

pub fn create_pitch_buffer_24khz_reader() -> FloatChunkReader {
    let data = bytes_to_f32_vec(include_bytes!(
        "../../../../tests/test_data/audio_processing/agc2/rnn_vad/pitch_buf_24k.dat"
    ));
    FloatChunkReader::new(data, BUF_SIZE_24_KHZ)
}

pub fn create_lp_residual_and_pitch_info_reader() -> FloatChunkReader {
    // LP residual chunk + 2 pitch info values (period, strength).
    let chunk_size = BUF_SIZE_24_KHZ + 2;
    let data = bytes_to_f32_vec(include_bytes!(
        "../../../../tests/test_data/audio_processing/agc2/rnn_vad/pitch_lp_res.dat"
    ));
    FloatChunkReader::new(data, chunk_size)
}

pub fn create_gru_input_reader() -> FloatChunkReader {
    let data = bytes_to_f32_vec(include_bytes!(
        "../../../../tests/test_data/audio_processing/agc2/rnn_vad/gru_in.dat"
    ));
    FloatChunkReader::new(data, 1)
}

pub fn create_vad_probs_reader() -> FloatChunkReader {
    let data = bytes_to_f32_vec(include_bytes!(
        "../../../../tests/test_data/audio_processing/agc2/rnn_vad/vad_prob.dat"
    ));
    FloatChunkReader::new(data, 1)
}

pub struct PitchTestData {
    pitch_buffer_24k: [f32; BUF_SIZE_24_KHZ],
    square_energies_24k: [f32; REFINE_NUM_LAGS_24_KHZ],
    auto_correlation_12k: [f32; NUM_LAGS_12_KHZ],
}

impl PitchTestData {
    pub fn new() -> Self {
        let data = bytes_to_f32_vec(include_bytes!(
            "../../../../tests/test_data/audio_processing/agc2/rnn_vad/pitch_search_int.dat"
        ));
        let needed = BUF_SIZE_24_KHZ + REFINE_NUM_LAGS_24_KHZ + NUM_LAGS_12_KHZ;
        assert!(
            data.len() >= needed,
            "pitch_search_int.dat too small: {} < {}",
            data.len(),
            needed
        );

        let mut offset = 0usize;

        let mut pitch_buffer_24k = [0.0f32; BUF_SIZE_24_KHZ];
        pitch_buffer_24k.copy_from_slice(&data[offset..offset + BUF_SIZE_24_KHZ]);
        offset += BUF_SIZE_24_KHZ;

        let mut square_energies_24k = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        square_energies_24k.copy_from_slice(&data[offset..offset + REFINE_NUM_LAGS_24_KHZ]);
        offset += REFINE_NUM_LAGS_24_KHZ;
        // Matches C++ test_utils.cc behavior.
        square_energies_24k.reverse();

        let mut auto_correlation_12k = [0.0f32; NUM_LAGS_12_KHZ];
        auto_correlation_12k.copy_from_slice(&data[offset..offset + NUM_LAGS_12_KHZ]);

        Self {
            pitch_buffer_24k,
            square_energies_24k,
            auto_correlation_12k,
        }
    }

    pub fn pitch_buffer_24khz_view(&self) -> &[f32; BUF_SIZE_24_KHZ] {
        &self.pitch_buffer_24k
    }

    pub fn square_energies_24khz_view(&self) -> &[f32; REFINE_NUM_LAGS_24_KHZ] {
        &self.square_energies_24k
    }

    pub fn auto_correlation_12khz_view(&self) -> &[f32; NUM_LAGS_12_KHZ] {
        &self.auto_correlation_12k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_sizes_are_valid() {
        let pitch = create_pitch_buffer_24khz_reader();
        assert_eq!(0, pitch.size() % BUF_SIZE_24_KHZ);

        let lp = create_lp_residual_and_pitch_info_reader();
        assert_eq!(0, lp.size() % (BUF_SIZE_24_KHZ + 2));

        let test_data = PitchTestData::new();
        assert_eq!(BUF_SIZE_24_KHZ, test_data.pitch_buffer_24khz_view().len());
        assert_eq!(
            REFINE_NUM_LAGS_24_KHZ,
            test_data.square_energies_24khz_view().len()
        );
        assert_eq!(
            NUM_LAGS_12_KHZ,
            test_data.auto_correlation_12khz_view().len()
        );
    }
}
