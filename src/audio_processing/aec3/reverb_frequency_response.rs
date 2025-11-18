use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;

pub struct ReverbFrequencyResponse {
    average_decay: f32,
    tail_response: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl ReverbFrequencyResponse {
    pub fn new() -> Self {
        Self {
            average_decay: 0.0,
            tail_response: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    pub fn update(
        &mut self,
        frequency_response: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        filter_delay_blocks: i32,
        linear_filter_quality: Option<f32>,
        stationary_block: bool,
    ) {
        if stationary_block {
            return;
        }
        let quality = match linear_filter_quality {
            Some(q) => q,
            None => return,
        };
        if frequency_response.is_empty() {
            return;
        }
        if filter_delay_blocks < 0 {
            return;
        }
        let delay = filter_delay_blocks as usize;
        if delay >= frequency_response.len() {
            return;
        }
        self.update_internal(frequency_response, delay, quality);
    }

    pub fn frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.tail_response
    }

    fn update_internal(
        &mut self,
        frequency_response: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        filter_delay_blocks: usize,
        linear_filter_quality: f32,
    ) {
        let freq_resp_tail = &frequency_response[frequency_response.len() - 1];
        let freq_resp_direct_path = &frequency_response[filter_delay_blocks];
        let average_decay = average_decay_within_filter(freq_resp_direct_path, freq_resp_tail);
        let smoothing = 0.2f32 * linear_filter_quality;
        self.average_decay += smoothing * (average_decay - self.average_decay);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            self.tail_response[k] = freq_resp_direct_path[k] * self.average_decay;
        }
        if FFT_LENGTH_BY_2_PLUS_1 > 2 {
            for k in 1..FFT_LENGTH_BY_2_PLUS_1 - 1 {
                let avg_neighbour =
                    0.5f32 * (self.tail_response[k - 1] + self.tail_response[k + 1]);
                if avg_neighbour > self.tail_response[k] {
                    self.tail_response[k] = avg_neighbour;
                }
            }
        }
    }
}

fn average_decay_within_filter(
    freq_resp_direct_path: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    freq_resp_tail: &[f32; FFT_LENGTH_BY_2_PLUS_1],
) -> f32 {
    const SKIP_BINS: usize = 1;
    let direct_energy: f32 = freq_resp_direct_path[SKIP_BINS..].iter().sum();
    if direct_energy == 0.0 {
        return 0.0;
    }
    let tail_energy: f32 = freq_resp_tail[SKIP_BINS..].iter().sum();
    tail_energy / direct_energy
}

impl Default for ReverbFrequencyResponse {
    fn default() -> Self {
        Self::new()
    }
}
