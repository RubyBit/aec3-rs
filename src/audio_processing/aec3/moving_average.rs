/// Simple moving average over vectors with fixed length.
pub struct MovingAverage {
    num_elements: usize,
    memory: Vec<f32>,
    history_len: usize,
    scaling: f32,
    mem_index: usize,
}

impl MovingAverage {
    pub fn new(num_elements: usize, window_length: usize) -> Self {
        assert!(
            num_elements > 0,
            "moving average requires at least one element"
        );
        assert!(
            window_length > 0,
            "moving average window length must be positive"
        );
        let history_len = window_length.saturating_sub(1);
        let memory = vec![0.0; num_elements * history_len];
        let scaling = 1.0 / window_length as f32;
        Self {
            num_elements,
            memory,
            history_len,
            scaling,
            mem_index: 0,
        }
    }

    pub fn average(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.num_elements);
        assert_eq!(output.len(), self.num_elements);

        output.copy_from_slice(input);
        if self.history_len > 0 {
            for chunk in self.memory.chunks_exact(self.num_elements) {
                for (dst, &value) in output.iter_mut().zip(chunk.iter()) {
                    *dst += value;
                }
            }
        }

        for value in output.iter_mut() {
            *value *= self.scaling;
        }

        if self.history_len == 0 {
            return;
        }

        let start = self.mem_index * self.num_elements;
        let end = start + self.num_elements;
        self.memory[start..end].copy_from_slice(input);
        self.mem_index = (self.mem_index + 1) % self.history_len;
    }
}

#[cfg(test)]
mod tests {
    use super::MovingAverage;

    #[test]
    fn averages_match_reference() {
        const NUM_ELEM: usize = 4;
        const MEM_LEN: usize = 3;
        let mut avg = MovingAverage::new(NUM_ELEM, MEM_LEN);
        let mut output = [0.0f32; NUM_ELEM];
        let data1 = [1.0f32, 2.0, 3.0, 4.0];
        let data2 = [5.0f32, 1.0, 9.0, 7.0];
        let data3 = [3.0f32, 3.0, 5.0, 6.0];
        let data4 = [8.0f32, 4.0, 2.0, 1.0];

        avg.average(&data1, &mut output);
        for i in 0..NUM_ELEM {
            assert!((output[i] - data1[i] / MEM_LEN as f32).abs() < 1e-6);
        }

        avg.average(&data2, &mut output);
        for i in 0..NUM_ELEM {
            assert!((output[i] - (data1[i] + data2[i]) / MEM_LEN as f32).abs() < 1e-6);
        }

        avg.average(&data3, &mut output);
        for i in 0..NUM_ELEM {
            assert!((output[i] - (data1[i] + data2[i] + data3[i]) / MEM_LEN as f32).abs() < 1e-6);
        }

        avg.average(&data4, &mut output);
        for i in 0..NUM_ELEM {
            assert!((output[i] - (data2[i] + data3[i] + data4[i]) / MEM_LEN as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn pass_through_when_unit_window() {
        const NUM_ELEM: usize = 4;
        const MEM_LEN: usize = 1;
        let mut avg = MovingAverage::new(NUM_ELEM, MEM_LEN);
        let mut output = [0.0f32; NUM_ELEM];
        let data1 = [1.0f32, 2.0, 3.0, 4.0];
        let data2 = [5.0f32, 1.0, 9.0, 7.0];
        let data3 = [3.0f32, 3.0, 5.0, 6.0];
        let data4 = [8.0f32, 4.0, 2.0, 1.0];

        avg.average(&data1, &mut output);
        assert_eq!(output, data1);

        avg.average(&data2, &mut output);
        assert_eq!(output, data2);

        avg.average(&data3, &mut output);
        assert_eq!(output, data3);

        avg.average(&data4, &mut output);
        assert_eq!(output, data4);
    }
}
