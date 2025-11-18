use std::f64::consts::PI;

pub const KERNEL_SIZE: usize = 32;
const KERNEL_OFFSET_COUNT: usize = 32;
const KERNEL_STORAGE_SIZE: usize = KERNEL_SIZE * (KERNEL_OFFSET_COUNT + 1);

fn sinc_scale_factor(io_ratio: f64) -> f64 {
    let factor = if io_ratio > 1.0 { 1.0 / io_ratio } else { 1.0 };
    factor * 0.9
}

fn convolve(input: &[f32], k1: &[f32], k2: &[f32], interp: f64) -> f32 {
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    for i in 0..KERNEL_SIZE {
        sum1 += input[i] * k1[i];
        sum2 += input[i] * k2[i];
    }
    ((1.0 - interp) * sum1 as f64 + interp * sum2 as f64) as f32
}

pub struct SincResampler {
    io_sample_rate_ratio: f64,
    request_frames: usize,
    input_buffer_size: usize,
    kernel_storage: Vec<f32>,
    kernel_pre_sinc_storage: Vec<f32>,
    kernel_window_storage: Vec<f32>,
    input_buffer: Vec<f32>,
    virtual_source_idx: f64,
    buffer_primed: bool,
    block_size: usize,
    r0: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    r4: usize,
}

impl SincResampler {
    pub fn new(io_sample_rate_ratio: f64, request_frames: usize) -> Self {
        assert!(request_frames > 0);
        let input_buffer_size = request_frames + KERNEL_SIZE;
        let mut resampler = Self {
            io_sample_rate_ratio,
            request_frames,
            input_buffer_size,
            kernel_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            kernel_pre_sinc_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            kernel_window_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            input_buffer: vec![0.0; input_buffer_size],
            virtual_source_idx: 0.0,
            buffer_primed: false,
            block_size: 0,
            r0: 0,
            r1: 0,
            r2: KERNEL_SIZE / 2,
            r3: 0,
            r4: 0,
        };
        resampler.initialize_kernel();
        resampler.flush();
        assert!(resampler.block_size > KERNEL_SIZE);
        resampler
    }

    fn update_regions(&mut self, second_load: bool) {
        self.r0 = if second_load {
            KERNEL_SIZE
        } else {
            KERNEL_SIZE / 2
        };
        self.r3 = self.r0 + self.request_frames - KERNEL_SIZE;
        self.r4 = self.r0 + self.request_frames - KERNEL_SIZE / 2;
        self.block_size = self.r4 - self.r2;
        self.r1 = 0;
    }

    fn initialize_kernel(&mut self) {
        const ALPHA: f64 = 0.16;
        const A0: f64 = 0.5 * (1.0 - ALPHA);
        const A1: f64 = 0.5;
        const A2: f64 = 0.5 * ALPHA;
        let sinc_factor = sinc_scale_factor(self.io_sample_rate_ratio);
        for offset_idx in 0..=KERNEL_OFFSET_COUNT {
            let subsample_offset = offset_idx as f64 / KERNEL_OFFSET_COUNT as f64;
            for i in 0..KERNEL_SIZE {
                let idx = i + offset_idx * KERNEL_SIZE;
                let pre_sinc = PI * ((i as f64) - (KERNEL_SIZE as f64 / 2.0) - subsample_offset);
                self.kernel_pre_sinc_storage[idx] = pre_sinc as f32;
                let x = (i as f64 - subsample_offset) / KERNEL_SIZE as f64;
                let window = (A0 - A1 * (2.0 * PI * x).cos() + A2 * (4.0 * PI * x).cos()) as f32;
                self.kernel_window_storage[idx] = window;
                let value = if pre_sinc == 0.0 {
                    sinc_factor
                } else {
                    (sinc_factor * pre_sinc).sin() / pre_sinc
                };
                self.kernel_storage[idx] = (window as f64 * value) as f32;
            }
        }
    }

    pub fn set_ratio(&mut self, io_sample_rate_ratio: f64) {
        if (self.io_sample_rate_ratio - io_sample_rate_ratio).abs() < f64::EPSILON {
            return;
        }
        self.io_sample_rate_ratio = io_sample_rate_ratio;
        let sinc_factor = sinc_scale_factor(self.io_sample_rate_ratio);
        for offset_idx in 0..=KERNEL_OFFSET_COUNT {
            for i in 0..KERNEL_SIZE {
                let idx = i + offset_idx * KERNEL_SIZE;
                let window = self.kernel_window_storage[idx] as f64;
                let pre_sinc = self.kernel_pre_sinc_storage[idx] as f64;
                let value = if pre_sinc == 0.0 {
                    sinc_factor
                } else {
                    (sinc_factor * pre_sinc).sin() / pre_sinc
                };
                self.kernel_storage[idx] = (window * value) as f32;
            }
        }
    }

    fn buffer_slice_mut(&mut self, offset: usize, len: usize) -> &mut [f32] {
        debug_assert!(offset + len <= self.input_buffer_size);
        &mut self.input_buffer[offset..offset + len]
    }

    pub fn resample<F>(&mut self, frames: usize, destination: &mut [f32], mut provide_input: F)
    where
        F: FnMut(&mut [f32]),
    {
        if frames == 0 {
            return;
        }
        assert!(destination.len() >= frames);
        if !self.buffer_primed {
            let slice = self.buffer_slice_mut(self.r0, self.request_frames);
            provide_input(slice);
            self.buffer_primed = true;
        }
        let current_ratio = self.io_sample_rate_ratio;
        let mut remaining = frames;
        let mut dest_index = 0;
        while remaining > 0 {
            let mut iterations =
                ((self.block_size as f64 - self.virtual_source_idx) / current_ratio).ceil() as i64;
            if iterations < 0 {
                iterations = 0;
            }
            while iterations > 0 {
                let source_idx = self.virtual_source_idx.floor() as usize;
                let subsample_remainder = self.virtual_source_idx - source_idx as f64;
                let virtual_offset_idx = subsample_remainder * KERNEL_OFFSET_COUNT as f64;
                let offset_idx = virtual_offset_idx.floor() as usize;
                let interp = virtual_offset_idx - offset_idx as f64;
                let k1_start = offset_idx * KERNEL_SIZE;
                let k1 = &self.kernel_storage[k1_start..k1_start + KERNEL_SIZE];
                let k2 = &self.kernel_storage[k1_start + KERNEL_SIZE..k1_start + 2 * KERNEL_SIZE];
                let input_idx = self.r1 + source_idx;
                let input = &self.input_buffer[input_idx..input_idx + KERNEL_SIZE];
                destination[dest_index] = convolve(input, k1, k2, interp);
                dest_index += 1;
                remaining -= 1;
                self.virtual_source_idx += current_ratio;
                if remaining == 0 {
                    return;
                }
                iterations -= 1;
            }
            self.virtual_source_idx -= self.block_size as f64;
            let src_range = self.r3..self.r3 + KERNEL_SIZE;
            self.input_buffer.copy_within(src_range, self.r1);
            if self.r0 == self.r2 {
                self.update_regions(true);
            }
            let slice = self.buffer_slice_mut(self.r0, self.request_frames);
            provide_input(slice);
        }
    }

    pub fn chunk_size(&self) -> usize {
        (self.block_size as f64 / self.io_sample_rate_ratio) as usize
    }

    pub fn request_frames(&self) -> usize {
        self.request_frames
    }

    pub fn flush(&mut self) {
        self.virtual_source_idx = 0.0;
        self.buffer_primed = false;
        self.input_buffer.fill(0.0);
        self.update_regions(false);
    }
}
