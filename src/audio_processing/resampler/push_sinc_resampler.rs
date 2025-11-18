use std::mem;

use super::sinc_resampler::{KERNEL_SIZE, SincResampler};
use crate::audio_processing::audio_util::{float_s16_slice_to_s16, s16_slice_to_float_s16};

/// Push-based wrapper around `SincResampler` mirroring the WebRTC implementation.
pub struct PushSincResampler {
    resampler: SincResampler,
    destination_frames: usize,
    needs_prime: bool,
    scratch: Vec<f32>,
}

impl PushSincResampler {
    pub fn new(source_frames: usize, destination_frames: usize) -> Self {
        assert!(source_frames > 0);
        assert!(destination_frames > 0);
        let io_ratio = source_frames as f64 / destination_frames as f64;
        let resampler = SincResampler::new(io_ratio, source_frames);
        Self {
            resampler,
            destination_frames,
            needs_prime: true,
            scratch: Vec::new(),
        }
    }

    pub fn resample_f32(&mut self, source: &[f32], destination: &mut [f32]) -> usize {
        assert_eq!(source.len(), self.resampler.request_frames());
        assert!(destination.len() >= self.destination_frames);
        self.ensure_prime();
        self.process_with_source(
            &mut destination[..self.destination_frames],
            source,
            |src, dest| dest.copy_from_slice(src),
        );
        self.destination_frames
    }

    pub fn resample_i16(&mut self, source: &[i16], destination: &mut [i16]) -> usize {
        assert_eq!(source.len(), self.resampler.request_frames());
        assert!(destination.len() >= self.destination_frames);
        let required = self.destination_frames.max(self.resampler.chunk_size());
        self.ensure_prime();
        let mut scratch = mem::take(&mut self.scratch);
        if scratch.len() < required {
            scratch.resize(required, 0.0);
        }
        let (output_slice, _) = scratch.split_at_mut(self.destination_frames);
        self.process_with_source(output_slice, source, |src, dest| {
            s16_slice_to_float_s16(src, dest);
        });
        float_s16_slice_to_s16(output_slice, &mut destination[..self.destination_frames]);
        self.scratch = scratch;
        self.destination_frames
    }

    fn process_with_source<T, F>(&mut self, destination: &mut [f32], source: &[T], mut fill: F)
    where
        F: FnMut(&[T], &mut [f32]),
    {
        let mut consumed = 0usize;
        let total = source.len();
        self.resampler
            .resample(self.destination_frames, destination, |buffer| {
                let len = buffer.len();
                assert!(consumed + len <= total);
                fill(&source[consumed..consumed + len], buffer);
                consumed += len;
            });
        assert_eq!(consumed, total);
    }

    fn ensure_prime(&mut self) {
        if !self.needs_prime {
            return;
        }
        let chunk = self.resampler.chunk_size();
        if chunk > 0 {
            let mut scratch = mem::take(&mut self.scratch);
            if scratch.len() < chunk {
                scratch.resize(chunk, 0.0);
            }
            self.resampler
                .resample(chunk, &mut scratch[..chunk], |buffer| buffer.fill(0.0));
            self.scratch = scratch;
        }
        self.needs_prime = false;
    }

    pub fn algorithmic_delay_seconds(source_rate_hz: i32) -> f32 {
        (1.0 / source_rate_hz as f32) * (KERNEL_SIZE as f32 / 2.0)
    }
}
