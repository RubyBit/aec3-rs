use std::cmp;

use crate::audio_processing::audio_frame::AudioFrame;
use crate::audio_processing::audio_util::{
    float_s16_slice_to_float, float_s16_slice_to_s16, float_slice_to_float_s16,
    float_slice_to_float_s16_in_place, s16_slice_to_float_s16,
};
use crate::audio_processing::channel_buffer::ChannelBuffer;
use crate::audio_processing::resampler::push_sinc_resampler::PushSincResampler;
use crate::audio_processing::splitting_filter::SplittingFilter;
use crate::audio_processing::stream_config::StreamConfig;

const SAMPLES_PER_32KHZ_CHANNEL: usize = 320;
const SAMPLES_PER_48KHZ_CHANNEL: usize = 480;
const MAX_SAMPLES_PER_CHANNEL: usize = AudioBuffer::MAX_SAMPLE_RATE / 100;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Band {
    Band0To8kHz = 0,
    Band8To16kHz = 1,
    Band16To24kHz = 2,
}

pub struct AudioBuffer {
    input_num_frames: usize,
    input_num_channels: usize,
    buffer_num_frames: usize,
    buffer_num_channels: usize,
    output_num_frames: usize,
    num_channels: usize,
    num_bands: usize,
    num_split_frames: usize,
    data: ChannelBuffer<f32>,
    split_data: Option<ChannelBuffer<f32>>,
    splitting_filter: Option<SplittingFilter>,
    input_resamplers: Vec<PushSincResampler>,
    output_resamplers: Vec<PushSincResampler>,
    downmix_by_averaging: bool,
    channel_for_downmixing: usize,
}

impl AudioBuffer {
    pub const SPLIT_BAND_SIZE: usize = 160;
    pub const MAX_SAMPLE_RATE: usize = 384_000;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_num_frames: usize,
        input_num_channels: usize,
        buffer_num_frames: usize,
        buffer_num_channels: usize,
        output_num_frames: usize,
    ) -> Self {
        assert!(input_num_frames > 0);
        assert!(buffer_num_frames > 0);
        assert!(output_num_frames > 0);
        assert!(input_num_channels > 0);
        assert!(buffer_num_channels > 0);
        assert!(buffer_num_channels <= input_num_channels);

        let num_bands = num_bands_from_frames(buffer_num_frames);
        let num_split_frames = buffer_num_frames / num_bands;
        let data = ChannelBuffer::new(buffer_num_frames, buffer_num_channels, 1);

        let (split_data, splitting_filter) = if num_bands > 1 {
            (
                Some(ChannelBuffer::new(
                    buffer_num_frames,
                    buffer_num_channels,
                    num_bands,
                )),
                Some(SplittingFilter::new(
                    buffer_num_channels,
                    num_bands,
                    buffer_num_frames,
                )),
            )
        } else {
            (None, None)
        };

        let input_resamplers = if input_num_frames != buffer_num_frames {
            (0..buffer_num_channels)
                .map(|_| PushSincResampler::new(input_num_frames, buffer_num_frames))
                .collect()
        } else {
            Vec::new()
        };

        let output_resamplers = if output_num_frames != buffer_num_frames {
            (0..buffer_num_channels)
                .map(|_| PushSincResampler::new(buffer_num_frames, output_num_frames))
                .collect()
        } else {
            Vec::new()
        };

        Self {
            input_num_frames,
            input_num_channels,
            buffer_num_frames,
            buffer_num_channels,
            output_num_frames,
            num_channels: buffer_num_channels,
            num_bands,
            num_split_frames,
            data,
            split_data,
            splitting_filter,
            input_resamplers,
            output_resamplers,
            downmix_by_averaging: true,
            channel_for_downmixing: 0,
        }
    }

    pub fn from_sample_rates(
        input_rate: usize,
        input_channels: usize,
        buffer_rate: usize,
        buffer_channels: usize,
        output_rate: usize,
    ) -> Self {
        Self::new(
            input_rate / 100,
            input_channels,
            buffer_rate / 100,
            buffer_channels,
            output_rate / 100,
        )
    }

    pub fn set_downmixing_to_specific_channel(&mut self, channel: usize) {
        self.downmix_by_averaging = false;
        self.channel_for_downmixing = cmp::min(channel, self.input_num_channels - 1);
    }

    pub fn set_downmixing_by_averaging(&mut self) {
        self.downmix_by_averaging = true;
    }

    pub fn set_num_channels(&mut self, num_channels: usize) {
        assert!(num_channels <= self.buffer_num_channels);
        self.num_channels = num_channels;
        self.data.set_num_channels(num_channels);
        if let Some(split) = self.split_data.as_mut() {
            split.set_num_channels(num_channels);
        }
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn num_frames(&self) -> usize {
        self.buffer_num_frames
    }

    pub fn num_frames_per_band(&self) -> usize {
        self.num_split_frames
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub fn channel(&self, channel: usize) -> &[f32] {
        self.data.channel(channel)
    }

    pub fn channel_mut(&mut self, channel: usize) -> &mut [f32] {
        self.data.channel_mut(channel)
    }

    pub fn split_band(&self, channel: usize, band: usize) -> &[f32] {
        self.split_data
            .as_ref()
            .unwrap_or(&self.data)
            .band(channel, band)
    }

    pub fn split_band_mut(&mut self, channel: usize, band: usize) -> &mut [f32] {
        if self.num_bands > 1 {
            self.split_data.as_mut().unwrap().band_mut(channel, band)
        } else {
            self.data.band_mut(channel, band)
        }
    }

    pub fn copy_from_frame(&mut self, frame: &AudioFrame) {
        assert_eq!(frame.num_channels, self.input_num_channels);
        assert_eq!(frame.samples_per_channel, self.input_num_frames);
        self.restore_num_channels();
        let resampling_required = self.input_num_frames != self.buffer_num_frames;
        let interleaved = frame.data();

        if self.num_channels == 1 {
            self.copy_from_frame_mono(interleaved, resampling_required);
        } else {
            self.copy_from_frame_multi(interleaved, resampling_required);
        }
    }

    fn copy_from_frame_mono(&mut self, interleaved: &[i16], resampling_required: bool) {
        if self.input_num_channels == 1 {
            if resampling_required {
                let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
                s16_slice_to_float_s16(interleaved, &mut temp[..self.input_num_frames]);
                self.input_resamplers[0]
                    .resample_f32(&temp[..self.input_num_frames], self.data.channel_mut(0));
            } else {
                s16_slice_to_float_s16(interleaved, self.data.channel_mut(0));
            }
            return;
        }

        if resampling_required {
            let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
            let working = &mut temp[..self.input_num_frames];
            if self.downmix_by_averaging {
                downmix_interleaved_to_mono_i16(
                    interleaved,
                    self.input_num_frames,
                    self.input_num_channels,
                    working,
                );
            } else {
                let channel = cmp::min(self.channel_for_downmixing, self.input_num_channels - 1);
                extract_channel_i16(
                    interleaved,
                    self.input_num_channels,
                    self.input_num_frames,
                    channel,
                    working,
                );
            }
            self.input_resamplers[0].resample_f32(working, self.data.channel_mut(0));
        } else {
            let dest = self.data.channel_mut(0);
            if self.downmix_by_averaging {
                downmix_interleaved_to_mono_i16(
                    interleaved,
                    self.input_num_frames,
                    self.input_num_channels,
                    dest,
                );
            } else {
                let channel = cmp::min(self.channel_for_downmixing, self.input_num_channels - 1);
                extract_channel_i16(
                    interleaved,
                    self.input_num_channels,
                    self.input_num_frames,
                    channel,
                    dest,
                );
            }
        }
    }

    fn copy_from_frame_multi(&mut self, interleaved: &[i16], resampling_required: bool) {
        if resampling_required {
            let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
            for ch in 0..self.num_channels {
                extract_channel_i16(
                    interleaved,
                    self.input_num_channels,
                    self.input_num_frames,
                    ch,
                    &mut temp[..self.input_num_frames],
                );
                self.input_resamplers[ch]
                    .resample_f32(&temp[..self.input_num_frames], self.data.channel_mut(ch));
            }
        } else {
            for ch in 0..self.num_channels {
                extract_channel_i16(
                    interleaved,
                    self.input_num_channels,
                    self.input_num_frames,
                    ch,
                    self.data.channel_mut(ch),
                );
            }
        }
    }

    pub fn copy_from(&mut self, data: &[&[f32]], stream_config: &StreamConfig) {
        assert_eq!(data.len(), stream_config.num_channels());
        assert_eq!(stream_config.num_frames(), self.input_num_frames);
        self.restore_num_channels();
        let downmix_needed = self.input_num_channels > 1 && self.num_channels == 1;
        let resampling_required = self.input_num_frames != self.buffer_num_frames;

        if downmix_needed {
            let mut scratch = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
            let downmixed = if self.downmix_by_averaging {
                average_channels(
                    data,
                    self.input_num_frames,
                    self.input_num_channels,
                    &mut scratch[..self.input_num_frames],
                )
            } else {
                let channel = cmp::min(self.channel_for_downmixing, self.input_num_channels - 1);
                &data[channel][..self.input_num_frames]
            };

            let dest = self.data.channel_mut(0);
            if resampling_required {
                self.input_resamplers[0].resample_f32(downmixed, dest);
                float_slice_to_float_s16_in_place(dest);
            } else {
                float_slice_to_float_s16(downmixed, dest);
            }
            return;
        }

        for ch in 0..self.num_channels {
            let dest = self.data.channel_mut(ch);
            if resampling_required {
                self.input_resamplers[ch].resample_f32(data[ch], dest);
                float_slice_to_float_s16_in_place(dest);
            } else {
                float_slice_to_float_s16(data[ch], dest);
            }
        }
    }

    pub fn copy_to_frame(&mut self, frame: &mut AudioFrame) {
        let frame_channels = frame.num_channels;
        assert!(frame_channels == self.num_channels || self.num_channels == 1);
        assert!(frame_channels > 0);
        assert_eq!(frame.samples_per_channel, self.output_num_frames);
        let resampling_required = self.buffer_num_frames != self.output_num_frames;
        let interleaved = frame.mutable_data();

        if self.num_channels == 1 {
            self.copy_mono_to_frame(interleaved, frame_channels, resampling_required);
        } else {
            self.copy_multi_to_frame(interleaved, frame_channels, resampling_required);
        }
    }

    fn copy_mono_to_frame(
        &mut self,
        interleaved: &mut [i16],
        frame_channels: usize,
        resampling_required: bool,
    ) {
        let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
        let source = if resampling_required {
            self.output_resamplers[0]
                .resample_f32(self.data.channel(0), &mut temp[..self.output_num_frames]);
            &temp[..self.output_num_frames]
        } else {
            self.data.channel(0)
        };

        if frame_channels == 1 {
            float_s16_slice_to_s16(source, interleaved);
        } else {
            for frame_idx in 0..self.output_num_frames {
                let value = float_s16_to_s16_sample(source[frame_idx]);
                for ch in 0..frame_channels {
                    interleaved[frame_idx * frame_channels + ch] = value;
                }
            }
        }
    }

    fn copy_multi_to_frame(
        &mut self,
        interleaved: &mut [i16],
        frame_channels: usize,
        resampling_required: bool,
    ) {
        let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
        if resampling_required {
            for ch in 0..self.num_channels {
                self.output_resamplers[ch]
                    .resample_f32(self.data.channel(ch), &mut temp[..self.output_num_frames]);
                interleave_channel(
                    ch,
                    frame_channels,
                    &temp[..self.output_num_frames],
                    interleaved,
                );
            }
        } else {
            for ch in 0..self.num_channels {
                interleave_channel(ch, frame_channels, self.data.channel(ch), interleaved);
            }
        }

        for ch in self.num_channels..frame_channels {
            for frame_idx in 0..self.output_num_frames {
                interleaved[frame_idx * frame_channels + ch] =
                    interleaved[frame_idx * frame_channels];
            }
        }
    }

    pub fn copy_to_stream(&mut self, stream_config: &StreamConfig, data: &mut [&mut [f32]]) {
        assert_eq!(stream_config.num_frames(), self.output_num_frames);
        assert!(data.len() >= stream_config.num_channels());
        assert!(stream_config.num_channels() >= self.num_channels);
        let resampling_required = self.output_num_frames != self.buffer_num_frames;
        let (existing, extra_all) = data.split_at_mut(self.num_channels);
        if resampling_required {
            let mut temp = [0.0f32; MAX_SAMPLES_PER_CHANNEL];
            for (ch, dest) in existing.iter_mut().enumerate() {
                self.output_resamplers[ch]
                    .resample_f32(self.data.channel(ch), &mut temp[..self.output_num_frames]);
                float_s16_slice_to_float(
                    &temp[..self.output_num_frames],
                    &mut dest[..self.output_num_frames],
                );
            }
        } else {
            for (ch, dest) in existing.iter_mut().enumerate() {
                float_s16_slice_to_float(
                    self.data.channel(ch),
                    &mut dest[..self.output_num_frames],
                );
            }
        }
        let extra_needed = stream_config
            .num_channels()
            .saturating_sub(self.num_channels);
        if extra_needed > 0 {
            let reference = &existing[0][..self.output_num_frames];
            for dest in extra_all.iter_mut().take(extra_needed) {
                dest[..self.output_num_frames].copy_from_slice(reference);
            }
        }
    }

    pub fn copy_to_audio_buffer(&mut self, buffer: &mut AudioBuffer) {
        assert_eq!(buffer.num_frames(), self.output_num_frames);
        assert!(buffer.num_channels() >= self.num_channels);
        let resampling_required = self.output_num_frames != self.buffer_num_frames;
        if resampling_required {
            for ch in 0..self.num_channels {
                self.output_resamplers[ch]
                    .resample_f32(self.data.channel(ch), buffer.channel_mut(ch));
            }
        } else {
            for ch in 0..self.num_channels {
                buffer
                    .channel_mut(ch)
                    .copy_from_slice(self.data.channel(ch));
            }
        }

        if buffer.num_channels() > self.num_channels {
            let reference = buffer.channel(0).to_vec();
            for ch in self.num_channels..buffer.num_channels() {
                buffer.channel_mut(ch).copy_from_slice(&reference);
            }
        }
    }

    pub fn split_into_frequency_bands(&mut self) {
        if self.num_bands > 1 {
            if let (Some(filter), Some(split)) =
                (self.splitting_filter.as_mut(), self.split_data.as_mut())
            {
                filter.analysis(&self.data, split);
            }
        }
    }

    pub fn merge_frequency_bands(&mut self) {
        if self.num_bands > 1 {
            if let (Some(filter), Some(split)) =
                (self.splitting_filter.as_mut(), self.split_data.as_ref())
            {
                filter.synthesis(split, &mut self.data);
            }
        }
    }

    pub fn export_split_channel_data(&self, channel: usize, split_band_data: &mut [&mut [i16]]) {
        assert_eq!(split_band_data.len(), self.num_bands);
        for band in 0..self.num_bands {
            float_s16_slice_to_s16(self.split_band(channel, band), split_band_data[band]);
        }
    }

    pub fn import_split_channel_data(&mut self, channel: usize, split_band_data: &[&[i16]]) {
        assert_eq!(split_band_data.len(), self.num_bands);
        for band in 0..self.num_bands {
            s16_slice_to_float_s16(split_band_data[band], self.split_band_mut(channel, band));
        }
    }

    fn restore_num_channels(&mut self) {
        self.num_channels = self.buffer_num_channels;
        self.data.set_num_channels(self.buffer_num_channels);
        if let Some(split) = self.split_data.as_mut() {
            split.set_num_channels(self.buffer_num_channels);
        }
    }
}

fn downmix_interleaved_to_mono_i16(
    interleaved: &[i16],
    num_frames: usize,
    num_channels: usize,
    dest: &mut [f32],
) {
    for frame in 0..num_frames {
        let mut sum: i32 = 0;
        for ch in 0..num_channels {
            sum += interleaved[frame * num_channels + ch] as i32;
        }
        dest[frame] = (sum / num_channels as i32) as f32;
    }
}

fn extract_channel_i16(
    interleaved: &[i16],
    num_channels: usize,
    samples_per_channel: usize,
    channel: usize,
    dest: &mut [f32],
) {
    let mut idx = channel;
    for sample in dest.iter_mut().take(samples_per_channel) {
        *sample = interleaved[idx] as f32;
        idx += num_channels;
    }
}

fn average_channels<'a>(
    data: &[&'a [f32]],
    num_frames: usize,
    num_channels: usize,
    dest: &'a mut [f32],
) -> &'a [f32] {
    for frame in 0..num_frames {
        let mut value = data[0][frame];
        for ch in 1..num_channels {
            value += data[ch][frame];
        }
        dest[frame] = value / num_channels as f32;
    }
    &dest[..num_frames]
}

fn interleave_channel(channel: usize, num_channels: usize, src: &[f32], dest: &mut [i16]) {
    let mut idx = channel;
    for &sample in src {
        dest[idx] = float_s16_to_s16_sample(sample);
        idx += num_channels;
    }
}

fn float_s16_to_s16_sample(sample: f32) -> i16 {
    let clamped = sample.clamp(-32768.0, 32767.0);
    (clamped + clamped.signum() * 0.5).trunc() as i16
}

fn num_bands_from_frames(num_frames: usize) -> usize {
    if num_frames == SAMPLES_PER_32KHZ_CHANNEL {
        2
    } else if num_frames == SAMPLES_PER_48KHZ_CHANNEL {
        3
    } else {
        1
    }
}
