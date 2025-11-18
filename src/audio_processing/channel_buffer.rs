//! Port of WebRTC's ChannelBuffer and IFChannelBuffer helpers.

use crate::audio_processing::audio_util::float_s16_slice_to_s16;

#[derive(Debug, Clone)]
pub struct ChannelBuffer<T: Copy + Default> {
    data: Vec<T>,
    num_frames: usize,
    num_frames_per_band: usize,
    num_allocated_channels: usize,
    num_channels: usize,
    num_bands: usize,
}

impl<T: Copy + Default> ChannelBuffer<T> {
    pub fn new(num_frames: usize, num_channels: usize, num_bands: usize) -> Self {
        assert!(num_frames > 0);
        assert!(num_channels > 0);
        assert!(num_bands >= 1);
        assert_eq!(num_frames % num_bands, 0);
        Self {
            data: vec![T::default(); num_frames * num_channels],
            num_frames,
            num_frames_per_band: num_frames / num_bands,
            num_allocated_channels: num_channels,
            num_channels,
            num_bands,
        }
    }

    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    pub fn num_frames_per_band(&self) -> usize {
        self.num_frames_per_band
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub fn set_num_channels(&mut self, num_channels: usize) {
        assert!(num_channels <= self.num_allocated_channels);
        self.num_channels = num_channels;
    }

    pub fn channel(&self, channel: usize) -> &[T] {
        assert!(channel < self.num_channels);
        let start = channel * self.num_frames;
        &self.data[start..start + self.num_frames]
    }

    pub fn channel_mut(&mut self, channel: usize) -> &mut [T] {
        assert!(channel < self.num_channels);
        let start = channel * self.num_frames;
        &mut self.data[start..start + self.num_frames]
    }

    pub fn band(&self, channel: usize, band: usize) -> &[T] {
        assert!(channel < self.num_channels);
        assert!(band < self.num_bands);
        let start = channel * self.num_frames + band * self.num_frames_per_band;
        &self.data[start..start + self.num_frames_per_band]
    }

    pub fn band_mut(&mut self, channel: usize, band: usize) -> &mut [T] {
        assert!(channel < self.num_channels);
        assert!(band < self.num_bands);
        let start = channel * self.num_frames + band * self.num_frames_per_band;
        &mut self.data[start..start + self.num_frames_per_band]
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

pub struct IFChannelBuffer {
    ivalid: bool,
    ibuf: ChannelBuffer<i16>,
    fvalid: bool,
    fbuf: ChannelBuffer<f32>,
}

impl IFChannelBuffer {
    pub fn new(num_frames: usize, num_channels: usize, num_bands: usize) -> Self {
        Self {
            ivalid: true,
            ibuf: ChannelBuffer::new(num_frames, num_channels, num_bands),
            fvalid: true,
            fbuf: ChannelBuffer::new(num_frames, num_channels, num_bands),
        }
    }

    pub fn ibuf(&mut self) -> &mut ChannelBuffer<i16> {
        self.refresh_i();
        self.fvalid = false;
        &mut self.ibuf
    }

    pub fn fbuf(&mut self) -> &mut ChannelBuffer<f32> {
        self.refresh_f();
        self.ivalid = false;
        &mut self.fbuf
    }

    pub fn ibuf_const(&mut self) -> &ChannelBuffer<i16> {
        self.refresh_i();
        &self.ibuf
    }

    pub fn fbuf_const(&mut self) -> &ChannelBuffer<f32> {
        self.refresh_f();
        &self.fbuf
    }

    pub fn set_num_channels(&mut self, num_channels: usize) {
        self.ibuf.set_num_channels(num_channels);
        self.fbuf.set_num_channels(num_channels);
    }

    fn refresh_f(&mut self) {
        if !self.fvalid {
            assert!(self.ivalid);
            for ch in 0..self.ibuf.num_channels() {
                let src = self.ibuf.channel(ch);
                let dst = self.fbuf.channel_mut(ch);
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = *s as f32;
                }
            }
            self.fbuf.set_num_channels(self.ibuf.num_channels());
            self.fvalid = true;
        }
    }

    fn refresh_i(&mut self) {
        if !self.ivalid {
            assert!(self.fvalid);
            for ch in 0..self.fbuf.num_channels() {
                let src = self.fbuf.channel(ch);
                let dst = self.ibuf.channel_mut(ch);
                float_s16_slice_to_s16(src, dst);
            }
            self.ibuf.set_num_channels(self.fbuf.num_channels());
            self.ivalid = true;
        }
    }
}
