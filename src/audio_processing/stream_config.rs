//! Minimal port of WebRTC's `StreamConfig` helper.

const CHUNK_SIZE_MS: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamConfig {
    sample_rate_hz: usize,
    num_channels: usize,
    has_keyboard: bool,
    num_frames: usize,
}

impl StreamConfig {
    pub fn new(sample_rate_hz: usize, num_channels: usize, has_keyboard: bool) -> Self {
        let num_frames = frames_from_rate(sample_rate_hz);
        Self {
            sample_rate_hz,
            num_channels,
            has_keyboard,
            num_frames,
        }
    }

    pub fn sample_rate_hz(&self) -> usize {
        self.sample_rate_hz
    }

    pub fn set_sample_rate_hz(&mut self, rate: usize) {
        self.sample_rate_hz = rate;
        self.num_frames = frames_from_rate(rate);
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn set_num_channels(&mut self, channels: usize) {
        self.num_channels = channels;
    }

    pub fn has_keyboard(&self) -> bool {
        self.has_keyboard
    }

    pub fn set_has_keyboard(&mut self, has_keyboard: bool) {
        self.has_keyboard = has_keyboard;
    }

    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    pub fn num_samples(&self) -> usize {
        self.num_frames * self.num_channels
    }
}

fn frames_from_rate(rate: usize) -> usize {
    if rate == 0 {
        return 0;
    }
    (CHUNK_SIZE_MS * rate) / 1000
}
