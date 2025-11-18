use std::time::Instant;

use crate::audio_processing::channel_layout::ChannelLayout;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VadActivity {
    Active,
    Passive,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SpeechType {
    NormalSpeech,
    Plc,
    Cng,
    PlcCng,
    CodecPlc,
    Undefined,
}

pub struct AudioFrame {
    data: [i16; AudioFrame::MAX_DATA_SIZE_SAMPLES],
    muted: bool,
    profile_timestamp: Option<Instant>,
    pub timestamp: u32,
    pub elapsed_time_ms: i64,
    pub ntp_time_ms: i64,
    pub samples_per_channel: usize,
    pub sample_rate_hz: i32,
    pub num_channels: usize,
    pub channel_layout: ChannelLayout,
    pub speech_type: SpeechType,
    pub vad_activity: VadActivity,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            data: [0; AudioFrame::MAX_DATA_SIZE_SAMPLES],
            muted: true,
            profile_timestamp: None,
            timestamp: 0,
            elapsed_time_ms: -1,
            ntp_time_ms: -1,
            samples_per_channel: 0,
            sample_rate_hz: 0,
            num_channels: 0,
            channel_layout: ChannelLayout::None,
            speech_type: SpeechType::Undefined,
            vad_activity: VadActivity::Unknown,
        }
    }
}

impl AudioFrame {
    pub const MAX_DATA_SIZE_SAMPLES: usize = 7680;
    pub const MAX_DATA_SIZE_BYTES: usize = Self::MAX_DATA_SIZE_SAMPLES * std::mem::size_of::<i16>();

    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn reset_without_muting(&mut self) {
        let muted = self.muted;
        self.reset();
        self.muted = muted;
    }

    pub fn update_frame(
        &mut self,
        timestamp: u32,
        data: &[i16],
        samples_per_channel: usize,
        sample_rate_hz: i32,
        speech_type: SpeechType,
        vad_activity: VadActivity,
        num_channels: usize,
    ) {
        assert!(samples_per_channel * num_channels <= Self::MAX_DATA_SIZE_SAMPLES);
        self.timestamp = timestamp;
        self.samples_per_channel = samples_per_channel;
        self.sample_rate_hz = sample_rate_hz;
        self.speech_type = speech_type;
        self.vad_activity = vad_activity;
        self.num_channels = num_channels;
        self.channel_layout = ChannelLayout::guess_from_channel_count(num_channels);
        self.muted = false;
        let len = samples_per_channel * num_channels;
        self.data[..len].copy_from_slice(&data[..len]);
    }

    pub fn copy_from(&mut self, other: &AudioFrame) {
        let len = other.data_len();
        self.timestamp = other.timestamp;
        self.elapsed_time_ms = other.elapsed_time_ms;
        self.ntp_time_ms = other.ntp_time_ms;
        self.samples_per_channel = other.samples_per_channel;
        self.sample_rate_hz = other.sample_rate_hz;
        self.num_channels = other.num_channels;
        self.channel_layout = other.channel_layout;
        self.speech_type = other.speech_type;
        self.vad_activity = other.vad_activity;
        self.muted = other.muted;
        self.profile_timestamp = other.profile_timestamp;
        self.data[..len].copy_from_slice(&other.data[..len]);
    }

    pub fn update_profile_timestamp(&mut self) {
        self.profile_timestamp = Some(Instant::now());
    }

    pub fn elapsed_profile_time_ms(&self) -> i64 {
        match self.profile_timestamp {
            Some(ts) => ts.elapsed().as_millis() as i64,
            None => -1,
        }
    }

    pub fn data(&self) -> &[i16] {
        if self.muted {
            static ZERO: [i16; AudioFrame::MAX_DATA_SIZE_SAMPLES] =
                [0; AudioFrame::MAX_DATA_SIZE_SAMPLES];
            &ZERO[..self.data_len()]
        } else {
            &self.data[..self.data_len()]
        }
    }

    pub fn mutable_data(&mut self) -> &mut [i16] {
        let len = self.data_len();
        if self.muted {
            self.data[..len].fill(0);
            self.muted = false;
        }
        &mut self.data[..len]
    }

    pub fn mute(&mut self) {
        self.muted = true;
    }

    pub fn muted(&self) -> bool {
        self.muted
    }

    fn data_len(&self) -> usize {
        self.samples_per_channel * self.num_channels
    }
}
