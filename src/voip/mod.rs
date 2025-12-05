use std::fmt;

use crate::api::{
    config::EchoCanceller3Config,
    control::{EchoControl, Metrics},
};
use crate::audio_processing::aec3::{
    echo_canceller3::EchoCanceller3,
};
use crate::audio_processing::audio_buffer::AudioBuffer;
use crate::audio_processing::high_pass_filter::HighPassFilter;
use crate::audio_processing::stream_config::StreamConfig;

/// Convenient result type used by the VoIP wrapper.
pub type VoipResult<T> = std::result::Result<T, VoipAec3Error>;

/// Builder for [`VoipAec3`].
#[derive(Debug, Clone)]
pub struct VoipAec3Builder {
    render_sample_rate_hz: i32,
    capture_sample_rate_hz: i32,
    render_channels: usize,
    capture_channels: usize,
    enable_high_pass: bool,
    config: Option<EchoCanceller3Config>,
    initial_delay_ms: Option<i32>,
}

impl VoipAec3Builder {
    /// Creates a new builder for the specified sample rate and channel layout.
    /// 
    /// sample_rate_hz: Sample rate in Hz for both devices (e.g., 16000, 32000, 44100, 48000).
    /// You can override this sample rate with the other functions if needed.
    pub fn new(sample_rate_hz: i32, render_channels: usize, capture_channels: usize) -> Self {
        Self {
            render_sample_rate_hz: sample_rate_hz, // TODO: Maybe change interface instead? Seems a bit odd
            capture_sample_rate_hz: sample_rate_hz,
            render_channels,
            capture_channels,
            enable_high_pass: true,
            config: None,
            initial_delay_ms: None,
        }
    }

    /// Overrides the default AEC3 configuration.
    pub fn with_config(mut self, config: EchoCanceller3Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Enables or disables the capture high-pass filter (enabled by default).
    pub fn enable_high_pass(mut self, enable: bool) -> Self {
        self.enable_high_pass = enable;
        self
    }

    /// Sets an optional initial buffer delay hint (in milliseconds).
    pub fn initial_delay_ms(mut self, delay_ms: i32) -> Self {
        self.initial_delay_ms = Some(delay_ms);
        self
    }

    pub fn capture_sample_rate_hz(mut self, sample_rate_hz: i32) -> Self {
        self.capture_sample_rate_hz = sample_rate_hz;
        self
    }

    pub fn render_sample_rate_hz(mut self, sample_rate_hz: i32) -> Self {
        self.render_sample_rate_hz = sample_rate_hz;
        self
    }

    /// Consumes the builder and creates the [`VoipAec3`] pipeline.
    pub fn build(self) -> VoipResult<VoipAec3> {
        if !self.valid_sample_rates(self.capture_sample_rate_hz, self.render_sample_rate_hz) {
            return Err(VoipAec3Error::UnsupportedSampleRate {
                render: self.render_sample_rate_hz,
                capture: self.capture_sample_rate_hz,
            });
        }
        
        if self.render_channels == 0 || self.capture_channels == 0 {
            return Err(VoipAec3Error::InvalidChannelCount {
                render: self.render_channels,
                capture: self.capture_channels,
            });
        }

        let chosen_config = self.config.unwrap_or_else(|| {
            EchoCanceller3::create_default_config(self.render_channels, self.capture_channels)
        });
        // Use an internal full-band rate for the AEC core. For 44.1 kHz input
        // we run the internal AEC at 48 kHz to match the library's multi-band
        // expectations while keeping the external stream configuration at
        // 44.1 kHz (as it will be resampled).
        let internal_sample_rate = {
            // check which sample rate is lower than the other if any
            if self.capture_sample_rate_hz == 44_100 || self.render_sample_rate_hz == 44_100 { // if any is 44.1kHz
                48_000
            } else {
                // if any is 16kHz, use 16kHz, else if any is 32kHz use 32kHz, else use 48kHz
                let min_rate = self.capture_sample_rate_hz.min(self.render_sample_rate_hz);
                if min_rate <= 16_000 {
                    16_000
                } else if min_rate <= 32_000 {
                    32_000
                } else {
                    48_000
                }
            }
        };

        let mut aec3 = EchoCanceller3::new(
            chosen_config,
            internal_sample_rate,
            self.render_channels,
            self.capture_channels,
        );

        if let Some(delay) = self.initial_delay_ms {
            aec3.set_audio_buffer_delay(delay);
        }

        let capture_stream_config =
            StreamConfig::new(self.capture_sample_rate_hz, self.capture_channels, false);
        let render_stream_config =
            StreamConfig::new(self.render_sample_rate_hz, self.render_channels, false);

        let capture_frame_samples = capture_stream_config.num_frames();
        let render_frame_samples = render_stream_config.num_frames();

        let capture_buffer = AudioBuffer::from_sample_rates(
            self.capture_sample_rate_hz as usize,
            self.capture_channels,
            internal_sample_rate as usize,
            self.capture_channels,
            self.capture_sample_rate_hz as usize,
        );
        let render_buffer = AudioBuffer::from_sample_rates(
            self.render_sample_rate_hz as usize,
            self.render_channels,
            internal_sample_rate as usize,
            self.render_channels,
            self.render_sample_rate_hz as usize,
        );

        let capture_scratch = allocate_planar_storage(self.capture_channels, capture_frame_samples);
        let render_scratch = allocate_planar_storage(self.render_channels, render_frame_samples);
        let output_scratch = allocate_planar_storage(self.capture_channels, capture_frame_samples);
        let hp_scratch = self
            .enable_high_pass
            .then(|| allocate_planar_storage(self.capture_channels, AudioBuffer::SPLIT_BAND_SIZE));
        let hp_filter = self
            .enable_high_pass
            .then(|| HighPassFilter::new(internal_sample_rate, self.capture_channels));

        Ok(VoipAec3 {
            sample_rate_hz: self.capture_sample_rate_hz,
            capture_frame_samples,
            render_frame_samples,
            render_channels: self.render_channels,
            capture_channels: self.capture_channels,
            aec3,
            capture_buffer,
            render_buffer,
            capture_stream_config,
            render_stream_config,
            capture_scratch,
            render_scratch,
            output_scratch,
            hp_scratch,
            hp_filter,
        })
    }

    fn valid_sample_rates(&self, capture_rate: i32, render_rate: i32) -> bool {
        if (capture_rate < 16_000 || capture_rate > 48_000)
            || (render_rate < 16_000 || render_rate > 48_000)
        {
            return false;
        }
        true
    }
}

/// High-level wrapper that mirrors the WebRTC AEC3 reference usage pattern
/// while exposing an ergonomic Rust API for VoIP pipelines.
pub struct VoipAec3 {
    sample_rate_hz: i32,
    capture_frame_samples: usize,
    render_frame_samples: usize,
    render_channels: usize,
    capture_channels: usize,
    aec3: EchoCanceller3,
    capture_buffer: AudioBuffer,
    render_buffer: AudioBuffer,
    capture_stream_config: StreamConfig,
    render_stream_config: StreamConfig,
    capture_scratch: Vec<Vec<f32>>,
    render_scratch: Vec<Vec<f32>>,
    output_scratch: Vec<Vec<f32>>,
    hp_scratch: Option<Vec<Vec<f32>>>,
    hp_filter: Option<HighPassFilter>,
}

impl VoipAec3 {
    /// Convenience constructor mirroring [`VoipAec3Builder::new`].
    pub fn builder(
        sample_rate_hz: i32,
        render_channels: usize,
        capture_channels: usize,
    ) -> VoipAec3Builder {
        VoipAec3Builder::new(sample_rate_hz, render_channels, capture_channels)
    }

    /// Number of samples per channel expected for each 10 ms frame.
    pub fn capture_frame_samples(&self) -> usize {
        self.capture_frame_samples
    }

    /// Number of samples per channel expected for each 10 ms render frame.
    pub fn render_frame_samples(&self) -> usize {
        self.render_frame_samples
    }

    /// Returns the capture sample rate configured for the pipeline.
    pub fn sample_rate_hz(&self) -> i32 {
        self.sample_rate_hz
    }

    /// Provides the current metrics without running another processing step.
    pub fn metrics(&self) -> Metrics {
        self.aec3.metrics()
    }

    /// Updates the audio buffer delay hint at runtime.
    pub fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.aec3.set_audio_buffer_delay(delay_ms);
    }

    /// Feeds a render (far-end) frame into the pipeline.
    pub fn handle_render_frame(&mut self, render_frame: &[f32]) -> VoipResult<()> {
        validate_frame_length(
            render_frame.len(),
            self.render_channels,
            self.render_frame_samples,
            FrameKind::Render,
        )?;
        copy_interleaved_to_planar(
            render_frame,
            self.render_channels,
            self.render_frame_samples,
            &mut self.render_scratch,
        );
        let render_refs: Vec<&[f32]> = self
            .render_scratch
            .iter()
            .map(|channel| &channel[..self.render_frame_samples])
            .collect();
        self.render_buffer
            .copy_from(&render_refs, &self.render_stream_config);
        self.render_buffer.split_into_frequency_bands();
        self.aec3.analyze_render(&mut self.render_buffer);
        self.render_buffer.merge_frequency_bands();
        Ok(())
    }

    /// Processes a capture (microphone) frame and writes the AEC output.
    pub fn process_capture_frame(
        &mut self,
        capture_frame: &[f32],
        level_change: bool,
        output: &mut [f32],
    ) -> VoipResult<Metrics> {
        validate_frame_length(
            capture_frame.len(),
            self.capture_channels,
            self.capture_frame_samples,
            FrameKind::Capture,
        )?;
        validate_output_length(output.len(), self.capture_channels, self.capture_frame_samples)?;

        copy_interleaved_to_planar(
            capture_frame,
            self.capture_channels,
            self.capture_frame_samples,
            &mut self.capture_scratch,
        );
        let capture_refs: Vec<&[f32]> = self
            .capture_scratch
            .iter()
            .map(|channel| &channel[..self.capture_frame_samples])
            .collect();
        self.capture_buffer
            .copy_from(&capture_refs, &self.capture_stream_config);
        self.aec3.analyze_capture(&mut self.capture_buffer);
        self.capture_buffer.split_into_frequency_bands();

        if let (Some(filter), Some(scratch)) = (self.hp_filter.as_mut(), self.hp_scratch.as_mut()) {
            for (ch, slot) in scratch.iter_mut().enumerate() {
                let src = self.capture_buffer.split_band(ch, 0);
                slot[..src.len()].copy_from_slice(src);
            }
            filter.process(scratch);
            for (ch, slot) in scratch.iter().enumerate() {
                let dst = self.capture_buffer.split_band_mut(ch, 0);
                dst.copy_from_slice(slot);
            }
        }

        self.aec3
            .process_capture(&mut self.capture_buffer, level_change);
        self.capture_buffer.merge_frequency_bands();

        let mut output_refs: Vec<&mut [f32]> = self
            .output_scratch
            .iter_mut()
            .map(|channel| &mut channel[..self.capture_frame_samples])
            .collect();
        self.capture_buffer
            .copy_to_stream(&self.capture_stream_config, &mut output_refs);
        planar_to_interleaved(
            &self.output_scratch,
            self.capture_channels,
            self.capture_frame_samples,
            output,
        );

        Ok(self.aec3.metrics())
    }

    /// Convenience helper that optionally feeds a render frame before processing capture audio.
    pub fn process(
        &mut self,
        capture_frame: &[f32],
        render_frame: Option<&[f32]>,
        level_change: bool,
        output: &mut [f32],
    ) -> VoipResult<Metrics> {
        if let Some(render) = render_frame {
            self.handle_render_frame(render)?;
        }
        self.process_capture_frame(capture_frame, level_change, output)
    }
}

#[derive(Debug, Clone, Copy)]
enum FrameKind {
    Capture,
    Render,
}

fn validate_frame_length(
    actual_samples: usize,
    channels: usize,
    frame_samples: usize,
    kind: FrameKind,
) -> VoipResult<()> {
    let expected = frame_samples * channels;
    if actual_samples != expected {
        let error = match kind {
            FrameKind::Capture => VoipAec3Error::CaptureFrameSize {
                expected,
                actual: actual_samples,
            },
            FrameKind::Render => VoipAec3Error::RenderFrameSize {
                expected,
                actual: actual_samples,
            },
        };
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_output_length(actual: usize, channels: usize, frame_samples: usize) -> VoipResult<()> {
    let required = channels * frame_samples;
    if actual < required {
        Err(VoipAec3Error::OutputBufferTooSmall { required, actual })
    } else {
        Ok(())
    }
}

fn allocate_planar_storage(channels: usize, frames: usize) -> Vec<Vec<f32>> {
    (0..channels).map(|_| vec![0.0; frames]).collect()
}

fn copy_interleaved_to_planar(
    interleaved: &[f32],
    channels: usize,
    frames: usize,
    planar: &mut [Vec<f32>],
) {
    for ch in 0..channels {
        let dst = &mut planar[ch][..frames];
        for frame in 0..frames {
            dst[frame] = interleaved[frame * channels + ch];
        }
    }
}

fn planar_to_interleaved(
    planar: &[Vec<f32>],
    channels: usize,
    frames: usize,
    interleaved: &mut [f32],
) {
    for frame in 0..frames {
        let base = frame * channels;
        for ch in 0..channels {
            interleaved[base + ch] = planar[ch][frame];
        }
    }
}

/// Error type produced by the VoIP wrapper.
#[derive(Debug, Clone, PartialEq)]
pub enum VoipAec3Error {
    UnsupportedSampleRate { render: i32, capture: i32 },
    InvalidChannelCount { render: usize, capture: usize },
    CaptureFrameSize { expected: usize, actual: usize },
    RenderFrameSize { expected: usize, actual: usize },
    OutputBufferTooSmall { required: usize, actual: usize },
}

impl fmt::Display for VoipAec3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoipAec3Error::UnsupportedSampleRate { render, capture } => {
                write!(
                    f,
                    "unsupported sample rate (render={render} Hz, capture={capture} Hz) (expected between 16kHz and 48kHz)"
                )
            }
            VoipAec3Error::InvalidChannelCount { render, capture } => write!(
                f,
                "channel counts must be > 0 (render={render}, capture={capture})"
            ),
            VoipAec3Error::CaptureFrameSize { expected, actual } => write!(
                f,
                "capture frame length mismatch: expected {expected} samples, got {actual}"
            ),
            VoipAec3Error::RenderFrameSize { expected, actual } => write!(
                f,
                "render frame length mismatch: expected {expected} samples, got {actual}"
            ),
            VoipAec3Error::OutputBufferTooSmall { required, actual } => write!(
                f,
                "output buffer too small: need {required} samples, have {actual}"
            ),
        }
    }
}

impl std::error::Error for VoipAec3Error {}
