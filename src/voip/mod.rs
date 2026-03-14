use std::fmt;
use std::path::PathBuf;

use crate::api::{
    config::EchoCanceller3Config,
    control::{EchoControl, Metrics},
};
use crate::audio_processing::aec3::{
    echo_canceller3::EchoCanceller3,
};
use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use crate::audio_processing::audio_buffer::AudioBuffer;
use crate::audio_processing::gain_controller2::{GainController2, GainController2Config};
use crate::audio_processing::high_pass_filter::HighPassFilter;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};
use crate::audio_processing::ns::{NoiseSuppressor, NsConfig};
use crate::audio_processing::stream_config::StreamConfig;

const DEFAULT_APPLIED_INPUT_VOLUME: i32 = 255;

/// Convenient result type used by the VoIP wrapper.
pub type VoipResult<T> = std::result::Result<T, VoipAec3Error>;

/// Builder for [`VoipAec3`].
#[derive(Debug, Clone)]
pub struct VoipAec3Builder {
    // Maybe u32 is better? I don't think it matters except if this is used in a 16bit chip somehow
    render_sample_rate_hz: usize,
    capture_sample_rate_hz: usize,
    render_channels: usize, // This could be u16
    capture_channels: usize,
    enable_high_pass: bool,
    enable_noise_suppression: bool,
    enable_gain_controller2: bool,
    noise_suppression_config: NsConfig,
    gain_controller2_config: GainController2Config,
    input_volume_controller_config: InputVolumeControllerConfig,
    config: Option<EchoCanceller3Config>,
    initial_delay_ms: Option<i32>,
    diagnostics_enabled: bool,
    diagnostics_output_dir: Option<PathBuf>,
    diagnostics_level: Option<DiagnosticLevel>,
}

impl VoipAec3Builder {
    /// Creates a new builder for the specified sample rate and channel layout.
    /// 
    /// sample_rate_hz: Sample rate in Hz for both devices (e.g., 16000, 32000, 44100, 48000).
    /// You can override this sample rate with the other functions if needed.
    pub fn new(sample_rate_hz: usize, render_channels: usize, capture_channels: usize) -> Self {
        Self {
            render_sample_rate_hz: sample_rate_hz, // TODO: Maybe change interface instead? Seems a bit odd
            capture_sample_rate_hz: sample_rate_hz,
            render_channels,
            capture_channels,
            enable_high_pass: true,
            enable_noise_suppression: false,
            enable_gain_controller2: false,
            noise_suppression_config: NsConfig::default(),
            gain_controller2_config: GainController2Config::default(),
            input_volume_controller_config: InputVolumeControllerConfig::default(),
            config: None,
            initial_delay_ms: None,
            diagnostics_enabled: false,
            diagnostics_output_dir: None,
            diagnostics_level: None,
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

    /// Enables or disables noise suppression on the capture path.
    pub fn enable_noise_suppression(mut self, enable: bool) -> Self {
        self.enable_noise_suppression = enable;
        self
    }

    /// Sets the noise suppression configuration.
    pub fn noise_suppression_config(mut self, config: NsConfig) -> Self {
        self.noise_suppression_config = config;
        self
    }

    /// Enables or disables AGC2 on the capture path (disabled by default).
    pub fn enable_gain_controller2(mut self, enable: bool) -> Self {
        self.enable_gain_controller2 = enable;
        self
    }

    /// Sets the AGC2 top-level configuration.
    pub fn gain_controller2_config(mut self, config: GainController2Config) -> Self {
        self.gain_controller2_config = config;
        self
    }

    /// Sets the AGC2 input volume controller tuning configuration.
    pub fn input_volume_controller_config(mut self, config: InputVolumeControllerConfig) -> Self {
        self.input_volume_controller_config = config;
        self
    }

    /// Sets an optional initial buffer delay hint (in milliseconds).
    pub fn initial_delay_ms(mut self, delay_ms: i32) -> Self {
        self.initial_delay_ms = Some(delay_ms);
        self
    }

    /// Enables or disables AEC3 diagnostic dumping (log + WAV files).
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn enable_diagnostics(mut self, enable: bool) -> Self {
        self.diagnostics_enabled = enable;
        self
    }

    /// Sets the output directory for diagnostic artifacts.
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn diagnostics_output_directory<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.diagnostics_output_dir = Some(path.into());
        self
    }

    /// Sets the diagnostic verbosity level.
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn diagnostics_level(mut self, level: DiagnosticLevel) -> Self {
        self.diagnostics_level = Some(level);
        self
    }

    pub fn capture_sample_rate_hz(mut self, sample_rate_hz: usize) -> Self {
        self.capture_sample_rate_hz = sample_rate_hz;
        self
    }

    pub fn render_sample_rate_hz(mut self, sample_rate_hz: usize) -> Self {
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

        let mut chosen_config = self.config.unwrap_or_else(|| {
            EchoCanceller3::create_default_config(self.render_channels, self.capture_channels)
        });

        let ns_analyze_linear_aec_output_when_available = self.enable_noise_suppression
            && self
                .noise_suppression_config
                .analyze_linear_aec_output_when_available;

        if ns_analyze_linear_aec_output_when_available {
            chosen_config.filter.export_linear_aec_output = true;
        }

        if self.diagnostics_enabled {
            if let Some(dir) = &self.diagnostics_output_dir {
                EchoCanceller3::set_diagnostics_output_directory(dir);
            }
            if let Some(level) = self.diagnostics_level {
                EchoCanceller3::set_diagnostics_level(level);
            }
            EchoCanceller3::set_diagnostics_enabled(true);
        }
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
            internal_sample_rate, // TODO: Update to usize throughout? Definitely at least u32
            self.render_channels,
            self.capture_channels,
        );

        if self.diagnostics_enabled {
            // Make it convenient to separate per-build captures in the same log.
            aec3.initiate_new_set_of_recordings();
        }

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
            self.capture_sample_rate_hz,
            self.capture_channels,
            internal_sample_rate as usize,
            self.capture_channels,
            self.capture_sample_rate_hz,
        );
        let render_buffer = AudioBuffer::from_sample_rates(
            self.render_sample_rate_hz,
            self.render_channels,
            internal_sample_rate as usize,
            self.render_channels,
            self.render_sample_rate_hz,
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
        let noise_suppressor = self.enable_noise_suppression.then(|| {
            NoiseSuppressor::new(
                self.noise_suppression_config,
                internal_sample_rate as usize,
                self.capture_channels,
            )
        });
        let linear_output_buffer = ns_analyze_linear_aec_output_when_available.then(|| {
            AudioBuffer::from_sample_rates(16_000, self.capture_channels, 16_000, self.capture_channels, 16_000)
        });

        let gain_controller2 = if self.enable_gain_controller2 {
            if !GainController2::validate(&self.gain_controller2_config) {
                return Err(VoipAec3Error::InvalidGainController2Config);
            }
            Some(GainController2::new(
                self.gain_controller2_config,
                self.input_volume_controller_config,
                internal_sample_rate as usize,
                self.capture_channels,
                true,
            ))
        } else {
            None
        };

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
            noise_suppressor,
            linear_output_buffer,
            ns_analyze_linear_aec_output_when_available,
            gain_controller2,
            applied_input_volume: DEFAULT_APPLIED_INPUT_VOLUME,
            applied_input_volume_changed: false,
        })
    }

    fn valid_sample_rates(&self, capture_rate: usize, render_rate: usize) -> bool {
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
    sample_rate_hz: usize,
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
    noise_suppressor: Option<NoiseSuppressor>,
    linear_output_buffer: Option<AudioBuffer>,
    ns_analyze_linear_aec_output_when_available: bool,
    gain_controller2: Option<GainController2>,
    applied_input_volume: i32,
    applied_input_volume_changed: bool,
}

impl VoipAec3 {
    /// Convenience constructor mirroring [`VoipAec3Builder::new`].
    pub fn builder(
        sample_rate_hz: usize,
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
    pub fn sample_rate_hz(&self) -> usize {
        self.sample_rate_hz
    }

    /// Provides the current metrics without running another processing step.
    pub fn metrics(&self) -> Metrics {
        self.aec3.metrics()
    }

    /// Updates the applied microphone input volume used by AGC2 input-volume analysis.
    pub fn set_applied_input_volume(&mut self, input_volume: i32) -> VoipResult<()> {
        if !(0..=255).contains(&input_volume) {
            return Err(VoipAec3Error::InvalidAppliedInputVolume {
                actual: input_volume,
            });
        }

        self.applied_input_volume_changed = self.applied_input_volume != input_volume;
        self.applied_input_volume = input_volume;
        Ok(())
    }

    /// Returns the last AGC2 recommended input volume, if AGC2 is enabled.
    pub fn recommended_input_volume(&self) -> Option<i32> {
        self.gain_controller2
            .as_ref()
            .and_then(|gc2| gc2.recommended_input_volume())
    }

    /// Updates AGC2 fixed digital gain in dB at runtime.
    pub fn set_fixed_gain_db(&mut self, gain_db: f32) {
        if let Some(gc2) = self.gain_controller2.as_mut() {
            gc2.set_fixed_gain_db(gain_db);
        }
    }

    /// Passes capture-output-used state to AGC2 input-volume controller.
    pub fn set_capture_output_used(&mut self, capture_output_used: bool) {
        if let Some(gc2) = self.gain_controller2.as_mut() {
            gc2.set_capture_output_used(capture_output_used);
        }
    }

    /// Returns AGC2 CPU feature flags when AGC2 is enabled.
    pub fn gain_controller2_cpu_features(&self) -> Option<AvailableCpuFeatures> {
        self.gain_controller2.as_ref().map(|gc2| gc2.cpu_features())
    }

    /// Updates the audio buffer delay hint at runtime.
    pub fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.aec3.set_audio_buffer_delay(delay_ms);
    }

    /// Enables or disables diagnostic dumping globally.
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn set_diagnostics_enabled(enabled: bool) {
        ApmDataDumper::set_activated(enabled);
    }

    /// Sets the global diagnostics verbosity level.
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn set_diagnostics_level(level: DiagnosticLevel) {
        ApmDataDumper::set_diagnostics_level(level);
    }

    /// Sets the global diagnostics output directory.
    ///
    /// This is a no-op unless the crate is built with the `diagnostics` feature.
    pub fn set_diagnostics_output_directory<P: AsRef<std::path::Path>>(path: P) {
        ApmDataDumper::set_output_directory(path);
    }

    /// Starts a new logical recording set in the diagnostics log.
    pub fn initiate_new_set_of_recordings(&self) {
        self.aec3.initiate_new_set_of_recordings();
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

        if let Some(gc2) = self.gain_controller2.as_mut() {
            gc2.analyze(self.applied_input_volume, &self.capture_buffer);
        }

        if let Some(ns) = self.noise_suppressor.as_mut()
            && (!self.ns_analyze_linear_aec_output_when_available
                || self.linear_output_buffer.is_none())
        {
            ns.analyze(&self.capture_buffer);
        }

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

        if let Some(linear_output_buffer) = self.linear_output_buffer.as_mut() {
            self.aec3.process_capture_with_linear_output(
                &mut self.capture_buffer,
                linear_output_buffer,
                level_change,
            );
        } else {
            self.aec3
                .process_capture(&mut self.capture_buffer, level_change);
        }

        if let Some(ns) = self.noise_suppressor.as_mut()
            && self.ns_analyze_linear_aec_output_when_available
            && let Some(linear_output_buffer) = self.linear_output_buffer.as_ref()
        {
            ns.analyze(linear_output_buffer);
        }

        if let Some(ns) = self.noise_suppressor.as_mut() {
            ns.process(&mut self.capture_buffer);
        }

        self.capture_buffer.merge_frequency_bands();

        if let Some(gc2) = self.gain_controller2.as_mut() {
            gc2.process(self.applied_input_volume_changed, &mut self.capture_buffer);
            self.applied_input_volume_changed = false;
        }

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
    UnsupportedSampleRate { render: usize, capture: usize },
    InvalidChannelCount { render: usize, capture: usize },
    InvalidGainController2Config,
    InvalidAppliedInputVolume { actual: i32 },
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
            VoipAec3Error::InvalidGainController2Config => write!(
                f,
                "invalid gain_controller2 configuration"
            ),
            VoipAec3Error::InvalidAppliedInputVolume { actual } => write!(
                f,
                "invalid applied input volume: expected in [0, 255], got {actual}"
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
