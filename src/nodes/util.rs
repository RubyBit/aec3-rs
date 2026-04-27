use crate::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use crate::audio_processing::audio_buffer::AudioBuffer;
use crate::audio_processing::stream_config::StreamConfig;
use crate::graph::{GraphError, GraphResult};

use super::audio::{AudioChunk, AudioFormat};

pub(crate) fn validate_audio_format(format: AudioFormat, node: &str) -> GraphResult<()> {
    if format.sample_rate_hz < 16_000 || format.sample_rate_hz > 48_000 {
        return Err(GraphError::NodeError(format!(
            "{} only supports sample rates between 16kHz and 48kHz (got {} Hz)",
            node, format.sample_rate_hz
        )));
    }

    if format.channels == 0 {
        return Err(GraphError::NodeError(format!(
            "{} requires at least one channel",
            node
        )));
    }

    if format.frames_per_channel as u32 * 100 != format.sample_rate_hz {
        return Err(GraphError::NodeError(format!(
            "{} currently only supports 10 ms audio chunks (got {} frames at {} Hz)",
            node, format.frames_per_channel, format.sample_rate_hz
        )));
    }

    Ok(())
}

pub(crate) fn internal_sample_rate(primary: AudioFormat, sidechain: Option<AudioFormat>) -> usize {
    let primary_rate = primary.sample_rate_hz as usize;
    let sidechain_rate = sidechain.map(|format| format.sample_rate_hz as usize);

    if primary_rate == 44_100 || sidechain_rate == Some(44_100) {
        return 48_000;
    }

    let min_rate = sidechain_rate.map_or(primary_rate, |rate| primary_rate.min(rate));
    if min_rate <= 16_000 {
        16_000
    } else if min_rate <= 32_000 {
        32_000
    } else {
        48_000
    }
}

pub(crate) struct ChunkIo {
    format: AudioFormat,
    stream_config: StreamConfig,
    audio_buffer: AudioBuffer,
    input_scratch: Vec<Vec<f32>>,
    output_scratch: Vec<Vec<f32>>,
}

impl ChunkIo {
    pub(crate) fn new(format: AudioFormat, buffer_sample_rate_hz: usize) -> Self {
        let stream_config = StreamConfig::new(
            format.sample_rate_hz as usize,
            format.channels as usize,
            false,
        );
        let audio_buffer = AudioBuffer::from_sample_rates(
            format.sample_rate_hz as usize,
            format.channels as usize,
            buffer_sample_rate_hz,
            format.channels as usize,
            format.sample_rate_hz as usize,
        );
        let frame_samples = stream_config.num_frames();
        Self {
            format,
            stream_config,
            audio_buffer,
            input_scratch: allocate_planar_storage(format.channels as usize, frame_samples),
            output_scratch: allocate_planar_storage(format.channels as usize, frame_samples),
        }
    }

    pub(crate) fn audio_buffer(&self) -> &AudioBuffer {
        &self.audio_buffer
    }

    pub(crate) fn audio_buffer_mut(&mut self) -> &mut AudioBuffer {
        &mut self.audio_buffer
    }

    pub(crate) fn load_chunk(&mut self, chunk: &AudioChunk, node: &str) -> GraphResult<()> {
        if chunk.format != self.format {
            return Err(GraphError::NodeError(format!(
                "{} expected {}, got {}",
                node,
                self.format.schema_key(),
                chunk.format.schema_key()
            )));
        }

        copy_interleaved_to_planar(
            chunk.samples(),
            self.format.channels as usize,
            self.stream_config.num_frames(),
            &mut self.input_scratch,
        );
        let refs: Vec<&[f32]> = self
            .input_scratch
            .iter()
            .map(|channel| &channel[..self.stream_config.num_frames()])
            .collect();
        self.audio_buffer.copy_from(&refs, &self.stream_config);
        Ok(())
    }

    pub(crate) fn store_chunk(&mut self, chunk: &mut AudioChunk) -> GraphResult<()> {
        if chunk.format != self.format {
            return Err(GraphError::NodeError(format!(
                "cannot store {} into {}",
                self.format.schema_key(),
                chunk.format.schema_key()
            )));
        }

        let frame_samples = self.stream_config.num_frames();
        let mut output_refs: Vec<&mut [f32]> = self
            .output_scratch
            .iter_mut()
            .map(|channel| &mut channel[..frame_samples])
            .collect();
        self.audio_buffer
            .copy_to_stream(&self.stream_config, &mut output_refs);
        planar_to_interleaved(
            &self.output_scratch,
            self.format.channels as usize,
            frame_samples,
            chunk.samples_mut(),
        );
        Ok(())
    }
}

pub(crate) struct AudioAdapter {
    input_format: AudioFormat,
    output_format: AudioFormat,
    input_stream_config: StreamConfig,
    output_stream_config: StreamConfig,
    audio_buffer: AudioBuffer,
    input_scratch: Vec<Vec<f32>>,
    output_scratch: Vec<Vec<f32>>,
}

impl AudioAdapter {
    pub(crate) fn new(input_format: AudioFormat, output_format: AudioFormat) -> Self {
        let buffer_channels = usize::from(input_format.channels.min(output_format.channels));
        let buffer_rate = output_format.sample_rate_hz as usize;
        let input_stream_config = StreamConfig::new(
            input_format.sample_rate_hz as usize,
            input_format.channels as usize,
            false,
        );
        let output_stream_config = StreamConfig::new(
            output_format.sample_rate_hz as usize,
            output_format.channels as usize,
            false,
        );
        let audio_buffer = AudioBuffer::from_sample_rates(
            input_format.sample_rate_hz as usize,
            input_format.channels as usize,
            buffer_rate,
            buffer_channels,
            output_format.sample_rate_hz as usize,
        );
        let input_frames = input_stream_config.num_frames();
        let output_frames = output_stream_config.num_frames();

        Self {
            input_format,
            output_format,
            input_stream_config,
            output_stream_config,
            audio_buffer,
            input_scratch: allocate_planar_storage(input_format.channels as usize, input_frames),
            output_scratch: allocate_planar_storage(output_format.channels as usize, output_frames),
        }
    }

    pub(crate) fn process(
        &mut self,
        input: &AudioChunk,
        output: &mut AudioChunk,
    ) -> GraphResult<()> {
        if input.format != self.input_format || output.format != self.output_format {
            return Err(GraphError::NodeError(
                "audio adapter received an unexpected format".to_string(),
            ));
        }

        copy_interleaved_to_planar(
            input.samples(),
            self.input_format.channels as usize,
            self.input_stream_config.num_frames(),
            &mut self.input_scratch,
        );
        let refs: Vec<&[f32]> = self
            .input_scratch
            .iter()
            .map(|channel| &channel[..self.input_stream_config.num_frames()])
            .collect();
        self.audio_buffer
            .copy_from(&refs, &self.input_stream_config);

        let frame_samples = self.output_stream_config.num_frames();
        let mut output_refs: Vec<&mut [f32]> = self
            .output_scratch
            .iter_mut()
            .map(|channel| &mut channel[..frame_samples])
            .collect();
        self.audio_buffer
            .copy_to_stream(&self.output_stream_config, &mut output_refs);
        planar_to_interleaved(
            &self.output_scratch,
            self.output_format.channels as usize,
            frame_samples,
            output.samples_mut(),
        );
        Ok(())
    }
}

pub(crate) fn maybe_enable_diagnostics(
    enabled: bool,
    output_dir: Option<&std::path::Path>,
    level: Option<crate::audio_processing::logging::apm_data_dumper::DiagnosticLevel>,
) {
    if !enabled {
        return;
    }

    if let Some(dir) = output_dir {
        EchoCanceller3::set_diagnostics_output_directory(dir);
    }
    if let Some(level) = level {
        EchoCanceller3::set_diagnostics_level(level);
    }
    EchoCanceller3::set_diagnostics_enabled(true);
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
    for channel in 0..channels {
        let dst = &mut planar[channel][..frames];
        for frame in 0..frames {
            dst[frame] = interleaved[frame * channels + channel];
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
        for channel in 0..channels {
            interleaved[base + channel] = planar[channel][frame];
        }
    }
}
