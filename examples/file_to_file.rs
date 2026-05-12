use ::aec3::nodes::audio::AudioFormat;
use ::aec3::pipelines::linear;
use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::env;
use std::path::{Path, PathBuf};

struct WavData {
    spec: WavSpec,
    samples: Vec<f32>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        bail!(
            "usage: cargo run --example file_to_file -- <render.wav> <capture.wav> <output.wav> [delay_ms]"
        );
    }

    let render_path = PathBuf::from(&args[1]);
    let capture_path = PathBuf::from(&args[2]);
    let output_path = PathBuf::from(&args[3]);
    let delay_ms = args
        .get(4)
        .map(|value| value.parse::<i32>().context("delay_ms must be an integer"))
        .transpose()?
        .unwrap_or(0);

    let render = read_wav(&render_path)
        .with_context(|| format!("failed to read {}", render_path.display()))?;
    let capture = read_wav(&capture_path)
        .with_context(|| format!("failed to read {}", capture_path.display()))?;
    ensure_matching_streams(&render.spec, &capture.spec)?;

    let format = AudioFormat::ten_ms(capture.spec.sample_rate, capture.spec.channels);
    let mut pipeline = linear::builder(format, format)
        .initial_delay_ms(delay_ms)
        .build()
        .context("failed to build linear AEC pipeline")?;

    let frame_samples = format.sample_count();
    let mut output = Vec::with_capacity(capture.samples.len());
    let mut render_offset = 0usize;
    let mut capture_offset = 0usize;

    while capture_offset < capture.samples.len() {
        let render_frame = padded_frame(&render.samples, render_offset, frame_samples);
        pipeline
            .handle_render_frame(&render_frame)
            .context("failed to feed render frame")?;
        render_offset += frame_samples;

        let remaining_capture = capture.samples.len() - capture_offset;
        let valid_samples = remaining_capture.min(frame_samples);
        let capture_frame = padded_frame(&capture.samples, capture_offset, frame_samples);
        capture_offset += valid_samples;

        let mut processed = vec![0.0f32; frame_samples];
        let produced = pipeline
            .process_capture_frame(&capture_frame, &mut processed)
            .context("failed to process capture frame")?;
        if produced {
            output.extend_from_slice(&processed[..valid_samples]);
        } else {
            output.resize(output.len() + valid_samples, 0.0);
        }
    }

    write_float_wav(
        &output_path,
        capture.spec.sample_rate,
        capture.spec.channels,
        &output,
    )
    .with_context(|| format!("failed to write {}", output_path.display()))?;

    println!(
        "processed {} capture samples at {} Hz / {} channel(s) -> {}",
        output.len(),
        capture.spec.sample_rate,
        capture.spec.channels,
        output_path.display()
    );
    Ok(())
}

fn ensure_matching_streams(render: &WavSpec, capture: &WavSpec) -> Result<()> {
    if render.sample_rate != capture.sample_rate {
        bail!(
            "render and capture sample rates must match ({} != {})",
            render.sample_rate,
            capture.sample_rate
        );
    }
    if render.channels != capture.channels {
        bail!(
            "render and capture channel counts must match ({} != {})",
            render.channels,
            capture.channels
        );
    }
    if capture.sample_rate < 16_000 || capture.sample_rate > 48_000 {
        bail!(
            "AEC3 expects sample rates between 16 kHz and 48 kHz, got {} Hz",
            capture.sample_rate
        );
    }
    Ok(())
}

fn padded_frame(samples: &[f32], offset: usize, frame_samples: usize) -> Vec<f32> {
    let mut frame = vec![0.0f32; frame_samples];
    if offset < samples.len() {
        let count = (samples.len() - offset).min(frame_samples);
        frame[..count].copy_from_slice(&samples[offset..offset + count]);
    }
    frame
}

fn read_wav(path: &Path) -> Result<WavData> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        SampleFormat::Float => reader // For the odd case :P
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to decode float WAV samples")?,
        SampleFormat::Int => {
            let scale = (1i64 << (u32::from(spec.bits_per_sample) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| (sample as f32 / scale).clamp(-1.0, 1.0)))
                .collect::<Result<Vec<_>, _>>()
                .context("failed to decode integer WAV samples")?
        }
    };
    Ok(WavData { spec, samples })
}

fn write_float_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}
