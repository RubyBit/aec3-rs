use aec3::api::config::EchoCanceller3Config;
use aec3::api::control::EchoControl;
use aec3::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use aec3::audio_processing::audio_buffer::AudioBuffer;

/// Simple integration test that constructs an AEC3 instance, feeds a
/// render and capture `AudioBuffer` and verifies that the render buffer
/// is not modified while the capture buffer is changed by the AEC.
#[test]
fn simple_integration_processes_render_and_capture() {
    let sample_rate_hz: i32 = 16_000;
    let sample_rate = sample_rate_hz as usize;
    let mut aec3 = EchoCanceller3::with_multichannel_config(
        EchoCanceller3Config::default(),
        Some(EchoCanceller3Config::create_default_multichannel_config()),
        sample_rate_hz,
        2,
        2,
    );

    // Create buffers for 16 kHz stereo (160 samples per channel).
    let mut render = AudioBuffer::from_sample_rates(sample_rate, 2, sample_rate, 2, sample_rate);
    let mut capture = AudioBuffer::from_sample_rates(sample_rate, 2, sample_rate, 2, sample_rate);

    let num_frames = render.num_frames();

    // Fill render with two different tones and capture as a nearend tone
    // plus a scaled version of the render (to simulate echo pickup).
    for i in 0..num_frames {
        let r0 = (i as f32 / 40.0).cos() * 0.4;
        let r1 = (i as f32 / 40.0).cos() * 0.2;
        render.channel_mut(0)[i] = r0;
        render.channel_mut(1)[i] = r1;

        let near0 = (i as f32 / 20.0).sin() * 0.4;
        let near1 = (i as f32 / 20.0).sin() * 0.2;
        capture.channel_mut(0)[i] = near0 + r0 * 0.2;
        capture.channel_mut(1)[i] = near1 + r1 * 0.2;
    }

    // Helper to flatten buffer contents for easy equality checks.
    let flatten = |b: &AudioBuffer| {
        let mut v = Vec::with_capacity(b.num_channels() * b.num_frames());
        for ch in 0..b.num_channels() {
            v.extend_from_slice(b.channel(ch));
        }
        v
    };

    let render_before = flatten(&render);
    let capture_before = flatten(&capture);

    // For multi-band rates, split into frequency bands before feeding
    // the buffers into the AEC. This mirrors how the crate's unit tests
    // prepare buffers for AEC3 usage.
    if render.num_bands() > 1 {
        render.split_into_frequency_bands();
        capture.split_into_frequency_bands();
    }

    // analyze_render should not modify the render buffer
    aec3.analyze_render(&mut render);
    let render_after = flatten(&render);
    assert_eq!(
        render_before, render_after,
        "render buffer must remain unmodified"
    );

    // A typical usage pattern calls analyze_capture before process_capture.
    aec3.analyze_capture(&mut capture);

    // process_capture is expected to modify the capture buffer
    aec3.process_capture(&mut capture, /* level_change= */ false);
    let capture_after = flatten(&capture);
    assert_ne!(
        capture_before, capture_after,
        "capture buffer should be modified by AEC"
    );
}
