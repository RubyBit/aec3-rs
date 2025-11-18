use aec3::voip::{VoipAec3, VoipAec3Error};

#[test]
fn voip_wrapper_processes_frame() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;
    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .initial_delay_ms(0)
        .build()
        .expect("failed to build pipeline");

    let frame_samples = pipeline.frame_samples();
    let frame_len = frame_samples * channels;

    let mut render = vec![0.0f32; frame_len];
    let mut capture = vec![0.0f32; frame_len];
    for i in 0..frame_samples {
        let t = i as f32 / 40.0;
        render[i] = (t).sin() * 0.8;
        capture[i] = (t * 2.0).cos() * 0.2 + render[i] * 0.5;
    }

    let mut output = vec![0.0f32; frame_len];
    let metrics = pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed");

    assert_ne!(capture, output, "AEC output should differ from raw capture");
    assert!(metrics.delay_ms >= 0);
}

#[test]
fn voip_wrapper_validates_frame_sizes() {
    let sample_rate_hz = 32_000;
    let channels = 2usize;
    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .build()
        .expect("failed to build pipeline");

    let frame_samples = pipeline.frame_samples();
    let frame_len = frame_samples * channels;
    let capture = vec![0.0f32; frame_len - 1];
    let render = vec![0.0f32; frame_len];
    let mut output = vec![0.0f32; frame_len];

    let err = pipeline
        .process(&capture, Some(&render), false, &mut output)
        .unwrap_err();
    assert!(matches!(err, VoipAec3Error::CaptureFrameSize { .. }));
}
