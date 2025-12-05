use aec3::voip::{VoipAec3, VoipAec3Error};

#[test]
fn voip_wrapper_processes_frame() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;
    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .initial_delay_ms(0)
        .build()
        .expect("failed to build pipeline");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture_frame_len = capture_frame_samples * channels;
    let render_frame_len = render_frame_samples * channels;

    let mut render = vec![0.0f32; render_frame_len];
    let mut capture = vec![0.0f32; capture_frame_len];
    for i in 0..capture_frame_samples {
        let t = i as f32 / 40.0;
        render[i] = (t).sin() * 0.8;
        capture[i] = (t * 2.0).cos() * 0.2 + render[i] * 0.5;
    }

    let mut output = vec![0.0f32; capture_frame_len];
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

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture_frame_len = capture_frame_samples * channels;
    let render_frame_len = render_frame_samples * channels;
    let capture = vec![0.0f32; capture_frame_len - 1];
    let render = vec![0.0f32; render_frame_len];
    let mut output = vec![0.0f32; capture_frame_len];
    let err = pipeline
        .process(&capture, Some(&render), false, &mut output)
        .unwrap_err();
    assert!(matches!(err, VoipAec3Error::CaptureFrameSize { .. }));
}

#[test]
fn voip_wrapper_supports_various_sample_rates() {
    let rates = [16_000, 32_000, 44_100, 48_000];
    
    for &rate in &rates {
        let channels = 1usize;
        let mut pipeline = VoipAec3::builder(rate, channels, channels)
            .build()
            .expect(&format!("failed to build pipeline for rate {}", rate));

        let capture_frame_samples = pipeline.capture_frame_samples();
        let render_frame_samples = pipeline.render_frame_samples();
        
        // Basic check that frame sizes make sense (10ms)
        let expected_samples = rate as usize / 100;
        assert_eq!(capture_frame_samples, expected_samples, "Incorrect capture frame size for rate {}", rate);
        assert_eq!(render_frame_samples, expected_samples, "Incorrect render frame size for rate {}", rate);

        let capture = vec![0.0f32; capture_frame_samples * channels];
        let render = vec![0.0f32; render_frame_samples * channels];
        let mut output = vec![0.0f32; capture_frame_samples * channels];

        pipeline
            .process(&capture, Some(&render), false, &mut output)
            .expect(&format!("processing failed for rate {}", rate));
    }
}

#[test]
fn voip_wrapper_supports_mixed_sample_rates() {
    let capture_rate = 48_000;
    let render_rate = 16_000;
    let channels = 1usize;
    
    let mut pipeline = VoipAec3::builder(capture_rate, channels, channels)
        .render_sample_rate_hz(render_rate)
        .build()
        .expect("failed to build pipeline with mixed rates");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();

    assert_eq!(capture_frame_samples, 480);
    assert_eq!(render_frame_samples, 160);

    let capture = vec![0.0f32; capture_frame_samples * channels];
    let render = vec![0.0f32; render_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing failed for mixed rates");
}
