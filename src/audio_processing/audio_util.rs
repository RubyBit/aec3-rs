//! Port of `audio_processing/include/audio_util.h` helper functions.

pub fn s16_to_float(v: i16) -> f32 {
    const SCALING: f32 = 1.0 / 32768.0;
    v as f32 * SCALING
}

pub fn float_s16_to_s16(v: f32) -> i16 {
    let mut clamped = v.min(32767.0).max(-32768.0);
    clamped += clamped.signum() * 0.5;
    clamped as i16
}

pub fn float_to_s16(v: f32) -> i16 {
    float_s16_to_s16(v * 32768.0)
}

pub fn float_to_float_s16(v: f32) -> f32 {
    let clamped = v.min(1.0).max(-1.0);
    clamped * 32768.0
}

pub fn float_s16_to_float(v: f32) -> f32 {
    let clamped = v.min(32768.0).max(-32768.0);
    clamped * (1.0 / 32768.0)
}

pub fn s16_slice_to_float_s16(src: &[i16], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len());
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        *d = *s as f32;
    }
}

pub fn float_s16_slice_to_s16(src: &[f32], dest: &mut [i16]) {
    assert_eq!(src.len(), dest.len());
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        *d = float_s16_to_s16(*s);
    }
}

pub fn float_slice_to_float_s16(src: &[f32], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len());
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        *d = float_to_float_s16(*s);
    }
}

pub fn float_slice_to_float_s16_in_place(data: &mut [f32]) {
    for sample in data.iter_mut() {
        *sample = float_to_float_s16(*sample);
    }
}

pub fn float_s16_slice_to_float(src: &[f32], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len());
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        *d = float_s16_to_float(*s);
    }
}

pub fn float_s16_slice_to_float_in_place(data: &mut [f32]) {
    for sample in data.iter_mut() {
        *sample = float_s16_to_float(*sample);
    }
}

pub fn copy_audio_if_needed<T: Copy>(src: &[&[T]], num_frames: usize, dest: &mut [&mut [T]]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        if !std::ptr::eq(s.as_ptr(), d.as_mut_ptr()) {
            d[..num_frames].copy_from_slice(&s[..num_frames]);
        }
    }
}

pub fn deinterleave<T: Copy>(
    interleaved: &[T],
    samples_per_channel: usize,
    num_channels: usize,
    deinterleaved: &mut [&mut [T]],
) {
    for channel in 0..num_channels {
        let output = &mut deinterleaved[channel][..samples_per_channel];
        let mut idx = channel;
        for sample in output.iter_mut() {
            *sample = interleaved[idx];
            idx += num_channels;
        }
    }
}

pub fn interleave<T: Copy>(
    deinterleaved: &[&[T]],
    samples_per_channel: usize,
    num_channels: usize,
    interleaved: &mut [T],
) {
    for channel in 0..num_channels {
        let input = &deinterleaved[channel][..samples_per_channel];
        let mut idx = channel;
        for &sample in input {
            interleaved[idx] = sample;
            idx += num_channels;
        }
    }
}

pub fn downmix_to_mono_f32(input_channels: &[&[f32]], num_frames: usize, out: &mut [f32]) {
    for i in 0..num_frames {
        let mut value = input_channels[0][i];
        for channel in input_channels.iter().skip(1) {
            value += channel[i];
        }
        out[i] = value / input_channels.len() as f32;
    }
}

pub fn downmix_to_mono_i16(input_channels: &[&[i16]], num_frames: usize, out: &mut [i16]) {
    for i in 0..num_frames {
        let mut sum: i32 = input_channels[0][i] as i32;
        for channel in input_channels.iter().skip(1) {
            sum += channel[i] as i32;
        }
        sum /= input_channels.len() as i32;
        out[i] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

pub fn downmix_interleaved_to_mono_i16(
    interleaved: &[i16],
    num_frames: usize,
    num_channels: usize,
    out: &mut [i16],
) {
    assert!(num_channels > 0);
    for frame in 0..num_frames {
        let mut sum: i32 = 0;
        for ch in 0..num_channels {
            sum += interleaved[frame * num_channels + ch] as i32;
        }
        sum /= num_channels as i32;
        out[frame] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}
