use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::reverb_model::ReverbModel;
use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;

pub(super) fn compute_avg_render_reverb(
    spectrum_buffer: &SpectrumBuffer,
    delay_blocks: i32,
    reverb_decay: f32,
    reverb_model: &mut ReverbModel,
    reverb_power_spectrum: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let num_render_channels = spectrum_buffer.buffer[0].len();
    let idx_at_delay = spectrum_buffer.offset_index(spectrum_buffer.read, delay_blocks as isize);
    let idx_past = spectrum_buffer.inc_index(idx_at_delay);

    let mut x2_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
    let averaged_power = if num_render_channels > 1 {
        let average_channels = |idx: usize, dst: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]| {
            dst.fill(0.0);
            for ch in 0..num_render_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    dst[k] += spectrum_buffer.buffer[idx][ch][k];
                }
            }
            let normalizer = 1.0 / num_render_channels as f32;
            for value in dst.iter_mut() {
                *value *= normalizer;
            }
        };
        average_channels(idx_past, &mut x2_data);
        reverb_model.update_reverb_no_freq_shaping(&x2_data, 1.0, reverb_decay);
        average_channels(idx_at_delay, &mut x2_data);
        &x2_data
    } else {
        reverb_model.update_reverb_no_freq_shaping(
            &spectrum_buffer.buffer[idx_past][0],
            1.0,
            reverb_decay,
        );
        &spectrum_buffer.buffer[idx_at_delay][0]
    };

    let reverb = reverb_model.reverb();
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        reverb_power_spectrum[k] = averaged_power[k] + reverb[k];
    }
}
