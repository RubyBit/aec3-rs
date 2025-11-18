//! Splitting filter capable of producing 2 or 3 sub-bands per channel.

use crate::audio_processing::audio_util::{float_s16_slice_to_s16, s16_slice_to_float_s16};
use crate::audio_processing::channel_buffer::ChannelBuffer;
use crate::audio_processing::three_band_filter_bank::ThreeBandFilterBank;

const SAMPLES_PER_BAND: usize = 160;
const TWO_BAND_FILTER_SAMPLES_PER_FRAME: usize = 320;
const STATE_SIZE: usize = 6;
const MAX_BAND_FRAME_LENGTH: usize = 320;
const ALL_PASS_FILTER1: [u16; 3] = [6418, 36982, 57261];
const ALL_PASS_FILTER2: [u16; 3] = [21333, 49062, 63010];

#[derive(Clone)]
pub struct TwoBandStates {
    analysis_state1: [i32; STATE_SIZE],
    analysis_state2: [i32; STATE_SIZE],
    synthesis_state1: [i32; STATE_SIZE],
    synthesis_state2: [i32; STATE_SIZE],
}

impl TwoBandStates {
    fn new() -> Self {
        Self {
            analysis_state1: [0; STATE_SIZE],
            analysis_state2: [0; STATE_SIZE],
            synthesis_state1: [0; STATE_SIZE],
            synthesis_state2: [0; STATE_SIZE],
        }
    }
}

pub struct SplittingFilter {
    num_bands: usize,
    two_band_states: Vec<TwoBandStates>,
    three_band_filter_banks: Vec<ThreeBandFilterBank>,
}

impl SplittingFilter {
    pub fn new(num_channels: usize, num_bands: usize, num_frames: usize) -> Self {
        assert!(num_bands == 2 || num_bands == 3);
        let mut two_band_states = Vec::new();
        let mut three_band_filter_banks = Vec::new();
        if num_bands == 2 {
            two_band_states = (0..num_channels).map(|_| TwoBandStates::new()).collect();
        } else {
            three_band_filter_banks = (0..num_channels)
                .map(|_| ThreeBandFilterBank::new(num_frames))
                .collect();
        }
        Self {
            num_bands,
            two_band_states,
            three_band_filter_banks,
        }
    }

    pub fn analysis(&mut self, data: &ChannelBuffer<f32>, bands: &mut ChannelBuffer<f32>) {
        assert_eq!(self.num_bands, bands.num_bands());
        assert_eq!(data.num_channels(), bands.num_channels());
        assert_eq!(
            data.num_frames(),
            bands.num_frames_per_band() * bands.num_bands()
        );
        if bands.num_bands() == 2 {
            self.two_bands_analysis(data, bands);
        } else {
            self.three_bands_analysis(data, bands);
        }
    }

    pub fn synthesis(&mut self, bands: &ChannelBuffer<f32>, data: &mut ChannelBuffer<f32>) {
        assert_eq!(self.num_bands, bands.num_bands());
        assert_eq!(data.num_channels(), bands.num_channels());
        assert_eq!(
            data.num_frames(),
            bands.num_frames_per_band() * bands.num_bands()
        );
        if bands.num_bands() == 2 {
            self.two_bands_synthesis(bands, data);
        } else {
            self.three_bands_synthesis(bands, data);
        }
    }

    fn two_bands_analysis(&mut self, data: &ChannelBuffer<f32>, bands: &mut ChannelBuffer<f32>) {
        assert_eq!(data.num_frames(), TWO_BAND_FILTER_SAMPLES_PER_FRAME);
        for (channel, state) in self.two_band_states.iter_mut().enumerate() {
            let mut full_band = [0i16; TWO_BAND_FILTER_SAMPLES_PER_FRAME];
            float_s16_slice_to_s16(data.channel(channel), &mut full_band);
            let mut low_band = [0i16; SAMPLES_PER_BAND];
            let mut high_band = [0i16; SAMPLES_PER_BAND];
            analysis_qmf(
                &full_band,
                &mut low_band,
                &mut high_band,
                &mut state.analysis_state1,
                &mut state.analysis_state2,
            );
            s16_slice_to_float_s16(&low_band, bands.band_mut(channel, 0));
            s16_slice_to_float_s16(&high_band, bands.band_mut(channel, 1));
        }
    }

    fn two_bands_synthesis(&mut self, bands: &ChannelBuffer<f32>, data: &mut ChannelBuffer<f32>) {
        assert_eq!(data.num_frames(), TWO_BAND_FILTER_SAMPLES_PER_FRAME);
        for (channel, state) in self.two_band_states.iter_mut().enumerate() {
            let mut low_band = [0i16; SAMPLES_PER_BAND];
            let mut high_band = [0i16; SAMPLES_PER_BAND];
            float_s16_slice_to_s16(bands.band(channel, 0), &mut low_band);
            float_s16_slice_to_s16(bands.band(channel, 1), &mut high_band);
            let mut full_band = [0i16; TWO_BAND_FILTER_SAMPLES_PER_FRAME];
            synthesis_qmf(
                &low_band,
                &high_band,
                &mut full_band,
                &mut state.synthesis_state1,
                &mut state.synthesis_state2,
            );
            s16_slice_to_float_s16(&full_band, data.channel_mut(channel));
        }
    }

    fn three_bands_analysis(&mut self, data: &ChannelBuffer<f32>, bands: &mut ChannelBuffer<f32>) {
        let band_length = bands.num_frames_per_band();
        for channel in 0..self.three_band_filter_banks.len() {
            let channel_slice = bands.channel_mut(channel);
            let (band0, rest) = channel_slice.split_at_mut(band_length);
            let (band1, band2) = rest.split_at_mut(band_length);
            let mut split: [&mut [f32]; 3] = [band0, band1, band2];
            self.three_band_filter_banks[channel].analysis(data.channel(channel), &mut split);
        }
    }

    fn three_bands_synthesis(&mut self, bands: &ChannelBuffer<f32>, data: &mut ChannelBuffer<f32>) {
        let band_length = bands.num_frames_per_band();
        for channel in 0..data.num_channels() {
            let channel_slice = bands.channel(channel);
            let (band0, rest) = channel_slice.split_at(band_length);
            let (band1, band2) = rest.split_at(band_length);
            let split: [&[f32]; 3] = [band0, band1, band2];
            self.three_band_filter_banks[channel].synthesis(&split, data.channel_mut(channel));
        }
    }
}

fn analysis_qmf(
    in_data: &[i16],
    low_band: &mut [i16],
    high_band: &mut [i16],
    filter_state1: &mut [i32; STATE_SIZE],
    filter_state2: &mut [i32; STATE_SIZE],
) {
    assert_eq!(in_data.len() % 2, 0);
    let band_length = in_data.len() / 2;
    assert_eq!(band_length, low_band.len());
    assert_eq!(band_length, high_band.len());
    assert!(band_length <= MAX_BAND_FRAME_LENGTH);

    let mut half_in1 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut half_in2 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut filter1 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut filter2 = [0i32; MAX_BAND_FRAME_LENGTH];

    for i in 0..band_length {
        half_in2[i] = in_data[2 * i] as i32 * (1 << 10);
        half_in1[i] = in_data[2 * i + 1] as i32 * (1 << 10);
    }

    all_pass_qmf(
        &mut half_in1[..band_length],
        &mut filter1[..band_length],
        &ALL_PASS_FILTER1,
        filter_state1,
    );
    all_pass_qmf(
        &mut half_in2[..band_length],
        &mut filter2[..band_length],
        &ALL_PASS_FILTER2,
        filter_state2,
    );

    for i in 0..band_length {
        let tmp = (filter1[i] + filter2[i] + 1024) >> 11;
        low_band[i] = sat_w32_to_w16(tmp);
        let tmp = (filter1[i] - filter2[i] + 1024) >> 11;
        high_band[i] = sat_w32_to_w16(tmp);
    }
}

fn synthesis_qmf(
    low_band: &[i16],
    high_band: &[i16],
    out_data: &mut [i16],
    filter_state1: &mut [i32; STATE_SIZE],
    filter_state2: &mut [i32; STATE_SIZE],
) {
    let band_length = low_band.len();
    assert_eq!(band_length, high_band.len());
    assert_eq!(out_data.len(), band_length * 2);

    let mut half_in1 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut half_in2 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut filter1 = [0i32; MAX_BAND_FRAME_LENGTH];
    let mut filter2 = [0i32; MAX_BAND_FRAME_LENGTH];

    for i in 0..band_length {
        let tmp = low_band[i] as i32 + high_band[i] as i32;
        half_in1[i] = tmp * (1 << 10);
        let tmp = low_band[i] as i32 - high_band[i] as i32;
        half_in2[i] = tmp * (1 << 10);
    }

    all_pass_qmf(
        &mut half_in1[..band_length],
        &mut filter1[..band_length],
        &ALL_PASS_FILTER2,
        filter_state1,
    );
    all_pass_qmf(
        &mut half_in2[..band_length],
        &mut filter2[..band_length],
        &ALL_PASS_FILTER1,
        filter_state2,
    );

    let mut k = 0;
    for i in 0..band_length {
        let tmp = (filter2[i] + 512) >> 10;
        out_data[k] = sat_w32_to_w16(tmp);
        k += 1;
        let tmp = (filter1[i] + 512) >> 10;
        out_data[k] = sat_w32_to_w16(tmp);
        k += 1;
    }
}

fn all_pass_qmf(
    in_data: &mut [i32],
    out_data: &mut [i32],
    coeffs: &[u16; 3],
    state: &mut [i32; STATE_SIZE],
) {
    let data_length = in_data.len();
    let mut diff = sub_sat_w32(in_data[0], state[1]);
    out_data[0] = scaled_diff32(coeffs[0], diff, state[0]);
    for k in 1..data_length {
        diff = sub_sat_w32(in_data[k], out_data[k - 1]);
        out_data[k] = scaled_diff32(coeffs[0], diff, in_data[k - 1]);
    }
    state[0] = in_data[data_length - 1];
    state[1] = out_data[data_length - 1];

    diff = sub_sat_w32(out_data[0], state[3]);
    in_data[0] = scaled_diff32(coeffs[1], diff, state[2]);
    for k in 1..data_length {
        diff = sub_sat_w32(out_data[k], in_data[k - 1]);
        in_data[k] = scaled_diff32(coeffs[1], diff, out_data[k - 1]);
    }
    state[2] = out_data[data_length - 1];
    state[3] = in_data[data_length - 1];

    diff = sub_sat_w32(in_data[0], state[5]);
    out_data[0] = scaled_diff32(coeffs[2], diff, state[4]);
    for k in 1..data_length {
        diff = sub_sat_w32(in_data[k], out_data[k - 1]);
        out_data[k] = scaled_diff32(coeffs[2], diff, in_data[k - 1]);
    }
    state[4] = in_data[data_length - 1];
    state[5] = out_data[data_length - 1];
}

#[inline]
fn sub_sat_w32(a: i32, b: i32) -> i32 {
    let diff = a.wrapping_sub(b);
    if (a < 0) != (b < 0) && (a < 0) != (diff < 0) {
        if diff < 0 { i32::MAX } else { i32::MIN }
    } else {
        diff
    }
}

#[inline]
fn scaled_diff32(coeff: u16, value: i32, accum: i32) -> i32 {
    let high = (value >> 16) as i32 * coeff as i32;
    let low = (((value as u32) & 0x0000_FFFF) * coeff as u32) >> 16;
    accum + high + low as i32
}

#[inline]
fn sat_w32_to_w16(value: i32) -> i16 {
    if value > i16::MAX as i32 {
        i16::MAX
    } else if value < i16::MIN as i32 {
        i16::MIN
    } else {
        value as i16
    }
}
