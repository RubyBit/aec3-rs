//! Three-band analysis/synthesis filter bank.

use std::f32::consts::PI;

use crate::audio_processing::sparse_fir_filter::SparseFIRFilter;

const NUM_BANDS: usize = 3;
const SPARSITY: usize = 4;
const NUM_COEFFS: usize = 4;

const LOWPASS_COEFFS: [[f32; NUM_COEFFS]; NUM_BANDS * SPARSITY] = [
    [-0.00047749, -0.00496888, 0.16547118, 0.00425496],
    [-0.00173287, -0.01585778, 0.14989004, 0.00994113],
    [-0.00304815, -0.02536082, 0.12154542, 0.01157993],
    [-0.00383509, -0.02982767, 0.08543175, 0.00983212],
    [-0.00346946, -0.02587886, 0.04760441, 0.00607594],
    [-0.00154717, -0.01136076, 0.01387458, 0.00186353],
    [0.00186353, 0.01387458, -0.01136076, -0.00154717],
    [0.00607594, 0.04760441, -0.02587886, -0.00346946],
    [0.00983212, 0.08543175, -0.02982767, -0.00383509],
    [0.01157993, 0.12154542, -0.02536082, -0.00304815],
    [0.00994113, 0.14989004, -0.01585778, -0.00173287],
    [0.00425496, 0.16547118, -0.00496888, -0.00047749],
];

pub struct ThreeBandFilterBank {
    in_buffer: Vec<f32>,
    out_buffer: Vec<f32>,
    analysis_filters: Vec<SparseFIRFilter>,
    synthesis_filters: Vec<SparseFIRFilter>,
    dct_modulation: Vec<[f32; NUM_BANDS]>,
}

impl ThreeBandFilterBank {
    pub fn new(length: usize) -> Self {
        assert_eq!(length % NUM_BANDS, 0);
        let split_length = length / NUM_BANDS;
        let mut analysis_filters = Vec::with_capacity(NUM_BANDS * SPARSITY);
        let mut synthesis_filters = Vec::with_capacity(NUM_BANDS * SPARSITY);
        for i in 0..SPARSITY {
            for j in 0..NUM_BANDS {
                analysis_filters.push(SparseFIRFilter::new(
                    &LOWPASS_COEFFS[i * NUM_BANDS + j],
                    SPARSITY,
                    i,
                ));
                synthesis_filters.push(SparseFIRFilter::new(
                    &LOWPASS_COEFFS[i * NUM_BANDS + j],
                    SPARSITY,
                    i,
                ));
            }
        }
        let mut dct_modulation = Vec::with_capacity(NUM_BANDS * SPARSITY);
        for idx in 0..NUM_BANDS * SPARSITY {
            let mut row = [0.0f32; NUM_BANDS];
            for band in 0..NUM_BANDS {
                row[band] = 2.0
                    * ((2.0 * PI * idx as f32 * (2.0 * band as f32 + 1.0))
                        / (NUM_BANDS * SPARSITY) as f32)
                        .cos();
            }
            dct_modulation.push(row);
        }

        Self {
            in_buffer: vec![0.0; split_length],
            out_buffer: vec![0.0; split_length],
            analysis_filters,
            synthesis_filters,
            dct_modulation,
        }
    }

    pub fn analysis(&mut self, input: &[f32], out: &mut [&mut [f32]; NUM_BANDS]) {
        let split_length = self.in_buffer.len();
        assert_eq!(input.len(), split_length * NUM_BANDS);
        for band in out.iter_mut() {
            (**band).fill(0.0);
        }
        for i in 0..NUM_BANDS {
            downsample(input, split_length, NUM_BANDS - i - 1, &mut self.in_buffer);
            for j in 0..SPARSITY {
                let offset = i + j * NUM_BANDS;
                self.analysis_filters[offset].filter(&self.in_buffer, &mut self.out_buffer);
                down_modulate(&self.out_buffer, offset, &self.dct_modulation, out);
            }
        }
    }

    pub fn synthesis(&mut self, input: &[&[f32]; NUM_BANDS], out: &mut [f32]) {
        let split_length = self.in_buffer.len();
        assert_eq!(out.len(), split_length * NUM_BANDS);
        out.fill(0.0);
        for i in 0..NUM_BANDS {
            for j in 0..SPARSITY {
                let offset = i + j * NUM_BANDS;
                up_modulate(
                    input,
                    split_length,
                    offset,
                    &self.dct_modulation,
                    &mut self.in_buffer,
                );
                self.synthesis_filters[offset].filter(&self.in_buffer, &mut self.out_buffer);
                upsample(&self.out_buffer, i, out);
            }
        }
    }
}

fn downsample(input: &[f32], split_length: usize, offset: usize, out: &mut [f32]) {
    for i in 0..split_length {
        out[i] = input[NUM_BANDS * i + offset];
    }
}

fn upsample(input: &[f32], offset: usize, out: &mut [f32]) {
    let split_length = input.len();
    for i in 0..split_length {
        out[NUM_BANDS * i + offset] += NUM_BANDS as f32 * input[i];
    }
}

fn down_modulate(
    input: &[f32],
    modulation_index: usize,
    dct_modulation: &[[f32; NUM_BANDS]],
    out: &mut [&mut [f32]; NUM_BANDS],
) {
    for band in 0..NUM_BANDS {
        let band_slice = &mut **out.get_mut(band).unwrap();
        for (j, sample) in input.iter().enumerate() {
            band_slice[j] += dct_modulation[modulation_index][band] * sample;
        }
    }
}

fn up_modulate(
    input: &[&[f32]; NUM_BANDS],
    split_length: usize,
    modulation_index: usize,
    dct_modulation: &[[f32; NUM_BANDS]],
    out: &mut [f32],
) {
    out.fill(0.0);
    for band in 0..NUM_BANDS {
        for i in 0..split_length {
            out[i] += dct_modulation[modulation_index][band] * input[band][i];
        }
    }
}
