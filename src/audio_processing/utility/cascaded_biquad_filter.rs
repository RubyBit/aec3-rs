use num_complex::Complex32;

/// Parameters describing the zeros, poles, and gain for a single biquad stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiQuadParam {
    pub zero: Complex32,
    pub pole: Complex32,
    pub gain: f32,
    pub mirror_zero_along_i_axis: bool,
}

impl BiQuadParam {
    pub const fn new(zero: Complex32, pole: Complex32, gain: f32) -> Self {
        Self {
            zero,
            pole,
            gain,
            mirror_zero_along_i_axis: false,
        }
    }

    pub const fn with_mirrored_zero(zero_real: f32, pole: Complex32, gain: f32) -> Self {
        Self {
            zero: Complex32::new(zero_real, 0.0),
            pole,
            gain,
            mirror_zero_along_i_axis: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiQuadCoefficients {
    pub b: [f32; 3],
    pub a: [f32; 2],
}

#[derive(Clone, Debug)]
struct BiQuad {
    coefficients: BiQuadCoefficients,
    x: [f32; 2],
    y: [f32; 2],
}

impl BiQuad {
    fn new(coefficients: BiQuadCoefficients) -> Self {
        Self {
            coefficients,
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn from_param(param: BiQuadParam) -> Self {
        let z_r = param.zero.re;
        let z_i = param.zero.im;
        let p_r = param.pole.re;
        let p_i = param.pole.im;

        let mut b = [0.0f32; 3];
        if param.mirror_zero_along_i_axis {
            b[0] = param.gain;
            b[1] = 0.0;
            b[2] = param.gain * -(z_r * z_r);
        } else {
            b[0] = param.gain;
            b[1] = param.gain * -2.0 * z_r;
            b[2] = param.gain * (z_r * z_r + z_i * z_i);
        }

        let a = [-2.0 * p_r, p_r * p_r + p_i * p_i];
        Self::new(BiQuadCoefficients { b, a })
    }

    fn reset(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

pub struct CascadedBiQuadFilter {
    biquads: Vec<BiQuad>,
}

impl CascadedBiQuadFilter {
    pub fn with_coefficients(coefficients: BiQuadCoefficients, num_biquads: usize) -> Self {
        let mut biquads = Vec::with_capacity(num_biquads);
        for _ in 0..num_biquads {
            biquads.push(BiQuad::new(coefficients));
        }
        Self { biquads }
    }

    pub fn from_params(params: &[BiQuadParam]) -> Self {
        let biquads = params.iter().map(|p| BiQuad::from_param(*p)).collect();
        Self { biquads }
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        if self.biquads.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        self.apply_biquad_stage(input, output, 0);
        for stage in 1..self.biquads.len() {
            self.apply_biquad_stage_in_place(output, stage);
        }
    }

    pub fn process_in_place(&mut self, data: &mut [f32]) {
        for stage in 0..self.biquads.len() {
            self.apply_biquad_stage_in_place(data, stage);
        }
    }

    pub fn reset(&mut self) {
        for biquad in &mut self.biquads {
            biquad.reset();
        }
    }

    fn apply_biquad_stage(&mut self, input: &[f32], output: &mut [f32], stage: usize) {
        assert_eq!(input.len(), output.len());
        let biquad = &mut self.biquads[stage];
        let b = biquad.coefficients.b;
        let a = biquad.coefficients.a;
        let x_hist = &mut biquad.x;
        let y_hist = &mut biquad.y;
        for (x, y) in input.iter().zip(output.iter_mut()) {
            let tmp = *x;
            let out = b[0] * tmp + b[1] * x_hist[0] + b[2] * x_hist[1]
                - a[0] * y_hist[0]
                - a[1] * y_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = tmp;
            y_hist[1] = y_hist[0];
            y_hist[0] = out;
            *y = out;
        }
    }

    fn apply_biquad_stage_in_place(&mut self, data: &mut [f32], stage: usize) {
        let biquad = &mut self.biquads[stage];
        let b = biquad.coefficients.b;
        let a = biquad.coefficients.a;
        let x_hist = &mut biquad.x;
        let y_hist = &mut biquad.y;
        for sample in data.iter_mut() {
            let tmp = *sample;
            let out = b[0] * tmp + b[1] * x_hist[0] + b[2] * x_hist[1]
                - a[0] * y_hist[0]
                - a[1] * y_hist[1];
            x_hist[1] = x_hist[0];
            x_hist[0] = tmp;
            y_hist[1] = y_hist[0];
            y_hist[0] = out;
            *sample = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_input_when_no_biquads() {
        let mut filter = CascadedBiQuadFilter::from_params(&[]);
        let input = [1.0f32, -2.0, 3.5, 0.25];
        let mut output = [0.0f32; 4];
        filter.process(&input, &mut output);
        assert_eq!(input, output);
    }

    #[test]
    fn identity_coefficients_behave_like_passthrough() {
        let coeffs = BiQuadCoefficients {
            b: [1.0, 0.0, 0.0],
            a: [0.0, 0.0],
        };
        let mut filter = CascadedBiQuadFilter::with_coefficients(coeffs, 1);
        let input = [0.5f32, -0.25, 1.5, -3.0, 0.0];
        let mut output = [0.0f32; 5];
        filter.process(&input, &mut output);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn mirrored_zero_construction_matches_reference_formulas() {
        let param = BiQuadParam::with_mirrored_zero(0.5, Complex32::new(0.25, 0.75), 0.8);
        let biquad = BiQuad::from_param(param);
        assert!((biquad.coefficients.b[0] - 0.8).abs() < 1e-6);
        assert!((biquad.coefficients.b[1]).abs() < f32::EPSILON);
        assert!((biquad.coefficients.b[2] - 0.8 * -0.25).abs() < 1e-6);
        assert!((biquad.coefficients.a[0] + 0.5).abs() < 1e-6);
        assert!((biquad.coefficients.a[1] - (0.25 * 0.25 + 0.75 * 0.75)).abs() < 1e-6);
    }

    #[test]
    fn process_in_place_matches_out_of_place() {
        let coeffs = BiQuadCoefficients {
            b: [0.5, 0.25, 0.0],
            a: [0.1, -0.05],
        };
        let mut filter_a = CascadedBiQuadFilter::with_coefficients(coeffs, 2);
        let mut filter_b = CascadedBiQuadFilter::with_coefficients(coeffs, 2);
        let input = [1.0f32, 0.5, -0.25, 0.0, 2.0, -1.0, 0.25, 0.75];
        let mut out_of_place = [0.0f32; 8];
        let mut in_place = input.clone();

        filter_a.process(&input, &mut out_of_place);
        filter_b.process_in_place(&mut in_place);

        for (a, b) in out_of_place.iter().zip(in_place.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
