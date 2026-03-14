//! Transposed direct form I biquad filter.

/// Normalized biquad filter coefficients.
#[derive(Debug, Copy, Clone)]
pub struct Config {
    /// b[0], b[1], b[2].
    pub b: [f32; 3],
    /// a[1], a[2].
    pub a: [f32; 2],
}

#[derive(Debug, Copy, Clone, Default)]
struct State {
    b: [f32; 2],
    a: [f32; 2],
}

/// Transposed direct form I implementation of a bi-quad filter.
#[derive(Debug, Copy, Clone)]
pub struct BiQuadFilter {
    config: Config,
    state: State,
}

impl BiQuadFilter {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            state: State::default(),
        }
    }

    /// Sets filter configuration and resets internal state.
    pub fn set_config(&mut self, config: Config) {
        self.config = config;
        self.state = State::default();
    }

    /// Zeroes filter state.
    pub fn reset(&mut self) {
        self.state = State::default();
    }

    /// Filters `x` into `y` (same length). In-place processing is supported.
    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), y.len());

        let config_a0 = self.config.a[0];
        let config_a1 = self.config.a[1];
        let config_b0 = self.config.b[0];
        let config_b1 = self.config.b[1];
        let config_b2 = self.config.b[2];

        let mut state_a0 = self.state.a[0];
        let mut state_a1 = self.state.a[1];
        let mut state_b0 = self.state.b[0];
        let mut state_b1 = self.state.b[1];

        for (k, yk) in y.iter_mut().enumerate() {
            // Temporary x value allows in-place processing.
            let tmp = x[k];
            let y_val = config_b0 * tmp + config_b1 * state_b0 + config_b2 * state_b1
                - config_a0 * state_a0
                - config_a1 * state_a1;
            state_b1 = state_b0;
            state_b0 = tmp;
            state_a1 = state_a0;
            state_a0 = y_val;
            *yk = y_val;
        }

        self.state.a[0] = state_a0;
        self.state.a[1] = state_a1;
        self.state.b[0] = state_b0;
        self.state.b[1] = state_b1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_SIZE: usize = 8;
    const NUM_FRAMES: usize = 4;
    type FloatArraySequence = [[f32; FRAME_SIZE]; NUM_FRAMES];

    const BIQUAD_INPUT_SEQ: FloatArraySequence = [
        [
            -87.166290,
            -8.029022,
            101.619583,
            -0.294296,
            -5.825764,
            -8.890625,
            10.310432,
            54.845333,
        ],
        [
            -64.647644,
            -6.883945,
            11.059189,
            -95.242538,
            -108.870834,
            11.024944,
            63.044102,
            -52.709583,
        ],
        [
            -32.350529,
            -18.108028,
            -74.022339,
            -8.986874,
            -1.525581,
            103.705513,
            6.346226,
            -14.319557,
        ],
        [
            22.645832,
            -64.597153,
            55.462521,
            -109.393188,
            10.117825,
            -40.019642,
            -98.612228,
            -8.330326,
        ],
    ];

    // Computed as scipy.signal.butter(N=2, Wn=60/24000, btype='highpass').
    const BIQUAD_CONFIG: Config = Config {
        b: [0.99446179, -1.98892358, 0.99446179],
        a: [-1.98889291, 0.98895425],
    };

    const BIQUAD_OUTPUT_SEQ: FloatArraySequence = [
        [
            -86.68354497,
            -7.02175351,
            102.10290352,
            -0.37487333,
            -5.87205847,
            -8.85521608,
            10.33772563,
            54.51157181,
        ],
        [
            -64.92531604,
            -6.76395978,
            11.15534507,
            -94.68073341,
            -107.18177856,
            13.24642474,
            64.84288941,
            -50.97822629,
        ],
        [
            -30.1579652,
            -15.64850899,
            -71.06662821,
            -5.5883229,
            1.91175353,
            106.5572003,
            8.57183046,
            -12.06298473,
        ],
        [
            24.84286614,
            -62.18094158,
            57.91488056,
            -106.65685933,
            13.38760103,
            -36.60367134,
            -94.44880104,
            -3.59920354,
        ],
    ];

    fn expect_near_relative(expected: &[f32], computed: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), computed.len());
        for (i, (&e, &c)) in expected.iter().zip(computed.iter()).enumerate() {
            let abs_diff = (e - c).abs();
            if abs_diff == 0.0 {
                continue;
            }
            let safe_den = if e == 0.0 { 1.0 } else { e.abs() };
            let rel = abs_diff / safe_den;
            assert!(
                rel <= tolerance,
                "index={i}, expected={e}, computed={c}, rel={rel}, tol={tolerance}"
            );
        }
    }

    #[test]
    fn filter_not_in_place() {
        let mut filter = BiQuadFilter::new(BIQUAD_CONFIG);
        let mut samples = [0.0f32; FRAME_SIZE];

        for i in 0..NUM_FRAMES {
            filter.process(&BIQUAD_INPUT_SEQ[i], &mut samples);
            expect_near_relative(&BIQUAD_OUTPUT_SEQ[i], &samples, 2e-4);
        }
    }

    #[test]
    fn filter_in_place() {
        let mut filter = BiQuadFilter::new(BIQUAD_CONFIG);
        let mut samples = [0.0f32; FRAME_SIZE];

        for i in 0..NUM_FRAMES {
            samples.copy_from_slice(&BIQUAD_INPUT_SEQ[i]);
            let input_snapshot = samples;
            filter.process(&input_snapshot, &mut samples);
            expect_near_relative(&BIQUAD_OUTPUT_SEQ[i], &samples, 2e-4);
        }
    }

    #[test]
    fn set_config_different_output() {
        let mut filter = BiQuadFilter::new(Config {
            b: [0.97803048, -1.95606096, 0.97803048],
            a: [-1.95557824, 0.95654368],
        });

        let mut samples1 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples1);
        }

        filter.set_config(Config {
            b: [0.09763107, 0.19526215, 0.09763107],
            a: [-0.94280904, 0.33333333],
        });

        let mut samples2 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples2);
        }

        assert_ne!(samples1, samples2);
    }

    #[test]
    fn set_config_resets_state() {
        let mut filter = BiQuadFilter::new(BIQUAD_CONFIG);

        let mut samples1 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples1);
        }

        filter.set_config(BIQUAD_CONFIG);

        let mut samples2 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples2);
        }

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn reset() {
        let mut filter = BiQuadFilter::new(BIQUAD_CONFIG);

        let mut samples1 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples1);
        }

        filter.reset();

        let mut samples2 = [0.0f32; FRAME_SIZE];
        for frame in BIQUAD_INPUT_SEQ.iter().take(NUM_FRAMES) {
            filter.process(frame, &mut samples2);
        }

        assert_eq!(samples1, samples2);
    }
}
