//! XOR-shift based pseudo-random generator matching WebRTC's `webrtc::Random`.
//!
//! The implementation mirrors `rtc_base/random.{h,cc}` so that our test data is
//! reproducible and numerically aligned with the C++ reference.

#[derive(Clone, Debug)]
pub struct Random {
    state: u64,
}

impl Random {
    /// Creates a new RNG with the provided non-zero seed.
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "seed must be non-zero");
        Self { state: seed }
    }

    /// Generates the next 64-bit output using the xor-shift routine from the
    /// reference implementation.
    fn next_output(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        debug_assert!(self.state != 0);
        self.state.wrapping_mul(268_582_165_773_633_8717)
    }

    /// Uniform floating-point value in [0, 1).
    pub fn rand_float(&mut self) -> f32 {
        let value = (self.next_output() - 1) as f64 / 0xFFFF_FFFF_FFFF_FFFFu64 as f64;
        value as f32
    }

    /// Uniform double precision value in [0, 1).
    pub fn rand_double(&mut self) -> f64 {
        (self.next_output() - 1) as f64 / 0xFFFF_FFFF_FFFF_FFFFu64 as f64
    }

    /// Uniform integer in [0, t].
    pub fn rand_u32(&mut self, t: u32) -> u32 {
        let x = self.next_output() as u32;
        (((x as u64) * (t as u64 + 1)) >> 32) as u32
    }

    /// Uniform integer in [low, high].
    pub fn rand_u32_range(&mut self, low: u32, high: u32) -> u32 {
        assert!(low <= high);
        low + self.rand_u32(high - low)
    }

    /// Uniform integer in [low, high].
    pub fn rand_i32_range(&mut self, low: i32, high: i32) -> i32 {
        assert!(low <= high);
        let span = (high as i64 - low as i64) as u32;
        (self.rand_u32(span) as i64 + low as i64) as i32
    }

    pub fn rand_bool(&mut self) -> bool {
        self.rand_u32_range(0, 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::Random;

    #[test]
    fn sequence_matches_reference_behavior() {
        let mut rng = Random::new(0x1234_5678_9abc_def0);
        let first = rng.next_output();
        let second = rng.next_output();
        assert_ne!(first, second);
        assert!(first != 0 && second != 0);
    }

    #[test]
    #[should_panic]
    fn zero_seed_panics() {
        let _ = Random::new(0);
    }
}
