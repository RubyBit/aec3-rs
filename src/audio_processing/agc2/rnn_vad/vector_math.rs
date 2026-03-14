//! Vector math helpers for RNN-VAD.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;

/// Provides mathematical operations over vectors with architecture-dispatch
/// structure equivalent to the WebRTC implementation.
#[derive(Debug, Copy, Clone)]
pub struct VectorMath {
    cpu_features: AvailableCpuFeatures,
}

impl VectorMath {
    pub fn new(cpu_features: AvailableCpuFeatures) -> Self {
        Self { cpu_features }
    }

    /// Computes the dot product between two equally sized vectors.
    pub fn dot_product(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());

        if self.cpu_features.avx2 {
            return self.dot_product_avx2(x, y);
        }
        if self.cpu_features.sse2 {
            return self.dot_product_sse2(x, y);
        }
        if self.cpu_features.neon {
            return self.dot_product_neon(x, y);
        }
        self.dot_product_scalar(x, y)
    }

    fn dot_product_scalar(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    // The implementation uses scalar fallback for now while preserving the
    // architecture dispatch shape one-to-one with WebRTC.
    fn dot_product_avx2(&self, x: &[f32], y: &[f32]) -> f32 {
        self.dot_product_scalar(x, y)
    }

    fn dot_product_sse2(&self, x: &[f32], y: &[f32]) -> f32 {
        self.dot_product_scalar(x, y)
    }

    fn dot_product_neon(&self, x: &[f32], y: &[f32]) -> f32 {
        self.dot_product_scalar(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::get_available_cpu_features;

    const X: [f32; 19] = [
        0.31593041,
        0.9350786,
        -0.25252445,
        -0.86956251,
        -0.9673632,
        0.54571901,
        -0.72504495,
        -0.79509912,
        -0.25525012,
        -0.73340473,
        0.15747377,
        -0.04370565,
        0.76135145,
        -0.57239645,
        0.68616848,
        0.3740298,
        0.34710799,
        -0.92207423,
        0.10738454,
    ];
    const SIZE_OF_X_SUBSPAN: usize = 16;
    const ENERGY_OF_X: f32 = 7.315563958160327;
    const ENERGY_OF_X_SUBSPAN: f32 = 6.333327669592963;

    fn get_cpu_features_to_test() -> Vec<AvailableCpuFeatures> {
        let mut v = vec![AvailableCpuFeatures::new(false, false, false)];
        let available = get_available_cpu_features();

        if available.avx2 {
            v.push(AvailableCpuFeatures::new(false, true, false));
        }
        if available.sse2 {
            v.push(AvailableCpuFeatures::new(true, false, false));
        }
        if available.neon {
            v.push(AvailableCpuFeatures::new(false, false, true));
        }
        v
    }

    #[test]
    fn test_dot_product() {
        for cpu_features in get_cpu_features_to_test() {
            let vector_math = VectorMath::new(cpu_features);
            let energy = vector_math.dot_product(&X, &X);
            let energy_subspan = vector_math.dot_product(&X[..SIZE_OF_X_SUBSPAN], &X[..SIZE_OF_X_SUBSPAN]);

            assert!(
                (energy - ENERGY_OF_X).abs() < 1e-6,
                "cpu_features={cpu_features}, got={energy}, expected={ENERGY_OF_X}"
            );
            assert!(
                (energy_subspan - ENERGY_OF_X_SUBSPAN).abs() < 1e-6,
                "cpu_features={cpu_features}, got={energy_subspan}, expected={ENERGY_OF_X_SUBSPAN}"
            );
        }
    }
}
