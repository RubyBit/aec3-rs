use std::fmt;

/// Collection of flags indicating which CPU features are available.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AvailableCpuFeatures {
    // Intel.
    pub sse2: bool,
    pub avx2: bool,
    // ARM.
    pub neon: bool,
}

impl AvailableCpuFeatures {
    pub const fn new(sse2: bool, avx2: bool, neon: bool) -> Self {
        Self { sse2, avx2, neon }
    }
}

impl fmt::Display for AvailableCpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.sse2 {
            parts.push("SSE2");
        }
        if self.avx2 {
            parts.push("AVX2");
        }
        if self.neon {
            parts.push("NEON");
        }

        if parts.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&parts.join("_"))
        }
    }
}

/// Detects available CPU features.
pub fn get_available_cpu_features() -> AvailableCpuFeatures {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return AvailableCpuFeatures::new(
            std::arch::is_x86_feature_detected!("sse2"),
            std::arch::is_x86_feature_detected!("avx2"),
            false,
        );
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "aarch64")]
        let neon = std::arch::is_aarch64_feature_detected!("neon");
        #[cfg(target_arch = "arm")]
        let neon = std::arch::is_arm_feature_detected!("neon");
        return AvailableCpuFeatures::new(false, false, neon);
    }

    #[allow(unreachable_code)]
    AvailableCpuFeatures::new(false, false, false)
}

/// Returns CPU feature flags all set to false.
pub const fn no_available_cpu_features() -> AvailableCpuFeatures {
    AvailableCpuFeatures::new(false, false, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_string_none() {
        assert_eq!("none", no_available_cpu_features().to_string());
    }

    #[test]
    fn to_string_feature_join_order() {
        assert_eq!(
            "SSE2_AVX2_NEON",
            AvailableCpuFeatures::new(true, true, true).to_string()
        );
        assert_eq!(
            "AVX2",
            AvailableCpuFeatures::new(false, true, false).to_string()
        );
    }
}
