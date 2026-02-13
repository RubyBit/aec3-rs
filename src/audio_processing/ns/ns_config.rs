//! Noise suppressor configuration.

/// Suppression aggressiveness target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionLevel {
    K6dB,
    K12dB,
    K18dB,
    K21dB,
}

/// Config struct for the noise suppressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NsConfig {
    pub target_level: SuppressionLevel,
    /// When enabled and the caller provides linear AEC output, NS analysis
    /// will use that signal instead of the raw capture signal.
    pub analyze_linear_aec_output_when_available: bool,
}

impl Default for NsConfig {
    fn default() -> Self {
        Self {
            target_level: SuppressionLevel::K12dB,
            analyze_linear_aec_output_when_available: false,
        }
    }
}
