//! Echo control API definitions ported from `api/echo_control.h`.

/// Metrics exposed by an echo controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            echo_return_loss: 0.0,
            echo_return_loss_enhancement: 0.0,
            delay_ms: 0,
        }
    }
}

/// Trait representing an acoustic echo cancellation submodule.
pub trait EchoControl {
    /// Buffer type used for render/capture processing.
    type Buffer;

    /// Analysis (without modification) of the render signal.
    fn analyze_render(&mut self, render: &mut Self::Buffer);

    /// Analysis (without modification) of the capture signal.
    fn analyze_capture(&mut self, capture: &mut Self::Buffer);

    /// Processes the capture signal in order to remove the echo.
    fn process_capture(&mut self, capture: &mut Self::Buffer, level_change: bool);

    /// Processes the capture signal and provides the linear filter output.
    fn process_capture_with_linear_output(
        &mut self,
        capture: &mut Self::Buffer,
        linear_output: &mut Self::Buffer,
        level_change: bool,
    );

    /// Collects current performance metrics.
    fn metrics(&self) -> Metrics;

    /// Provides an optional external estimate of the audio buffer delay.
    fn set_audio_buffer_delay(&mut self, delay_ms: i32);

    /// Whether the signal is altered.
    fn active_processing(&self) -> bool;
}

/// Trait for factories that create echo controllers.
pub trait EchoControlFactory {
    type Controller: EchoControl;

    fn create(
        &self,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self::Controller;
}
