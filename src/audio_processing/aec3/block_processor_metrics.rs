use crate::audio_processing::aec3::aec3_common::METRICS_REPORTING_INTERVAL_BLOCKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderUnderrunCategory {
    None,
    Few,
    Several,
    Many,
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOverrunCategory {
    None,
    Few,
    Several,
    Many,
    Constant,
}

/// Handles the reporting of metrics for the block processor.
#[derive(Debug, Default)]
pub struct BlockProcessorMetrics {
    capture_block_counter: usize,
    metrics_reported: bool,
    render_buffer_underruns: usize,
    render_buffer_overruns: usize,
    buffer_render_calls: usize,
    last_underrun_category: Option<RenderUnderrunCategory>,
    last_overrun_category: Option<RenderOverrunCategory>,
}

impl BlockProcessorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_capture(&mut self, underrun: bool) {
        self.capture_block_counter += 1;
        if underrun {
            self.render_buffer_underruns += 1;
        }

        if self.capture_block_counter == METRICS_REPORTING_INTERVAL_BLOCKS {
            self.metrics_reported = true;
            self.last_underrun_category = Some(self.categorize_underruns());
            self.last_overrun_category = Some(self.categorize_overruns());
            self.reset_metrics();
            self.capture_block_counter = 0;
        } else {
            self.metrics_reported = false;
        }
    }

    pub fn update_render(&mut self, overrun: bool) {
        self.buffer_render_calls += 1;
        if overrun {
            self.render_buffer_overruns += 1;
        }
    }

    pub fn metrics_reported(&self) -> bool {
        self.metrics_reported
    }

    pub fn last_underrun_category(&self) -> Option<RenderUnderrunCategory> {
        self.last_underrun_category
    }

    pub fn last_overrun_category(&self) -> Option<RenderOverrunCategory> {
        self.last_overrun_category
    }

    fn reset_metrics(&mut self) {
        self.render_buffer_underruns = 0;
        self.render_buffer_overruns = 0;
        self.buffer_render_calls = 0;
    }

    fn categorize_underruns(&self) -> RenderUnderrunCategory {
        if self.render_buffer_underruns == 0 {
            RenderUnderrunCategory::None
        } else if self.render_buffer_underruns > (self.capture_block_counter >> 1) {
            RenderUnderrunCategory::Constant
        } else if self.render_buffer_underruns > 100 {
            RenderUnderrunCategory::Many
        } else if self.render_buffer_underruns > 10 {
            RenderUnderrunCategory::Several
        } else {
            RenderUnderrunCategory::Few
        }
    }

    fn categorize_overruns(&self) -> RenderOverrunCategory {
        if self.render_buffer_overruns == 0 {
            RenderOverrunCategory::None
        } else if self.render_buffer_overruns > (self.buffer_render_calls >> 1) {
            RenderOverrunCategory::Constant
        } else if self.render_buffer_overruns > 100 {
            RenderOverrunCategory::Many
        } else if self.render_buffer_overruns > 10 {
            RenderOverrunCategory::Several
        } else {
            RenderOverrunCategory::Few
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_reporting_cycles() {
        let mut metrics = BlockProcessorMetrics::new();
        for _ in 0..(METRICS_REPORTING_INTERVAL_BLOCKS * 3) {
            metrics.update_render(false);
            metrics.update_render(false);
            metrics.update_capture(false);
        }
        assert!(metrics.metrics_reported());
    }
}
