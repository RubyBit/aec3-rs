//! Rust representation of WebRTC's `ChannelLayout` helper.

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    None = 0,
    Unsupported = 1,
    Mono = 2,
    Stereo = 3,
    Layout2_1 = 4,
    Surround = 5,
    Layout4_0 = 6,
    Layout2_2 = 7,
    Quad = 8,
    Layout5_0 = 9,
    Layout5_1 = 10,
    Layout5_0Back = 11,
    Layout5_1Back = 12,
    Layout7_0 = 13,
    Layout7_1 = 14,
    Layout7_1Wide = 15,
    StereoDownmix = 16,
    Layout2Point1 = 17,
    Layout3_1 = 18,
    Layout4_1 = 19,
    Layout6_0 = 20,
    Layout6_0Front = 21,
    Hexagonal = 22,
    Layout6_1 = 23,
    Layout6_1Back = 24,
    Layout6_1Front = 25,
    Layout7_0Front = 26,
    Layout7_1WideBack = 27,
    Octagonal = 28,
    Discrete = 29,
    StereoAndKeyboardMic = 30,
    Layout4_1QuadSide = 31,
    Bitstream = 32,
}

impl ChannelLayout {
    pub fn channel_count(self) -> usize {
        match self {
            ChannelLayout::None | ChannelLayout::Unsupported => 0,
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo
            | ChannelLayout::StereoDownmix
            | ChannelLayout::Layout2Point1
            | ChannelLayout::StereoAndKeyboardMic
            | ChannelLayout::Layout4_1QuadSide => 2,
            ChannelLayout::Layout2_1 | ChannelLayout::Layout3_1 | ChannelLayout::Layout2_2 => 3,
            ChannelLayout::Surround
            | ChannelLayout::Layout4_0
            | ChannelLayout::Quad
            | ChannelLayout::Layout4_1
            | ChannelLayout::Layout6_0Front
            | ChannelLayout::Hexagonal => 4,
            ChannelLayout::Layout5_0
            | ChannelLayout::Layout5_1
            | ChannelLayout::Layout5_0Back
            | ChannelLayout::Layout5_1Back
            | ChannelLayout::Layout6_0
            | ChannelLayout::Layout6_1
            | ChannelLayout::Layout6_1Back
            | ChannelLayout::Layout6_1Front => 6,
            ChannelLayout::Layout7_0
            | ChannelLayout::Layout7_1
            | ChannelLayout::Layout7_1Wide
            | ChannelLayout::Layout7_0Front
            | ChannelLayout::Layout7_1WideBack
            | ChannelLayout::Octagonal => 8,
            ChannelLayout::Discrete | ChannelLayout::Bitstream => 0,
        }
    }

    pub fn guess_from_channel_count(channels: usize) -> ChannelLayout {
        match channels {
            0 => ChannelLayout::None,
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            3 => ChannelLayout::Layout2_1,
            4 => ChannelLayout::Quad,
            5 => ChannelLayout::Layout5_0,
            6 => ChannelLayout::Layout5_1,
            7 => ChannelLayout::Layout7_0,
            _ => ChannelLayout::Layout7_1,
        }
    }
}
