// The library target exposes DMA calibration helpers without constructing the
// full runner, so some production-only DMA accounting types are intentionally
// unused in this target. They remain used by the emulator binary.
#[allow(dead_code)]
mod dma;

pub mod dma_calibration;
