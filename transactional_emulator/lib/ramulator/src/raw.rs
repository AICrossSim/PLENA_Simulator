use anyhow::Result;
use core::ffi::c_void;
use core::mem::ManuallyDrop;
use std::ffi::CString;

use crate::config::*;

mod bindings {
    use core::ffi::{c_char, c_void};

    // Extern type
    #[allow(non_camel_case_types)]
    #[repr(transparent)]
    pub struct ramulator(c_void);

    #[link(name = "ramulator")]
    unsafe extern "C-unwind" {
        pub fn ramulator_new(config: *const c_char) -> *mut ramulator;
        pub fn ramulator_finalize(val: *mut ramulator);
        pub fn ramulator_request(
            val: *mut ramulator,
            addr: u64,
            write: bool,
            callback: extern "C" fn(*mut c_void),
            data: *mut c_void,
            size: i32,
        ) -> bool;
        pub fn ramulator_period(val: *mut ramulator) -> f32;
        pub fn ramulator_tick(val: *mut ramulator);
    }
}

mod config {
    use serde::Serialize;

    use crate::config::*;

    #[derive(Serialize)]
    pub struct Config {
        pub frontend: Frontend,
        pub memory_system: MemorySystem,
    }

    #[derive(Serialize)]
    #[serde(tag = "impl")]
    pub enum Frontend {
        External { clock_ratio: u32 },
    }

    #[derive(Serialize)]
    #[serde(tag = "impl")]
    pub enum MemorySystem {
        #[serde(rename = "GenericDRAM")]
        GenericDRAM {
            clock_ratio: u32,
            controllers: Vec<Controller>,
            channel_mapper: ChannelMapper,
        },
    }
}
pub struct Ramulator {
    raw: *mut bindings::ramulator,
    burst_size: u32,
    channel_width: u32,
    num_channels: u32,
    transfer_size: u32,
}

// SAFETY: Ramulator is single-threaded.
unsafe impl Send for Ramulator {}

impl Drop for Ramulator {
    fn drop(&mut self) {
        unsafe {
            bindings::ramulator_finalize(self.raw);
        }
    }
}

impl Ramulator {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let dram_impl = match &config.controllers[0] {
            Controller::GenericDDR(c) => &c.dram,
            Controller::HBM12(c) => &c.dram,
        };

        let internal_prefetch_size = match dram_impl {
            // "DDR3" => 8,
            DRAM::DDR4(_) => 8,
            // "DDR4-VRR" => 8,
            // "DDR5" => 16,
            // "DDR5-VRR" => 16,
            // "DDR5-RVRR" => 16,
            // "LPDDR5" => 8,
            // "GDDR6" => 8,
            // "HBM" => 2,
            DRAM::HBM2(_) => 2,
            // "HBM3" => 2,
        };

        let default_channel_width = match dram_impl {
            // "DDR3" => 64,
            DRAM::DDR4(_) => 64,
            // "DDR4-VRR" => 64,
            // "DDR5" => 32,
            // "DDR5-VRR" => 32,
            // "DDR5-RVRR" => 32,
            // "LPDDR5" => 32,
            // "GDDR6" => 64,
            // "HBM" => 128,
            DRAM::HBM2(_) => 64,
            // "HBM3" => 64,
        };

        let num_channels = config.controllers.len() as u32;

        // Ramulator can exhibit undefined behaviour if config is incorrect.
        // Synthesis a working config here.
        let ramulator_config = config::Config {
            frontend: config::Frontend::External { clock_ratio: 1 },
            memory_system: config::MemorySystem::GenericDRAM {
                clock_ratio: 1,
                // dram: config.dram,
                controllers: config.controllers,
                channel_mapper: config.channel_mapper,
            },
        };

        let cfg = serde_json::to_string(&ramulator_config).unwrap();
        let c_str = CString::new(cfg).unwrap();
        let raw = unsafe { bindings::ramulator_new(c_str.as_ptr()) };
        if raw.is_null() {
            anyhow::bail!("Ramulator failed to initialize");
        }
        Ok(Ramulator {
            raw,
            burst_size: internal_prefetch_size,
            channel_width: default_channel_width,
            transfer_size: internal_prefetch_size * default_channel_width / 8,
            num_channels,
        })
    }

    pub fn period(&mut self) -> u32 {
        let period_in_ns = unsafe { bindings::ramulator_period(self.raw) };
        (period_in_ns * 1000.).round() as u32
    }

    pub fn burst_size(&self) -> u32 {
        self.burst_size
    }

    pub fn channel_width(&self) -> u32 {
        self.channel_width
    }

    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }

    pub fn tick(&mut self) {
        unsafe { bindings::ramulator_tick(self.raw) }
    }

    fn access_thin<F: FnOnce()>(&mut self, addr: u64, write: bool, callback: F) -> bool {
        extern "C" fn bridge<F: FnOnce()>(data: *mut c_void) {
            unsafe { core::mem::transmute_copy::<_, F>(&data)() }
        }

        union PtrUnion<F> {
            ptr: *mut c_void,
            data: ManuallyDrop<F>,
        }

        let mut ptr = PtrUnion {
            ptr: core::ptr::null_mut(),
        };
        ptr.data = ManuallyDrop::new(callback);

        let success = unsafe {
            bindings::ramulator_request(
                self.raw,
                addr,
                write,
                bridge::<F>,
                ptr.ptr,
                self.transfer_size as _,
            )
        };
        if !success {
            unsafe { ManuallyDrop::drop(&mut ptr.data) }
        }
        success
    }

    pub fn access<F: FnOnce()>(&mut self, addr: u64, write: bool, callback: F) -> bool {
        if core::mem::size_of::<F>() > core::mem::size_of::<*mut c_void>() {
            return self.access_thin(addr, write, Box::new(callback));
        }

        return self.access_thin(addr, write, callback);
    }

    pub fn read<F: FnOnce()>(&mut self, addr: u64, callback: F) -> bool {
        self.access(addr, false, callback)
    }

    pub fn write<F: FnOnce()>(&mut self, addr: u64, callback: F) -> bool {
        self.access(addr, true, callback)
    }
}
