use anyhow::{Context, Result};
use core::ffi::c_void;
use core::mem::ManuallyDrop;
use std::ffi::CString;

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
            callback: Option<extern "C" fn(*mut c_void)>,
            data: *mut c_void,
        ) -> bool;
        pub fn ramulator_period(val: *mut ramulator) -> f32;
        pub fn ramulator_tick(val: *mut ramulator);
    }
}

mod config {
    use serde::Serialize;

    #[derive(Serialize)]
    #[serde(rename_all = "PascalCase")]
    pub struct Config {
        pub frontend: Frontend,
        pub memory_system: MemorySystem,
    }

    #[derive(Serialize)]
    #[serde(tag = "impl")]
    pub enum Frontend {
        GEM5,
    }

    #[derive(Serialize)]
    #[serde(tag = "impl")]
    pub enum MemorySystem {
        GenericDRAM {
            clock_ratio: u32,
            #[serde(rename = "DRAM")]
            dram: serde_json::Value,
            #[serde(rename = "Controller")]
            controller: serde_json::Value,
            #[serde(rename = "AddrMapper")]
            addr_mapper: serde_json::Value,
        },
    }
}
pub struct Ramulator {
    raw: *mut bindings::ramulator,
    burst_size: u32,
    channel_width: u32,
    num_channels: u32,
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
        let dram_impl = config.dram["impl"]
            .as_str()
            .context("config's DRAM.impl does not exist")?;

        let internal_prefetch_size = match dram_impl {
            "DDR3" => 8,
            "DDR4" => 8,
            "DDR4-VRR" => 8,
            "DDR5" => 16,
            "DDR5-VRR" => 16,
            "DDR5-RVRR" => 16,
            "LPDDR5" => 8,
            "GDDR6" => 8,
            "HBM" => 2,
            "HBM2" => 2,
            "HBM3" => 2,
            _ => unreachable!(),
        };

        // TODO: It looks like this can be set from config file, but ramulator2 has no example
        // of doing so. For now assume the default will be used.
        let default_channel_width = match dram_impl {
            "DDR3" => 64,
            "DDR4" => 64,
            "DDR4-VRR" => 64,
            "DDR5" => 32,
            "DDR5-VRR" => 32,
            "DDR5-RVRR" => 32,
            "LPDDR5" => 32,
            "GDDR6" => 64,
            "HBM" => 128,
            "HBM2" => 64,
            "HBM3" => 64,
            _ => unreachable!(),
        };

        // TODO: allow presets?
        let num_channels = config.dram["org"]["channel"].as_u64().context(
            "config's DRAM.org.channel is not set. Please set it explicitly even if preset is used",
        )? as u32;

        // For HBM, Ramulator does not support ClosedRowPolicy
        if matches!(dram_impl, "GDDR6" | "HBM" | "HBM2" | "HBM3")
            && config.controller["RowPolicy"]["impl"] == "ClosedRowPolicy"
        {
            anyhow::bail!("Ramulator would crash if HBM is used with ClosedRowPolicy");
        }

        // Ramulator can exhibit undefined behaviour if config is incorrect.
        // Synthesis a working config here.
        let ramulator_config = config::Config {
            frontend: config::Frontend::GEM5,
            memory_system: config::MemorySystem::GenericDRAM {
                clock_ratio: 1,
                dram: config.dram,
                controller: config.controller,
                addr_mapper: config.addr_mapper,
            },
        };

        let c_str = CString::new(serde_json::to_string(&ramulator_config).unwrap()).unwrap();
        let raw = unsafe { bindings::ramulator_new(c_str.as_ptr()) };
        if raw.is_null() {
            anyhow::bail!("Ramulator failed to initialize");
        }
        Ok(Ramulator {
            raw,
            burst_size: internal_prefetch_size,
            channel_width: default_channel_width,
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

    fn read_thin<F: FnOnce()>(&mut self, addr: u64, callback: F) -> bool {
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
            bindings::ramulator_request(self.raw, addr, false, Some(bridge::<F>), ptr.ptr)
        };
        if !success {
            unsafe { ManuallyDrop::drop(&mut ptr.data) }
        }
        success
    }

    pub fn read<F: FnOnce()>(&mut self, addr: u64, callback: F) -> bool {
        if core::mem::size_of::<F>() > core::mem::size_of::<*mut c_void>() {
            return self.read_thin(addr, Box::new(callback));
        }

        return self.read_thin(addr, callback);
    }

    pub fn write(&mut self, addr: u64) -> bool {
        let success = unsafe {
            bindings::ramulator_request(self.raw, addr, true, None, core::ptr::null_mut())
        };
        success
    }
}
