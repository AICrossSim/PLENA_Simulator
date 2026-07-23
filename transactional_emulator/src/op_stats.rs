//! Records timing and HBM memory traffic for each instruction, enabled by `--op-stats`.
//!
//! Each instruction is bracketed by a read of the clock and the HBM read/write
//! counters; the difference gives its elapsed time (dt_ps) and HBM bytes moved.
//!
//! Output is JSON Lines - one line per instruction, plus a final summary line:
//!
//!   {"pc":12,"op":"M_MM","dt_ps":72000,"hbm_rd":0,"hbm_wr":0}
//!   ...
//!   {"aggregate":true,"start_ps":0,"end_ps":321504000,
//!    "total_dt_ps":321504000,"total_hbm_rd":270336,"total_hbm_wr":16384,
//!    "ops":[{"op":"H_PREFETCH_M","count":40,...}, ...]}
//!
//! Instructions with zero time and zero traffic are omitted from the per-line
//! output to keep it small, but still count toward the summary totals.
//!
//! Attribution: HBM traffic is charged to whichever instruction waited on it.
//! `H_PREFETCH_M/V` issue their DMA in the background, so the cost lands on the
//! first SRAM consumer that stalls on the load barrier. Under
//! `--blocking-prefetch` the prefetch instruction is charged directly instead.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memory::Statistics;
use runtime::Executor;

/// Snapshot taken immediately before an instruction dispatches.
pub(crate) struct OpMark {
    start_ps: u64,
    hbm: Statistics,
}

#[derive(Default)]
struct OpAgg {
    count: u64,
    dt_ps: u64,
    hbm_rd: u64,
    hbm_wr: u64,
}

pub(crate) struct OpStatsRecorder {
    writer: BufWriter<File>,
    /// Reads the live HBM byte counters; injected by the runner because the
    /// accelerator only holds a type-erased memory handle without them.
    hbm_stats: Box<dyn Fn() -> Statistics + Send>,
    agg: BTreeMap<&'static str, OpAgg>,
    run_start_ps: Option<u64>,
    last_end_ps: u64,
}

impl OpStatsRecorder {
    pub(crate) fn new(
        path: &Path,
        hbm_stats: Box<dyn Fn() -> Statistics + Send>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            writer: BufWriter::new(File::create(path)?),
            hbm_stats,
            agg: BTreeMap::new(),
            run_start_ps: None,
            last_end_ps: 0,
        })
    }

    /// Snapshot the executor clock and HBM counters before dispatching an op.
    pub(crate) fn begin(&self) -> OpMark {
        OpMark {
            start_ps: Executor::current().now().as_picos(),
            hbm: (self.hbm_stats)(),
        }
    }

    /// Record the deltas for one completed instruction.
    pub(crate) fn record(&mut self, pc: usize, op: &'static str, mark: OpMark) {
        let end_ps = Executor::current().now().as_picos();
        let after = (self.hbm_stats)();
        let dt_ps = end_ps - mark.start_ps;
        let hbm_rd = after.total_bytes_read - mark.hbm.total_bytes_read;
        let hbm_wr = after.total_bytes_written - mark.hbm.total_bytes_written;

        self.run_start_ps.get_or_insert(mark.start_ps);
        self.last_end_ps = end_ps;

        let entry = self.agg.entry(op).or_default();
        entry.count += 1;
        entry.dt_ps += dt_ps;
        entry.hbm_rd += hbm_rd;
        entry.hbm_wr += hbm_wr;

        // Zero-cost instructions (most scalar/control ops) are aggregate-only.
        if dt_ps != 0 || hbm_rd != 0 || hbm_wr != 0 {
            let _ = writeln!(
                self.writer,
                "{{\"pc\":{pc},\"op\":\"{op}\",\"dt_ps\":{dt_ps},\"hbm_rd\":{hbm_rd},\"hbm_wr\":{hbm_wr}}}",
            );
        }
    }

    /// Write the trailing aggregate line and flush. Call once, after `do_ops`.
    pub(crate) fn finish(&mut self) {
        let (total_dt, total_rd, total_wr) = self.agg.values().fold((0u64, 0u64, 0u64), |acc, a| {
            (acc.0 + a.dt_ps, acc.1 + a.hbm_rd, acc.2 + a.hbm_wr)
        });
        let ops: Vec<String> = self
            .agg
            .iter()
            .map(|(op, a)| {
                format!(
                    "{{\"op\":\"{op}\",\"count\":{},\"dt_ps\":{},\"hbm_rd\":{},\"hbm_wr\":{}}}",
                    a.count, a.dt_ps, a.hbm_rd, a.hbm_wr
                )
            })
            .collect();
        let _ = writeln!(
            self.writer,
            "{{\"aggregate\":true,\"start_ps\":{},\"end_ps\":{},\"total_dt_ps\":{total_dt},\"total_hbm_rd\":{total_rd},\"total_hbm_wr\":{total_wr},\"ops\":[{}]}}",
            self.run_start_ps.unwrap_or(0),
            self.last_end_ps,
            ops.join(","),
        );
        let _ = self.writer.flush();
    }
}
