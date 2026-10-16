"""Reproduce the complete corrected Shared + routed MoE analytical study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from research.moe_dispatch.plena65536_moe import search, report, plot, timing_audit, dispatch_analysis, delivery
from research.moe_dispatch.plena65536_moe.workloads import load_workloads,write_workload_artifacts
from research.moe_dispatch.plena65536_moe.split_diagnostic import write_frozen_diagnostics


def main(output):
    output=Path(output)
    # Search never opens heldout routes; load complete input provenance only
    # after both development searches have frozen identical hardware.
    search.main(output)
    search.main(output/'repeat_search')
    delivery.verify_search_repeat(output)
    inputs=load_workloads()
    write_workload_artifacts(inputs,output)
    report.main(output)
    timing_audit.main(output)
    selected=json.loads((output/'FROZEN_SELECTION.json').read_text())
    write_frozen_diagnostics(selected,inputs['heldout'],output,repeats=2)
    dispatch_analysis.main(output)
    plot.main(output)
    delivery.package(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
