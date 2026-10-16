"""Run full Dedicated Shared/Routed DSE and all verification/delivery steps."""
import argparse
from pathlib import Path
from research.moe_dispatch.plena65536_dedicated import search,report,delivery,split_frontier
from research.moe_dispatch.plena65536_dedicated.independent_audit import audit_saved_search


def main(output):
    output=Path(output)
    search.main(output)
    search.main(output/'repeat_search')
    delivery.verify_repeat(output)
    audit_saved_search(output,repeat_directory=output/'repeat_search')
    report.run(output)
    split_frontier.main(output)
    delivery.package(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
