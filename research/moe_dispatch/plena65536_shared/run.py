"""Reproduce the full compute DSE, dataflow report and shareable figures."""
import argparse
from pathlib import Path
from .search import main as search
from .report import main as report
from .plot import main as plot

def main(out):
    search(out)
    report(out)
    plot(out)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
