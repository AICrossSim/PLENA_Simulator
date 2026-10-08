"""Recorded presentation correction; leave numerical CSV/source run untouched."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from ...common import sha, write_json

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/"TABLES_ZH.md"
    before=sha(path)
    text=path.read_text()
    assert text.count("新异构/单核")==text.count("新异构/同构")==1
    text=text.replace("新异构/单核","该设计/单核").replace("新异构/同构","该设计/同构")
    text=text.replace("两列跨架构比值的分子都是该行设计；只有 best_hetero 行才是异构/基线。\n\n","")
    path.write_text(text.rstrip()+"\n")
    write_json(HERE/"REPORT_PRESENTATION_FIX.json",{
        "command":sys.argv,"completed_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha(Path(__file__)),"report_before_sha256":before,"report_after_sha256":sha(path),
        "numerical_csv_unchanged":True,"change":"table labels describe each row design; no numerical changes"})


if __name__=="__main__": main()
