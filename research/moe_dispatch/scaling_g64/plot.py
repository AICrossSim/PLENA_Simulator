"""Standalone scientific plot from measured simulator tables, no fitted data."""
import argparse
import csv
from pathlib import Path
import os


def generate(out):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/plena-scaling-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (out/"all_aggregates.csv").open(encoding="utf-8-sig") as f:
        rows=[r for r in csv.DictReader(f) if r["batch"]=="7-Batch GM" and
              r["profile"]=="pipelined_tree1" and r["policy"]=="full"]
    by={(int(r["scale"]),r["role"],r["variant"]):r for r in rows}
    scales=(65536,131072,262144)
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.7))
    styles=[("single","main","Compact single","#555555","o"),
            ("homo","main","Compact homogeneous","#2879b9","s"),
            ("hetero","main","Heterogeneous, monolithic E*","#d76c2e","^"),
            ("hetero","split_e","Heterogeneous, independent E lanes","#9a55bb","D"),
            ("fair_homo_stream","main","Homogeneous stream control","#2a9e79","v"),
            ("fair_stream_cluster","split_e","Independent stream control","#77aa33","P")]
    for role,variant,label,color,marker in styles:
        pairs=[(s,by[(s,role,variant)]) for s in scales if (s,role,variant) in by]
        xs=[s//1024 for s,_ in pairs]
        axes[0].plot(xs,[1000*float(r["compute_ms"]) for _,r in pairs],
                     label=label,color=color,marker=marker,lw=1.8)
        if role=="hetero":
            axes[1].plot(xs,[float(by[(s,"homo","main")]["compute_ms"])/float(r["compute_ms"])
                            for s,r in pairs],label=label,color=color,marker=marker,lw=1.8)
    axes[0].set_ylabel("7-batch GM latency (microseconds)")
    axes[1].set_ylabel("Speedup over compact homogeneous")
    axes[1].axhline(1,color="#777777",lw=1,ls="--")
    for ax in axes:
        ax.set_xticks([64,128,256]);ax.set_xlabel("Total multiplier budget (Ki MACs)")
        ax.grid(True,alpha=.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Fixed G=64 compute scaling: ideal data supply, timed controller",fontsize=13)
    fig.text(.5,.015,"* 256K monolithic E has PM=4; independent-lane variant retains PM<=2. Analytical timing, not silicon measurements.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(out/"scaling_comparison.png",dpi=180)
    fig.savefig(out/"scaling_comparison.pdf")
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    generate(p.parse_args().output.resolve())
