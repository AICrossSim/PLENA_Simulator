"""Separate row padding, column padding, K padding, and dynamic ownership."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from .report import gm, mean, write_csv, BATCHES


def generate(output):
    grouped=defaultdict(list)
    for line in (output/"records.jsonl").open():
        row=json.loads(line)
        if row["profile"]!="pipelined_tree1" or row["policy"]!="full":continue
        useful=m_pad=mn_pad=mnk_pad=0
        cd=lambda a,b:(a+b-1)//b
        cores=row["cores"]
        group_of={i:g for g,workers in enumerate(row["groups"]) for i in workers}
        shared_group=None
        for j in row["result"]["jobs"]:
            c=cores[j["core_id"]]
            m,h,f=j["M"],j["H"],j["F"]
            for k,n,multiplicity in ((h,f,2),(f,h,1)):
                mp=cd(m,c["pm"])*c["pm"]
                np=cd(n,c["pn"])*c["pn"]
                kp=cd(k,c["r"])*c["r"]
                useful+=multiplicity*m*n*k
                m_pad+=multiplicity*mp*n*k
                mn_pad+=multiplicity*mp*np*k
                mnk_pad+=multiplicity*mp*np*kp
            if j["is_shared"]:shared_group=group_of[j["core_id"]]
        assert mnk_pad==row["result"]["issued_macs"]
        grouped[row["design_id"]].append(dict(
            window_id=row["window_id"],batch=row["batch"],design_id=row["design_id"],
            scale=row["scale"],role=row["role"],variant=row["variant"],
            M_fill_pct=100*useful/m_pad,N_fill_given_M_pct=100*m_pad/mn_pad,
            K_fill_given_MN_pct=100*mn_pad/mnk_pad,spatial_util_pct=100*useful/mnk_pad,
            whole_chip_util_pct=100*row["result"]["wall_peak_utilization"],
            Shared_on_P_pct=100*int(shared_group==0),
            Shared_on_E_pct=100*int(shared_group==1)))
    output_rows=[]
    for design,rows in grouped.items():
        batch_rows=[]
        for b in BATCHES:
            selected=[r for r in rows if r["batch"]==b]
            record={k:rows[0][k] for k in ("design_id","scale","role","variant")}
            record["batch"]=b
            for k in ("M_fill_pct","N_fill_given_M_pct","K_fill_given_MN_pct","spatial_util_pct","whole_chip_util_pct"):
                record[k]=gm(r[k] for r in selected)
            for k in ("Shared_on_P_pct","Shared_on_E_pct"):
                record[k]=mean(r[k] for r in selected)
            batch_rows.append(record)
        total={**batch_rows[0],"batch":"7-Batch GM"}
        for k in ("M_fill_pct","N_fill_given_M_pct","K_fill_given_MN_pct","spatial_util_pct","whole_chip_util_pct"):
            total[k]=gm(r[k] for r in batch_rows)
        for k in ("Shared_on_P_pct","Shared_on_E_pct"):
            total[k]=mean(r[k] for r in batch_rows)
        output_rows.extend(batch_rows+[total])
    write_csv(output/"padding_and_ownership.csv",output_rows)
    lines=["# 行、列与归约填充的独立归因","", 
           "所有数据来自完整 M-DPU 主时序；逐窗口计算后按七个 batch 等权汇总。M 填充率只计算行浪费；N、K 列是依次展开后的条件利用率，三者相乘等于算术发射槽位利用率。Shared 分配比例为窗口数比例。", "",
           "| 配置 | Batch | 行填充率 % | 列填充率 % | K 填充率 % | 发射槽位利用率 % | 整片有效利用率 % | Shared 给 P % |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in output_rows:
        lines.append("| "+" | ".join([r["design_id"],str(r["batch"])]+[f"{r[k]:.3f}" for k in
          ("M_fill_pct","N_fill_given_M_pct","K_fill_given_MN_pct","spatial_util_pct","whole_chip_util_pct","Shared_on_P_pct")])+" |")
    (output/"MECHANISM.md").write_text("\n".join(lines)+"\n")
    return output_rows

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    generate(p.parse_args().output.resolve())
