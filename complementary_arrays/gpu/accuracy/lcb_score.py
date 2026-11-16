"""LiveCodeBench scoring with the official runner (lcb_runner.evaluation.compute_code_generation_metrics, commit
28fef95), run inside a container without network (lcb_score.sh). Tests = the public and private tests of each problem,
6 s per test; a sample passes when every test passes.

usage (inside the container): python lcb_score.py TASKS.jsonl GENERATED.jsonl OUT.jsonl [processes]
"""

import base64
import json
import pickle
import sys
import zlib

sys.path.insert(0, "/lcb")
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def tests(r):
    public = json.loads(r["public_test_cases"])
    try:
        private = json.loads(r["private_test_cases"])
    except Exception:
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode("utf-8")))))
    t = public + private
    io = dict(
        inputs=[x["input"] for x in t],
        outputs=[x["output"] for x in t],
        fn_name=json.loads(r["metadata"]).get("func_name"),
    )
    return {"input_output": json.dumps(io)}


def main():
    data_path, gen_path, out_path = sys.argv[1:4]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    gens = [json.loads(line) for line in open(gen_path)]
    want = {g["id"] for g in gens}
    data = {r["id"]: r for r in map(json.loads, open(data_path)) if r["id"] in want}
    samples = [tests(data[g["id"]]) for g in gens]
    generations = [[s["pred"] or "" for s in g["samples"]] for g in gens]
    metrics, results, _ = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=procs, timeout=6)
    with open(out_path, "w") as f:
        for i, g in enumerate(gens):
            ok = [
                bool(res) and all(x is True or (not isinstance(x, bool) and x > 0) for x in res) for res in results[i]
            ]
            rec = dict(
                id=g["id"],
                passed=ok,
                score=sum(ok) / len(ok),
                difficulty=data[g["id"]]["difficulty"],
                finish=[s["finish"] for s in g["samples"]],
            )
            f.write(json.dumps(rec) + "\n")
    print(
        "LCB_SCORE_DONE", json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list | dict)}), flush=True
    )


if __name__ == "__main__":
    main()
