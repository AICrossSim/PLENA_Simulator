"""BFCL (bfcl-eval 2026.3.23: official generation and AST / state-based checkers) against a running serve.py server.

Prompt mode with the official handlers: BFCL's Qwen3 prompt handler for Qwen3.5 (thinking removed before parsing) and its
generic chat-template handler for DeepSeek-V2-Lite-Chat. One change: the output cap per call is raised from 4,096 tokens
to 32,768 (Qwen3.5) or 8,192 (DeepSeek), so that thinking is not cut. Temperature is the model's recommended value
(Qwen3.5 1.0, DeepSeek 0.3); top-p and top-k come from the server's generation_config. Categories: non_live and live
(AST), multi_turn (state-based execution); web search and memory are not run.

usage: python -m complementary_arrays.gpu.accuracy.bfcl {qwen35,deepseek} MODE PORT OUT_DIR MODEL_DIR
       [--cats non_live,live,multi_turn] [--threads 64]
"""

import argparse
import os
import sys
import time

CAP = {"qwen35": 32768, "deepseek": 8192}
SERVER_MAX = {"qwen35": 65536, "deepseek": 32768}
TEMPERATURE = {"qwen35": "1.0", "deepseek": "0.3"}


def capped_query(kind):
    def query(self, inference_data):
        prompt = self._format_prompt(inference_data["message"], inference_data["function"])
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        left = max(16, min(CAP[kind], SERVER_MAX[kind] - len(self.tokenizer.tokenize(prompt)) - 2))
        extra = {k: getattr(self, k) for k in ("stop_token_ids", "skip_special_tokens") if hasattr(self, k)}
        t0 = time.time()
        r = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=prompt,
            max_tokens=left,
            timeout=72000,
            **({"extra_body": extra} if extra else {}),
        )
        return r, time.time() - t0

    return query


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("mode")
    p.add_argument("port")
    p.add_argument("out")
    p.add_argument("model_dir", help="model directory served (for the tokenizer)")
    p.add_argument("--cats", default="non_live,live,multi_turn")
    p.add_argument("--threads", default="64")
    a = p.parse_args()
    os.environ.update(
        LOCAL_SERVER_PORT=a.port,
        LOCAL_SERVER_ENDPOINT="127.0.0.1",
        BFCL_PROJECT_ROOT=os.path.join(os.path.abspath(a.out), "bfcl"),
    )
    os.makedirs(os.environ["BFCL_PROJECT_ROOT"], exist_ok=True)
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    from bfcl_eval.model_handler.local_inference.quick_testing_oss import QuickTestingOSSHandler

    base = QwenHandler if a.kind == "qwen35" else QuickTestingOSSHandler
    handler = type(f"Capped{base.__name__}", (base,), {"_query_prompting": capped_query(a.kind)})
    name = f"plena-{a.kind}"
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=name,
        display_name=f"{a.kind} {a.mode}",
        url="-",
        org="-",
        license="-",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )
    from bfcl_eval.__main__ import cli

    def run(argv):
        sys.argv = ["bfcl", *argv]
        try:
            cli()
        except SystemExit as e:
            if e.code not in (0, None):
                raise

    run(
        [
            "generate",
            "--model",
            name,
            "--test-category",
            a.cats,
            "--temperature",
            TEMPERATURE[a.kind],
            "--num-threads",
            a.threads,
            "--skip-server-setup",
            "--local-model-path",
            a.model_dir,
        ]
    )
    run(["evaluate", "--model", name, "--test-category", a.cats])
    print("BFCL_DONE", a.kind, a.mode, flush=True)


if __name__ == "__main__":
    main()
