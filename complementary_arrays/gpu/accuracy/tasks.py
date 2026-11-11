"""Accuracy tasks scored from model likelihoods and greedy generations, with the lm-eval-harness prompt formats.

The data directory holds wikitext2/test.arrow, lambada/test.arrow, arc_e/test.arrow, arc_c/test.arrow,
hellaswag/validation.arrow, piqa/validation.arrow, winogrande/validation.arrow, mmlu/test.parquet, mmlu/dev.parquet
and gsm8k/test.arrow. Every task returns its per-item correctness for paired comparisons between weight formats.
"""

import math
import re

import numpy as np

TASKS = ("wikitext2", "lambada", "arc_e", "arc_c", "hellaswag", "piqa", "winogrande", "mmlu", "gsm8k")


class Scorer:
    """Batched log-likelihoods and next-token logits from a decoder (`base` returns hidden states, `head` logits)."""

    def __init__(self, model, tok, base, head, max_len=4096, tokens_per_batch=24576):
        self.m, self.tok, self.base, self.head = model, tok, base, head
        self.max_len, self.tpb = max_len, tokens_per_batch
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def hidden(self, seqs):
        import torch

        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            m[i, : len(s)] = 1
        return self.base(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False).last_hidden_state

    def loglik(self, reqs):
        """reqs: (context ids, continuation ids) pairs -> (sum log-probability, greedy match, tokens) per pair."""
        import torch

        out = [None] * len(reqs)
        order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][0]) + len(reqs[i][1])))
        i = 0
        with torch.inference_mode():
            while i < len(order):
                T = min(self.max_len, len(reqs[order[i]][0]) + len(reqs[order[i]][1]))
                n = max(1, min(128, self.tpb // max(T, 1)))
                batch = order[i : i + n]
                i += n
                seqs, spans = [], []
                for j in batch:
                    c, t = reqs[j]
                    s = (c + t)[-self.max_len :]
                    seqs.append(s)
                    spans.append((len(s) - len(t), len(t)))
                h = self.hidden(seqs)
                rows, tgt, owner = [], [], []
                for b, (st, nt) in enumerate(spans):
                    rows.append(h[b, st - 1 : st - 1 + nt])
                    tgt += seqs[b][st : st + nt]
                    owner += [b] * nt
                lp = torch.log_softmax(self.head(torch.cat(rows)).float(), -1)
                tg = torch.tensor(tgt, device=lp.device)
                tok_lp = lp.gather(1, tg[:, None])[:, 0].cpu().numpy()
                greedy = (lp.argmax(-1) == tg).cpu().numpy()
                owner = np.array(owner)
                for b, j in enumerate(batch):
                    k = owner == b
                    out[j] = (float(tok_lp[k].sum()), bool(greedy[k].all()), int(k.sum()))
                del h, lp
        return out

    def next_token_logits(self, ctxs, token_ids):
        """Logits of the candidate tokens after each context."""
        import torch

        out = [None] * len(ctxs)
        order = sorted(range(len(ctxs)), key=lambda i: -len(ctxs[i]))
        i = 0
        with torch.inference_mode():
            while i < len(order):
                T = min(self.max_len, len(ctxs[order[i]]))
                n = max(1, min(128, self.tpb // max(T, 1)))
                batch = order[i : i + n]
                i += n
                seqs = [ctxs[j][-self.max_len :] for j in batch]
                h = self.hidden(seqs)
                last = torch.stack([h[b, len(s) - 1] for b, s in enumerate(seqs)])
                lg = self.head(last).float()[:, token_ids].cpu().numpy()
                for b, j in enumerate(batch):
                    out[j] = lg[b]
                del h
        return out


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False)


def split_continuation(tok, context, continuation):
    """Context and continuation ids, tokenized jointly when the joint tokens start with the context's own tokens."""
    c = enc(tok, context)
    whole = enc(tok, context + continuation)
    return (whole[: len(c)], whole[len(c) :]) if whole[: len(c)] == c else (c, enc(tok, continuation))


def multiple_choice(sc, tok, items):
    """items: (context, [continuations], gold index) -> acc, length-normalized acc_norm, per-item correctness."""
    reqs, idx = [], []
    for qi, (ctx, conts, _) in enumerate(items):
        for k, t in enumerate(conts):
            reqs.append(split_continuation(tok, ctx, t))
            idx.append((qi, k))
    ll = {key: r[0] for key, r in zip(idx, sc.loglik(reqs), strict=True)}
    acc, accn = [], []
    for qi, (_, conts, gold) in enumerate(items):
        v = np.array([ll[(qi, k)] for k in range(len(conts))])
        ln = np.array([max(1, len(t)) for t in conts], float)
        acc.append(int(np.argmax(v) == gold))
        accn.append(int(np.argmax(v / ln) == gold))
    return dict(acc=float(np.mean(acc)), acc_norm=float(np.mean(accn)), n=len(items), correct=acc, correct_norm=accn)


def arrow(data, name, split):
    from datasets import Dataset

    return Dataset.from_file(str(data / name / f"{split}.arrow"))


def limited(items, limit):
    return items[:limit] if limit else items


def wikitext2(sc, tok, a):
    """Perplexity over disjoint 1024-token windows of the test split."""
    import torch

    ids = enc(tok, "\n\n".join(arrow(a.data, "wikitext2", "test")["text"]))
    W = 1024
    wins = [w for w in (ids[o : o + W + 1] for o in range(0, len(ids) - 1, W)) if len(w) >= 2]
    wins = limited(wins, a.limit)
    nll, nt = 0.0, 0
    with torch.inference_mode():
        for s in range(0, len(wins), 8):
            chunk = wins[s : s + 8]
            h = sc.hidden([w[:-1] for w in chunk])
            for b, w in enumerate(chunk):
                n = len(w) - 1
                tg = torch.tensor(w[1:], device=h.device)
                for p0 in range(0, n, 512):
                    lp = torch.log_softmax(sc.head(h[b, p0 : min(n, p0 + 512)]).float(), -1)
                    nll -= float(lp.gather(1, tg[p0 : min(n, p0 + 512), None]).sum())
                nt += n
            del h
    return dict(perplexity=math.exp(nll / nt), tokens=nt, windows=len(wins), nll=nll)


def lambada(sc, tok, a):
    """Greedy accuracy and perplexity of each passage's last word."""
    reqs = []
    for t in limited(arrow(a.data, "lambada", "test")["text"], a.limit):
        ctx, last = t.rsplit(" ", 1)
        reqs.append(split_continuation(tok, ctx, " " + last))
    r = sc.loglik(reqs)
    acc = [int(x[1]) for x in r]
    return dict(acc=float(np.mean(acc)), perplexity=math.exp(-np.mean([x[0] for x in r])), n=len(r), correct=acc)


def arc(sc, tok, a, name):
    items = []
    for x in arrow(a.data, name, "test"):
        labels = x["choices"]["label"]
        if x["answerKey"] not in labels:
            continue
        conts = [" " + t for t in x["choices"]["text"]]
        items.append((f"Question: {x['question']}\nAnswer:", conts, labels.index(x["answerKey"])))
    return multiple_choice(sc, tok, limited(items, a.limit))


def _hellaswag_text(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def hellaswag(sc, tok, a):
    items = []
    for x in arrow(a.data, "hellaswag", "validation"):
        ctx = x["ctx_a"] + " " + x["ctx_b"].capitalize()
        conts = [" " + _hellaswag_text(e) for e in x["endings"]]
        items.append((_hellaswag_text(x["activity_label"] + ": " + ctx), conts, int(x["label"])))
    return multiple_choice(sc, tok, limited(items, a.limit))


def piqa(sc, tok, a):
    items = [
        (f"Question: {x['goal']}\nAnswer:", [" " + x["sol1"], " " + x["sol2"]], int(x["label"]))
        for x in arrow(a.data, "piqa", "validation")
    ]
    return multiple_choice(sc, tok, limited(items, a.limit))


def winogrande(sc, tok, a):
    """Partial scoring: likelihood of the text after the blank given each filled-in option."""
    reqs, golds = [], []
    for x in limited(list(arrow(a.data, "winogrande", "validation")), a.limit):
        i = x["sentence"].index("_")
        cont = " " + x["sentence"][i + 1 :].strip()
        for opt in (x["option1"], x["option2"]):
            reqs.append((enc(tok, x["sentence"][:i] + opt), enc(tok, cont)))
        golds.append(int(x["answer"]) - 1)
    r = sc.loglik(reqs)
    acc = [int(int(np.argmax([r[2 * q][0], r[2 * q + 1][0]])) == g) for q, g in enumerate(golds)]
    return dict(acc=float(np.mean(acc)), n=len(golds), correct=acc)


def mmlu(sc, tok, a):
    """5-shot; the answer is the most likely of the tokens ' A' to ' D'."""
    import pandas as pd

    test = pd.read_parquet(a.data / "mmlu" / "test.parquet")
    dev = pd.read_parquet(a.data / "mmlu" / "dev.parquet")

    def fmt(r):
        options = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"]))
        return r["question"].strip() + "\n" + options + "\nAnswer:"

    shots = {
        s: "".join(fmt(r) + " " + "ABCD"[int(r["answer"])] + "\n\n" for _, r in g.head(5).iterrows())
        for s, g in dev.groupby("subject")
    }
    rows = limited(list(test.iterrows()), a.limit)
    header = "The following are multiple choice questions (with answers) about {}.\n\n"
    ctxs = [enc(tok, header.format(r["subject"].replace("_", " ")) + shots[r["subject"]] + fmt(r)) for _, r in rows]
    cand = [enc(tok, " " + letter) for letter in "ABCD"]
    assert all(len(c) == 1 for c in cand)
    lg = sc.next_token_logits(ctxs, [c[0] for c in cand])
    correct = [int(int(np.argmax(logits)) == int(r["answer"])) for logits, (_, r) in zip(lg, rows, strict=True)]
    subjects = {}
    for c, (_, r) in zip(correct, rows, strict=True):
        subjects.setdefault(r["subject"], []).append(c)
    return dict(
        acc=float(np.mean(correct)),
        acc_macro=float(np.mean([np.mean(v) for v in subjects.values()])),
        n=len(rows),
        correct=correct,
        per_subject={k: float(np.mean(v)) for k, v in subjects.items()},
    )


def chat_ids(tok, text):
    x = tok.apply_chat_template(
        [dict(role="user", content=text)], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if hasattr(x, "keys"):
        x = x["input_ids"]
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def _last_number(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", "").strip())
    return float(m[-1]) if m else None


def gsm8k(sc, tok, a, model):
    """0-shot chat prompt asking for step-by-step reasoning and a boxed answer; greedy generation; exact match of the
    last boxed number (or the last number)."""
    import torch

    rows = limited(list(arrow(a.data, "gsm8k", "test")), a.limit)
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [chat_ids(tok, r["question"] + suffix) for r in rows]
    golds = [_last_number(r["answer"].split("####")[-1]) for r in rows]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    outs = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))
    with torch.inference_mode():
        for s in range(0, len(order), a.gen_batch):
            b = order[s : s + a.gen_batch]
            T = max(len(prompts[i]) for i in b)
            x = torch.full((len(b), T), sc.pad, dtype=torch.long)
            m = torch.zeros_like(x)
            for k, i in enumerate(b):
                x[k, T - len(prompts[i]) :] = torch.tensor(prompts[i])
                m[k, T - len(prompts[i]) :] = 1
            g = model.generate(
                input_ids=x.cuda(),
                attention_mask=m.cuda(),
                max_new_tokens=a.max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=eos,
                pad_token_id=sc.pad,
            )
            for k, i in enumerate(b):
                outs[i] = tok.decode(g[k, T:], skip_special_tokens=True)
    correct, lens = [], []
    for o, gold in zip(outs, golds, strict=True):
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", o)
        pred = _last_number(boxed[-1]) if boxed else _last_number(o)
        correct.append(int(pred is not None and gold is not None and abs(pred - gold) < 1e-6))
        lens.append(len(enc(tok, o)))
    return dict(
        acc=float(np.mean(correct)),
        n=len(rows),
        correct=correct,
        mean_output_tokens=float(np.mean(lens)),
        truncated=int(sum(n >= a.max_new for n in lens)),
        max_new_tokens=a.max_new,
    )


def run_task(name, sc, tok, model, a):
    """`a` carries data (directory), limit (0 = all items), gen_batch and max_new (GSM8K)."""
    if name in ("arc_e", "arc_c"):
        return arc(sc, tok, a, name)
    if name == "gsm8k":
        return gsm8k(sc, tok, a, model)
    return {
        "wikitext2": wikitext2,
        "lambada": lambada,
        "hellaswag": hellaswag,
        "piqa": piqa,
        "winogrande": winogrande,
        "mmlu": mmlu,
    }[name](sc, tok, a)
