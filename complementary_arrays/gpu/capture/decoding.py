"""Pieces shared by the routing captures: router hooks, left-padded prompt batches, manifests and shard files."""

import contextlib
import functools
import json

import numpy as np

from .requests import file_sha256, write_json


class RouterRecorder:
    """Forward hooks on every MoE router that keep the top-k expert ids and scores of one decode step."""

    def __init__(self, layers, topk):
        self.layers, self.topk = list(layers), topk
        self.batch, self.ids, self.scores = None, {}, {}
        for li, moe in layers.items():
            moe.gate.register_forward_hook(functools.partial(self._hook, li))

    def _hook(self, li, module, args, result):
        if self.batch is None:
            return
        _, scores, ids = result
        assert ids.shape == (self.batch, self.topk), (li, tuple(ids.shape))
        self.ids[li] = ids.detach().cpu().numpy().astype(np.int16)
        self.scores[li] = scores.detach().float().cpu().numpy()

    @contextlib.contextmanager
    def step(self, batch):
        """Record the forward passes inside the block; each must route `batch` tokens."""
        self.batch, self.ids, self.scores = batch, {}, {}
        try:
            yield self
        finally:
            self.batch = None


def eos_and_pad(tokenizer):
    """EOS token ids and the padding id (the smallest EOS id when the tokenizer has no pad token)."""
    eos = tokenizer.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos)
    return eos, pad


def left_padded(requests, pad):
    """Token ids and attention mask of a request group on the GPU, left-padded to the longest prompt."""
    import torch

    width = max(r["prompt_tokens"] for r in requests)
    ids = torch.full((len(requests), width), pad, device="cuda", dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, r in enumerate(requests):
        ids[j, -r["prompt_tokens"] :] = torch.tensor(r["input_ids"], device="cuda")
        mask[j, -r["prompt_tokens"] :] = 1
    return ids, mask


def record_live_step(arrays, step, active, recorder, rows=None):
    """Add one decode step of live requests to a shard: their batch indices, expert ids and scores (`rows` selects
    them when the batch still holds finished requests)."""
    arrays[f"request_indices_s{step}"] = active.copy()
    for li in recorder.layers:
        ids, scores = recorder.ids[li], recorder.scores[li]
        if rows is not None:
            ids, scores = ids[rows], scores[rows]
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        arrays[f"ids_l{li}_s{step}"] = ids
        arrays[f"scores_l{li}_s{step}"] = scores


def runtime():
    import torch
    import transformers

    return dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name())


def start_manifest(out, manifest, resume, same):
    """Write a new manifest, or continue an interrupted capture whose manifest agrees on the keys in `same`."""
    path = out / "manifest.json"
    if path.exists():
        if not resume:
            raise SystemExit(f"{path} exists; pass --resume to continue that capture")
        old = json.loads(path.read_text())
        for key in same:
            if old[key] != manifest[key]:
                raise SystemExit(f"cannot resume {path}: {key} differs")
        for shard in old["shards"]:
            if file_sha256(out / shard["file"]) != shard["sha256"]:
                raise SystemExit(f"cannot resume {path}: {shard['file']} changed")
        manifest["shards"] = old["shards"]
    write_json(path, manifest)
    return path


def save_shard(path, manifest, group, arrays, **record):
    """Write one request group's arrays next to the manifest and register the file in it."""
    file = path.parent / f"{group['id']}.npz"
    np.savez_compressed(file, **arrays)
    shard = {k: group[k] for k in ("id", "source", "split", "batch", "cohort")}
    shard.update(file=file.name, sha256=file_sha256(file), **record)
    manifest["shards"].append(shard)
    write_json(path, manifest)
    print("CAPTURE", json.dumps({k: v for k, v in shard.items() if k != "live_counts"}), flush=True)


def finish_manifest(path, manifest, groups):
    manifest["complete"] = len(manifest["shards"]) == len(groups)
    write_json(path, manifest)
    print("COMPLETE", manifest["kind"], len(manifest["shards"]), "of", len(groups), "groups", flush=True)
