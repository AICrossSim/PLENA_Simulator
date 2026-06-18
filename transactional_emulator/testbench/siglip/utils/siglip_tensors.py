"""Shared SigLIP tensor preparation helpers for encoder-style harnesses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.model_loader import load_siglip_config, resolve_siglip_model_spec
from transactional_emulator.testbench.siglip.utils.core import resolve_vision_encoder_layer


def build_runtime_hidden_positions(hidden_size: int, num_heads: int, padded_head_dim: int) -> torch.Tensor:
	"""Map visible hidden lanes into per-head padded runtime positions."""
	if hidden_size % num_heads != 0:
		raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

	visible_head_dim = hidden_size // num_heads
	positions: list[int] = []
	for head_idx in range(num_heads):
		base = head_idx * padded_head_dim
		for dim_idx in range(visible_head_dim):
			positions.append(base + dim_idx)

	return torch.tensor(positions, dtype=torch.long)


def _scatter_hidden_vector(vector: torch.Tensor, hidden_size_padded: int, runtime_hidden_positions: torch.Tensor) -> torch.Tensor:
	runtime = torch.zeros(hidden_size_padded, dtype=vector.dtype)
	runtime[runtime_hidden_positions[: vector.numel()]] = vector
	return runtime


def _scatter_hidden_square_matrix(
	matrix: torch.Tensor,
	hidden_size_padded: int,
	runtime_hidden_positions: torch.Tensor,
) -> torch.Tensor:
	runtime = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=matrix.dtype)
	idx = runtime_hidden_positions[: matrix.shape[0]]
	runtime[idx.unsqueeze(1), idx.unsqueeze(0)] = matrix
	return runtime


def _scatter_seq_hidden_tensor(
	tensor: torch.Tensor,
	hidden_size_padded: int,
	runtime_hidden_positions: torch.Tensor,
) -> torch.Tensor:
	runtime = torch.zeros(tensor.shape[0], hidden_size_padded, dtype=tensor.dtype)
	runtime[:, runtime_hidden_positions[: tensor.shape[1]]] = tensor
	return runtime


def _scatter_hidden_to_inter_matrix(
	matrix: torch.Tensor,
	hidden_size_padded: int,
	runtime_hidden_positions: torch.Tensor,
) -> torch.Tensor:
	runtime = torch.zeros(hidden_size_padded, matrix.shape[1], dtype=matrix.dtype)
	runtime[runtime_hidden_positions[: matrix.shape[0]], :] = matrix
	return runtime


def _scatter_inter_to_hidden_matrix(
	matrix: torch.Tensor,
	hidden_size_padded: int,
	runtime_hidden_positions: torch.Tensor,
) -> torch.Tensor:
	runtime = torch.zeros(matrix.shape[0], hidden_size_padded, dtype=matrix.dtype)
	runtime[:, runtime_hidden_positions[: matrix.shape[1]]] = matrix
	return runtime


def prepare_full_siglip_tensors(
	*,
	mlen: int = 128,
	inter_dim: int | None = None,
	variant: str | None = None,
	config_path: str | Path | None = None,
	model_id: str | None = None,
) -> dict:
	"""Build full-width SigLIP encoder tensors used by multiple harnesses."""
	from transformers import AutoModel

	repo_root = Path(__file__).parents[4]
	resolved_config_path, resolved_model_id, resolved_variant = resolve_siglip_model_spec(
		variant=variant,
		config_path=config_path,
		model_id=model_id,
	)
	config_file = Path(resolved_config_path)
	if not config_file.is_absolute():
		config_file = repo_root / config_file

	siglip_config = load_siglip_config(str(config_file))
	image_size = int(siglip_config["image_size"])
	patch_size = int(siglip_config["patch_size"])
	s_full = (image_size // patch_size) ** 2
	hq = int(siglip_config["num_attention_heads"])
	hidden_size = int(siglip_config["hidden_size"])
	if inter_dim is None:
		inter_dim = int(os.environ.get("SIGLIP_INTER_DIM", str(siglip_config.get("intermediate_size", 1024))))
	if inter_dim <= 0:
		raise ValueError("SIGLIP_INTER_DIM must be > 0")
	h_qkv = hidden_size // hq
	d_padded = mlen
	hidden_size_padded = hq * d_padded
	runtime_hidden_positions = build_runtime_hidden_positions(hidden_size, hq, d_padded)

	print(f"Loading SigLIP weights from {resolved_model_id} (variant={resolved_variant}) ...")
	model = AutoModel.from_pretrained(resolved_model_id, torch_dtype=torch.float32)
	vision_root = getattr(model, "vision_model", model)
	layer0 = resolve_vision_encoder_layer(model, layer_idx=0)

	torch.manual_seed(0)
	pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
	embed_out = vision_root.embeddings(pixel_values).detach().contiguous()
	if embed_out.shape[1] != s_full or embed_out.shape[2] != int(siglip_config["hidden_size"]):
		raise ValueError(f"Unexpected embedding output shape {tuple(embed_out.shape)}")

	x_in_full = embed_out[0, :, :hidden_size].to(torch.bfloat16).float().contiguous()

	ln1_weight = layer0.layer_norm1.weight[:hidden_size].detach().float()
	ln1_bias = layer0.layer_norm1.bias[:hidden_size].detach().float()
	x_ln1_full = F.layer_norm(
		x_in_full,
		(hidden_size,),
		weight=ln1_weight,
		bias=ln1_bias,
		eps=layer0.layer_norm1.eps,
	)

	attn_mod = layer0.self_attn
	wq = attn_mod.q_proj.weight[:hidden_size, :hidden_size].detach().float().t().contiguous()
	q_bias = attn_mod.q_proj.bias[:hidden_size].detach().float().contiguous()
	out_bias = attn_mod.out_proj.bias[:hidden_size].detach().float().contiguous()
	wo = attn_mod.out_proj.weight[:hidden_size, :hidden_size].detach().float().t().contiguous()
	k_weight = attn_mod.k_proj.weight[:hidden_size, :hidden_size].detach().float()
	k_bias = attn_mod.k_proj.bias[:hidden_size].detach().float()
	v_weight = attn_mod.v_proj.weight[:hidden_size, :hidden_size].detach().float()
	v_bias = attn_mod.v_proj.bias[:hidden_size].detach().float()
	k_all = F.linear(
		x_ln1_full,
		k_weight,
		k_bias,
	)
	v_all = F.linear(
		x_ln1_full,
		v_weight,
		v_bias,
	)

	hkv = int(getattr(attn_mod, "num_key_value_heads", hq))
	k_full = k_all.reshape(1, s_full, hkv, h_qkv).contiguous()
	v_full = v_all.reshape(1, s_full, hkv, h_qkv).contiguous()

	aligned_inter_dim = ((inter_dim + mlen - 1) // mlen) * mlen

	mlp_mod = layer0.mlp
	w1 = mlp_mod.fc1.weight[:inter_dim, :hidden_size].detach().float().t().contiguous()
	w2 = mlp_mod.fc2.weight[:hidden_size, :inter_dim].detach().float().t().contiguous()
	fc1_bias = mlp_mod.fc1.bias[:inter_dim].detach().float().contiguous()
	fc2_bias = mlp_mod.fc2.bias[:hidden_size].detach().float().contiguous()
	if aligned_inter_dim != inter_dim:
		w1 = F.pad(w1, (0, aligned_inter_dim - inter_dim)).contiguous()
		w2 = F.pad(w2, (0, 0, 0, aligned_inter_dim - inter_dim)).contiguous()
		fc1_bias = F.pad(fc1_bias, (0, aligned_inter_dim - inter_dim)).contiguous()

	ln1_weight_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	ln1_weight_padded[:hidden_size] = ln1_weight
	ln1_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	ln1_bias_padded[:hidden_size] = ln1_bias
	ln2_weight = layer0.layer_norm2.weight[:hidden_size].detach().float()
	ln2_bias = layer0.layer_norm2.bias[:hidden_size].detach().float()
	ln2_weight_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	ln2_weight_padded[:hidden_size] = ln2_weight
	ln2_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	ln2_bias_padded[:hidden_size] = ln2_bias
	wq_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wq_padded[:hidden_size, :hidden_size] = wq
	wo_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wo_padded[:hidden_size, :hidden_size] = wo

	q_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	q_bias_padded[:hidden_size] = q_bias
	out_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	out_bias_padded[:hidden_size] = out_bias
	fc2_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	fc2_bias_padded[:hidden_size] = fc2_bias

	x_in_runtime_full = _scatter_seq_hidden_tensor(x_in_full, hidden_size_padded, runtime_hidden_positions)
	ln1_weight_runtime = _scatter_hidden_vector(ln1_weight, hidden_size_padded, runtime_hidden_positions)
	ln1_bias_runtime = _scatter_hidden_vector(ln1_bias, hidden_size_padded, runtime_hidden_positions)
	ln2_weight_runtime = _scatter_hidden_vector(ln2_weight, hidden_size_padded, runtime_hidden_positions)
	ln2_bias_runtime = _scatter_hidden_vector(ln2_bias, hidden_size_padded, runtime_hidden_positions)
	wq_runtime = _scatter_hidden_square_matrix(wq, hidden_size_padded, runtime_hidden_positions)
	wo_runtime = _scatter_hidden_square_matrix(wo, hidden_size_padded, runtime_hidden_positions)
	q_bias_runtime = _scatter_hidden_vector(q_bias, hidden_size_padded, runtime_hidden_positions)
	out_bias_runtime = _scatter_hidden_vector(out_bias, hidden_size_padded, runtime_hidden_positions)
	fc2_bias_runtime = _scatter_hidden_vector(fc2_bias, hidden_size_padded, runtime_hidden_positions)
	w1_runtime = _scatter_hidden_to_inter_matrix(w1, hidden_size_padded, runtime_hidden_positions)
	w2_runtime = _scatter_inter_to_hidden_matrix(w2, hidden_size_padded, runtime_hidden_positions)

	x_ln1_runtime = F.layer_norm(
		x_in_runtime_full.to(torch.bfloat16).float(),
		(hidden_size_padded,),
		weight=ln1_weight_runtime,
		bias=ln1_bias_runtime,
		eps=layer0.layer_norm1.eps,
	)
	x_ln1_runtime_visible = x_ln1_runtime[:, runtime_hidden_positions]
	k_all_runtime = F.linear(x_ln1_runtime_visible, k_weight, k_bias)
	v_all_runtime = F.linear(x_ln1_runtime_visible, v_weight, v_bias)
	k_full_runtime = k_all_runtime.reshape(1, s_full, hkv, h_qkv).contiguous()
	v_full_runtime = v_all_runtime.reshape(1, s_full, hkv, h_qkv).contiguous()

	return {
		"s_full": s_full,
		"hidden_size": hidden_size,
		"hq": hq,
		"hkv": hkv,
		"h_qkv": h_qkv,
		"d_padded": d_padded,
		"hidden_size_padded": hidden_size_padded,
		"runtime_hidden_positions": runtime_hidden_positions,
		"inter_dim": inter_dim,
		"aligned_inter_dim": aligned_inter_dim,
		"x_in_full": x_in_full,
		"x_in_runtime_full": x_in_runtime_full,
		"k_full": k_full,
		"v_full": v_full,
		"k_full_runtime": k_full_runtime,
		"v_full_runtime": v_full_runtime,
		"w1_raw": w1,
		"w2_raw": w2,
		"w1_runtime": w1_runtime,
		"w2_runtime": w2_runtime,
		"wq_padded": wq_padded,
		"wq_runtime": wq_runtime,
		"out_proj_weight": wo_padded,
		"out_proj_weight_runtime": wo_runtime,
		"q_bias_padded": q_bias_padded,
		"q_bias_runtime": q_bias_runtime,
		"ln1_weight_padded": ln1_weight_padded,
		"ln1_weight_runtime": ln1_weight_runtime,
		"ln1_bias_padded": ln1_bias_padded,
		"ln1_bias_runtime": ln1_bias_runtime,
		"ln2_weight_padded": ln2_weight_padded,
		"ln2_weight_runtime": ln2_weight_runtime,
		"ln2_bias_padded": ln2_bias_padded,
		"ln2_bias_runtime": ln2_bias_runtime,
		"out_proj_bias_padded": out_bias_padded,
		"out_proj_bias_runtime": out_bias_runtime,
		"fc1_bias_padded": fc1_bias,
		"fc2_bias_padded": fc2_bias_padded,
		"fc2_bias_runtime": fc2_bias_runtime,
	}


def prepare_reduced_siglip_tensors(
	*,
	mlen: int = 64,
	hidden_per_head: int = 64,
	variant: str | None = None,
	config_path: str | Path | None = None,
	model_id: str | None = None,
) -> dict:
	"""Build reduced-width SigLIP tensors while keeping full sequence length."""
	from transformers import AutoModel

	repo_root = Path(__file__).parents[4]
	resolved_config_path, resolved_model_id, resolved_variant = resolve_siglip_model_spec(
		variant=variant,
		config_path=config_path,
		model_id=model_id,
	)
	config_file = Path(resolved_config_path)
	if not config_file.is_absolute():
		config_file = repo_root / config_file

	with open(config_file) as f:
		siglip_config = json.load(f)
	vision_cfg = siglip_config["vision_config"]

	image_size = int(vision_cfg["image_size"])
	patch_size = int(vision_cfg["patch_size"])
	s_full = (image_size // patch_size) ** 2
	hq = int(vision_cfg["num_attention_heads"])
	full_hidden_size = int(vision_cfg["hidden_size"])
	hidden_size = hq * hidden_per_head
	if hidden_size >= full_hidden_size:
		raise ValueError(f"Reduced hidden_size={hidden_size} must be smaller than full hidden_size={full_hidden_size}")

	inter_dim = int(os.environ.get("SIGLIP_INTER_DIM", "256"))
	if inter_dim <= 0:
		raise ValueError("SIGLIP_INTER_DIM must be > 0")
	h_qkv = hidden_size // hq
	d_padded = mlen
	hidden_size_padded = hq * d_padded
	runtime_hidden_positions = build_runtime_hidden_positions(hidden_size, hq, d_padded)

	print(f"Loading SigLIP weights from {resolved_model_id} (variant={resolved_variant}) ...")
	model = AutoModel.from_pretrained(resolved_model_id, torch_dtype=torch.float32)
	vision_root = getattr(model, "vision_model", model)
	layer0 = resolve_vision_encoder_layer(model, layer_idx=0)

	torch.manual_seed(0)
	pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
	embed_out = vision_root.embeddings(pixel_values).detach().contiguous()
	if embed_out.shape[1] != s_full or embed_out.shape[2] != full_hidden_size:
		raise ValueError(f"Unexpected embedding output shape {tuple(embed_out.shape)}")

	x_in_full = embed_out[0, :, :hidden_size].to(torch.bfloat16).float().contiguous()

	ln1_weight = layer0.layer_norm1.weight[:hidden_size].detach().float()
	ln1_bias = layer0.layer_norm1.bias[:hidden_size].detach().float()
	x_ln1_full = F.layer_norm(
		x_in_full,
		(hidden_size,),
		weight=ln1_weight,
		bias=ln1_bias,
		eps=layer0.layer_norm1.eps,
	)

	attn_mod = layer0.self_attn
	wq = attn_mod.q_proj.weight[:hidden_size, :hidden_size].detach().float().t().contiguous()
	q_bias = attn_mod.q_proj.bias[:hidden_size].detach().float().contiguous()
	wo = attn_mod.out_proj.weight[:hidden_size, :hidden_size].detach().float().t().contiguous()
	k_weight = attn_mod.k_proj.weight[:hidden_size, :hidden_size].detach().float()
	k_bias = attn_mod.k_proj.bias[:hidden_size].detach().float()
	v_weight = attn_mod.v_proj.weight[:hidden_size, :hidden_size].detach().float()
	v_bias = attn_mod.v_proj.bias[:hidden_size].detach().float()
	k_all = F.linear(
		x_ln1_full,
		k_weight,
		k_bias,
	)
	v_all = F.linear(
		x_ln1_full,
		v_weight,
		v_bias,
	)

	hkv = int(getattr(attn_mod, "num_key_value_heads", hq))
	k_full = k_all.reshape(1, s_full, hkv, h_qkv).contiguous()
	v_full = v_all.reshape(1, s_full, hkv, h_qkv).contiguous()

	aligned_inter_dim = ((inter_dim + mlen - 1) // mlen) * mlen

	mlp_mod = layer0.mlp
	w1 = mlp_mod.fc1.weight[:inter_dim, :hidden_size].detach().float().t().contiguous()
	w2 = mlp_mod.fc2.weight[:hidden_size, :inter_dim].detach().float().t().contiguous()
	if aligned_inter_dim != inter_dim:
		w1 = F.pad(w1, (0, aligned_inter_dim - inter_dim)).contiguous()
		w2 = F.pad(w2, (0, 0, 0, aligned_inter_dim - inter_dim)).contiguous()

	wq_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wq_padded[:hidden_size, :hidden_size] = wq
	wo_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wo_padded[:hidden_size, :hidden_size] = wo

	q_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	q_bias_padded[:hidden_size] = q_bias

	x_in_runtime_full = _scatter_seq_hidden_tensor(x_in_full, hidden_size_padded, runtime_hidden_positions)
	wq_runtime = _scatter_hidden_square_matrix(wq, hidden_size_padded, runtime_hidden_positions)
	wo_runtime = _scatter_hidden_square_matrix(wo, hidden_size_padded, runtime_hidden_positions)
	q_bias_runtime = _scatter_hidden_vector(q_bias, hidden_size_padded, runtime_hidden_positions)
	w1_runtime = _scatter_hidden_to_inter_matrix(w1, hidden_size_padded, runtime_hidden_positions)
	w2_runtime = _scatter_inter_to_hidden_matrix(w2, hidden_size_padded, runtime_hidden_positions)

	ln1_weight_runtime = _scatter_hidden_vector(ln1_weight, hidden_size_padded, runtime_hidden_positions)
	ln1_bias_runtime = _scatter_hidden_vector(ln1_bias, hidden_size_padded, runtime_hidden_positions)
	x_ln1_runtime = F.layer_norm(
		x_in_runtime_full.to(torch.bfloat16).float(),
		(hidden_size_padded,),
		weight=ln1_weight_runtime,
		bias=ln1_bias_runtime,
		eps=layer0.layer_norm1.eps,
	)
	x_ln1_runtime_visible = x_ln1_runtime[:, runtime_hidden_positions]
	k_all_runtime = F.linear(x_ln1_runtime_visible, k_weight, k_bias)
	v_all_runtime = F.linear(x_ln1_runtime_visible, v_weight, v_bias)
	k_full_runtime = k_all_runtime.reshape(1, s_full, hkv, h_qkv).contiguous()
	v_full_runtime = v_all_runtime.reshape(1, s_full, hkv, h_qkv).contiguous()

	return {
		"s_full": s_full,
		"hidden_size": hidden_size,
		"hq": hq,
		"hkv": hkv,
		"h_qkv": h_qkv,
		"d_padded": d_padded,
		"hidden_size_padded": hidden_size_padded,
		"runtime_hidden_positions": runtime_hidden_positions,
		"inter_dim": inter_dim,
		"aligned_inter_dim": aligned_inter_dim,
		"x_in_full": x_in_full,
		"x_in_runtime_full": x_in_runtime_full,
		"k_full": k_full,
		"v_full": v_full,
		"k_full_runtime": k_full_runtime,
		"v_full_runtime": v_full_runtime,
		"w1_raw": w1,
		"w2_raw": w2,
		"w1_runtime": w1_runtime,
		"w2_runtime": w2_runtime,
		"wq_padded": wq_padded,
		"wq_runtime": wq_runtime,
		"out_proj_weight": wo_padded,
		"out_proj_weight_runtime": wo_runtime,
		"q_bias_padded": q_bias_padded,
		"q_bias_runtime": q_bias_runtime,
	}
