"""Shared SigLIP tensor preparation helpers for encoder-style harnesses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.utils.core import resolve_vision_encoder_layer


def prepare_full_siglip_tensors(*, mlen: int = 128) -> dict:
	"""Build full-width SigLIP encoder tensors used by multiple harnesses."""
	from transformers import AutoModel

	repo_root = Path(__file__).parents[4]
	config_path = repo_root / "compiler" / "doc" / "Model_Lib" / "siglip-so400m-patch14-384.json"
	model_id = "google/siglip-so400m-patch14-384"

	with open(config_path) as f:
		siglip_config = json.load(f)
	vision_cfg = siglip_config["vision_config"]

	image_size = int(vision_cfg["image_size"])
	patch_size = int(vision_cfg["patch_size"])
	s_full = (image_size // patch_size) ** 2
	hq = int(vision_cfg["num_attention_heads"])
	hidden_size = int(vision_cfg["hidden_size"])
	inter_dim = int(os.environ.get("SIGLIP_INTER_DIM", "1024"))
	if inter_dim <= 0:
		raise ValueError("SIGLIP_INTER_DIM must be > 0")
	h_qkv = hidden_size // hq

	print(f"Loading SigLIP weights from {model_id} ...")
	model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
	vision_root = getattr(model, "vision_model", model)
	layer0 = resolve_vision_encoder_layer(model, layer_idx=0)

	torch.manual_seed(0)
	pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
	embed_out = vision_root.embeddings(pixel_values).detach().contiguous()
	if embed_out.shape[1] != s_full or embed_out.shape[2] != int(vision_cfg["hidden_size"]):
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
	k_all = F.linear(
		x_ln1_full,
		attn_mod.k_proj.weight[:hidden_size, :hidden_size].detach().float(),
		attn_mod.k_proj.bias[:hidden_size].detach().float(),
	)
	v_all = F.linear(
		x_ln1_full,
		attn_mod.v_proj.weight[:hidden_size, :hidden_size].detach().float(),
		attn_mod.v_proj.bias[:hidden_size].detach().float(),
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

	d_padded = mlen
	hidden_size_padded = hq * d_padded
	wq_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wq_padded[:hidden_size, :hidden_size] = wq
	wo_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wo_padded[:hidden_size, :hidden_size] = wo

	q_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	q_bias_padded[:hidden_size] = q_bias

	return {
		"s_full": s_full,
		"hidden_size": hidden_size,
		"hq": hq,
		"hkv": hkv,
		"h_qkv": h_qkv,
		"inter_dim": inter_dim,
		"aligned_inter_dim": aligned_inter_dim,
		"x_in_full": x_in_full,
		"k_full": k_full,
		"v_full": v_full,
		"w1_raw": w1,
		"w2_raw": w2,
		"wq_padded": wq_padded,
		"out_proj_weight": wo_padded,
		"q_bias_padded": q_bias_padded,
	}


def prepare_reduced_siglip_tensors(*, mlen: int = 64, hidden_per_head: int = 64) -> dict:
	"""Build reduced-width SigLIP tensors while keeping full sequence length."""
	from transformers import AutoModel

	repo_root = Path(__file__).parents[4]
	config_path = repo_root / "compiler" / "doc" / "Model_Lib" / "siglip-so400m-patch14-384.json"
	model_id = "google/siglip-so400m-patch14-384"

	with open(config_path) as f:
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

	print(f"Loading SigLIP weights from {model_id} ...")
	model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
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
	k_all = F.linear(
		x_ln1_full,
		attn_mod.k_proj.weight[:hidden_size, :hidden_size].detach().float(),
		attn_mod.k_proj.bias[:hidden_size].detach().float(),
	)
	v_all = F.linear(
		x_ln1_full,
		attn_mod.v_proj.weight[:hidden_size, :hidden_size].detach().float(),
		attn_mod.v_proj.bias[:hidden_size].detach().float(),
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

	d_padded = mlen
	hidden_size_padded = hq * d_padded
	wq_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wq_padded[:hidden_size, :hidden_size] = wq
	wo_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
	wo_padded[:hidden_size, :hidden_size] = wo

	q_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
	q_bias_padded[:hidden_size] = q_bias

	return {
		"s_full": s_full,
		"hidden_size": hidden_size,
		"hq": hq,
		"hkv": hkv,
		"h_qkv": h_qkv,
		"inter_dim": inter_dim,
		"aligned_inter_dim": aligned_inter_dim,
		"x_in_full": x_in_full,
		"k_full": k_full,
		"v_full": v_full,
		"w1_raw": w1,
		"w2_raw": w2,
		"wq_padded": wq_padded,
		"out_proj_weight": wo_padded,
		"q_bias_padded": q_bias_padded,
	}
