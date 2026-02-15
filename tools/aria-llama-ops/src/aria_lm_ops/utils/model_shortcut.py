from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig


def load_tinyllama_cfg() -> LlamaConfig:
    _model_name = "Cheng98/TinyLlama_v1.1"
    config = AutoConfig.from_pretrained(_model_name)
    return config


def load_tinyllama_ckpt() -> dict[str, torch.Tensor]:
    _model_name = "Cheng98/TinyLlama_v1.1"
    model_path = Path(snapshot_download(_model_name, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict


def load_llama2_7b_cfg() -> LlamaConfig:
    _model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(_model_name)
    return config


def load_llama2_7b_ckpt() -> dict[str, torch.Tensor]:
    _model_name = "meta-llama/Llama-2-7b-hf"
    model_path = Path(snapshot_download(_model_name, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict


def load_llama3_8b_cfg() -> LlamaConfig:
    _model_name = "meta-llama/Llama-3.1-8B"
    config = AutoConfig.from_pretrained(_model_name)
    return config


def load_llama3_8b_ckpt() -> dict[str, torch.Tensor]:
    _model_name = "meta-llama/Llama-3.1-8B"
    model_path = Path(snapshot_download(_model_name, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict
