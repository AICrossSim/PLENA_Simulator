from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_tinyllama_cfg() -> LlamaConfig:
    _MODEL_NAME = "Cheng98/TinyLlama_v1.1"
    config = AutoConfig.from_pretrained(_MODEL_NAME)
    return config


def load_tinyllama_ckpt() -> dict[str, torch.Tensor]:
    _MODEL_NAME = "Cheng98/TinyLlama_v1.1"
    model_path = Path(snapshot_download(_MODEL_NAME, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict


def load_llama2_7b_cfg() -> LlamaConfig:
    _MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(_MODEL_NAME)
    return config


def load_llama2_7b_ckpt() -> dict[str, torch.Tensor]:
    _MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    model_path = Path(snapshot_download(_MODEL_NAME, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict


def load_llama3_8b_cfg() -> LlamaConfig:
    _MODEL_NAME = "meta-llama/Llama-3.1-8B"
    config = AutoConfig.from_pretrained(_MODEL_NAME)
    return config


def load_llama3_8b_ckpt() -> dict[str, torch.Tensor]:
    _MODEL_NAME = "meta-llama/Llama-3.1-8B"
    model_path = Path(snapshot_download(_MODEL_NAME, local_files_only=True))
    state_dict = load_file(model_path.joinpath("model.safetensors"))
    return state_dict
