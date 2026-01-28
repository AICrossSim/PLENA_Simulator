import math
import random

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward as _flash_attention_forward_ref

from aria_lm_ops.config import CheckConfig
from aria_lm_ops.utils.model_shortcut import load_tinyllama_cfg, load_llama2_7b_cfg, load_llama3_8b_cfg
from aria_lm_ops.models.llama import flash_attn2_gemv


@torch.no_grad()
def check_flash_attn2_gemv():
    for config in [load_tinyllama_cfg(), load_llama2_7b_cfg(), load_llama3_8b_cfg()]:
        print(f"{config._name_or_path}")
        # config = load_tinyllama_cfg()

        b = batch_size = 1
        h = hidden_size = config.hidden_size
        s_q = seq_len_q = 1
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        h_qkv = config.hidden_size // config.num_attention_heads
        Bc = 16
        qk_scale = 1.0 / math.sqrt(h_qkv)

        for i in range(CheckConfig.check_n_times):
            s_kv = random.randint(1, 512)

            q = torch.randn(b, s_q, num_q_heads, h_qkv, dtype=torch.bfloat16).cuda()
            k = torch.randn(b, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16).cuda()
            v = torch.randn(b, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16).cuda()

            o_ref = _flash_attention_forward_ref(
                q,
                k,
                v,
                attention_mask=None,
                query_length=s_q,
                is_causal=True,
                dropout=0.0,
                position_ids=None,
                sliding_window=None,
                use_top_left_mask=False,
            )

            o = flash_attn2_gemv(
                q,
                k,
                v,
                qk_scale=qk_scale,
                s_q=s_q,
                s_kv=s_kv,
                h_qkv=h_qkv,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                Bc=Bc,
            )

            print(
                f"  {i+1}/{CheckConfig.check_n_times}: hidden_size={h}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, h_qkv={h_qkv}, s_kv={s_kv}, mean_error={(o - o_ref).abs().mean()}"
            )
            assert torch.allclose(o, o_ref.float(), atol=1e-1), "Mismatch between custom and reference FlashAttention2"


if __name__ == "__main__":
    check_flash_attn2_gemv()
