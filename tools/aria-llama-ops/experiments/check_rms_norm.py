import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from aria_lm_ops.models.llama import rms_norm
from aria_lm_ops.utils.model_shortcut import load_tinyllama_cfg, load_tinyllama_ckpt
from aria_lm_ops.config import CheckConfig


@torch.no_grad()
def check_rms_norm():
    config = load_tinyllama_cfg()
    state_dict = load_tinyllama_ckpt()
    w_name = "model.layers.0.input_layernorm.weight"

    b = batch_size = 1
    h = hidden_size = config.hidden_size
    s = seq_len = 1
    w = state_dict[w_name]
    x = torch.randn(b, s, h)
    var_eps = config.rms_norm_eps

    rms_ref = LlamaRMSNorm(h, var_eps)
    rms_ref.load_state_dict({"weight": w})

    x = x.cuda()
    w = w.cuda()
    rms_ref.cuda()

    for _ in range(CheckConfig.check_n_times):
        out = rms_norm(x, w, s, h, var_eps)
        out_ref = rms_ref(x)
        assert torch.allclose(out, out_ref, atol=1e-5), "Mismatch between custom and reference RMSNorm"
        x = torch.randn(b, s, h).cuda()

    print(f"{CheckConfig.check_n_times} RMSNorm checks passed")
    return True


if __name__ == "__main__":
    check_rms_norm()
