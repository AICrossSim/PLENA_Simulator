# ARIA Launage Models



## Example Models
### Tiny-Llama

| Weight Name | Shape |
| --- | --- |
| `model.embed_tokens.weight` | [32 000, 2 048] |
| `model.layers.0.input_layernorm.weight` | [2 048] |
| `model.layers.0.mlp.down_proj.weight` | [2 048, 5 632] |
| `model.layers.0.mlp.gate_proj.weight` | [5 632, 2 048] |
| `model.layers.0.mlp.up_proj.weight` | [5 632, 2 048] |
| `model.layers.0.post_attention_layernorm.weight` | [2 048] |
| `model.layers.0.self_attn.k_proj.weight` | [256, 2 048] |
| `model.layers.0.self_attn.o_proj.weight` | [2 048, 2 048] |
| `model.layers.0.self_attn.q_proj.weight` | [2 048, 2 048] |
| `model.layers.0.self_attn.v_proj.weight` | [256, 2 048] |