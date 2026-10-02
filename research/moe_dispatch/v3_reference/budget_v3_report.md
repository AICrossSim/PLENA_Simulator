
## tile_bytes

| main | L | main_tile_B | augmented_tile_B | requests_32B |
|---|---|---|---|---|
| BF16 | 0 | 4096 | 4096 | 128 |
| MXINT4 | 8 | 1088 | 1152 | 36 |
| MXINT4 | 16 | 1088 | 1216 | 38 |
| MXINT3 | 8 | 832 | 896 | 28 |
| MXINT3 | 16 | 832 | 960 | 30 |
| MXINT2 | 8 | 576 | 640 | 20 |
| MXINT2 | 16 | 576 | 704 | 22 |

## rank_capacity

| model | expert | proj | K | segments | cap_L8 | cap_L16 | P1_slack |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | routed | gate | 2048 | 4 | 32 | 64 | 0 |
| DeepSeek-V2-Lite | routed | up | 2048 | 4 | 32 | 64 | 0 |
| DeepSeek-V2-Lite | routed | down | 1408 | 3 | 24 | 48 | 128 |
| DeepSeek-V2-Lite | shared | gate | 2048 | 4 | 32 | 64 | 0 |
| DeepSeek-V2-Lite | shared | up | 2048 | 4 | 32 | 64 | 0 |
| DeepSeek-V2-Lite | shared | down | 2816 | 6 | 48 | 96 | 256 |
| Qwen2-57B-A14B | routed | gate | 3584 | 7 | 56 | 112 | 0 |
| Qwen2-57B-A14B | routed | up | 3584 | 7 | 56 | 112 | 0 |
| Qwen2-57B-A14B | routed | down | 2560 | 5 | 40 | 80 | 0 |
| Qwen2-57B-A14B | shared | gate | 3584 | 7 | 56 | 112 | 0 |
| Qwen2-57B-A14B | shared | up | 3584 | 7 | 56 | 112 | 0 |
| Qwen2-57B-A14B | shared | down | 20480 | 40 | 320 | 640 | 0 |
| GLM-4.5-Air | routed | gate | 4096 | 8 | 64 | 128 | 0 |
| GLM-4.5-Air | routed | up | 4096 | 8 | 64 | 128 | 0 |
| GLM-4.5-Air | routed | down | 1408 | 3 | 24 | 48 | 128 |
| GLM-4.5-Air | shared | gate | 4096 | 8 | 64 | 128 | 0 |
| GLM-4.5-Air | shared | up | 4096 | 8 | 64 | 128 | 0 |
| GLM-4.5-Air | shared | down | 1408 | 3 | 24 | 48 | 128 |

## expert_bytes

| model | expert | format | ranks | bytes | bf16_bytes | ratio | factor_share |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | routed | MXINT4 only | 0/0/0 | 4603904 | 17301504 | 3.758 | 0.0 |
| DeepSeek-V2-Lite | routed | W4+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 4970112 | 17301504 | 3.481 | 0.074 |
| DeepSeek-V2-Lite | routed | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/24 | 5052544 | 17301504 | 3.424 | 0.089 |
| DeepSeek-V2-Lite | routed | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/24 | 5212160 | 17301504 | 3.319 | 0.117 |
| DeepSeek-V2-Lite | routed | W4+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 5336320 | 17301504 | 3.242 | 0.137 |
| DeepSeek-V2-Lite | routed | W3+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 3888768 | 17301504 | 4.449 | 0.094 |
| DeepSeek-V2-Lite | routed | W3+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 4254976 | 17301504 | 4.066 | 0.172 |
| DeepSeek-V2-Lite | shared | MXINT4 only | 0/0/0 | 9191424 | 34603008 | 3.765 | 0.0 |
| DeepSeek-V2-Lite | shared | W4+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 9889920 | 34603008 | 3.499 | 0.071 |
| DeepSeek-V2-Lite | shared | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/48 | 10023040 | 34603008 | 3.452 | 0.083 |
| DeepSeek-V2-Lite | shared | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/48 | 10280960 | 34603008 | 3.366 | 0.106 |
| DeepSeek-V2-Lite | shared | W4+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 10588416 | 34603008 | 3.268 | 0.132 |
| DeepSeek-V2-Lite | shared | W3+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 7727232 | 34603008 | 4.478 | 0.09 |
| DeepSeek-V2-Lite | shared | W3+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 8425728 | 34603008 | 4.107 | 0.166 |
| Qwen2-57B-A14B | routed | MXINT4 only | 0/0/0 | 14622720 | 55050240 | 3.765 | 0.0 |
| Qwen2-57B-A14B | routed | W4+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 15485824 | 55050240 | 3.555 | 0.056 |
| Qwen2-57B-A14B | routed | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/24 | 15631232 | 55050240 | 3.522 | 0.065 |
| Qwen2-57B-A14B | routed | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/24 | 15912960 | 55050240 | 3.459 | 0.081 |
| Qwen2-57B-A14B | routed | W4+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 16221952 | 55050240 | 3.394 | 0.099 |
| Qwen2-57B-A14B | routed | W3+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 12045184 | 55050240 | 4.57 | 0.072 |
| Qwen2-57B-A14B | routed | W3+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 12781312 | 55050240 | 4.307 | 0.125 |
| Qwen2-57B-A14B | shared | MXINT4 only | 0/0/0 | 116981760 | 440401920 | 3.765 | 0.0 |
| Qwen2-57B-A14B | shared | W4+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 122377216 | 440401920 | 3.599 | 0.044 |
| Qwen2-57B-A14B | shared | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/48 | 122983424 | 440401920 | 3.581 | 0.049 |
| Qwen2-57B-A14B | shared | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/48 | 124157952 | 440401920 | 3.547 | 0.058 |
| Qwen2-57B-A14B | shared | W4+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 126298112 | 440401920 | 3.487 | 0.074 |
| Qwen2-57B-A14B | shared | W3+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 94852096 | 440401920 | 4.643 | 0.057 |
| Qwen2-57B-A14B | shared | W3+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 98772992 | 440401920 | 4.459 | 0.094 |
| GLM-4.5-Air | routed | MXINT4 only | 0/0/0 | 9207808 | 34603008 | 3.758 | 0.0 |
| GLM-4.5-Air | routed | W4+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 9741952 | 34603008 | 3.552 | 0.055 |
| GLM-4.5-Air | routed | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/24 | 9889920 | 34603008 | 3.499 | 0.069 |
| GLM-4.5-Air | routed | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/24 | 10176512 | 34603008 | 3.4 | 0.095 |
| GLM-4.5-Air | routed | W4+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 10276096 | 34603008 | 3.367 | 0.104 |
| GLM-4.5-Air | routed | W3+LR (A MXINT4, B BF16, L=8) | 32/32/24 | 7579264 | 34603008 | 4.565 | 0.07 |
| GLM-4.5-Air | routed | W3+LR (A MXINT4, B BF16, L=16) | 64/64/48 | 8113408 | 34603008 | 4.265 | 0.132 |
| GLM-4.5-Air | shared | MXINT4 only | 0/0/0 | 9207808 | 34603008 | 3.758 | 0.0 |
| GLM-4.5-Air | shared | W4+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 9956608 | 34603008 | 3.475 | 0.075 |
| GLM-4.5-Air | shared | W4+LR (A MXINT8-split, B BF16, L=8) | 32/32/48 | 10121472 | 34603008 | 3.419 | 0.09 |
| GLM-4.5-Air | shared | W4+LR (A BF16, B BF16, L=8) [P1 only] | 32/32/48 | 10440704 | 34603008 | 3.314 | 0.118 |
| GLM-4.5-Air | shared | W4+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 10705408 | 34603008 | 3.232 | 0.14 |
| GLM-4.5-Air | shared | W3+LR (A MXINT4, B BF16, L=8) | 32/32/48 | 7793920 | 34603008 | 4.44 | 0.096 |
| GLM-4.5-Air | shared | W3+LR (A MXINT4, B BF16, L=16) | 64/64/96 | 8542720 | 34603008 | 4.051 | 0.175 |

## legacy_w4_prediction

| case | org | BF16_us | W4_bytes_only_us | speedup | ideal | R | bound_after |
|---|---|---|---|---|---|---|---|
| bfcl_b2 | 6 | 1081.3 | 1058.4 | 1.02 | 3.49 | 0.29 | on-chip |
| bfcl_b2 | 4+2 | 1081.3 | 529.2 | 2.04 | 3.49 | 0.59 | on-chip |
| bfcl_b16 | 6 | 7144.2 | 7144.2 | 1.0 | 3.48 | 0.29 | on-chip |
| bfcl_b16 | 4+2 | 6758.4 | 4233.6 | 1.6 | 3.48 | 0.46 | on-chip |
| swe_b16 | 6 | 4365.9 | 4365.9 | 1.0 | 3.48 | 0.29 | on-chip |
| swe_b16 | 4+2 | 3307.5 | 3307.5 | 1.0 | 3.48 | 0.29 | on-chip |
| mixed_T64_zipf | 6 | 15346.9 | 15346.9 | 1.0 | 3.48 | 0.29 | on-chip |
| mixed_T64_zipf | 4+2 | 12700.9 | 12700.9 | 1.0 | 3.48 | 0.29 | on-chip |
| mixed_T96_zipf | 6 | 20109.7 | 20109.7 | 1.0 | 3.48 | 0.29 | on-chip |
| mixed_T96_zipf | 4+2 | 18389.8 | 18389.8 | 1.0 | 3.48 | 0.29 | on-chip |

## v3_prediction

| case | org | point | time_us | bound | onchip_over_hbm | speedup_vs_current_4p2 |
|---|---|---|---|---|---|---|
| bfcl_b2 | 6 | BF16 @128 | 1081.3 | HBM | 0.12 | 1.0 |
| bfcl_b2 | 6 | BF16 @256 | 540.7 | HBM | 0.25 | 2.0 |
| bfcl_b2 | 6 | W4+LR P1 @256 | 155.1 | HBM | 0.9 | 6.97 |
| bfcl_b2 | 6 | W4+LR P2 @256 | 155.1 | HBM | 0.27 | 6.97 |
| bfcl_b2 | 6 | W4+LR A8-split P2 @256 | 157.6 | HBM | 0.27 | 6.86 |
| bfcl_b2 | 6 | W3+LR P2 @256 | 121.3 | HBM | 0.35 | 8.91 |
| bfcl_b2 | 6 | W4+LR P2 @512 | 77.6 | HBM | 0.55 | 13.94 |
| bfcl_b2 | 3+3 | BF16 @128 | 1081.3 | HBM | 0.12 | 1.0 |
| bfcl_b2 | 3+3 | BF16 @256 | 540.7 | HBM | 0.25 | 2.0 |
| bfcl_b2 | 3+3 | W4+LR P1 @256 | 155.1 | HBM | 0.45 | 6.97 |
| bfcl_b2 | 3+3 | W4+LR P2 @256 | 155.1 | HBM | 0.27 | 6.97 |
| bfcl_b2 | 3+3 | W4+LR A8-split P2 @256 | 157.6 | HBM | 0.27 | 6.86 |
| bfcl_b2 | 3+3 | W3+LR P2 @256 | 121.3 | HBM | 0.35 | 8.91 |
| bfcl_b2 | 3+3 | W4+LR P2 @512 | 77.6 | HBM | 0.55 | 13.94 |
| bfcl_b2 | 4+2 | BF16 @128 | 1081.3 | HBM | 0.12 | 1.0 |
| bfcl_b2 | 4+2 | BF16 @256 | 540.7 | HBM | 0.25 | 2.0 |
| bfcl_b2 | 4+2 | W4+LR P1 @256 | 155.1 | HBM | 0.45 | 6.97 |
| bfcl_b2 | 4+2 | W4+LR P2 @256 | 155.1 | HBM | 0.25 | 6.97 |
| bfcl_b2 | 4+2 | W4+LR A8-split P2 @256 | 157.6 | HBM | 0.25 | 6.86 |
| bfcl_b2 | 4+2 | W3+LR P2 @256 | 121.3 | HBM | 0.25 | 8.91 |
| bfcl_b2 | 4+2 | W4+LR P2 @512 | 77.6 | HBM | 0.5 | 13.94 |
| bfcl_b16 | 6 | BF16 @128 | 6758.4 | HBM | 0.13 | 1.0 |
| bfcl_b16 | 6 | BF16 @256 | 3379.2 | HBM | 0.25 | 2.0 |
| bfcl_b16 | 6 | W4+LR P1 @256 | 970.5 | HBM | 0.9 | 6.96 |
| bfcl_b16 | 6 | W4+LR P2 @256 | 970.5 | HBM | 0.3 | 6.96 |
| bfcl_b16 | 6 | W4+LR A8-split P2 @256 | 986.5 | HBM | 0.29 | 6.85 |
| bfcl_b16 | 6 | W3+LR P2 @256 | 759.3 | HBM | 0.38 | 8.9 |
| bfcl_b16 | 6 | W4+LR P2 @512 | 485.3 | HBM | 0.59 | 13.93 |
| bfcl_b16 | 3+3 | BF16 @128 | 6758.4 | HBM | 0.13 | 1.0 |
| bfcl_b16 | 3+3 | BF16 @256 | 3379.2 | HBM | 0.26 | 2.0 |
| bfcl_b16 | 3+3 | W4+LR P1 @256 | 970.5 | HBM | 0.51 | 6.96 |
| bfcl_b16 | 3+3 | W4+LR P2 @256 | 970.5 | HBM | 0.37 | 6.96 |
| bfcl_b16 | 3+3 | W4+LR A8-split P2 @256 | 986.5 | HBM | 0.37 | 6.85 |
| bfcl_b16 | 3+3 | W3+LR P2 @256 | 759.3 | HBM | 0.47 | 8.9 |
| bfcl_b16 | 3+3 | W4+LR P2 @512 | 485.3 | HBM | 0.74 | 13.93 |
| bfcl_b16 | 4+2 | BF16 @128 | 6758.4 | HBM | 0.12 | 1.0 |
| bfcl_b16 | 4+2 | BF16 @256 | 3379.2 | HBM | 0.25 | 2.0 |
| bfcl_b16 | 4+2 | W4+LR P1 @256 | 970.5 | HBM | 0.45 | 6.96 |
| bfcl_b16 | 4+2 | W4+LR P2 @256 | 970.5 | HBM | 0.26 | 6.96 |
| bfcl_b16 | 4+2 | W4+LR A8-split P2 @256 | 986.5 | HBM | 0.26 | 6.85 |
| bfcl_b16 | 4+2 | W3+LR P2 @256 | 759.3 | HBM | 0.27 | 8.9 |
| bfcl_b16 | 4+2 | W4+LR P2 @512 | 485.3 | HBM | 0.52 | 13.93 |
| swe_b16 | 6 | BF16 @128 | 3108.9 | HBM | 0.13 | 1.06 |
| swe_b16 | 6 | BF16 @256 | 1554.4 | HBM | 0.25 | 2.13 |
| swe_b16 | 6 | W4+LR P1 @256 | 446.3 | HBM | 0.9 | 7.41 |
| swe_b16 | 6 | W4+LR P2 @256 | 446.3 | HBM | 0.39 | 7.41 |
| swe_b16 | 6 | W4+LR A8-split P2 @256 | 453.6 | HBM | 0.39 | 7.29 |
| swe_b16 | 6 | W3+LR P2 @256 | 349.2 | HBM | 0.5 | 9.47 |
| swe_b16 | 6 | W4+LR P2 @512 | 223.2 | HBM | 0.78 | 14.82 |
| swe_b16 | 3+3 | BF16 @128 | 3108.9 | HBM | 0.14 | 1.06 |
| swe_b16 | 3+3 | BF16 @256 | 1554.4 | HBM | 0.28 | 2.13 |
| swe_b16 | 3+3 | W4+LR P1 @256 | 446.3 | HBM | 0.71 | 7.41 |
| swe_b16 | 3+3 | W4+LR P2 @256 | 446.3 | HBM | 0.61 | 7.41 |
| swe_b16 | 3+3 | W4+LR A8-split P2 @256 | 453.6 | HBM | 0.61 | 7.29 |
| swe_b16 | 3+3 | W3+LR P2 @256 | 349.2 | HBM | 0.78 | 9.47 |
| swe_b16 | 3+3 | W4+LR P2 @512 | 274.1 | on-chip | 1.23 | 12.07 |
| swe_b16 | 4+2 | BF16 @128 | 3108.9 | HBM | 0.13 | 1.06 |
| swe_b16 | 4+2 | BF16 @256 | 1554.4 | HBM | 0.26 | 2.13 |
| swe_b16 | 4+2 | W4+LR P1 @256 | 446.3 | HBM | 0.47 | 7.41 |
| swe_b16 | 4+2 | W4+LR P2 @256 | 446.3 | HBM | 0.29 | 7.41 |
| swe_b16 | 4+2 | W4+LR A8-split P2 @256 | 453.6 | HBM | 0.29 | 7.29 |
| swe_b16 | 4+2 | W3+LR P2 @256 | 349.2 | HBM | 0.34 | 9.47 |
| swe_b16 | 4+2 | W4+LR P2 @512 | 223.2 | HBM | 0.59 | 14.82 |
| mixed_T64_zipf | 6 | BF16 @128 | 8921.1 | HBM | 0.14 | 1.42 |
| mixed_T64_zipf | 6 | BF16 @256 | 4460.5 | HBM | 0.28 | 2.85 |
| mixed_T64_zipf | 6 | W4+LR P1 @256 | 1285.4 | on-chip | 1.0 | 9.88 |
| mixed_T64_zipf | 6 | W4+LR P2 @256 | 1281.2 | HBM | 0.48 | 9.91 |
| mixed_T64_zipf | 6 | W4+LR A8-split P2 @256 | 1302.3 | HBM | 0.48 | 9.75 |
| mixed_T64_zipf | 6 | W3+LR P2 @256 | 1002.4 | HBM | 0.61 | 12.67 |
| mixed_T64_zipf | 6 | W4+LR P2 @512 | 640.6 | HBM | 0.96 | 19.83 |
| mixed_T64_zipf | 3+3 | BF16 @128 | 8921.1 | HBM | 0.17 | 1.42 |
| mixed_T64_zipf | 3+3 | BF16 @256 | 4460.5 | HBM | 0.34 | 2.85 |
| mixed_T64_zipf | 3+3 | W4+LR P1 @256 | 1281.2 | HBM | 0.87 | 9.91 |
| mixed_T64_zipf | 3+3 | W4+LR P2 @256 | 1281.2 | HBM | 0.78 | 9.91 |
| mixed_T64_zipf | 3+3 | W4+LR A8-split P2 @256 | 1302.3 | HBM | 0.78 | 9.75 |
| mixed_T64_zipf | 3+3 | W3+LR P2 @256 | 1002.4 | HBM | 1.0 | 12.67 |
| mixed_T64_zipf | 3+3 | W4+LR P2 @512 | 1001.6 | on-chip | 1.56 | 12.68 |
| mixed_T64_zipf | 4+2 | BF16 @128 | 8921.1 | HBM | 0.13 | 1.42 |
| mixed_T64_zipf | 4+2 | BF16 @256 | 4460.5 | HBM | 0.26 | 2.85 |
| mixed_T64_zipf | 4+2 | W4+LR P1 @256 | 1281.2 | HBM | 0.53 | 9.91 |
| mixed_T64_zipf | 4+2 | W4+LR P2 @256 | 1281.2 | HBM | 0.39 | 9.91 |
| mixed_T64_zipf | 4+2 | W4+LR A8-split P2 @256 | 1302.3 | HBM | 0.39 | 9.75 |
| mixed_T64_zipf | 4+2 | W3+LR P2 @256 | 1002.4 | HBM | 0.48 | 12.67 |
| mixed_T64_zipf | 4+2 | W4+LR P2 @512 | 640.6 | HBM | 0.77 | 19.83 |
| mixed_T96_zipf | 6 | BF16 @128 | 8921.1 | HBM | 0.15 | 2.06 |
| mixed_T96_zipf | 6 | BF16 @256 | 4460.5 | HBM | 0.31 | 4.12 |
| mixed_T96_zipf | 6 | W4+LR P1 @256 | 1403.1 | on-chip | 1.1 | 13.11 |
| mixed_T96_zipf | 6 | W4+LR P2 @256 | 1281.2 | HBM | 0.63 | 14.35 |
| mixed_T96_zipf | 6 | W4+LR A8-split P2 @256 | 1302.3 | HBM | 0.63 | 14.12 |
| mixed_T96_zipf | 6 | W3+LR P2 @256 | 1002.4 | HBM | 0.8 | 18.35 |
| mixed_T96_zipf | 6 | W4+LR P2 @512 | 806.0 | on-chip | 1.26 | 22.82 |
| mixed_T96_zipf | 3+3 | BF16 @128 | 8921.1 | HBM | 0.2 | 2.06 |
| mixed_T96_zipf | 3+3 | BF16 @256 | 4460.5 | HBM | 0.4 | 4.12 |
| mixed_T96_zipf | 3+3 | W4+LR P1 @256 | 1504.8 | on-chip | 1.17 | 12.22 |
| mixed_T96_zipf | 3+3 | W4+LR P2 @256 | 1463.0 | on-chip | 1.14 | 12.57 |
| mixed_T96_zipf | 3+3 | W4+LR A8-split P2 @256 | 1474.0 | on-chip | 1.13 | 12.48 |
| mixed_T96_zipf | 3+3 | W3+LR P2 @256 | 1463.0 | on-chip | 1.46 | 12.57 |
| mixed_T96_zipf | 3+3 | W4+LR P2 @512 | 1463.0 | on-chip | 2.28 | 12.57 |
| mixed_T96_zipf | 4+2 | BF16 @128 | 8921.1 | HBM | 0.17 | 2.06 |
| mixed_T96_zipf | 4+2 | BF16 @256 | 4460.5 | HBM | 0.35 | 4.12 |
| mixed_T96_zipf | 4+2 | W4+LR P1 @256 | 1281.2 | HBM | 0.8 | 14.35 |
| mixed_T96_zipf | 4+2 | W4+LR P2 @256 | 1281.2 | HBM | 0.66 | 14.35 |
| mixed_T96_zipf | 4+2 | W4+LR A8-split P2 @256 | 1302.3 | HBM | 0.66 | 14.12 |
| mixed_T96_zipf | 4+2 | W3+LR P2 @256 | 1002.4 | HBM | 0.82 | 18.35 |
| mixed_T96_zipf | 4+2 | W4+LR P2 @512 | 840.0 | on-chip | 1.31 | 21.89 |

## per_tile_onchip

| Me | legacy_any_M | dense4_P2 | dense4_P1 | stream2_P2 | stream2_P1 | single6_P2 | single6_P1 | hbm_W4_256 | hbm_BF16_128 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 30.4 | 2.25 | 4.0 | 2.25 | 4.0 | 1.2 | 4.0 | 4.5 | 32.0 |
| 2 | 30.4 | 2.25 | 4.0 | 2.25 | 4.0 | 1.2 | 4.0 | 4.5 | 32.0 |
| 4 | 30.4 | 2.25 | 4.0 | 2.25 | 4.0 | 1.2 | 4.0 | 4.5 | 32.0 |
| 8 | 60.8 | 2.25 | 4.0 | inf | inf | 2.4 | 4.0 | 4.5 | 32.0 |
| 16 | 121.6 | 4.0 | 4.0 | inf | inf | 3.6 | 4.0 | 4.5 | 32.0 |
| 32 | 243.2 | 8.0 | 8.0 | inf | inf | 7.2 | 7.2 | 4.5 | 32.0 |
| 64 | 486.4 | 16.0 | 16.0 | inf | inf | 13.2 | 13.2 | 4.5 | 32.0 |

## storage

| T | prec | L | z_mode | total_KiB | capacity_KiB | slack_KiB | fits | ingress_fifo | landing_pool | dense_W_register | stream_W_register | dense_X_register | stream_X_register | dense_acc_buffer | stream_acc_sram | U_buffers_bf16 | U_d_partial_fp32 | X_activation | Z_dense | Z_stream | combine_fp32 | route_state | control_reserve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | P2 | 8 | full | 476.3 | 2108.0 | 1631.7 | True | 8.0 | 64.0 | 18.0 | 2.2 | 8.0 | 8.0 | 4.0 | 44.0 | 4.2 | 3.4 | 64.0 | 88.0 | 11.0 | 128.0 | 5.5 | 16.0 |
| 16 | P2 | 16 | full | 485.0 | 2108.0 | 1623.0 | True | 8.0 | 64.0 | 19.0 | 2.4 | 8.0 | 8.0 | 4.0 | 44.0 | 8.4 | 6.8 | 64.0 | 88.0 | 11.0 | 128.0 | 5.5 | 16.0 |
| 16 | P1 | 8 | full | 528.1 | 2108.0 | 1579.9 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 4.0 | 44.0 | 4.2 | 3.4 | 64.0 | 88.0 | 11.0 | 128.0 | 5.5 | 16.0 |
| 16 | P1 | 16 | full | 535.6 | 2108.0 | 1572.4 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 4.0 | 44.0 | 8.4 | 6.8 | 64.0 | 88.0 | 11.0 | 128.0 | 5.5 | 16.0 |
| 64 | P2 | 8 | full | 1352.3 | 2108.0 | 755.7 | True | 8.0 | 64.0 | 18.0 | 2.2 | 8.0 | 8.0 | 16.0 | 44.0 | 14.7 | 12.4 | 256.0 | 352.0 | 11.0 | 512.0 | 10.0 | 16.0 |
| 64 | P2 | 16 | full | 1380.5 | 2108.0 | 727.5 | True | 8.0 | 64.0 | 19.0 | 2.4 | 8.0 | 8.0 | 16.0 | 44.0 | 29.4 | 24.8 | 256.0 | 352.0 | 11.0 | 512.0 | 10.0 | 16.0 |
| 64 | P1 | 8 | full | 1404.1 | 2108.0 | 703.9 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 16.0 | 44.0 | 14.7 | 12.4 | 256.0 | 352.0 | 11.0 | 512.0 | 10.0 | 16.0 |
| 64 | P1 | 16 | full | 1431.1 | 2108.0 | 676.9 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 16.0 | 44.0 | 29.4 | 24.8 | 256.0 | 352.0 | 11.0 | 512.0 | 10.0 | 16.0 |
| 96 | P2 | 8 | streamed | 1600.3 | 2108.0 | 507.7 | True | 8.0 | 64.0 | 18.0 | 2.2 | 8.0 | 8.0 | 24.0 | 44.0 | 21.7 | 18.4 | 384.0 | 192.0 | 11.0 | 768.0 | 13.0 | 16.0 |
| 96 | P2 | 16 | streamed | 1641.5 | 2108.0 | 466.5 | True | 8.0 | 64.0 | 19.0 | 2.4 | 8.0 | 8.0 | 24.0 | 44.0 | 43.4 | 36.8 | 384.0 | 192.0 | 11.0 | 768.0 | 13.0 | 16.0 |
| 96 | P1 | 8 | streamed | 1652.1 | 2108.0 | 455.9 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 24.0 | 44.0 | 21.7 | 18.4 | 384.0 | 192.0 | 11.0 | 768.0 | 13.0 | 16.0 |
| 96 | P1 | 16 | streamed | 1692.1 | 2108.0 | 415.9 | True | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 24.0 | 44.0 | 43.4 | 36.8 | 384.0 | 192.0 | 11.0 | 768.0 | 13.0 | 16.0 |
| 128 | P2 | 8 | streamed | 2072.3 | 2108.0 | 35.7 | True | 8.0 | 64.0 | 18.0 | 2.2 | 8.0 | 8.0 | 32.0 | 44.0 | 28.7 | 24.4 | 512.0 | 256.0 | 11.0 | 1024.0 | 16.0 | 16.0 |
| 128 | P2 | 16 | streamed | 2126.5 | 2108.0 | -18.5 | False | 8.0 | 64.0 | 19.0 | 2.4 | 8.0 | 8.0 | 32.0 | 44.0 | 57.4 | 48.8 | 512.0 | 256.0 | 11.0 | 1024.0 | 16.0 | 16.0 |
| 128 | P1 | 8 | streamed | 2124.1 | 2108.0 | -16.1 | False | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 32.0 | 44.0 | 28.7 | 24.4 | 512.0 | 256.0 | 11.0 | 1024.0 | 16.0 | 16.0 |
| 128 | P1 | 16 | streamed | 2177.1 | 2108.0 | -69.1 | False | 8.0 | 64.0 | 64.0 | 8.0 | 8.0 | 8.0 | 32.0 | 44.0 | 57.4 | 48.8 | 512.0 | 256.0 | 11.0 | 1024.0 | 16.0 | 16.0 |

## credits_pool

| bw | latency | credits_32B | inflight_bytes | pool_min_bytes | pool_min_KiB |
|---|---|---|---|---|---|
| 128.0 | 64.0 | 272 | 8704 | 22016 | 21.5 |
| 128.0 | 150.0 | 616 | 19712 | 33024 | 32.2 |
| 256.0 | 64.0 | 544 | 17408 | 39424 | 38.5 |
| 256.0 | 150.0 | 1232 | 39424 | 61440 | 60.0 |

## compensation

| Me | scheme | big_core_issues | small_core_issues | extra_issue_pct_on_big | small_core_busy_vs_big_pct | bytes_per_expert | cross_core_KiB | sync |
|---|---|---|---|---|---|---|---|---|
| 1 | W4 only (no compensation) | 4352 | 0 | 0.0 | 0.0 | 4603904 | 0.0 | none |
| 1 | rank lanes L=8 | 4434 | 0 | 1.9 | 0.0 | 4970112 | 0.0 | none |
| 1 | separate short-K pass, same core | 5650 | 0 | 29.8 | 0.0 | 4970112 | 0.0 | accumulate before SiLU |
| 1 | K-extension (P1 only) | 5138 | 0 | 18.1 | 0.0 | 4970112 | 0.0 | none |
| 1 | offload X*A and U*B to small core | 4352 | 1298 | 0.0 | 29.8 | 4970112 | 25.8 | per output column block, before SiLU (gate/up) and before combine (down) |
| 2 | W4 only (no compensation) | 4352 | 0 | 0.0 | 0.0 | 4603904 | 0.0 | none |
| 2 | rank lanes L=8 | 4434 | 0 | 1.9 | 0.0 | 4970112 | 0.0 | none |
| 2 | separate short-K pass, same core | 5650 | 0 | 29.8 | 0.0 | 4970112 | 0.0 | accumulate before SiLU |
| 2 | K-extension (P1 only) | 5138 | 0 | 18.1 | 0.0 | 4970112 | 0.0 | none |
| 2 | offload X*A and U*B to small core | 4352 | 1298 | 0.0 | 29.8 | 4970112 | 51.5 | per output column block, before SiLU (gate/up) and before combine (down) |
| 4 | W4 only (no compensation) | 4352 | 0 | 0.0 | 0.0 | 4603904 | 0.0 | none |
| 4 | rank lanes L=8 | 4434 | 0 | 1.9 | 0.0 | 4970112 | 0.0 | none |
| 4 | separate short-K pass, same core | 5650 | 0 | 29.8 | 0.0 | 4970112 | 0.0 | accumulate before SiLU |
| 4 | K-extension (P1 only) | 5138 | 0 | 18.1 | 0.0 | 4970112 | 0.0 | none |
| 4 | offload X*A and U*B to small core | 4352 | 2596 | 0.0 | 59.7 | 4970112 | 103.0 | per output column block, before SiLU (gate/up) and before combine (down) |
| 16 | W4 only (no compensation) | 17408 | 0 | 0.0 | 0.0 | 4603904 | 0.0 | none |
| 16 | rank lanes L=8 | 17736 | 0 | 1.9 | 0.0 | 4970112 | 0.0 | none |
| 16 | separate short-K pass, same core | 22600 | 0 | 29.8 | 0.0 | 4970112 | 0.0 | accumulate before SiLU |
| 16 | K-extension (P1 only) | 20552 | 0 | 18.1 | 0.0 | 4970112 | 0.0 | none |
| 16 | offload X*A and U*B to small core | 17408 | 10384 | 0.0 | 59.7 | 4970112 | 412.0 | per output column block, before SiLU (gate/up) and before combine (down) |

## shared_split_B16

| split | extra_onchip_bytes | sync |
|---|---|---|
| none (v3 default: whole Shared on dense core) | 0 | none |
| I-split (each core: Gate/Up columns + Down rows) | 262144 | Down partial outputs must be summed (extra combine RMW Me*d*FP32); each core must apply all r_d rank terms with its own partial U_d, so per-core Down capacity L*S_c must be >= r_d |
| N-split (each core: half of every projection's output columns) | 157184 | X broadcast, full Z broadcast before Down, U_d broadcast |

## timeline

| scenario | variant | bw | latency | credits_B | pool_B | silu_barrier | credit_release | cycles | bytes_bound | efficiency | barrier_stall | peak_pool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| routed Me=2 on stream core | v3 | 256.0 | 64.0 | 17408 | 65536 | 44.0 | ingress | 19785 | 19414 | 0.981 | 51 | 29632 |
| routed Me=2 on stream core | v3 | 256.0 | 150.0 | 39424 | 65536 | 44.0 | ingress | 19758 | 19414 | 0.983 | 56 | 51584 |
| routed Me=2 on stream core | v3, pool 16 KiB | 256.0 | 150.0 | 39424 | 16384 | 44.0 | ingress | 50870 | 19414 | 0.382 | 53 | 16384 |
| routed Me=2 on stream core | v3, credits 256 at landing | 256.0 | 64.0 | 8192 | 65536 | 44.0 | landing | 47219 | 19414 | 0.411 | 52 | 11968 |
| routed Me=2 on stream core | legacy feed 30.4 ns/issue | 256.0 | 64.0 | 17408 | 65536 | 44.0 | ingress | 134930 | 19414 | 0.144 | 60 | 65536 |
| routed Me=4 on stream core | v3 | 256.0 | 64.0 | 17408 | 65536 | 88.0 | ingress | 19785 | 19414 | 0.981 | 95 | 41152 |
| routed Me=4 on stream core | v3 | 256.0 | 150.0 | 39424 | 65536 | 88.0 | ingress | 19758 | 19414 | 0.983 | 100 | 63040 |
| routed Me=4 on stream core | v3, pool 16 KiB | 256.0 | 150.0 | 39424 | 16384 | 88.0 | ingress | 50877 | 19414 | 0.382 | 97 | 16384 |
| routed Me=4 on stream core | v3, credits 256 at landing | 256.0 | 64.0 | 8192 | 65536 | 88.0 | landing | 47219 | 19414 | 0.411 | 96 | 18112 |
| routed Me=4 on stream core | legacy feed 30.4 ns/issue | 256.0 | 64.0 | 17408 | 65536 | 88.0 | ingress | 269767 | 19414 | 0.072 | 104 | 65536 |
| Shared Me=16 on dense core | v3 | 256.0 | 64.0 | 17408 | 65536 | 4.0 | ingress | 40111 | 38632 | 0.963 | 15 | 65056 |
| Shared Me=16 on dense core | v3 | 256.0 | 150.0 | 39424 | 65536 | 4.0 | ingress | 40011 | 38632 | 0.966 | 20 | 65056 |
| Shared Me=16 on dense core | v3, pool 16 KiB | 256.0 | 150.0 | 39424 | 16384 | 4.0 | ingress | 102276 | 38632 | 0.378 | 20 | 16384 |
| Shared Me=16 on dense core | v3, credits 256 at landing | 256.0 | 64.0 | 8192 | 65536 | 4.0 | landing | 93940 | 38632 | 0.411 | 20 | 11808 |
| Shared Me=16 on dense core | legacy feed 30.4 ns/issue | 256.0 | 64.0 | 17408 | 65536 | 4.0 | ingress | 1075040 | 38632 | 0.036 | 20 | 65536 |
| Shared Me=64 on dense core | v3 | 256.0 | 64.0 | 17408 | 65536 | 16.0 | ingress | 141548 | 38632 | 0.273 | 32 | 65536 |
| Shared Me=64 on dense core | v3 | 256.0 | 150.0 | 39424 | 65536 | 16.0 | ingress | 141634 | 38632 | 0.273 | 32 | 65536 |
| Shared Me=64 on dense core | v3, pool 16 KiB | 256.0 | 150.0 | 39424 | 16384 | 16.0 | ingress | 141634 | 38632 | 0.273 | 32 | 16384 |
| Shared Me=64 on dense core | v3, credits 256 at landing | 256.0 | 64.0 | 8192 | 65536 | 16.0 | landing | 141548 | 38632 | 0.273 | 32 | 65536 |
| Shared Me=64 on dense core | legacy feed 30.4 ns/issue | 256.0 | 64.0 | 17408 | 65536 | 16.0 | ingress | 4299884 | 38632 | 0.009 | 32 | 65536 |

## ports_W4_256

| path | required_Bpc | provided_Bpc | note |
|---|---|---|---|
| HBM -> ingress FIFO | 256.0 | 256.0 | 8 x 32 B grants per cycle |
| ingress FIFO -> pool (write) | 256.0 | 256 | 16 of the 128 pool banks per cycle |
| pool -> dense core (burst) | 512 | 512 | one compressed tile per W-register slot |
| pool -> stream core (burst) | 512 | 512 | one compressed tile per issue |
| pool aggregate (1 write + 2 read streams, peak) | 1280.0 | 2048 | 128 banks x 16 B; bank conflicts must be simulated, not assumed away |
| activation store -> dense X | 455.1 | 512 | Me=16: X block per (group, segment, M block) / G |
| activation store -> stream X | 1.3 | 128 | X stationary for a whole K segment |
| activation store Z write + U read | 56.9 | 256 | SiLU output and rank-lane U operands |
| activation store aggregate | 513.3 | 1024 | 64 banks x 16 B |
| stream acc SRAM RMW | 14.2 | 128 | per issue 32 B read + 32 B write |
| combine buffer RMW | 37.9 | 256 | FP32 Down output per (expert, column tile), gate folded |
