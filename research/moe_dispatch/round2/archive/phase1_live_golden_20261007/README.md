# Preserved live expectations before integration corrections

The exact pre-repair test sources and their SHA256 values are retained here. Historical performance artifacts are unchanged.

The live Matrix baseline gains one safe-zero shift-amount setup instruction per recurrent head: 128 heads × 23 Mamba layers = 2,944 dynamic instructions. The full-layer baseline changes from 5,416,009 to 5,418,953 cycles; the L_TILE path remains 4,320,036. The live test also checks that the current baseline contains 128 V_SHFT_V zero writes per recurrent layer. This fixes NaN/Inf-safe zero initialization and does not change the archived historical evidence.

The live Nemotron formal MoE logical-traffic ratio now counts the explicit output-combine reads: 23 layers × (6 routed + 1 Shared expert) × hidden 2,688 × BF16 2 B = 865,536 B. Physical B200 profiler reads remain 1,114,920,410 B. Logical reads become 1,039,508,736 B, so the ratio is 1.0725454932588465. The historical ratio without combine reads is checked separately.

Targeted repair receipt: round2/results/E0/tests/live_golden_fix.xml (6 passed). Full current files have a separate receipt.
