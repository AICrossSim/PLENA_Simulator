# Round A: prospective equal-MAC geometry screening

post-Router compute/vector analytical layer makespan, ideal memory; not HBM E2E or complete inference

96 geometries ×9 policies ×18 development layers ×2 repeats = 31,104 evaluations.
Frozen selection, then135 heldout layers ×9 selected points ×2 = 2,430 evaluations.

| Selection | M×N×K | Policy | Heldout total ms | p95 ms | Spatial util | Speedup vs6×4×512 | vsbest single | vsbest homo |
|---|---|---|---:|---:|---:|---:|---:|---:|
|best_single_eft|1x24x512|eft|25.023207|0.834306|96.504%|2.506×|1.000×|1.125×|
|best_single_threshold|1x24x512|threshold_1|25.023207|0.834306|96.504%|2.506×|1.000×|1.125×|
|best_homogeneous_eft|1x12x512+1x12x512|eft|28.153740|0.990560|96.903%|2.227×|0.889×|1.000×|
|best_homogeneous_threshold|1x12x512+1x12x512|threshold_1|28.153740|0.990560|96.903%|2.227×|0.889×|1.000×|
|best_heterogeneous_eft|1x2x512+1x22x512|eft|25.323792|0.847256|96.742%|2.476×|0.988×|1.112×|
|best_heterogeneous_threshold|1x2x512+1x22x512|threshold_1|54.467930|0.896080|96.744%|1.151×|0.459×|0.517×|
|fixed_single_6|6x4x512|eft|62.705784|1.222056|61.834%|1.000×|0.399×|0.449×|
|fixed_homogeneous_3_3|3x4x512+3x4x512|eft|42.292132|1.120488|79.902%|1.483×|0.592×|0.666×|
|fixed_heterogeneous_4_2|2x4x512+4x4x512|eft|40.363354|1.042322|81.314%|1.554×|0.620×|0.698×|

The fastest heldout point is not used to revise the development-selected hardware.
A new PN width changes buffers and ports. All rows have12,288 main MACs, but full SRAM/port feasibility and PPA are deferred to Round B.
Latency includes the modeled dot/result dependency and one global vector queue. It excludes HBM, SRAM bank service, paid runtime decision service, attention and Router.
The captured heldout data was previously exposed for correctness in the v3 campaign; the new geometry selection is development-only but this is not a pristine blind test.
No quantitative full-model accuracy claim is made by this metadata-only screening.
