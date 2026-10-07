# 第二轮交付状态

分支：`research/moe-supply-first-v3`；报告生成时提交：`3bb0be26c3411fd37ba363f45b57e852af08efdd`。

全部要求完成：否。存在文件不等于完整范围已完成。

| 用户节 | 状态 | 必须文件 | 原因 |
|---|---|---|---|
| 1 | 完成 | BRANCHES_BEFORE.md、MERGE_LOG.md、BRANCHES_AFTER.md、results/E0/reproduce_check.csv | 本节范围检查通过 |
| 2 | 完成 | results/E0/frozen_inputs.json、results/E0/SOURCES.md | 本节范围检查通过 |
| 3 | 完成 | results/E1/bounds_per_window.csv、results/E1/headroom_by_batch.csv、results/E1/SUMMARY.md | 本节范围检查通过 |
| 4 | 完成 | results/E2/micro.csv、results/E2/layer_grid.csv、results/E2/SUMMARY.md | 本节范围检查通过 |
| 5.1 | 部分完成 | results/E3/bnb_certificate.csv、results/E3/bnb_leaves.csv、results/E3/bnb_summary.json、results/E3/lb_validity.csv | pipelined/A/5+1 仍有未剪区域，族最优未证明；pipelined/B/4+2 仍有未剪区域，族最优未证明；pipelined/B/5+1 仍有未剪区域，族最优未证明；pipelined/B/heterogeneous 仍有未剪区域，族最优未证明；pipelined/B/homogeneous 仍有未剪区域，族最优未证明；pipelined/B/single 通用证书前沿仍开放；有限单核域另已直接穷举；port_tight/A/5+1 仍有未剪区域，族最优未证明；port_tight/B/4+2 仍有未剪区域，族最优未证明；port_tight/B/5+1 仍有未剪区域，族最优未证明；port_tight/B/heterogeneous 仍有未剪区域，族最优未证明；port_tight/B/homogeneous 仍有未剪区域，族最优未证明；port_tight/B/single 通用证书前沿仍开放；有限单核域另已直接穷举；fixed_issue/A/4+2 仍有未剪区域，族最优未证明；fixed_issue/A/5+1 仍有未剪区域，族最优未证明；fixed_issue/A/heterogeneous 仍有未剪区域，族最优未证明；fixed_issue/A/homogeneous 仍有未剪区域，族最优未证明；fixed_issue/A/single 通用证书前沿仍开放；有限单核域另已直接穷举；fixed_issue/B/4+2 仍有未剪区域，族最优未证明；fixed_issue/B/5+1 仍有未剪区域，族最优未证明；fixed_issue/B/heterogeneous 仍有未剪区域，族最优未证明；fixed_issue/B/homogeneous 仍有未剪区域，族最优未证明；fixed_issue/B/single 通用证书前沿仍开放；有限单核域另已直接穷举 |
| 5.2 | 部分完成 | results/E3/schedule_gaps.csv | pipelined 已评估搜索种子仍含未证最优内层分配；不能把全部叶子称为精确最优分配；port_tight 已评估搜索种子仍含未证最优内层分配；不能把全部叶子称为精确最优分配；fixed_issue 已评估搜索种子仍含未证最优内层分配；不能把全部叶子称为精确最优分配；完整评估 BnB 叶子仍有内层分配未证最优，已保留在开放前沿 |
| 5.3 | 部分完成 | results/E3/workload_map.csv、results/E3/workload_extreme.json | 部分负载点硬件搜索证明仍开放；Δ 仅为已评估候选比；极端负载的 δ=0 全域复验未闭合 |
| 5.4 | 部分完成 | results/E3/robust_objectives.csv、results/E3/selection_stability.csv | 稳健目标只覆盖开发集已评估近优候选，尚非经证明的各族 1% 近优集合 |
| 5.5 | 部分完成 | results/E3/sobol.csv、results/E3/flip_boundary.csv | 部分敏感性采样的硬件搜索未闭合，Sobol 是候选估计的指数 |
| 6 | 部分完成 | results/E4/heldout_main_table.csv、results/E4/breakdown.csv、results/E4/hbm512_sensitivity.csv、results/E4/SUMMARY.md | 族最优搜索未闭合，E4 当前按冻结已评估候选交付 |
| 7 | 完成 | results/E5/dispatch_table.csv、results/E5/predictor_table.csv、results/E5/dispatcher_state_bits.csv、results/E5/SUMMARY.md | 本节范围检查通过 |
| 8 | 部分完成 | results/E6/moe_layer_e2e.csv、results/E6/model_token_e2e.csv、results/E6/SUMMARY.md | 缺少冻结 DeepSeek 模型匹配的 attention/router/norm 每层计时与层映射；整模型每 token 无法生成 |
| 9–10 | 完成 | REPORT_ZH.md、figures/fig_headroom.pdf、figures/fig_headroom.png、figures/fig_main_bars.pdf、figures/fig_main_bars.png、figures/fig_breakdown.pdf、figures/fig_breakdown.png、figures/fig_bnb_coverage.pdf、figures/fig_bnb_coverage.png、figures/fig_workload_map.pdf、figures/fig_workload_map.png、figures/fig_sobol.pdf、figures/fig_sobol.png、figures/fig_flip_boundary.pdf、figures/fig_flip_boundary.png、figures/fig_dataflow_grid.pdf、figures/fig_dataflow_grid.png、figures/fig_me_crossover.pdf、figures/fig_me_crossover.png、figures/fig_predictor.pdf、figures/fig_predictor.png | 本节范围检查通过 |

## 1

已完成范围：

```json
{
  "reproduction_rows": 1350,
  "all_abs_diff_zero": true,
  "cohorts": {
    "historical945": {
      "all_exact": true,
      "cohort": "historical945",
      "repeats": 2,
      "rows": 945,
      "sha256": "bcd3897ed29b5a68d891c2822a521caf0cd56ff26719ff79b926165d37f945eb"
    },
    "previous_BF16_c256_three": {
      "all_exact": true,
      "cohort": "previous_BF16_c256_three",
      "repeats": 2,
      "rows": 405,
      "sha256": "476e8fbf849678e108e9d739ebeb3f0c627c9a52966040c570cbba9ccfd9f93a"
    }
  },
  "reproduction_receipt": "results/E0/post_integration_replay_20261007/reproduction_receipt.json",
  "tests": {
    "required_suites": [
      "research",
      "analytical",
      "rust",
      "main_rust",
      "round2"
    ],
    "latest_by_suite": {
      "research": {
        "command": [
          "/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python",
          "-m",
          "pytest",
          "research/moe_dispatch",
          "-q",
          "--ignore=research/moe_dispatch/archive",
          "--ignore=research/moe_dispatch/results",
          "--ignore=research/moe_dispatch/round2/results",
          "--ignore=research/moe_dispatch/round2/archive",
          "--junitxml=research/moe_dispatch/round2/results/E0/tests/20261007T114345980637Z/research.xml"
        ],
        "errors": 0,
        "failed_tests": [],
        "failures": 0,
        "finished_utc": "2026-10-07T11:48:20.271424+00:00",
        "initial_classification": "passed",
        "junit_sha256": "6ca33ba365add076973eff1c1c4d26ab6d01ab07a7738f3591330de246e98dde",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T114345980637Z/research.log",
        "log_sha256": "b32a6359e650772712c2ff70d9505f33fe21df8dcafbcc784b19f8586be5ef7f",
        "returncode": 0,
        "skipped": 0,
        "started_utc": "2026-10-07T11:43:46.004320+00:00",
        "suite": "research",
        "tests": 281,
        "receipt": "results/E0/tests/20261007T114345980637Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "analytical": {
        "returncode": 0,
        "tests": 478,
        "passed": 469,
        "failures": 0,
        "errors": 0,
        "skipped": 9,
        "skip_details": [
          {
            "node_id": "analytic_models/performance/test_b200_formal_campaign.py::test_local_kda_stage2_independently_matches_the_formal_core_traffic",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_KDA_STAGE2_ROOT to cross-check the optional raw archive",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_b200_formal_campaign.py::test_raw_campaign_rebuilds_the_pinned_contract_and_routing_trace",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_B200_CAMPAIGN_ROOT to rebuild from the optional raw archive",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_gpu_evidence.py::test_local_archives_reproduce_every_imported_file",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "raw GPU archives are not part of a fresh clone",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-fsm-expected0]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-row-expected1]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-fsm-expected2]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-row-expected3]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-old_isa-expected4]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-old_isa-expected5]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          }
        ],
        "receipt": "results/E0/PHASE1_GATE.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae",
        "aggregation": "exact current node-ID union of broad historical-pin receipt, broad remaining receipt, and full repaired Matrix/formal files; no duplicate test counting",
        "uncovered": 0
      },
      "rust": {
        "command": [
          "/nix/store/qqq4l5xqxm4s009vcdm1p6kic2jga311-rust-default-1.93.1/bin/cargo",
          "test",
          "--locked",
          "--manifest-path",
          "research/moe_dispatch/rust/Cargo.toml",
          "--target-dir",
          "/tmp/plena-round2-cargo-target",
          "--",
          "--test-threads=2"
        ],
        "finished_utc": "2026-10-07T11:03:19.400537+00:00",
        "initial_classification": "passed",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T110253385260Z/rust.log",
        "log_sha256": "ec00fab049eb198b9800ce04b9c991380e60a8126101b03ef143e1e51e4536ce",
        "returncode": 0,
        "started_utc": "2026-10-07T11:02:53.400831+00:00",
        "suite": "rust",
        "receipt": "results/E0/tests/20261007T110253385260Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "main_rust": {
        "command": [
          "/nix/store/qqq4l5xqxm4s009vcdm1p6kic2jga311-rust-default-1.93.1/bin/cargo",
          "test",
          "--locked",
          "--manifest-path",
          "transactional_emulator/Cargo.toml",
          "--workspace",
          "--target-dir",
          "/tmp/mcl123-plena-layout-async-target",
          "--",
          "--test-threads=2"
        ],
        "finished_utc": "2026-10-07T10:55:35.509722+00:00",
        "initial_classification": "passed",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T105511300902Z/main_rust.log",
        "log_sha256": "8aa4a3a26a8fde392d3270794e11fe0f2e3f4ea0cb45d07656164f53b59fb87c",
        "returncode": 0,
        "started_utc": "2026-10-07T10:55:11.316392+00:00",
        "suite": "main_rust",
        "receipt": "results/E0/tests/20261007T105511300902Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "round2_subset_round2_delivery_metadata_unit_suite": {
        "returncode": 0,
        "tests": 9,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_delivery_metadata_unit_suite_20261007T152454Z.json",
        "commit": "2c95619bfc0d9130afdfb8b353a10e4ca9901e44",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      },
      "round2": {
        "returncode": 0,
        "tests": 126,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_final_engine_unit_suite_20261007T135716Z.json",
        "commit": "045161dd7e9c8fa44f4ef421cd324cfa3a0cb007",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      },
      "round2_subset_round2_final_reader_unit_suite": {
        "returncode": 0,
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_final_reader_unit_suite_20261007T145951Z.json",
        "commit": "cdd193e966b9ca35c23a88ca55631e87a8474581",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      }
    },
    "all_required_recorded_suites_passed": true,
    "phase1_gate_validated": true,
    "gate_receipt_hash_checks": {
      "compiler_integration/COPY_MANIFEST.json": true,
      "compiler_integration/TEST_RESULTS.md": true,
      "compiler_integration/tests/evidence_manifest.json": true,
      "post_integration_replay_20261007/reproduction_receipt.json": true,
      "tests/20261007T105511300902Z/UNIT_CHECKS.json": true,
      "tests/20261007T110253385260Z/UNIT_CHECKS.json": true,
      "tests/20261007T114345980637Z/UNIT_CHECKS.json": true,
      "tests/analytical_collection_current.log": true,
      "tests/analytical_full_historical_pin.xml": true,
      "tests/analytical_gate_coverage.csv": true,
      "tests/analytical_remaining.xml": true,
      "tests/live_golden_fix.xml": true,
      "tests/live_golden_full_files.xml": true
    },
    "gate_source_hash_checks": {
      "analytic_models/performance/hybrid_lcompute_campaign.py": true,
      "analytic_models/performance/profiles/historical_hybrid_compiler_evidence.json": true,
      "analytic_models/performance/test_matrix_lcompute_campaign.py": true,
      "analytic_models/performance/test_nemotron3_formal_dse.py": true,
      "research/moe_dispatch/round2/run_unit_checks.py": true
    },
    "all_supported_current_tests_passed": true,
    "all_collected_tests_executed_without_skips": false,
    "compiler_gate": {
      "additional_final_lifetime_tests_passed": 41,
      "affected_final_integration_passed": 45,
      "current_tvm_passed": 143,
      "evidence": "compiler_integration/TEST_RESULTS.md",
      "historical_precision_ablation_passed": 1,
      "ordinary_passed": 1143,
      "ordinary_retired_profile_skipped": 29,
      "ordinary_subtests_passed": 11,
      "research_passed": 33,
      "research_subtests_passed": 3,
      "scope": "Counts are suite-specific and overlap; do not add reruns to unique total. Archived-profile skips and numerical-precision correction are explicit in TEST_RESULTS.md.",
      "slow_connected_passed": 3,
      "slow_connected_retired_profile_skipped": 1,
      "tvm_retired_demos_skipped": 5,
      "tvm_standalone_entrypoints_passed": 13
    },
    "phase1_limitations": [
      "No new checkpoint inference or GPU benchmarks for round2 BF16 MoE.",
      "Optional raw GPU archive rebuilds and six pinned historical-E-Compiler oracles remain explicitly skipped.",
      "Existing archive native integration binary is hash/source checked; fresh Rust unit builds are separately recorded.",
      "New-model searches, native timing calibration and synthesis/PPA are outside this phase-1 gate."
    ]
  }
}
```

## 2

已完成范围：

```json
{
  "development_windows": 18,
  "heldout_windows": 135,
  "heldout_batch_counts": {
    "128": 9,
    "16": 27,
    "2": 27,
    "4": 27,
    "64": 9,
    "8": 27,
    "96": 9
  },
  "input_sha256": {
    "development.json": "168522072cb2fd80e2689af9ad370b25fbad022072f4abc94e4aa512223c95e0",
    "heldout.json": "0628f6e1e34bb936b0bac9be6682263ce53c5f313e3c776b26d5856f9e96191a",
    "mixed_development.json": "c2c4feb51d83abf6b8d224be6bafb9cc9432797f2740eac40ce76503cc8f3f55",
    "mixed_heldout.json": "2910033d86a3346a55ddfdf8781b86039a3726e07515581820bf92f03d233b8f"
  }
}
```

## 3

已完成范围：

```json
{
  "bounds_rows": 2430,
  "expected_rows": 2430,
  "covered_unique_keys": 2430
}
```

## 4

已完成范围：

```json
{
  "micro_rows": 1188,
  "micro_expected_rows": 1188,
  "layer_grid_rows": 720,
  "micro_unique_keys": 1188
}
```

## 5.1

已完成范围：

```json
{
  "proof_runs": 6,
  "lower_bound_checks": 36000,
  "lower_bound_unique_checks": 36000,
  "all_lb_ok": true,
  "seed_single_geometry_flow_counts": {
    "pipelined": 132,
    "port_tight": 132,
    "fixed_issue": 132
  },
  "direct_single_exhaustion": {
    "all_modes_complete": true,
    "dataflows": [
      "OS",
      "WS",
      "IS"
    ],
    "development_sha256": "4d1fc9142babb091199c49a31028220203230eeaf2ecd3845a66e175c7392e3e",
    "development_windows": 18,
    "dimension_order": "PM×PN×PK",
    "domain": {
      "PK": [
        32,
        64,
        128,
        256,
        512,
        1024
      ],
      "PM": [
        1,
        16
      ],
      "PN": [
        1,
        192
      ],
      "multiplier_product": 12288
    },
    "engine_sha256": "943211ec3825b344fa3358dd503f34ca4e452262c4a205b637d52c4b6c05c123",
    "expected_points_per_mode": 132,
    "geometry_inventory_sha256": "aa00801687626cb97ad6ed7ea4e7c5ebf0fb360aeae82d0d8281ac867436d2fd",
    "modes": {
      "fixed_issue": {
        "all_allocations_OPTIMAL": true,
        "all_repeats_identical": true,
        "best": {
          "design": {
            "acc_banks": [
              12
            ],
            "acc_bytes": [
              98304
            ],
            "cores": [
              {
                "pk": 128,
                "pm": 2,
                "pn": 48
              }
            ],
            "flows": [
              "WS"
            ],
            "label": "",
            "records": 8,
            "total_macs": 12288,
            "vector_lanes": [
              64
            ],
            "w_banks": [
              64
            ],
            "w_bytes": [
              40960
            ],
            "x_banks": [
              24
            ],
            "x_bytes": [
              12288
            ],
            "z_bytes": [
              393216
            ]
          },
          "flows": [
            "WS"
          ],
          "geomean_ms": 5.356828222055863,
          "geometry": "2x48x128"
        },
        "exact_allocation_windows": 2322,
        "executable_repeats_per_legal_point": 2,
        "invalid_templates": 3,
        "invalids": [
          {
            "flows": [
              "OS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "WS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "IS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          }
        ],
        "legal_points": 129,
        "points": 132,
        "seed_file_sha256": "53f929528f789ecbf57d11dc255c87d7ceb838e83db35339c89cae70a841077d",
        "single_family_proof_complete": true,
        "solver_algorithm_window_counts": {
          "exact_grouped_count_enumeration": 2322
        }
      },
      "pipelined": {
        "all_allocations_OPTIMAL": true,
        "all_repeats_identical": true,
        "best": {
          "design": {
            "acc_banks": [
              12
            ],
            "acc_bytes": [
              98304
            ],
            "cores": [
              {
                "pk": 128,
                "pm": 3,
                "pn": 32
              }
            ],
            "flows": [
              "WS"
            ],
            "label": "",
            "records": 8,
            "total_macs": 12288,
            "vector_lanes": [
              64
            ],
            "w_banks": [
              64
            ],
            "w_bytes": [
              40960
            ],
            "x_banks": [
              24
            ],
            "x_bytes": [
              12288
            ],
            "z_bytes": [
              393216
            ]
          },
          "flows": [
            "WS"
          ],
          "geomean_ms": 4.033368266089851,
          "geometry": "3x32x128"
        },
        "exact_allocation_windows": 2322,
        "executable_repeats_per_legal_point": 2,
        "invalid_templates": 3,
        "invalids": [
          {
            "flows": [
              "OS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "WS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "IS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          }
        ],
        "legal_points": 129,
        "points": 132,
        "seed_file_sha256": "01ed41419cd6039e4328e25c215eda223d927b69cf8bd5aec28c2d58900e84f9",
        "single_family_proof_complete": true,
        "solver_algorithm_window_counts": {
          "exact_grouped_count_enumeration": 2322
        }
      },
      "port_tight": {
        "all_allocations_OPTIMAL": true,
        "all_repeats_identical": true,
        "best": {
          "design": {
            "acc_banks": [
              12
            ],
            "acc_bytes": [
              98304
            ],
            "cores": [
              {
                "pk": 128,
                "pm": 3,
                "pn": 32
              }
            ],
            "flows": [
              "WS"
            ],
            "label": "",
            "records": 8,
            "total_macs": 12288,
            "vector_lanes": [
              64
            ],
            "w_banks": [
              64
            ],
            "w_bytes": [
              40960
            ],
            "x_banks": [
              24
            ],
            "x_bytes": [
              12288
            ],
            "z_bytes": [
              393216
            ]
          },
          "flows": [
            "WS"
          ],
          "geomean_ms": 11.243781734366843,
          "geometry": "3x32x128"
        },
        "exact_allocation_windows": 2322,
        "executable_repeats_per_legal_point": 2,
        "invalid_templates": 3,
        "invalids": [
          {
            "flows": [
              "OS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "WS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          },
          {
            "flows": [
              "IS"
            ],
            "geometry": "12x1x1024",
            "reason": "an expert has no physical core"
          }
        ],
        "legal_points": 129,
        "points": 132,
        "seed_file_sha256": "cdfb89bc81925e2adb62782511d7607490daecfe98a2e5d22e064dbabc92d549",
        "single_family_proof_complete": true,
        "solver_algorithm_window_counts": {
          "exact_grouped_count_enumeration": 2322
        }
      }
    },
    "objective_scope": "exhaustive declared single geometry×OS/WS/IS family with fixed pool/port ledger, exact integer resource relaxation and executable deterministic finite LPT replay; not globally optimal temporal schedule or arbitrary dataflow/hardware",
    "physical_budget": {
      "W_X_acc_banks": [
        64,
        24,
        12
      ],
      "bank_B_per_cycle": 16,
      "other_fixed_storage_B": 1613824,
      "private_W_B": 40960,
      "private_X_B": 12288,
      "private_Z_B": 393216,
      "private_acc_B": 98304,
      "vector_lanes": 64
    },
    "single_geometry_count": 44,
    "single_inventory_sha256": "c25db204f7d0a2b0ccd85c6475f79c01c1eecd1654b4bbfc3c3afb947f901a21"
  }
}
```

未完成范围的下界/差距：

```json
{
  "family_frontiers": [
    {
      "onchip_mode": "pipelined",
      "proof": "A",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 106646251841749620,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 4.0136142207695915,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.19697423210236487,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "pipelined",
      "proof": "A",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 4.48836609894005,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 12.048811276512339,
      "open_frontiers": 7653,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "pipelined",
      "proof": "A",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 2352706598170238310,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 4.0136142207695915,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.19697423210236487,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "pipelined",
      "proof": "A",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 5783725298295945,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 4.16228176038323,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 3.9083517627912956,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "pipelined",
      "proof": "A",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 132,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 4.033368266089851,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.6901196768467566,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "pipelined",
      "proof": "B",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 4.0136142207695915,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.19697423210236487,
      "open_frontiers": 220,
      "open_lattice_points": 106646251841749620
    },
    {
      "onchip_mode": "pipelined",
      "proof": "B",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 4.48836609894005,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 12.048811276512339,
      "open_frontiers": 220,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "pipelined",
      "proof": "B",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 4.0136142207695915,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.19697423210236487,
      "open_frontiers": 220,
      "open_lattice_points": 2352706598170238310
    },
    {
      "onchip_mode": "pipelined",
      "proof": "B",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 4.16228176038323,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 3.9083517627912956,
      "open_frontiers": 220,
      "open_lattice_points": 5783725298295945
    },
    {
      "onchip_mode": "pipelined",
      "proof": "B",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 89,
      "coverage_pct": 67.42424242424242,
      "proof_complete": false,
      "incumbent_ms": 4.033368266089851,
      "remaining_lower_bound_ms": 4.005723976726294,
      "certified_global_lower_bound_ms": 4.005723976726294,
      "gap_pct": 0.6901196768467566,
      "open_frontiers": 43,
      "open_lattice_points": 43
    },
    {
      "onchip_mode": "port_tight",
      "proof": "A",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 106646251841749620,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 11.52605269895404,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 3.2703519638834333,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "port_tight",
      "proof": "A",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 11.77964481480477,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 5.542469552027662,
      "open_frontiers": 5887,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "port_tight",
      "proof": "A",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 2352706598170238310,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 11.52605269895404,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 3.2703519638834333,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "port_tight",
      "proof": "A",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 5783725298295945,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 11.687989828917384,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 4.721265372326666,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "port_tight",
      "proof": "A",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 132,
      "coverage_pct": 100.0,
      "proof_complete": true,
      "incumbent_ms": 11.243781734366843,
      "remaining_lower_bound_ms": null,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 0.7412795551870799,
      "open_frontiers": 0,
      "open_lattice_points": 0
    },
    {
      "onchip_mode": "port_tight",
      "proof": "B",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 11.52605269895404,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 3.2703519638834333,
      "open_frontiers": 200,
      "open_lattice_points": 106646251841749620
    },
    {
      "onchip_mode": "port_tight",
      "proof": "B",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 11.77964481480477,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 5.542469552027662,
      "open_frontiers": 200,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "port_tight",
      "proof": "B",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 11.52605269895404,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 3.2703519638834333,
      "open_frontiers": 200,
      "open_lattice_points": 2352706598170238310
    },
    {
      "onchip_mode": "port_tight",
      "proof": "B",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 11.687989828917384,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 4.721265372326666,
      "open_frontiers": 200,
      "open_lattice_points": 5783725298295945
    },
    {
      "onchip_mode": "port_tight",
      "proof": "B",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 72,
      "coverage_pct": 54.54545454545455,
      "proof_complete": false,
      "incumbent_ms": 11.243781734366843,
      "remaining_lower_bound_ms": 11.161047173524718,
      "certified_global_lower_bound_ms": 11.161047173524718,
      "gap_pct": 0.7412795551870799,
      "open_frontiers": 57,
      "open_lattice_points": 60
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "A",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.261347663135061,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.212461896006438,
      "open_frontiers": 204,
      "open_lattice_points": 106646251841749620
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "A",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.289680242835185,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.784420495682353,
      "open_frontiers": 204,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "A",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.261347663135061,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.212461896006438,
      "open_frontiers": 204,
      "open_lattice_points": 2352706598170238310
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "A",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.3266498138096345,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 7.530736724877474,
      "open_frontiers": 204,
      "open_lattice_points": 5783725298295945
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "A",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 75,
      "coverage_pct": 56.81818181818182,
      "proof_complete": false,
      "incumbent_ms": 5.356828222055863,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 8.139957639585994,
      "open_frontiers": 55,
      "open_lattice_points": 57
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "B",
      "family": "4+2",
      "declared_lattice_points": 106646251841749620,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.261347663135061,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.212461896006438,
      "open_frontiers": 206,
      "open_lattice_points": 106646251841749620
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "B",
      "family": "5+1",
      "declared_lattice_points": 118495835379721800,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.289680242835185,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.784420495682353,
      "open_frontiers": 206,
      "open_lattice_points": 118495835379721800
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "B",
      "family": "heterogeneous",
      "declared_lattice_points": 2352706598170238310,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.261347663135061,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 6.212461896006438,
      "open_frontiers": 206,
      "open_lattice_points": 2352706598170238310
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "B",
      "family": "homogeneous",
      "declared_lattice_points": 5783725298295945,
      "covered_lattice_points": 0,
      "coverage_pct": 0.0,
      "proof_complete": false,
      "incumbent_ms": 5.3266498138096345,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 7.530736724877474,
      "open_frontiers": 206,
      "open_lattice_points": 5783725298295945
    },
    {
      "onchip_mode": "fixed_issue",
      "proof": "B",
      "family": "single",
      "declared_lattice_points": 132,
      "covered_lattice_points": 76,
      "coverage_pct": 57.57575757575758,
      "proof_complete": false,
      "incumbent_ms": 5.356828222055863,
      "remaining_lower_bound_ms": 4.9536067324063096,
      "certified_global_lower_bound_ms": 4.9536067324063096,
      "gap_pct": 8.139957639585994,
      "open_frontiers": 55,
      "open_lattice_points": 56
    }
  ]
}
```

继续命令（原始结果保留）：

```sh
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_pipelined_A.json --seconds 3600
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_pipelined_B.json --seconds 3600
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_port_tight_A.json --seconds 3600
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_port_tight_B.json --seconds 3600
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_fixed_issue_A.json --seconds 3600
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/bnb_fixed_issue_B.json --seconds 3600
```

## 5.2

已完成范围：

```json
{
  "rows": 1377,
  "expected_rows": 1377,
  "solver_status_counts": {
    "OPTIMAL": 1377
  },
  "visited_seed_assignment_statuses": {
    "pipelined": {
      "OPTIMAL": 10815,
      "FEASIBLE": 255
    },
    "port_tight": {
      "OPTIMAL": 10874,
      "FEASIBLE": 196
    },
    "fixed_issue": {
      "OPTIMAL": 10581,
      "FEASIBLE": 489
    }
  },
  "visited_leaf_statuses": {
    "evaluated/True": 329,
    "evaluated/False": 6,
    "invalid_or_unresolved/False": 12
  },
  "higher_effort_diagnostic": {
    "all_full_object_repeats_identical": true,
    "changed_owner_cases": 218,
    "completed": true,
    "completed_cases": 940,
    "deterministic_work_limit": 1.0,
    "diagnostic_source_sha256": "c4d1fa20d6cfed8c10c1b56344050fbbfc4728db12919714444b8b8153ba6e25",
    "effort_units": 100.0,
    "elapsed_seconds": 532.769666230306,
    "engine_sha256": "943211ec3825b344fa3358dd503f34ca4e452262c4a205b637d52c4b6c05c123",
    "frozen_evidence_sha256": {
      "FROZEN_SELECTION.json": "f461deaba2fce1d86aeaaefe205d8e4ab502ba69ff6a3b890b8dbad7360d0373",
      "bnb_certificate.csv": "f4ca2914c970e0bfa93e833e80a883ae7193c832ced11bf5db01b0e157dbf747",
      "bnb_fixed_issue_A.json": "90fa06ad53eff64bb9d39ec588016a23751962db1a539a0e4daa62f7e4a65245",
      "bnb_fixed_issue_B.json": "35a4a5194f14421abf73c07878226dfff624a1992decde0935532e9714df9300",
      "bnb_leaves.csv": "02be72212df9a8e0cb537d187e1d25abe314ada5d50bfcc7616b3b015194097d",
      "bnb_pipelined_A.json": "0ffa19cfd737c61a4f881478e4059b38cd940f062e2cfdddcde3cc9c95b47962",
      "bnb_pipelined_B.json": "03a267f8e74ecc3db0f524b7e75abd41d2d246a9f236bc520e04656101f45306",
      "bnb_port_tight_A.json": "25827c33b4fb18b6f1c46b94564f3d67cf3e44d6da16555bebcb585d6e7c42dd",
      "bnb_port_tight_B.json": "984f304cce6f53c895ecfb2fab1480078f045a63a8d942e52209c0e83481ad3e",
      "bnb_summary.json": "9dc6faa0dd1c1f44be62c2a06909b12d65469759eb16cb7f6ad5e1159fd11e84",
      "schedule_gaps.csv": "40ad3e1873a96fbf62c499ebc5ba21485f7996291583cd8151e41967c40659c3",
      "schedule_gaps_protocol.json": "b95fb967de74f600f1710604447f9a32d1354e969d5ed0dce55549c739bd3057",
      "seed_leaves.csv": "8fd265d5f5e17d5e2f63519b42da1d715c7a046fef7ead3f7ca36d4fb3195bec",
      "seed_points_fixed_issue.json": "53f929528f789ecbf57d11dc255c87d7ceb838e83db35339c89cae70a841077d",
      "seed_points_pipelined.json": "01ed41419cd6039e4328e25c215eda223d927b69cf8bd5aec28c2d58900e84f9",
      "seed_points_port_tight.json": "cdfb89bc81925e2adb62782511d7607490daecfe98a2e5d22e064dbabc92d549",
      "single_exhaustion_receipt.json": "1a6687b3868134502d2d48ed222949866dcfceacd9bd1c31d419a01389f5913a"
    },
    "frozen_evidence_unchanged": true,
    "full_unresolved_seed_cases": 940,
    "new_status_counts": {
      "FEASIBLE": 718,
      "OPTIMAL": 222
    },
    "old_effort_units": 10.0,
    "output_csv_sha256": "9d3a6e69b650d40be4b26dd08ad4a6f6a0043e791b06ca062beeba8fb503827c",
    "planned_cases": 940,
    "preflight_subset": false,
    "quantum_cycles": 1e-06,
    "query_limit": 64,
    "repeats": 2,
    "scope": "higher finite effort diagnostic; no retroactive headline/proof update",
    "unresolved_cases": 718,
    "unresolved_seed_counts_by_mode": {
      "fixed_issue": 489,
      "pipelined": 255,
      "port_tight": 196
    },
    "workers": 4
  }
}
```

未完成范围的下界/差距：

```json
{
  "scope": "冻结三组织族的 1,377 个调度差距行与主搜索所有访问叶子的内层求解是两个范围；后者未证分配保留可行见证和资源下界。",
  "diagnostic_unresolved_cases": 718
}
```

继续命令（原始结果保留）：

```sh
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.repair_inner --jobs 4 --effort-units 1000 --output-directory research/moe_dispatch/round2/results/E3/inner_effort1000
```

## 5.3

已完成范围：

```json
{
  "grid_rows": 4320,
  "grid_unique_points": 4320,
  "planned_grid_points": 4320,
  "certified_grid_points": 1131,
  "CMA_evaluations": 500,
  "extreme_delta0_proof_complete": false
}
```

未完成范围的下界/差距：

```json
{
  "extreme_open_lower_bound_ms": 1.67844866130092,
  "extreme_gap_pct": 3.9579093888636674,
  "grid_uncertified_points": 3189,
  "pointwise_bounds_and_gaps": "results/E3/workload_map.csv: open_lb_ms, gap_pct, proof_complete; exact open frontiers in search_certificates/grid/<point_index:04d>.json"
}
```

继续命令（原始结果保留）：

```sh
# 新 checkout 先恢复原证书；现有完整原始目录保持不动
if [ ! -d /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/grid ]; then
  for archive in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/certificate_archives/grid/part_*.tar.gz; do
    [ -f "$archive" ] || continue
    tar -xzf "$archive" -C /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3
  done
fi
for certificate in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/grid/*.json; do
  [ -f "$certificate" ] || continue
  case "$certificate" in *_continued.json) continue;; esac
  while [ -f "${certificate%.json}_continued.json" ]; do
    certificate="${certificate%.json}_continued.json"
  done
  /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate "$certificate" --seconds 3600
done
# 从 E3 目录提取原始完整精度检查点；workload_extreme.json 是派生摘要，不用它恢复
if [ ! -f /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/cma_verification/final_delta0.json ]; then
  tar -xzf /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/certificate_archives/extreme_final/part_000.tar.gz -C /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3 cma_verification/final_delta0.json
fi
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/cma_verification/final_delta0.json --seconds 3600
```

## 5.4

已完成范围：

```json
{
  "objective_rows": 396,
  "stability_rows": 396,
  "bootstrap_draw_counts": [
    "200"
  ]
}
```

## 5.5

已完成范围：

```json
{
  "Saltelli_base_N": 256,
  "sample_rows": 1792,
  "expected_sample_rows": 1792,
  "certified_samples": 5
}
```

未完成范围的下界/差距：

```json
{
  "pointwise_bounds_and_gaps": "results/E3/sobol_samples.csv: open_lb_ms, gap_pct, proof_complete; exact open frontiers in search_certificates/sobol/*.json and search_certificates/flip/*.json",
  "uncertified_Saltelli_samples": 1787
}
```

继续命令（原始结果保留）：

```sh
# 新 checkout 先恢复原证书；现有完整原始目录保持不动
if [ ! -d /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/sobol ]; then
  for archive in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/certificate_archives/sobol/part_*.tar.gz; do
    [ -f "$archive" ] || continue
    tar -xzf "$archive" -C /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3
  done
fi
for certificate in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/sobol/*.json; do
  [ -f "$certificate" ] || continue
  case "$certificate" in *_continued.json) continue;; esac
  while [ -f "${certificate%.json}_continued.json" ]; do
    certificate="${certificate%.json}_continued.json"
  done
  /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate "$certificate" --seconds 3600
done
# 新 checkout 先恢复原证书；现有完整原始目录保持不动
if [ ! -d /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/flip ]; then
  for archive in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/certificate_archives/flip/part_*.tar.gz; do
    [ -f "$archive" ] || continue
    tar -xzf "$archive" -C /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3
  done
fi
for certificate in /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/results/E3/search_certificates/flip/*.json; do
  [ -f "$certificate" ] || continue
  case "$certificate" in *_continued.json) continue;; esac
  while [ -f "${certificate%.json}_continued.json" ]; do
    certificate="${certificate%.json}_continued.json"
  done
  /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate "$certificate" --seconds 3600
done
```

## 6

已完成范围：

```json
{
  "main_table_rows": 66,
  "modes": [
    "fixed_issue",
    "pipelined",
    "port_tight"
  ],
  "schedulers": [
    "milp",
    "runtime"
  ]
}
```

## 7

已完成范围：

```json
{
  "dispatch_rows": 288,
  "predictor_rows": 36,
  "state_rows": 36,
  "conditional_oracle_replay_rows": 810
}
```

## 8

已完成范围：

```json
{
  "moe_rows": 168,
  "model_rows": 168,
  "model_token_timing_rows_available": 0
}
```

未完成范围的下界/差距：

```json
{
  "required_input": "matched DeepSeek non-MoE per-layer timing and layer-to-token mapping"
}
```

继续命令（原始结果保留）：

```sh
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E6  # 提供并接入匹配非 MoE 计时后运行
```

## 9–10

已完成范围：

```json
{}
```

## 重复与测试

```json
{
  "repeat_checks": {
    "receipt": {
      "all_configurations_identical": true,
      "caveat": "This is an execution/repetition receipt, not RTL calibration, full search closure, or complete-model timing.",
      "checks": [
        {
          "name": "historical_reproduction",
          "ok": true,
          "rows": 1350
        },
        {
          "name": "regional_bound_audit",
          "ok": true,
          "protocol": {
            "all_ok": true,
            "checks": 36000,
            "concrete_feasible_designs": 2000,
            "development_windows": 18,
            "repeats": 2,
            "sampling_attempts": 2953,
            "seed": 20261007,
            "warning": "Random checks support implementation confidence, not a mathematical proof"
          }
        },
        {
          "declared_points": 636,
          "evaluated_valid_points": 615,
          "name": "seed_pipelined",
          "ok": true
        },
        {
          "declared_points": 636,
          "evaluated_valid_points": 615,
          "name": "seed_port_tight",
          "ok": true
        },
        {
          "declared_points": 636,
          "evaluated_valid_points": 615,
          "name": "seed_fixed_issue",
          "ok": true
        },
        {
          "evidence": "results/executions/E2micro_cold_recompute_20261007T135717Z.json",
          "name": "E2micro_cold_recompute",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_main_search_20261007T135855Z.json",
          "name": "E3_main_search",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_schedule_gaps_20261007T142405Z.json",
          "name": "E3_schedule_gaps",
          "ok": true
        },
        {
          "evidence": "results/executions/E1_20261007T142609Z.json",
          "name": "E1",
          "ok": true
        },
        {
          "evidence": "results/executions/E2layer_20261007T142651Z.json",
          "name": "E2layer",
          "ok": true
        },
        {
          "evidence": "results/executions/E4_20261007T143032Z.json",
          "name": "E4",
          "ok": true
        },
        {
          "evidence": "results/executions/E5_20261007T143441Z.json",
          "name": "E5",
          "ok": true
        },
        {
          "evidence": "results/executions/E6_20261007T145008Z.json",
          "name": "E6",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_grid_20261007T142449Z.json",
          "name": "E3_grid",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_extreme_20261007T145104Z.json",
          "name": "E3_extreme",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_robust_20261007T145915Z.json",
          "name": "E3_robust",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_sobol_20261007T150110Z.json",
          "name": "E3_sobol",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_flip_20261007T152135Z.json",
          "name": "E3_flip",
          "ok": true
        },
        {
          "evidence": "results/executions/E3_inner_assignment_verification_20261007T144406Z.json",
          "name": "E3_inner_assignment_verification",
          "ok": true
        },
        {
          "actual": 4320,
          "expected": 4320,
          "name": "workload_map_coverage",
          "ok": true
        },
        {
          "actual": 1792,
          "expected": 1792,
          "name": "sobol_samples_coverage",
          "ok": true
        },
        {
          "name": "flip_coverage",
          "ok": true,
          "rows": 105
        },
        {
          "cases": 940,
          "name": "higher_effort_inner_repetition",
          "ok": true,
          "unresolved_cases": 718
        }
      ],
      "current_engine_sha256": "943211ec3825b344fa3358dd503f34ca4e452262c4a205b637d52c4b6c05c123",
      "execution_receipt_sha256": {
        "results/executions/E1_20261007T142609Z.json": "22891704e79038e0b210d44d7458a6ebdb785c295b340f4f1a845f3900259234",
        "results/executions/E2layer_20261007T142651Z.json": "af3ddce4aa861633a3dde64efa2ab628dcab7bc91c7f21f6168480910bace7f2",
        "results/executions/E2micro_cold_recompute_20261007T135717Z.json": "c31d995b84cda2d606e8e050eb7b94b951aba95ea0589cbd9b0d90719eb9613a",
        "results/executions/E3_extreme_20261007T145104Z.json": "2a0b13ff38f6ab756b7a28f8e8feb08a960e2f3b6ef69dd65e54777755a78dde",
        "results/executions/E3_flip_20261007T152135Z.json": "c001913f67aab499b7ca56ecc728d463b56e1a6e1aa9f43628aba692ef9870fd",
        "results/executions/E3_grid_20261007T142449Z.json": "5db0175955944c87cfd8a27d0928cb28e4d3f927ea70c4ffc42c1c28d46b6280",
        "results/executions/E3_inner_assignment_verification_20261007T144406Z.json": "43ec9846eed020de1c8cb787f7f155101d883303ceb415ffdc3ad861df1c9c29",
        "results/executions/E3_main_search_20261007T135855Z.json": "6475506a57a3a32fd2a0d7badc1ee9c83ef43b91e291b2967f678e5ae9a4b27c",
        "results/executions/E3_robust_20261007T145915Z.json": "9d20e190c4aa5e80d4f6d192642c7111042e231caa7f8dfa54fa1ff8b1e5fca8",
        "results/executions/E3_schedule_gaps_20261007T142405Z.json": "a1f59122fdefc375c780eec7b1c14d3605a961d26d33c5637206ed174d7a0544",
        "results/executions/E3_sobol_20261007T150110Z.json": "658f9c45ccb50e7f77a9e4162400a4c69f0eb46f2ee6265b922ec62e69a5dede",
        "results/executions/E4_20261007T143032Z.json": "69251bdf5c72593320e076ecebbb2bbf4d5860429a186e153e6071f846910949",
        "results/executions/E5_20261007T143441Z.json": "06a9f638337898e955dfcff5541007ba2a4b30731c5720a490b1e61a2ad1c735",
        "results/executions/E6_20261007T145008Z.json": "35e0727240a3cdb99ba79736e14478ebe805a46a96e6216d6bb765c2a216928c"
      },
      "generated_utc": "2026-10-07T15:23:48.042664+00:00",
      "protocol": "Performance drivers assert equality of two full result objects; E5 repeats warmup and heldout in two fresh identical states. Search wall-clock trajectories need not be identical; witness repetitions are.",
      "scope": "All actually evaluated physical configurations and complete learned predictor sequences; unopened search regions are not certified or claimed evaluated",
      "source_sha256": "c9a91d0c831a764f63a7baf5a9b829cf7ce9c8920eb361718368e1755bfedf7e"
    },
    "historical_cohorts_repeated": true,
    "main_seed_flags": {
      "pipelined": true,
      "port_tight": true,
      "fixed_issue": true
    },
    "protocol": "model evaluations execute deterministic two-run assertions; certificates explicitly record incumbent repeat flags; frontier traversal is time-limited and may visit different nodes",
    "all_configurations_identical_certified": true
  },
  "tests": {
    "required_suites": [
      "research",
      "analytical",
      "rust",
      "main_rust",
      "round2"
    ],
    "latest_by_suite": {
      "research": {
        "command": [
          "/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python",
          "-m",
          "pytest",
          "research/moe_dispatch",
          "-q",
          "--ignore=research/moe_dispatch/archive",
          "--ignore=research/moe_dispatch/results",
          "--ignore=research/moe_dispatch/round2/results",
          "--ignore=research/moe_dispatch/round2/archive",
          "--junitxml=research/moe_dispatch/round2/results/E0/tests/20261007T114345980637Z/research.xml"
        ],
        "errors": 0,
        "failed_tests": [],
        "failures": 0,
        "finished_utc": "2026-10-07T11:48:20.271424+00:00",
        "initial_classification": "passed",
        "junit_sha256": "6ca33ba365add076973eff1c1c4d26ab6d01ab07a7738f3591330de246e98dde",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T114345980637Z/research.log",
        "log_sha256": "b32a6359e650772712c2ff70d9505f33fe21df8dcafbcc784b19f8586be5ef7f",
        "returncode": 0,
        "skipped": 0,
        "started_utc": "2026-10-07T11:43:46.004320+00:00",
        "suite": "research",
        "tests": 281,
        "receipt": "results/E0/tests/20261007T114345980637Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "analytical": {
        "returncode": 0,
        "tests": 478,
        "passed": 469,
        "failures": 0,
        "errors": 0,
        "skipped": 9,
        "skip_details": [
          {
            "node_id": "analytic_models/performance/test_b200_formal_campaign.py::test_local_kda_stage2_independently_matches_the_formal_core_traffic",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_KDA_STAGE2_ROOT to cross-check the optional raw archive",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_b200_formal_campaign.py::test_raw_campaign_rebuilds_the_pinned_contract_and_routing_trace",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_B200_CAMPAIGN_ROOT to rebuild from the optional raw archive",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_gpu_evidence.py::test_local_archives_reproduce_every_imported_file",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "raw GPU archives are not part of a fresh clone",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-fsm-expected0]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-row-expected1]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-fsm-expected2]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-row-expected3]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[mamba-old_isa-expected4]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          },
          {
            "node_id": "analytic_models/performance/test_ltile_cost.py::test_published_r3_component_oracles[kda-old_isa-expected5]",
            "receipt": "tests/analytical_remaining.xml",
            "skip_reason": "set PLENA_E_COMPILER to the pinned E compiler worktree",
            "status": "skipped"
          }
        ],
        "receipt": "results/E0/PHASE1_GATE.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae",
        "aggregation": "exact current node-ID union of broad historical-pin receipt, broad remaining receipt, and full repaired Matrix/formal files; no duplicate test counting",
        "uncovered": 0
      },
      "rust": {
        "command": [
          "/nix/store/qqq4l5xqxm4s009vcdm1p6kic2jga311-rust-default-1.93.1/bin/cargo",
          "test",
          "--locked",
          "--manifest-path",
          "research/moe_dispatch/rust/Cargo.toml",
          "--target-dir",
          "/tmp/plena-round2-cargo-target",
          "--",
          "--test-threads=2"
        ],
        "finished_utc": "2026-10-07T11:03:19.400537+00:00",
        "initial_classification": "passed",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T110253385260Z/rust.log",
        "log_sha256": "ec00fab049eb198b9800ce04b9c991380e60a8126101b03ef143e1e51e4536ce",
        "returncode": 0,
        "started_utc": "2026-10-07T11:02:53.400831+00:00",
        "suite": "rust",
        "receipt": "results/E0/tests/20261007T110253385260Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "main_rust": {
        "command": [
          "/nix/store/qqq4l5xqxm4s009vcdm1p6kic2jga311-rust-default-1.93.1/bin/cargo",
          "test",
          "--locked",
          "--manifest-path",
          "transactional_emulator/Cargo.toml",
          "--workspace",
          "--target-dir",
          "/tmp/mcl123-plena-layout-async-target",
          "--",
          "--test-threads=2"
        ],
        "finished_utc": "2026-10-07T10:55:35.509722+00:00",
        "initial_classification": "passed",
        "log": "research/moe_dispatch/round2/results/E0/tests/20261007T105511300902Z/main_rust.log",
        "log_sha256": "8aa4a3a26a8fde392d3270794e11fe0f2e3f4ea0cb45d07656164f53b59fb87c",
        "returncode": 0,
        "started_utc": "2026-10-07T10:55:11.316392+00:00",
        "suite": "main_rust",
        "receipt": "results/E0/tests/20261007T105511300902Z/UNIT_CHECKS.json",
        "commit": "6130f924c3417ffda19cd0bf4c98d6f4c10b3cae"
      },
      "round2_subset_round2_delivery_metadata_unit_suite": {
        "returncode": 0,
        "tests": 9,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_delivery_metadata_unit_suite_20261007T152454Z.json",
        "commit": "2c95619bfc0d9130afdfb8b353a10e4ca9901e44",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      },
      "round2": {
        "returncode": 0,
        "tests": 126,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_final_engine_unit_suite_20261007T135716Z.json",
        "commit": "045161dd7e9c8fa44f4ef421cd324cfa3a0cb007",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      },
      "round2_subset_round2_final_reader_unit_suite": {
        "returncode": 0,
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "receipt": "results/executions/round2_final_reader_unit_suite_20261007T145951Z.json",
        "commit": "cdd193e966b9ca35c23a88ca55631e87a8474581",
        "source_hash_checked": [
          "model.py",
          "optimizer.py",
          "search.py",
          "run.py",
          "predictors.py",
          "oracle_replay.py",
          "regions.py",
          "sensitivity.py"
        ]
      }
    },
    "all_required_recorded_suites_passed": true,
    "phase1_gate_validated": true,
    "gate_receipt_hash_checks": {
      "compiler_integration/COPY_MANIFEST.json": true,
      "compiler_integration/TEST_RESULTS.md": true,
      "compiler_integration/tests/evidence_manifest.json": true,
      "post_integration_replay_20261007/reproduction_receipt.json": true,
      "tests/20261007T105511300902Z/UNIT_CHECKS.json": true,
      "tests/20261007T110253385260Z/UNIT_CHECKS.json": true,
      "tests/20261007T114345980637Z/UNIT_CHECKS.json": true,
      "tests/analytical_collection_current.log": true,
      "tests/analytical_full_historical_pin.xml": true,
      "tests/analytical_gate_coverage.csv": true,
      "tests/analytical_remaining.xml": true,
      "tests/live_golden_fix.xml": true,
      "tests/live_golden_full_files.xml": true
    },
    "gate_source_hash_checks": {
      "analytic_models/performance/hybrid_lcompute_campaign.py": true,
      "analytic_models/performance/profiles/historical_hybrid_compiler_evidence.json": true,
      "analytic_models/performance/test_matrix_lcompute_campaign.py": true,
      "analytic_models/performance/test_nemotron3_formal_dse.py": true,
      "research/moe_dispatch/round2/run_unit_checks.py": true
    },
    "all_supported_current_tests_passed": true,
    "all_collected_tests_executed_without_skips": false,
    "compiler_gate": {
      "additional_final_lifetime_tests_passed": 41,
      "affected_final_integration_passed": 45,
      "current_tvm_passed": 143,
      "evidence": "compiler_integration/TEST_RESULTS.md",
      "historical_precision_ablation_passed": 1,
      "ordinary_passed": 1143,
      "ordinary_retired_profile_skipped": 29,
      "ordinary_subtests_passed": 11,
      "research_passed": 33,
      "research_subtests_passed": 3,
      "scope": "Counts are suite-specific and overlap; do not add reruns to unique total. Archived-profile skips and numerical-precision correction are explicit in TEST_RESULTS.md.",
      "slow_connected_passed": 3,
      "slow_connected_retired_profile_skipped": 1,
      "tvm_retired_demos_skipped": 5,
      "tvm_standalone_entrypoints_passed": 13
    },
    "phase1_limitations": [
      "No new checkpoint inference or GPU benchmarks for round2 BF16 MoE.",
      "Optional raw GPU archive rebuilds and six pinned historical-E-Compiler oracles remain explicitly skipped.",
      "Existing archive native integration binary is hash/source checked; fresh Rust unit builds are separately recorded.",
      "New-model searches, native timing calibration and synthesis/PPA are outside this phase-1 gate."
    ]
  }
}
```
