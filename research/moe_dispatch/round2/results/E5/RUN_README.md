# Reproducible round-two evidence

Execution commit: `cdd193e966b9ca35c23a88ca55631e87a8474581`. BF16 only. 1 hypothetical cycle = 1 ns; ms = cycles / 1e6.

Boundary: post-router MoE Gate/Up, SiLU/Z, Down, combine. Phase-fluid analytical estimates, not RTL/native HBM measurements.

Command:
```sh
env PYTHONHASHSEED=20261007 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E5 --jobs 16 --onchip-mode all
```

Source hashes:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "849fe6ec1832f1c1d78e2cd56109b949b0021fcce23b1f1cdf399e9a8b35e1b7",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "a6ed6b80da0c5079a6111acad50c9841158da1993c4c7228d7c4ed0cd744da94",
  "final_receipts.py": "8f6a8e919c5b10a5f949e9b536295246a006c46cdf781fe1e535ac6041ec33e7",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "repair_inner.py": "c4d1fa20d6cfed8c10c1b56344050fbbfc4728db12919714444b8b8153ba6e25",
  "report.py": "2ba4e209f5aab56578b8cf1640563042d1e0327ffc90d9ae17e38aa551b979ee",
  "reproduce_e0.py": "10924fe0e87872992dbdd9ac26349a5bd31d85ada9dc1fa9eedfbae0ac94fd3f",
  "reproduce_historical_compiler_fixture.py": "c3983204e62b2d9720a3a5db0459ea17f13160440388dc3ca67b08ce16a62ddd",
  "resume.py": "08753bee95f60f2546b2cc6364f1d615ae0cd4f6140688705f2ec179aa1e5a15",
  "robust.py": "a3c4164b561def1958e1f2a827036126aec381e8236b319bf00d531ff2e3f482",
  "run.py": "970ac171f86afd66c644ae80c7dededae32539f8087b78cafbeb4ffeb3806622",
  "run_unit_checks.py": "30b9af2dbd34bd94cb459b18673a896d129519bf5b87f101284a0c1703e7d076",
  "search.py": "f8f40f99a0d246212869e6baa19019241a70f6d8c05094c739e138f806458710",
  "sensitivity.py": "3e5391729e908545456c9e38e7244af00b9109216e604681375ea4da57c20d89",
  "test_campaign_contract.py": "f68bfeae4c15fd92552b2a90c5a79bfee2c3efeb6d861eff187117c069bc140e",
  "test_figures.py": "6ef21780b94dd59d1342dace52b09104362586b49c75ff5191f58701832c1f5b",
  "test_model.py": "c97784538bc9ec23c788908caf54199147df2674d38f85b62ddfcaace09dccae",
  "test_optimizer.py": "92a3eef602e5ab20b3f98b1ed50e2392759e2eb13d6ced260157835a5bb3e943",
  "test_oracle_replay.py": "d497cc4215bc00488ec6a41fdf2b2621af104edd16f9f3fdd7fba4381156aa74",
  "test_regions.py": "e9032213e39546c50f7b4f4dc5b19148ff26529ad5ec617bb86f4c2b6fe60325",
  "test_robust.py": "594b4ab3a84ce9c719869cd7d651ad97a0e16a8dad9fe5bfbaa6276c4bcc235a",
  "test_search.py": "63cd6f73a78adcfe7798768eaf87d5150bc2721771e661c7b0e42b2c8c4f16c5",
  "test_sensitivity.py": "7744447403a6a2a8473c7241503a19f0ffca5d613c676106d4f91aceec587a78"
}
```

Frozen input hashes:
```json
{
  "development.json": "168522072cb2fd80e2689af9ad370b25fbad022072f4abc94e4aa512223c95e0",
  "heldout.json": "0628f6e1e34bb936b0bac9be6682263ce53c5f313e3c776b26d5856f9e96191a",
  "mixed_development.json": "c2c4feb51d83abf6b8d224be6bafb9cc9432797f2740eac40ce76503cc8f3f55",
  "mixed_heldout.json": "2910033d86a3346a55ddfdf8781b86039a3726e07515581820bf92f03d233b8f"
}
```

MAE=平均|预测时长−实测时长|/实测时长。成功窗口为Current结束前两个权重块的近似计算服务时间；WS每块服务ceil(Me/PM)个M发射，OS/IS每次装入服务一个M发射。排除HBM/端口，不能称精确tile时序。late=第一块晚于Current结束；stall为暴露权重等待，不与端口占用相加。oracle在第二遍以共享HBM、私有端口和有限W槽重放ours第一遍的固定归属/绑定/预取/相位释放，重新算完成时间并核对，不强制写MAE=0；保留浮点残差。其E2E等于固定ours计划，不能作为完美预测派工的性能上界；未选择核的反事实时长不在此oracle定义内。旧两遍profile-guided先知另存profile_guided_reference.csv，允许残差与改变调度。状态账本仅为可量化状态，不声称综合面积。
