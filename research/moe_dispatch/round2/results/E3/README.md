# E3 delivery evidence

This is a metadata refresh, not a simulation or reproduction rerun. Original producer text and hash snapshots are preserved verbatim in [RUN_README.md](RUN_README.md) and [PROVENANCE.execution.csv](PROVENANCE.execution.csv) when originally present.

Metadata generation commit: `8247e7c9ccc0db622eb2dbf378612581ef215f4b`. Frozen input manifest SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

BF16 only; 18 development and 135 held-out windows. New performance results cover post-router MoE Gate/Up, SiLU/Z, Down and combine; one hypothetical cycle = 1 ns, ms = cycles / 1e6. The phase-fluid/shared-credit analytical model is not RTL or native HBM calibration. E0 contains separately identified historical reproduction and test evidence. Search process completion is not proof closure. Complete-model timing remains unavailable without matching non-MoE timings.

Actual execution commits, exact commands, source/input hashes and original receipts are listed in [ACTUAL_EXECUTIONS.json](ACTUAL_EXECUTIONS.json). Old successful source versions are retained there but are not silently assigned as final numerical producers.

## E3_main_search

Execution commit: `045161dd7e9c8fa44f4ef421cd324cfa3a0cb007`. Receipt SHA-256: `6475506a57a3a32fd2a0d7badc1ee9c83ef43b91e291b2967f678e5ae9a4b27c`.

Actual successful receipt: [E3_main_search_20261007T135855Z.json](../executions/E3_main_search_20261007T135855Z.json); exit code 0; completed `2026-10-07T14:22:56.102282+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.main_search --stage search --jobs 16 --search-seconds 120
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "29281e5c9320626aa6ef8ead237091b460d62a4a303dd7cb3126bcff2f5128de",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "44f601132a4b6edef39501f3dfa981fb1396c7761c34fba9189cf25badc0c21f",
  "final_receipts.py": "8f6a8e919c5b10a5f949e9b536295246a006c46cdf781fe1e535ac6041ec33e7",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "report.py": "95d563b64df7ea91273401bf7a407bbcf2b6d448a8075cf9dfb1715fc369a85c",
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

## E3_lb_validity

Execution commit: `045161dd7e9c8fa44f4ef421cd324cfa3a0cb007`. Receipt SHA-256: `7d052edaa99fcefe21b935f1a7f186f74b4339afecadde97e7e6ec83e5616124`.

Actual successful receipt: [E3_lb_validity_20261007T135840Z.json](../executions/E3_lb_validity_20261007T135840Z.json); exit code 0; completed `2026-10-07T14:03:30.226057+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.main_search --stage validity --jobs 24
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "29281e5c9320626aa6ef8ead237091b460d62a4a303dd7cb3126bcff2f5128de",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "44f601132a4b6edef39501f3dfa981fb1396c7761c34fba9189cf25badc0c21f",
  "final_receipts.py": "8f6a8e919c5b10a5f949e9b536295246a006c46cdf781fe1e535ac6041ec33e7",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "report.py": "95d563b64df7ea91273401bf7a407bbcf2b6d448a8075cf9dfb1715fc369a85c",
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

## E3_schedule_gaps

Execution commit: `cd1b7554f7595080656e3bdf3a8c4c2b101a6f54`. Receipt SHA-256: `a1f59122fdefc375c780eec7b1c14d3605a961d26d33c5637206ed174d7a0544`.

Actual successful receipt: [E3_schedule_gaps_20261007T142405Z.json](../executions/E3_schedule_gaps_20261007T142405Z.json); exit code 0; completed `2026-10-07T14:24:23.072601+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.main_search --stage gaps --jobs 16
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "29281e5c9320626aa6ef8ead237091b460d62a4a303dd7cb3126bcff2f5128de",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "8a6906f1ef336e7af6faff8a00a0abb29d78f84e2525d1f87effbd10f2905189",
  "final_receipts.py": "8f6a8e919c5b10a5f949e9b536295246a006c46cdf781fe1e535ac6041ec33e7",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "report.py": "b45a71a0efcfc24759cc6d433e56366955e5bca7e6687f7164ad0ef07b048b62",
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

## E3_grid

Execution commit: `cd1b7554f7595080656e3bdf3a8c4c2b101a6f54`. Receipt SHA-256: `5db0175955944c87cfd8a27d0928cb28e4d3f927ea70c4ffc42c1c28d46b6280`.

Actual successful receipt: [E3_grid_20261007T142449Z.json](../executions/E3_grid_20261007T142449Z.json); exit code 0; completed `2026-10-07T14:40:06.880438+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.regions --stage grid --jobs 24 --point-seconds 2
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "29281e5c9320626aa6ef8ead237091b460d62a4a303dd7cb3126bcff2f5128de",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "8a6906f1ef336e7af6faff8a00a0abb29d78f84e2525d1f87effbd10f2905189",
  "final_receipts.py": "8f6a8e919c5b10a5f949e9b536295246a006c46cdf781fe1e535ac6041ec33e7",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "report.py": "b45a71a0efcfc24759cc6d433e56366955e5bca7e6687f7164ad0ef07b048b62",
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

## E3_extreme

Execution commit: `cdd193e966b9ca35c23a88ca55631e87a8474581`. Receipt SHA-256: `2a0b13ff38f6ab756b7a28f8e8feb08a960e2f3b6ef69dd65e54777755a78dde`.

Actual successful receipt: [E3_extreme_20261007T145104Z.json](../executions/E3_extreme_20261007T145104Z.json); exit code 0; completed `2026-10-07T14:59:09.241635+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.extreme --jobs 24 --max-evaluations 500 --point-seconds 2 --verify-seconds 120
```

Captured source SHA-256 snapshot:
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

## E3_robust

Execution commit: `cdd193e966b9ca35c23a88ca55631e87a8474581`. Receipt SHA-256: `9d20e190c4aa5e80d4f6d192642c7111042e231caa7f8dfa54fa1ff8b1e5fca8`.

Actual successful receipt: [E3_robust_20261007T145915Z.json](../executions/E3_robust_20261007T145915Z.json); exit code 0; completed `2026-10-07T15:01:06.159813+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.robust --jobs 28
```

Captured source SHA-256 snapshot:
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
  "restore_inputs.py": "92d07cbc624814ef3545c7a58bb1def4c1c8c938d6cc2e7d8d488e28ceac625a",
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

## E3_sobol

Execution commit: `fcca9c3af7da3e317db8b305e1c4b55b6babb256`. Receipt SHA-256: `658f9c45ccb50e7f77a9e4162400a4c69f0eb46f2ee6265b922ec62e69a5dede`.

Actual successful receipt: [E3_sobol_20261007T150110Z.json](../executions/E3_sobol_20261007T150110Z.json); exit code 0; completed `2026-10-07T15:21:30.484265+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.regions --stage sobol --jobs 28 --point-seconds 2
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "849fe6ec1832f1c1d78e2cd56109b949b0021fcce23b1f1cdf399e9a8b35e1b7",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "bebfd112b51752b84c47538db8e89b4bab76e5a9c07ee0ca94d047866ed16d61",
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
  "report.py": "d247320fe8800d6128d7abdf58bc051da2037f1c0ba5d1da3e9fb3a56760029b",
  "reproduce_e0.py": "10924fe0e87872992dbdd9ac26349a5bd31d85ada9dc1fa9eedfbae0ac94fd3f",
  "reproduce_historical_compiler_fixture.py": "c3983204e62b2d9720a3a5db0459ea17f13160440388dc3ca67b08ce16a62ddd",
  "restore_inputs.py": "92d07cbc624814ef3545c7a58bb1def4c1c8c938d6cc2e7d8d488e28ceac625a",
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

## E3_flip

Execution commit: `5d0855e7127740d0d6d5baf8100a16c5f048b31c`. Receipt SHA-256: `c001913f67aab499b7ca56ecc728d463b56e1a6e1aa9f43628aba692ef9870fd`.

Actual successful receipt: [E3_flip_20261007T152135Z.json](../executions/E3_flip_20261007T152135Z.json); exit code 0; completed `2026-10-07T15:22:37.435668+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.regions --stage flip --jobs 28 --point-seconds 2
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "849fe6ec1832f1c1d78e2cd56109b949b0021fcce23b1f1cdf399e9a8b35e1b7",
  "common.py": "3342422c57757019415473f3af15a2bb67b6398d165c767254e860858eb31d61",
  "conftest.py": "2432741d02c4b91029171d13ef564c6dfd6a023a275c741ac979ebedf1ee99d6",
  "execute.py": "c98661b9ec8a3d4389896b15774cafcbb9c0ea65bdeb0f01966d2578341aecfd",
  "extreme.py": "d8f34f96660e3373c510b20204541c8d262b1fea98e3178c4e5d7bdd97ee8804",
  "figures.py": "bebfd112b51752b84c47538db8e89b4bab76e5a9c07ee0ca94d047866ed16d61",
  "final_receipts.py": "c9a91d0c831a764f63a7baf5a9b829cf7ce9c8920eb361718368e1755bfedf7e",
  "finalize_delivery.py": "f26f6ae32b7bbe3c410f04069dd601a52ceb2bdd1435c1a8518c280d01cb41ae",
  "heldout_runner.py": "216d5797ce33ed31f194a96f383b87dbe8dd6d562d0231f9f6a9ea2e98a7a936",
  "main_search.py": "31d3b95ba6cdac60c15d3ed61a41196f126c448083a62cfc7a91c308f3410b3d",
  "model.py": "3ab8b0d33aaa57a387c2665229ebd21d5bd1b2d5d485424252fffaf394a0e2ca",
  "optimizer.py": "eb6f0de8f8411ce5f1758e24b4e077d39a88d858043e1e30dbfa333f5203f73d",
  "oracle_replay.py": "3cdb013bb5e704a3f4c999eb10573f300834ac1b8fa9dc92b5db8b7c75c4389b",
  "predictors.py": "bdd7b3c735368e4f5f7495ef2e83f300c8a45d3147b93eac19272d2a532ce2b2",
  "profile_inputs.py": "593e63fe27ef302d1e022486d1522f49427364c12a3145b576a6d42fca7a4278",
  "regions.py": "313a2032b630cb0a5343f9c19759eca2d6ad98daab3bb364dcec7f62ab70f346",
  "repair_inner.py": "c4d1fa20d6cfed8c10c1b56344050fbbfc4728db12919714444b8b8153ba6e25",
  "report.py": "745788d72460ea7aa19b942f194716b021a8c4830b991dbdc65bf7c0c0c3a59b",
  "reproduce_e0.py": "10924fe0e87872992dbdd9ac26349a5bd31d85ada9dc1fa9eedfbae0ac94fd3f",
  "reproduce_historical_compiler_fixture.py": "c3983204e62b2d9720a3a5db0459ea17f13160440388dc3ca67b08ce16a62ddd",
  "restore_inputs.py": "92d07cbc624814ef3545c7a58bb1def4c1c8c938d6cc2e7d8d488e28ceac625a",
  "resume.py": "08753bee95f60f2546b2cc6364f1d615ae0cd4f6140688705f2ec179aa1e5a15",
  "robust.py": "a3c4164b561def1958e1f2a827036126aec381e8236b319bf00d531ff2e3f482",
  "run.py": "970ac171f86afd66c644ae80c7dededae32539f8087b78cafbeb4ffeb3806622",
  "run_unit_checks.py": "30b9af2dbd34bd94cb459b18673a896d129519bf5b87f101284a0c1703e7d076",
  "search.py": "f8f40f99a0d246212869e6baa19019241a70f6d8c05094c739e138f806458710",
  "sensitivity.py": "3e5391729e908545456c9e38e7244af00b9109216e604681375ea4da57c20d89",
  "test_campaign_contract.py": "f68bfeae4c15fd92552b2a90c5a79bfee2c3efeb6d861eff187117c069bc140e",
  "test_figures.py": "6ef21780b94dd59d1342dace52b09104362586b49c75ff5191f58701832c1f5b",
  "test_finalize_delivery.py": "50df470dd4f8ae87581190ba070e5b04771bd8c64b0a172407346936f49aa905",
  "test_model.py": "c97784538bc9ec23c788908caf54199147df2674d38f85b62ddfcaace09dccae",
  "test_optimizer.py": "92a3eef602e5ab20b3f98b1ed50e2392759e2eb13d6ced260157835a5bb3e943",
  "test_oracle_replay.py": "d497cc4215bc00488ec6a41fdf2b2621af104edd16f9f3fdd7fba4381156aa74",
  "test_regions.py": "e9032213e39546c50f7b4f4dc5b19148ff26529ad5ec617bb86f4c2b6fe60325",
  "test_robust.py": "594b4ab3a84ce9c719869cd7d651ad97a0e16a8dad9fe5bfbaa6276c4bcc235a",
  "test_search.py": "63cd6f73a78adcfe7798768eaf87d5150bc2721771e661c7b0e42b2c8c4f16c5",
  "test_sensitivity.py": "7744447403a6a2a8473c7241503a19f0ffca5d613c676106d4f91aceec587a78"
}
```

## E3_inner_assignment_verification

Execution commit: `cdd193e966b9ca35c23a88ca55631e87a8474581`. Receipt SHA-256: `43ec9846eed020de1c8cb787f7f155101d883303ceb415ffdc3ad861df1c9c29`.

Actual successful receipt: [E3_inner_assignment_verification_20261007T144406Z.json](../executions/E3_inner_assignment_verification_20261007T144406Z.json); exit code 0; completed `2026-10-07T14:53:00.612963+00:00`. Frozen input SHA-256: `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

```sh
env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=20261007 /tmp/plena-round2-venv/bin/python -u -m research.moe_dispatch.round2.repair_inner --jobs 4 --effort-units 100
```

Captured source SHA-256 snapshot:
```json
{
  "__init__.py": "f6583f8e03d088078e11a1e9c62314fb009e01d3e6c169b71bb9cebdb707718b",
  "audit_tables.py": "29281e5c9320626aa6ef8ead237091b460d62a4a303dd7cb3126bcff2f5128de",
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

## Output inventory and portable references

[PROVENANCE.csv](PROVENANCE.csv) hashes every regular delivered file recursively, including audits, preserved producer snapshots and archives. `execution_commit` retains known original producer commits; `metadata_generation_commit` identifies this refresh separately. Metadata/audit files are not claimed to have performed numerical execution. The manifest excludes its own self-referential hash; the outer delivery receipt hashes that manifest.

[DELIVERY_ARCHIVE_MANIFEST.json](DELIVERY_ARCHIVE_MANIFEST.json) explicitly records raw/tmp/build/cache trees and symbolic-link dependencies that are not traversed or bundled. Referenced external payloads remain external; no incomplete bundle is described as self-contained.

Large original driver outputs may be stored as exact archive members while their old paths hold compact derived summaries. `sha256` always hashes the currently delivered file; `original_driver_output_sha256`/`archive_member_sha256` identify the exact original bytes. A derivative or compressed container is not assigned the historical numerical execution commit as its own producer. Linked archive indexes retain original execution receipts and restoration commands.

Frozen input source file hashes:
```json
{
  "development.json": "168522072cb2fd80e2689af9ad370b25fbad022072f4abc94e4aa512223c95e0",
  "heldout.json": "0628f6e1e34bb936b0bac9be6682263ce53c5f313e3c776b26d5856f9e96191a",
  "mixed_development.json": "c2c4feb51d83abf6b8d224be6bafb9cc9432797f2740eac40ce76503cc8f3f55",
  "mixed_heldout.json": "2910033d86a3346a55ddfdf8781b86039a3726e07515581820bf92f03d233b8f"
}
```
