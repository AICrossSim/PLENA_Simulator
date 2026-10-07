# Original report-reader metadata

This directory preserves the exact initial reader output before the compact delivery index removed duplicated full proof bodies. It is metadata packaging, not a new performance experiment. The actual producer receipt is `../executions/round2_final_report_20261007T153827Z.json`, execution commit `8247e7c9ccc0db622eb2dbf378612581ef215f4b`, frozen input SHA `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`.

`manifest.json` records the original 85,685,575-byte JSON, its exact SHA, archive/member names and successful bytewise roundtrip. `REPORT_ZH_verified_initial.md` is the exact report independently audited in `../FINAL_RENDERED_REPORT_AUDIT.json`. The final report differs only in its generation-commit label. `../FINAL_PACKAGE_AUDIT.json` verifies unchanged scientific sections, test/repeat evidence and proof scalar fields in the compact index.

To inspect the original full metadata without replacing the current compact delivery index, extract into a separate directory:

```sh
mkdir -p /tmp/plena-round2-report-reader-original
tar -xzf research/moe_dispatch/round2/results/report_reader_archives/initial_full_delivery_status.tar.gz -C /tmp/plena-round2-report-reader-original
```

Complete proof/frontier payloads remain in the separately SHA-linked E3 certificate files and archives; the compact index does not replace them.
