# Main-search summary payload

The required `bnb_summary.json` is an explicitly derived compact index preserving all aggregate fields and the six proof summaries. Its empty `runs` list makes the existing `report.proof_runs` reader load the six intact individual proof JSON files, including every original frontier and leaf. Those individual files and their resume paths are unchanged. The complete original 85.6 MB driver output is preserved byte-for-byte in this independent archive. SHA-256 values for the original output, derived index, archive, individual certificates, and successful main execution receipt are recorded separately.

All six summary runs were checked for equality to their individual proof files, current engine, frozen 18-development workload hash, saved parameter reconstruction and every accepted/incumbent repeat flag. The entire archive passed a bytewise roundtrip and the report reader returned the identical full proof objects after compaction. This packaging does not rerun or alter numerical optimization.

From the E3 directory, restore the exact original summary (overwriting the compact index):

```sh
tar -xzf certificate_archives/main_summary/part_000.tar.gz -C . bnb_summary.json
```

The six existing `bnb_{pipelined,port_tight,fixed_issue}_{A,B}.json` files retain the existing continuation paths. For example, from the repository root:

```sh
python -m research.moe_dispatch.round2.resume --certificate research/moe_dispatch/round2/results/E3/bnb_pipelined_B.json --seconds 3600
```
