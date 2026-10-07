# Portable search certificates

All original JSON bytes, complete open frontiers, workloads, parameters and repeated witnesses are preserved in independent gzip-compressed tar archives. Each chunk is below 45 MiB. `manifest.json` records every original relative path and SHA-256, each chunk hash, and the verified complete roundtrip.

From the E3 directory, restore the JSON names used by the existing resume command:

```sh
for archive in certificate_archives/sobol/part_*.tar.gz; do tar -xzf "$archive" -C .; done
```

Use `python -m research.moe_dispatch.round2.resume --help` for the unchanged continuation interface after extraction. The numerical model and search drivers were not changed for archiving.
