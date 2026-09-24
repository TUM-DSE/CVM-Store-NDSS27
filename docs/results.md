# Results, parsing and plotting

Where runs land under `bench-result/`, and the parse and plot tasks that turn them into the CSV tables and figures of the paper.

## Output layout

Results land under `bench-result/`, one tree per benchmark family, following
the OmaArvio action lifecycle (one timestamp per run, shared with SAR/perf/VFIO
trace files):

```
bench-result/
├── accelstore/{setup}-disk-{size}/{client}/{job}/…       run-all, run-epoch-sweep, run-ablation
├── bench/{setup}-disk-{size}/{arm}/{personality}/…       run-bench
├── ycsb/{setup}-disk-{size}/{arm}/{point}/…              run-ycsb
├── redis/{config}/{arm}/{point}/…                        run-redis
├── mlperf/…                                              run-mlperf
├── fio/{config}/{job}_bs512/…                            run-fio-iops, run-fio-bandwidth
├── {tcrypt,openssl}/{config}/…                           run-crypto
├── attestation/snp-disk-{size}/{timestamp}_*/…           attestation.run
├── parsed/                                               exported CSVs and trace JSON
└── plots/                                                the figures
```

Within a workload directory, variant runs are kept apart by path components so
no row can overwrite a differently-configured one: `sync-…` (metadata
persistence mode), `abl-…` (ablation arm), `io-…` (kernel-bypass arm, above
the other two), `pipe-…` (chunk-pipeline geometry), and on the bench path a
variant leaf naming `direct`/`buffered`, `threads-N`, `split-loops`,
`qd-matched`, `intra-op-<n>`, durability and fsync modes. The workload
directory itself holds the baseline. Next to every log, the actions write a
`*_flags.json` sidecar recording what the arm actually ran. The parsers merge
it so a CSV row carries its own configuration, not what its directory name
claims.

## Parsing and plotting

All parse/plot tasks operate on `bench-result/`. Figures land in
`bench-result/plots/`, exported tables in `bench-result/parsed/`. The `paper.*`
tasks chain them: each figure's parse and plot tasks run after its sweeps, and
`paper.<figure> --plot-only` is the parse-and-plot half alone.

| Task | Reads | Writes |
|---|---|---|
| `accelstore.parse-bench` | `bench-result/bench` | `parsed/bench-summary.csv`, `parsed/bench-flowops.csv`, the newest run per point. `--all-runs` for every run ever taken |
| `accelstore.parse-ablation` | `bench-result/accelstore` | `parsed/ablation-summary.csv` + relative/absolute/breakdown tables |
| `accelstore.parse-mlperf` | `bench-result/mlperf` | `parsed/mlperf-checkpointing.csv` |
| `accelstore.parse-redis` | `bench-result/redis` | `parsed/redis-summary.csv`, `parsed/redis-tests.csv` |
| `accelstore.parse-ycsb` | `bench-result/ycsb` | `parsed/ycsb-summary.csv`, `parsed/ycsb-kinds.csv` |
| `accelstore.parse-seal-unseal` | `bench-result/accelstore` | `parsed/seal-unseal-summary.csv` |
| `accelstore.parse-fio-bandwidth` | `bench-result/fio` | `parsed/fio-bandwidth.csv` |
| `accelstore.parse-crypto` | `bench-result/tcrypt`, `bench-result/openssl` | `parsed/crypto-summary.csv` |
| `accelstore.parse-amplification` | `bench-result/accelstore` | `parsed/amplification.csv`, API vs bdev bytes and the read/write/total ratio of the newest trace per leaf (the job directory as `default`, each `sync-…`/`abl-…` leaf as its own row) |
| `accelstore.parse-trace <json>` | one trace JSON | console metrics, optional `--output-json` |

The paper's six figures, one plot task each
(`modules/accelstore/tasks/plots/`, one file per figure, the two fio
figures share `fio.py`):

| Task | Figure | Output |
|---|---|---|
| `accelstore.plot-ablation` | the single-panel ablation figure | `bench-result/plots/ablation.pdf` |
| `accelstore.plot-synthetic` | YCSB-A beside its per-op GiB/s, the FS/LOG size sweep, the FS random reads, one three-panel stack | `bench-result/plots/synthetic.pdf` |
| `accelstore.plot-apps` | checkpointing + redis + personalities, one column | `bench-result/plots/apps.pdf` |
| `accelstore.plot-epoch` | overhead vs the durability window, plus the overhead grid as a table | `bench-result/plots/epoch-overhead.pdf` |
| `accelstore.plot-fio-iops` | fio IOPS on host, VM and CVM, libaio beside SPDK | `bench-result/plots/fio-iops_iops.pdf` |
| `accelstore.plot-fio-luks` | fio IOPS of the LUKS ciphers against plain libaio, VM beside CVM | `bench-result/plots/fio-luks_iops.pdf` |

The plot tasks take `--fill-missing` to mark cells with no run so the full
layout can be judged, and `--png` where a PNG next to the PDF is useful. The
two fio tasks take only `--output-dir` (their file names carry the plot
module's `_iops` suffix).

