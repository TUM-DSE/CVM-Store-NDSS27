# Benchmark families

Every `oma-arvio accelstore.*` run task, what it measures and the flags that shape a sweep. The [README](../README.md) lists the `paper.*` tasks that chain them. Every run task boots one VM per point, keeps going past a failed point, and takes `--hostname` to run under another host's configuration.

## Common axes

**Crypto build variants.** The CVM-Store clients exist in three builds. The build is a prefix of the client binary, and that prefix keeps the arms apart in `bench-result/` and on the guest `PATH`.

| `--crypto` | Prefix | Build |
|---|---|---|
| `aegis` | none | AEGIS-128X4, the default |
| `aesgcm` | `aesgcm_` | AES-256-GCM |
| `none` | `no_` | `DISABLE_CIF`: no encryption, integrity or freshness |

`--crypto` takes a comma-separated list or `all` and defaults to `aegis` on every run task. `--clients` takes unprefixed names, and every client runs in every build named.

**Block stacks.** The POSIX baselines run on the same NVMe as CVM-Store, formatted per run. One table, `tasks/storage_stacks.py`, declares them for `run-bench --backends`, `run-ycsb --arms`, `run-redis --arms` and `run-mlperf --stacks`.

| Stack | What it is |
|---|---|
| `ext4` | ext4 on the raw NVMe, O_DIRECT |
| `ext4-aes-xts-plain` | ext4 on dm-crypt with AES-XTS |
| `rollbaccine` | ext4 on Rollbaccine, f=0 |
| `rollbaccine-f1m` | ext4 on Rollbaccine, f=1 with a local checksum replica |

A Rollbaccine arm needs a capacity window (`--stack-capacity-gb`), since its checksum tree lives in guest RAM.

**Metadata persistence.** CVM-Store's durability and freshness window is a client setting, shared by `run-all`, `run-bench`, `run-ycsb` and `run-mlperf` as one `--sync-*` group.

| Flag | Meaning |
|---|---|
| `--sync-trigger` | what starts an epoch: `on-shutdown` (default), `on-barrier`, `every-n`, `every-t`, `per-op` |
| `--sync-every-n`, `--sync-interval-us` | the operand of `every-n` and `every-t` |
| `--sync-blocking` | hold mutators until the epoch covering them is durable |
| `--sync-metadata-only` | persist metadata without a counter commit |
| `--tc-backend mock-remote --tc-rtt-us <n>` | an asynchronous counter service with a round trip behind it |

A run's mode names its result directory, e.g. `sync-every-t-101000us_tc-mock-remote_rtt-5000us/`, below the workload directory that holds the baseline.

**Chunk pipeline geometry.** `run-all`, `run-bench` and `run-mlperf` expose the crypto pipeline's geometry as `--chunk-size`, `--pipeline-bytes`, `--io-depth`, `--reorder-window`, `--burst-bytes`, `--burst-adaptive`, `--io-span-bytes`, `--cluster-size` and `--md-pages`. Each maps onto the client flag of the same name, is bounds-checked before a VM boots, and adds a `pipe-*` token to the result path. A run that passes none of them writes the paths it always did.

## Storage abstractions alone

### `run-all`

The three clients (`kvs_client`, `log_client`, `fs_client`) on their workload files, one boot per (setup, size, client, job, crypto). `--setups` selects `snp`, `amd` or `host`, `--sizes` the VM preset, `--jobs` the workload files. `--tracing` records an SPDK trace, which `parse-amplification` and `parse-trace` read.

```bash
oma-arvio-root accelstore.run-all --setups snp --sizes medium --clients kvs_client
oma-arvio-root accelstore.run-all --io-path kernel --clients kvs_client
```

### `run-seal-unseal`

Seal-then-unseal pairs for all three clients at 100mb, 1gb, 10gb, 100gb and 1000gb (`--data-sizes`), each pair sharing a persisted trusted counter. `parse-seal-unseal` writes the per-phase breakdown of Table X.

### `run-epoch-sweep`

The 256 MB workload per client at each point of an epoch ladder, against a mock counter with a 5 ms round trip (`--rtt-us`). Two arms, `full` (epochs plus counter commits) and `metadata-only`, both relative to the same on-shutdown baseline. `per-op` runs the worst case as the upper bound.

| Flag | Default |
|---|---|
| `--trigger` | `every-t`, 1 to 281 ms in 20 ms steps (`--min/max/step-epoch-ms`). `every-n` and `both` are the alternatives |
| `--blocking` | `on`. `off` runs without the durability gate, `both` runs each point twice |
| `--arms` | `full,metadata-only` |

`plot-epoch` draws Fig. 10 and prints the overhead grid from the trace's epoch tracepoints.

### `run-ablation`

One arm takes away one property of the data path. The mask applies to every crypto pipeline operation, the metadata path runs unablated.

| Token | Takes away | Build |
|---|---|---|
| `no-overlap` | the pipeline's overlap, one chunk in flight at a time | CIF or no-CIF |
| `bounce-copy` | the avoided copy, one extra staged copy per chunk | CIF or no-CIF |
| `no-encrypt` | the AEAD, the MAC, Merkle and counter machinery stay | CIF only |
| `no-verify` | the read path's MAC, Merkle and counter checks | CIF only |
| `zero-copy` | every copy, the region goes straight to the caller's buffer | no-CIF only |

The default sweep is 12 arms on 3 clients: `none`, `no-overlap`, `bounce-copy`, `no-encrypt` and `no-verify` in the `aegis` and `aesgcm` builds, and `none` and `zero-copy` in the `none` build. Every build's unablated `abl-none` run is part of the sweep, as the denominator of its arms. `--arms` selects `<crypto>:<token>` pairs, `--repeat N` repeats the whole list. Each arm writes and reads its own device, since a device written under `no-encrypt` reads only under `no-encrypt`.

`--io-path kernel` runs the sweep over the guest's NVMe driver (`bdev_aio`) instead of SPDK's userspace one (`bdev_nvme` over VFIO), the "kernel NVMe drv" bar of Fig. 9. The same binary, blobstore and crypto path, only the driver changes. Rows land under an `io-kernel/` component above the `abl-*` slug.

```bash
oma-arvio-root accelstore.run-ablation --print-only
oma-arvio-root accelstore.run-ablation --io-path spdk
oma-arvio-root accelstore.run-ablation --io-path kernel --arms aegis:none
oma-arvio accelstore.parse-ablation && oma-arvio accelstore.plot-ablation
```

The runtime of an arm is the window between the client's `WORKLOAD_START` and `WORKLOAD_END` trace markers, cross-checked against the log. A row whose two sources disagree is marked, and a row without a ratio keeps a `status` (`failed-ops`, `no-runtime`, `stale-window`, `no-baseline`).

## Baseline comparisons

### `run-bench`

One `bench_client` personality (a `.wl` file under `modules/accelstore/jobs/`) per boot, on the CVM-Store backend once per crypto build and on each block stack named by `--backends`. The personality suites are the panels of Fig. 7 and Fig. 8. There is no default suite.

| `--suite` | Personalities | Panel |
|---|---|---|
| `random-read` | `fs_random_read_{4k,16k,64k,256k,1m}` | Fig. 7 (c), filesystem random reads |
| `log-small` | `log_small_{16k,64k,256k,1m}` | Fig. 7 (b), shared log small entries |
| `log-large` | `log_write_read_{64mb,256mb,1gb,4gb}` | Fig. 7 (b), shared log large entries |
| `fs-large` | `fs_write_read_{64m,256m,1g,4g}` | Fig. 7 (b), filesystem large files |
| `apps` | `varmail_log`, `videoaudit_log_64m_fsync`, `trainload_fs_paced`, `webserver_fs` | Fig. 8, the Filebench-style workloads |

The comparison is kept fair by construction, and every decision lands in the row's `*_flags.json` sidecar:

- POSIX arms run with O_DIRECT (`--direct false` for a buffered arm), so no arm serves a fileset out of the page cache.
- The crypto chunk size follows the personality: the largest chunk not above the smallest ranged read, so a read never decrypts more than it returns. `--chunk-size` overrides, `--derive-chunk-size false` measures the amplification on purpose.
- A personality without barriers gets `--barrier-mode none` on the POSIX arm instead of bulk fsyncs.
- `--threads N` sets the worker count on every arm. `--split-loops` reads an iteration bound as the run's total instead of per worker, so eight threads move the same bytes as one. It is turned on automatically, capped at 8 threads, for the LOG and random-read suites.
- `log-small` derives the small-object store geometry (chunk = object size, 16 KiB clusters, an `--md-pages` reservation), without which a small-object run spends up to 128 times its payload in device space.
- `--intra-op-depth N` cuts one operation into N pieces on both arms, `--match-io-depth` gives the POSIX arms the CVM-Store arm's in-flight byte budget. Each lands in its own result leaf.

```bash
oma-arvio-root accelstore.run-bench --suite apps --backends ext4,rollbaccine-f1m,accelstore --split-loops
oma-arvio-root accelstore.run-bench --suite log-small --threads 8
oma-arvio accelstore.parse-bench
```

### `run-ycsb`

The KVS against RocksDB. One binary, `ycsb_client`, runs the YCSB mixes on the CVM-Store KVS over SPDK and on RocksDB on a block stack, from one op stream. RocksDB runs with its write-ahead log disabled and BlobDB for values of 4 MB and above. Three point families, 23 points by default:

| `--points` | Points | Shape |
|---|---|---|
| `grid` | `{B,C,F}_{4k,16k,64k,256k,1m}` | 100k ops over 10k keys, ops/s and latency |
| `small` | `A_{4k,16k,64k,256k,1m}` | 128 GB per run-phase direction, MiB/s |
| `large` | `A_{64m,256m,1g}` | 256 GiB load phase, MiB/s, Fig. 7 (a) |

`--points` also takes single names or a bare mix letter. `--arms` defaults to `accelstore,ext4`. `--direction-bytes-gb` and `--large-total-gb` scale the two A halves, which is what `--quick` uses. Pinned client buffers, RocksDB's value size limit and the stack capacity are checked before the first boot.

```bash
oma-arvio-root accelstore.run-ycsb --points large --arms ext4,rollbaccine,accelstore --crypto none,aegis
oma-arvio accelstore.parse-ycsb
```

## Applications

### `run-redis`

Redis from [TUM-DSE/redis-CVM-Store](https://github.com/TUM-DSE/redis-CVM-Store), branch `cvm-store`, built against the AEGIS variant. On the `accelstore` arm every persistence artifact (AOF files, RDB snapshots) is a shared log entry on the store. On a block stack the same binary runs with the `--accelstore-*` flags omitted, so a row compares two storage paths and not two Redis builds.

| Point | `appendfsync` | `redis-benchmark` | Isolates |
|---|---|---|---|
| `always_c1` | always | 1 client, no pipelining, 200k requests | barrier latency, one fsync per command |
| `always_c8p4` | always | 8 clients, pipeline 4, 5M requests | group commit |
| `everysec_c8p4` | everysec | 8 clients, pipeline 4, 5M requests | the same load with fsync off the command path |
| `everysec_c50p16` | everysec | 50 clients, pipeline 16, 10M requests | append throughput |
| `dataset` | everysec | 1.5 GB fill, then 200k requests | timed BGSAVE, BGREWRITEAOF and restart-load walls |

`--points` takes names, the families `load` and `persist`, or `all`. `--requests-scale 100` is a smoke run. `--tc-rtt-us` puts the mock counter behind the store's barriers. The guest needs at least 7 vCPUs: the core layout separates housekeeping, Redis, the SPDK reactors, the rewrite child and `redis-benchmark`, derived from the VM size. Every run starts from a fresh data directory and a destroyed blobstore, except the `restart_load` phase of `dataset`, which is the recovery measurement.

```bash
oma-arvio-root accelstore.run-redis --arms accelstore,ext4,rollbaccine-f1m
oma-arvio accelstore.parse-redis
```

### `run-mlperf`

MLPerf Storage checkpointing inside the guest's container, one boot per (variant, arm, stack). The table in `tasks/mlperf_variants.py` declares the points, the task only selects rows. It refuses to start without the `--mlperf` guest.

| `--variants` | What it is |
|---|---|
| `ckpt-llama3-8b-subset` | Llama3-8B, closed division, 8 ranks through one store master. The cross-arm measurement row of Fig. 8 |
| `ckpt-llama3-8b-1rank` | the same model with one rank and no IPC, the smoke and iteration row |

| `--arms` | Storage |
|---|---|
| `accelstore-log`, `accelstore-fs` | the shared log or the filesystem, through the `pyaccelstore` bridge |
| `posix-direct` | O_DIRECT files on a block stack (`--stacks`, default `ext4`) |
| `posix-buffered`, `direct` | reference rows, on request or with `--arms all` |

Hugepages for a CVM-Store arm are derived from the store geometry and checked against the guest size. `parse-mlperf` reads mlpstorage's own `summary.json` per point and nothing else, so no published number is re-derived.

```bash
oma-arvio-root accelstore.run-mlperf --variants ckpt-llama3-8b-subset --arms accelstore-log,accelstore-fs,posix-direct --stacks ext4,rollbaccine-f1m --sizes large
oma-arvio accelstore.parse-mlperf
```

## Microbenchmarks

`run-fio-iops` runs the fio job files of OmaArvio's fio module on the host, a normal VM and the CVM, over `libaio`, SPDK and the LUKS ciphers (AES-XTS, AES-XTS with SHA-256, AEGIS-128). `run-fio-bandwidth` runs the bandwidth jobs on VM and CVM. `run-crypto` runs tcrypt and openssl on all three. Results land under `bench-result/fio/`, `bench-result/tcrypt/` and `bench-result/openssl/`. `plot-fio-iops` and `plot-fio-luks` draw Fig. 2 and Fig. 3, `parse-fio-bandwidth` and `parse-crypto` write Table II and Table XII.

## Attestation

`attestation.run` boots an SNP guest per iteration, has Veraison appraise its evidence and times each step on both sides. Per iteration it starts Veraison, clears its stores, provisions the AMD root key of the host's processor model as trust anchor, turns a report from the guest into a reference-value CoRIM with `go-gen-ref`, and runs `attestation-server` on the host against `attestation-client` in the guest. `attestation.parse` averages the latest three runs into Table IX.

```bash
oma-arvio veraison.setup                        # once: clone and start the docker deployment
oma-arvio-root attestation.run --iterations 3   # snp-disk-small, --size to change
oma-arvio attestation.parse                     # bench-result/attestation-summary.csv
```

`veraison.*` exposes the steps by hand: `start`, `stop`, `status`, `logs`, `stores`, `clear-stores`, `provision-trust-anchors`, `provision-reference-value`. The processor model, verifier port and host gateway are `attestation_*` keys in `config.toml`.
