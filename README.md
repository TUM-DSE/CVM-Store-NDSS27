# CVM-Store Evaluation NDSS '27

This repository contains the evaluation framework for the NDSS submission of
*CVM-Store*. CVM-Store is a storage system for
confidential VMs (CVMs) that combines a userspace I/O engine with
confidentiality, integrity and freshness (CIF) protection. The source code
of CVM-Store itself is at
[TUM-DSE/CVM-Store](https://github.com/TUM-DSE/CVM-Store).

The framework builds the required software and guest image, handles running
the experiments using the correct VM configurations and generating the final
output and figures. It is built on
[OmaArvio](https://github.com/TUM-DSE/OmaArvio) and this repo adds the CVM-Store
specific benchmarks, workloads, attestation flow and plotting. *AccelStore* is
the internal project name of CVM-Store and therefore is used interchangeably.

**Note for reviewers.** The evaluation needs SEV-SNP hardware and a Gen5 NVMe.
We can provide anonymous SSH access to a machine in our cluster with the
correct hardware, configuration and all dependencies already in place. To
request access, please send us your public SSH key through the submission
system (the key only, without any identifier). Cluster details are at
[TUM-DSE/doctor-cluster-config](https://github.com/TUM-DSE/doctor-cluster-config).

> [!WARNING]
> Experiments cannot run in parallel. Only one run at a time can use the
> same hardware.

## Hardware and software requirements

| Component | Requirement |
|---|---|
| CPU | AMD SEV-SNP enabled CPU, e.g. AMD EPYC 9334 (Genoa or later) |
| Memory | 320 GB DDR5 RAM |
| Storage device | KIOXIA CM7-R PCIe NVMe at Gen5 speeds (or a similar Gen5 NVMe) |
| Free disk space | Up to about 400 GB for traces and build artifacts |

The software side needs the following on the host.

- SEV-SNP enabled upstream Linux kernel 7.2 (older kernels require patches to run attestation experiments)
- Nix with flakes, which builds the software and the evaluation framework
- Docker, for the MLPerf Storage container in the guest image and the Veraison verifier
- NVMe must not hold any data (gets erased)

## Getting started

All commands run inside `nix develop`. Tasks that boot a VM or bind the NVMe
need root and use `oma-arvio-root`, everything else `oma-arvio`.

```bash
git clone https://github.com/TUM-DSE/CVM-Store-NDSS27.git
cd CVM-Store-NDSS27
nix develop # might take some time
```

Add the host in `config.toml` (see [Configuration](#configuration)),
then build the base dependencies in this order.

```bash
oma-arvio build.all                          # OVMF, QEMU and the SNP guest image
oma-arvio accelstore.build-guest --mlperf    # the CVM guest with the MLPerf container
oma-arvio veraison.setup                     # clone and start the Veraison verifier
oma-arvio veraison.status                    # every service should be up
```

Kernel builds go through ccache and need a cache directory mounted into the
Nix sandbox. [docs/guest-stack.md](docs/guest-stack.md) explains that setup,
the kernel patches and the container build.

## Running the paper experiments

The `paper.*` tasks reproduce the paper: one task per figure or table, each
running the sweeps behind it, parsing the results and drawing the output.
`paper.all` runs and generates parsed data and plots for all of them.

> [!IMPORTANT]
> Most `paper.*` tasks take hours. Start it inside a `tmux` session.

**Run the `--quick` set.** The full parameter set takes several weeks, the
two fio figures take a large part of it. `--quick` runs a representative
subset over fewer axes and less data per axis.

```bash
oma-arvio-root paper.all --quick                          # every figure and table, reduced
oma-arvio-root paper.all --quick --only synthetic,apps    # a subset
oma-arvio-root paper.all                                  # the full set, weeks of runtime
```

| Paper | Task | Output under `bench-result/` | Runtime `--quick` |
|---|---|---|---|
| Fig. 2, fio IOPS on host, VM and CVM | `paper.fio-iops` | `plots/fio-iops_iops.pdf` | skipped |
| Fig. 3 and Table II, block stacks and LUKS ciphers | `paper.fio-stacks` | `parsed/fio-bandwidth.csv`, `plots/fio-luks_iops.pdf` | skipped |
| Table XII, encryption throughput | `paper.crypto` | `parsed/crypto-summary.csv` | ~3.5 h |
| Table IX, attestation and provisioning time | `paper.attestation` | `attestation-summary.csv` | ~3 min |
| Table VI, read/write amplification | `paper.amplification` | `parsed/amplification.csv` | ~1 h |
| Fig. 10, metadata epoch overhead | `paper.epoch` | `plots/epoch-overhead.pdf` | ~3.5 h |
| Fig. 9, ablation study | `paper.ablation` | `plots/ablation.pdf` | ~1 h 40 min |
| Table X, seal, unseal and recovery time | `paper.seal-unseal` | `parsed/seal-unseal-summary.csv` | ~40 min |
| Fig. 7, I/O analysis of the three abstractions | `paper.synthetic` | `plots/synthetic.pdf` | ~8 h |
| Fig. 8, end-to-end applications | `paper.apps` | `plots/apps.pdf` | ~3 h (not reduced) |
| All of the above | `paper.all` | | ~1 day |

Runtimes are from one `--quick` run on an AMD EPYC Genoa host with SEV-SNP.


### Changes of `--quick` compared to paper

| Experiment | Full | `--quick` |
|---|---|---|
| Fig. 2 and Fig. 3, fio | run | skipped |
| Fig. 10, epoch window | 1 to 281 ms, 15 points | 1, 21, 41 and 61 ms |
| Fig. 7, KVS (YCSB-A) | 4k to 1m and 64m to 1g, 128 GB and 256 GiB | 16K, 1M, 256M, 1G at 64 GB and 128 GiB |
| Fig. 7, shared log | 16k to 1m and 64mb to 4gb, 128 and 256 GiB per phase | 16K, 1M, 256M, 1G at 64 and 128 GiB |
| Fig. 7, filesystem | 64m to 4g | 64M, 256M, 1G |
| Fig. 7, filesystem random read | 4k to 1m, 5 points | 4K, 64K, 1M |
| Table X, seal and unseal | 100mb to 1000gb, 5 points | 100MB, 10GB, 1000GB |
| Fig. 9, ablation | 256 GiB per direction | 64 GiB per direction |
| Table XII, Table IX, Table VI, Fig. 8 | run | unchanged |

Absolute numbers of a quick run differ from the paper where a point moves
less data. Ratios between arms and the shapes should be preserved.

## Results and output

All results land under `bench-result/`. The figures are in `plots/` and the
tables in `parsed/`, the experiments table above names the file per figure.

| Path | Content |
|---|---|
| `bench-result/plots/` | the figures as PDF, one per `plot-*` task |
| `bench-result/parsed/` | the tables as CSV, one per `parse-*` task |
| `bench-result/attestation-summary.csv` | the attestation table |
| `bench-result/<family>/` | the raw runs behind them |

A `paper.*` task outputs a summary table with the status of the tasks that ran and their runtime:
```
   6 ok    0 failed    42m  oma-arvio-root accelstore.run-ycsb --points large ...
  12 ok    1 failed   118m  oma-arvio-root accelstore.run-bench --suite random-read ...
                       3m  oma-arvio-root accelstore.plot-synthetic
```

[docs/results.md](docs/results.md) documents the directory layout and every
parse and plot task.

## Command overview

`oma-arvio --list` prints every task with a one-line help. The tasks come in
families, all documented in [docs/benchmarks.md](docs/benchmarks.md).

| Family | Tasks | Purpose |
|---|---|---|
| Paper | `paper.<figure>`, `paper.all` | run, parse and plot one figure or all of them |
| Build | `build.all`, `accelstore.build-guest`, `accelstore.build-mlperf-image`, `accelstore.active-guest-kernel` | firmware, QEMU, the guest image and its container |
| Storage abstractions | `accelstore.run-all`, `accelstore.run-seal-unseal`, `accelstore.run-epoch-sweep`, `accelstore.run-ablation` | the CVM-Store clients alone, with the durability window, seal and unseal, and the ablation arms as axes |
| Baseline comparisons | `accelstore.run-bench`, `accelstore.run-ycsb` | the personalities and the KVS against ext4 and Rollbaccine on the same NVMe |
| Applications | `accelstore.run-mlperf`, `accelstore.run-redis` | MLPerf Storage checkpointing and Redis persistence |
| Microbenchmarks | `accelstore.run-fio-iops`, `accelstore.run-fio-bandwidth`, `accelstore.run-crypto` | fio and cipher throughput on host, VM and CVM |
| Attestation | `attestation.run`, `attestation.parse`, `veraison.*` | SEV-SNP attestation against the Veraison verifier |
| Parse and plot | `accelstore.parse-*`, `accelstore.plot-*` | CSV tables and figures from `bench-result/` |

Every run task takes `--print-only` to show its plan without booting a VM, and
`--hostname` to run under another host's configuration. Sweeps are selected
with `--setups`, `--sizes`, `--clients`, `--crypto` and the family's own
point flags.

## Configuration

`config.toml` holds the per-host settings the harness needs to boot a guest
with the NVMe passed through. A host is a `[hosts.<hostname>]` table:

| Key | Meaning |
|---|---|
| `nvme_pci` | PCI address of the NVMe under test, bound to VFIO for a run |
| `dev_path` | the device node of the same NVMe, used by the POSIX baselines and the kernel I/O path |
| `valid_pcie_speeds` | PCIe generations the device is accepted at, a run refuses a downtrained link |
| `vm_resources` | the VM size presets `small`, `medium`, `large` and `numa` as vCPU count, memory in GB, NUMA nodes and pin base |

## Further documentation

- [docs/guest-stack.md](docs/guest-stack.md): the guest image, kernel patches, MLPerf container and ccache
- [docs/benchmarks.md](docs/benchmarks.md): every benchmark family and its flags, the crypto build variants and the chunk pipeline geometry
- [docs/results.md](docs/results.md): the result directory layout, parse and plot tasks
- [OmaArvio](https://github.com/TUM-DSE/OmaArvio): the VM orchestration this framework builds on

## Source repositories

Main software components included via the flakes:

- [TUM-DSE/CVM-Store](https://github.com/TUM-DSE/CVM-Store): the storage system under evaluation
- [TUM-DSE/OmaArvio](https://github.com/TUM-DSE/OmaArvio): VM orchestration, guest image, fio and crypto modules
- [TUM-DSE/dpdk](https://github.com/TUM-DSE/dpdk) and [TUM-DSE/spdk](https://github.com/TUM-DSE/spdk): the CVM-enabled DPDK and SPDK forks
- [TUM-DSE/redis-CVM-Store](https://github.com/TUM-DSE/redis-CVM-Store): Redis with CVM-Store persistence
- [TUM-DSE/MLPerf-CVM-Store](https://github.com/TUM-DSE/MLPerf-CVM-Store) and [TUM-DSE/DLIO-CVM-Store](https://github.com/TUM-DSE/DLIO-CVM-Store): MLPerf Storage checkpointing on CVM-Store
