# Building the guest stack

What `oma-arvio accelstore.build-guest --mlperf` puts into the CVM guest image, and the one host-side setup it needs.

## What the image carries

| Part | Source | Why |
|---|---|---|
| DPDK and SPDK | the TUM-DSE forks `TUM-DSE/dpdk` @ `cvm-dev-mempool` and `TUM-DSE/spdk` @ `cvm-dev-master-hardend`, pinned in the root flake | allocate DMA memory as shared (unencrypted) guest pages via `MAP_CVM_SHARED`. The `nvme`, `fio` and `accelstore` modules all build against them |
| Guest kernel 6.17 | 6.17.13 from a pinned nixpkgs, patches in `modules/accelstore/nix/patches/6.17/` | `MAP_CVM_SHARED` mmap support, user-space MMIO in the `#VC` handler for SPDK, and an exported `walk_page_range` |
| CVM-Store clients and baselines | `modules/accelstore/flake.nix` | the three crypto build variants, `bench_client`, `ycsb_client`, Redis, Rollbaccine |
| MLPerf Storage container (`--mlperf`) | `modules/accelstore/nix/mlperf/` | Docker plus the patched `mlpstorage` and DLIO forks, driving CVM-Store through the `pyaccelstore` bridge |

The MMIO patch reopens CVE-2023-46813, so the image is for benchmark guests only.

## Build commands

```bash
oma-arvio accelstore.build-guest --mlperf                            # base container, then the guest
oma-arvio accelstore.build-guest --mlperf --no-build-mlperf-base-image  # reuse build/mlperf-base-image.tar
oma-arvio accelstore.build-mlperf-image                              # refresh the container only
oma-arvio accelstore.active-guest-kernel                             # what is installed
```

## ccache for kernel builds

Guest kernels build through ccache and need a shared cache directory mounted
into the Nix build sandbox. The build fails with instructions if it is missing.

On NixOS (not together with `programs.ccache.enable`, whose tmpfiles rule conflicts):

```nix
nix.settings.extra-sandbox-paths = [ "/var/cache/ccache" ];
systemd.tmpfiles.rules = [ "d /var/cache/ccache 1777 root root -" ];
```

On other distributions, create `/nix/var/cache/ccache` owned by `root:nixbld`
with mode `0770`, add `extra-sandbox-paths = /nix/var/cache/ccache` to
`/etc/nix/nix.conf`, and restart `nix-daemon`.
