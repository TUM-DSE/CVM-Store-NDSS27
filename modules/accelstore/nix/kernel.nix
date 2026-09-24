# Requires CCache enabled on the host

{ pkgs
, pkgs-kernel-617
}:

let
  lib = pkgs.lib;

  ccacheExtraConfig = ''
    export CCACHE_COMPRESS=1
    # 000, not 007: with `auto-allocate-uids` every build runs as a fresh
    # unrelated uid, so cache entries must stay world-rw to be shared. The
    # cache directory's own permissions are the access boundary.
    export CCACHE_UMASK=000
    # NixOS `programs.ccache.enable` route first, then the non-NixOS
    # wiki route. Only the configured one is bind-mounted in the sandbox.
    for dir in /var/cache/ccache /nix/var/cache/ccache; do
      if [ -d "$dir" ] && [ -w "$dir" ]; then
        export CCACHE_DIR="$dir"
        break
      fi
    done
    if [ -z "''${CCACHE_DIR:-}" ]; then
      echo "====="
      echo "No writable ccache directory in the build sandbox (checked"
      echo "/var/cache/ccache and /nix/var/cache/ccache for user $(whoami))."
      echo "On NixOS set:  programs.ccache.enable = true;"
      echo "  (with auto-allocate-uids also:  chmod 1777 /var/cache/ccache"
      echo "   — builds run as throwaway uids outside the nixbld group)"
      echo "Elsewhere create /nix/var/cache/ccache (root:nixbld, 0770) and"
      echo "add to nix.conf:  extra-sandbox-paths = /nix/var/cache/ccache"
      echo "See https://wiki.nixos.org/wiki/CCache"
      echo "====="
      exit 1
    fi
  '';

  ccacheStdenvFor = p:
    p.ccacheStdenv.override { extraConfig = ccacheExtraConfig; stdenv = p.stdenv; };

  snpConfig = {
    name = "enable-sev-snp";
    patch = null;
    extraConfig = ''
      AMD_MEM_ENCRYPT y
      VIRT_DRIVERS y
      SEV_GUEST m
      X86_CPUID m
      VFIO_NOIOMMU y
      DMA_API_DEBUG n
      FS_VERITY y
    '';
  };

  cvmPatches = series: [
    {
      name = "sev_mmio";
      patch = ./patches + "/${series}/linux_sev_mmio.patch";
    }
    {
      name = "linux_walk_page";
      patch = ./patches + "/${series}/linux_walk_page.patch";
    }
    {
      name = "linux_shared_mmap";
      patch = ./patches + "/${series}/linux_cvm_shared.patch";
    }
  ];

  patchKernel = pkgsK: series: kernel:
    kernel.override {
      kernelPatches = (kernel.kernelPatches or [ ]) ++ [ snpConfig ] ++ cvmPatches series;
      stdenv = ccacheStdenvFor pkgsK;
    };

  kernels = {
    "6.17" = {
      pkgs = pkgs-kernel-617;
      kernel = patchKernel pkgs-kernel-617 "6.17" pkgs-kernel-617.linux_6_17;
    };
  };
in
{
  variants = builtins.attrNames kernels;

  packagesFor = variant:
    let
      entry = kernels.${variant} or (throw
        "Unknown kernel variant '${variant}'. Known: ${toString (builtins.attrNames kernels)}");
    in
    entry.pkgs.linuxPackagesFor entry.kernel;

  kernelFor = variant: kernels.${variant}.kernel;
}
