{
  description = "accelstore module: AccelStore client benchmarks and paper plotting";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Pinned nixpkgs that still packages linux 6.17 
    nixpkgs-kernel-617.url = "github:NixOS/nixpkgs/52e64396e98aef211f4127416dbdae17a01b3d3f";
    core = {
      url = "github:TUM-DSE/OmaArvio?dir=core";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # AccelStore needs to be build against harness SPDK
    dpdk-spdk = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/dpdk_spdk";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
    };
    nvme = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/nvme";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
    };
    fio = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/fio";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
    };
    crypto = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/crypto";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
    };
    accelstore-src = {
      url = "github:TUM-DSE/CVM-Store/main";
    };

    redis-src = {
      url = "github:TUM-DSE/redis-CVM-Store/cvm-store";
      flake = false;
    };
    mlperf-src = {
      url = "github:TUM-DSE/MLPerf-CVM-Store/cvm-store";
      flake = false;
    };
    dlio-src = {
      url = "github:TUM-DSE/DLIO-CVM-Store/cvm-store";
      flake = false;
    };

    # The base image `nix run .#build-mlperf-image` writes. The default is a
    # placeholder directory so the flake always evaluates; see
    # nix/mlperf/base-image-placeholder/PLACEHOLDER.
    mlperf-base-image = {
      url = "path:./nix/mlperf/base-image-placeholder";
      flake = false;
    };
  };

  outputs =
    { self
    , nixpkgs
    , nixpkgs-kernel-617
    , core
    , dpdk-spdk
    , nvme
    , fio
    , crypto
    , accelstore-src
    , redis-src
    , mlperf-src
    , dlio-src
    , mlperf-base-image
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      pkgs-kernel-617 = import nixpkgs-kernel-617 { inherit system; config.allowUnfree = true; };

      kernels = import ./nix/kernel.nix {
        inherit pkgs pkgs-kernel-617;
      };

      dm-rollbaccine = pkgs-kernel-617.callPackage ./nix/rollbaccine-module.nix {
        kernel = kernels.kernelFor "6.17";
      };

      redis-cvm-store = pkgs.callPackage ./nix/redis-cvm-store.nix {
        inherit redis-src;
        accelstore = accelstoreVariants.aegis;
      };

      build-mlperf-image = pkgs.callPackage ./nix/mlperf/build-base-image.nix {
        inherit mlperf-src dlio-src;
      };

      mlperf = pkgs.callPackage ./nix/mlperf/mlperf.nix {
        inherit accelstorePackages pyaccelstore;
        baseImage =
          if builtins.pathExists "${mlperf-base-image}/PLACEHOLDER"
          then null
          else mlperf-base-image;
      };

      pyaccelstore = upstream.pyaccelstore.override {
        accelstore = accelstoreVariants.aegis;
      };

      defaultKernel = "6.17";

      # Build harness SPDK with our custom blob patch
      # We carry it locally and then just assert it matches upstream to minimize rebuilds
      spdkBlobPatch = ./nix/patches/spdk/blob-create-and-open.patch;
      upstreamSpdkBlobPatch = "${accelstore-src}/nix/spdk-blob-create-and-open.patch";
      harnessSpdk =
        assert pkgs.lib.assertMsg
          (builtins.hashFile "sha256" spdkBlobPatch
            == builtins.hashFile "sha256" upstreamSpdkBlobPatch) ''
          modules/accelstore/nix/patches/spdk/blob-create-and-open.patch has drifted from
          ${upstreamSpdkBlobPatch}.
          Resync it with
            cp ${upstreamSpdkBlobPatch} modules/accelstore/nix/patches/spdk/blob-create-and-open.patch
          or drop the copy and this override if the patch has landed in the SPDK fork.
        '';
        dpdk-spdk.packages.${system}.spdk.spdkSrc.overrideAttrs (old: {
          patches = (old.patches or [ ]) ++ [ spdkBlobPatch ];
        });

      # Override to build with patched SPDK
      withHarnessSpdk = drv:
        (drv.override {
          spdk = harnessSpdk;
        }).overrideAttrs (old: {
          # Needed for newer SPDK versions than AccelStore's own pin.
          dontStrip = true;
          NIX_CFLAGS_COMPILE = (old.NIX_CFLAGS_COMPILE or "") + " -Wno-error";
          buildInputs = builtins.map
            (dep:
              if (dep.pname or "") == "dpdk"
              then dpdk-spdk.packages.${system}.dpdk
              else dep)
            old.buildInputs;
        });

      # AccelStore configs
      upstream = accelstore-src.packages.${system};
      cryptoVariants = {
        aegis = { attr = "accelstore"; args = { }; };
        aesgcm = { attr = "accelstore-aesgcm"; args = { cryptoBackend = "aesgcm"; }; };
        none = { attr = "accelstore-no-crypto"; args = { noCrypto = true; }; };
      };
      accelstoreVariants = builtins.mapAttrs
        (_id: v: withHarnessSpdk (upstream.accelstore.override v.args))
        cryptoVariants;

      accelstorePackages = builtins.attrValues accelstoreVariants;
    in
    {
      packages.${system} = (pkgs.lib.mapAttrs'
        (id: v: pkgs.lib.nameValuePair v.attr accelstoreVariants.${id})
        cryptoVariants) // {

        guest-kernel-6-17 = kernels.kernelFor "6.17";

        inherit dm-rollbaccine;
        inherit redis-cvm-store;

        inherit mlperf build-mlperf-image pyaccelstore;

        dpdk-cvm-shared = dpdk-spdk.packages.${system}.dpdk.overrideAttrs (old: {
          mesonFlags = map
            (flag:
              if flag == "-Dexamples=helloworld"
              then "-Dexamples=helloworld,cvm_shared"
              else flag)
            (old.mesonFlags or [ ]);
        });

        # Workload files, exposed at /shared/modules/accelstore/ in the guest
        # and host runner.
        accelstore-jobs = pkgs.runCommand "accelstore-jobs" { } ''
          # No -a/-p: archive mode would stamp the store source's read-only
          # mode onto $out itself; plain -r still carries the exec bits.
          mkdir -p $out
          cp -r ${./jobs}/. $out/
        '';
      };

      nixosModules.guest = self.lib.guestFor defaultKernel;

      lib = {
        hostPackages = pkgs:
          import ./nix/host.nix {
            inherit pkgs accelstorePackages redis-cvm-store;
          };
        sharedData = self.packages.${system}.accelstore-jobs;

        guestFor = variant: self.lib.guestWith { kernel = variant; };

        guestWith = { kernel ? defaultKernel, mlperf ? false }:
          let variant = kernel; in
          import ./nix/guest.nix {
            mlperfImage = if mlperf then self.packages.${system}.mlperf else null;
            accelstorePackages = accelstorePackages ++ pkgs.lib.optionals (variant == "6.17") [ dm-rollbaccine ];
            inherit redis-cvm-store;
            kernelPackages = kernels.packagesFor variant;
            extraModulePackages = [ ]
              ++ pkgs.lib.optionals (variant == "6.17") [ dm-rollbaccine ];
          };
        kernelVariants = kernels.variants;
        inherit defaultKernel;
        mlperfVariants = [ false true ];
      } // core.lib.mkModulePythonLib { name = "accelstore"; inherit self; };
    };
}
