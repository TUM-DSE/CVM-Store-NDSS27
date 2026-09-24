{
  description = "cvm-store: AccelStore benchmark harness built on OmaArvio";

  inputs = {
    omaarvio = {
      # Not `github:`: OmaArvio's own flake sets `self.submodules`, which the
      # tarball scheme rejects. The subdirectory module flakes below do not.
      url = "git+https://github.com/TUM-DSE/OmaArvio.git";
    };

    # Reuse OmaArvio's pins
    nixpkgs.follows = "omaarvio/nixpkgs";
    flake-utils.follows = "omaarvio/flake-utils";
    pre-commit-hooks.follows = "omaarvio/pre-commit-hooks";
    core.follows = "omaarvio/core";

    # CVM-enabled DPDK/SPDK
    dpdk-src = {
      url = "github:TUM-DSE/dpdk/cvm-dev-mempool";
      flake = false;
    };
    spdk-src = {
      url = "git+https://github.com/TUM-DSE/spdk.git?ref=cvm-dev-master-hardend&submodules=1";
      flake = false;
    };

    mod-dpdk-spdk = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/dpdk_spdk";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
      inputs.dpdk-src.follows = "dpdk-src";
      inputs.spdk-src.follows = "spdk-src";
    };
    mod-nvme = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/nvme";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
      inputs.dpdk-spdk.follows = "mod-dpdk-spdk";
    };
    mod-fio = {
      url = "github:TUM-DSE/OmaArvio?dir=modules/fio";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
      inputs.nvme.follows = "mod-nvme";
      inputs.dpdk-spdk.follows = "mod-dpdk-spdk";
    };
    mod-crypto.follows = "omaarvio/mod-crypto";

    mod-accelstore = {
      url = "path:./modules/accelstore";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
      inputs.dpdk-spdk.follows = "mod-dpdk-spdk";
      inputs.nvme.follows = "mod-nvme";
      inputs.fio.follows = "mod-fio";
      inputs.crypto.follows = "mod-crypto";
    };

    # SEV-SNP attestation
    mod-attestation = {
      url = "path:./modules/attestation";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.core.follows = "core";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , pre-commit-hooks
    , core
    , mod-accelstore
    , ...
    }@inputs:
    let
      # Auto-discover active modules: any mod-* input is active.
      activeModules =
        map (name: inputs.${name})
          (builtins.filter
            (name: builtins.match "^mod-.*" name != null)
            (builtins.attrNames inputs));

      kernelVariants = mod-accelstore.lib.kernelVariants;
      defaultKernel = mod-accelstore.lib.defaultKernel;
      mlperfVariants = mod-accelstore.lib.mlperfVariants;

      modulesForGuest = point:
        map
          (m:
            if m ? lib.guestWith
            then m // { nixosModules = (m.nixosModules or { }) // { guest = m.lib.guestWith point; }; }
            else m)
          activeModules;

      kernelSuffix = variant: builtins.replaceStrings [ "." ] [ "-" ] variant;

      guestSuffix = point:
        "-linux-${kernelSuffix point.kernel}"
        + (if point.mlperf then "-mlperf" else "");

      guestPoints = builtins.concatMap
        (kernel: map (mlperf: { inherit kernel mlperf; }) mlperfVariants)
        kernelVariants;

      guestSystems = builtins.listToAttrs (map
        (point: {
          name = guestSuffix point;
          value = core.lib.mkGuestSystem { modules = modulesForGuest point; };
        })
        guestPoints);

      guestSystem = guestSystems.${guestSuffix { kernel = defaultKernel; mlperf = false; }};

      guestImageNames = [ "snp-guest-image" "direct-guest-image" ];

      variantImages = builtins.listToAttrs (builtins.concatMap
        (point: map
          (image: {
            name = "${image}${guestSuffix point}";
            value = guestSystems.${guestSuffix point}.packages.${image};
          })
          guestImageNames)
        guestPoints);
    in
    (flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [ core.overlays.qemu-qmp ];
      };

      pre-commit-check = pre-commit-hooks.lib.${system}.run {
        src = ./.;
        hooks = {
          nixpkgs-fmt.enable = true;
          black.enable = true;
        };
      };

      oma-arvio = core.lib.mkOmaArvio { inherit pkgs; modules = activeModules; };

      update-accelstore = pkgs.writeShellApplication {
        name = "update-accelstore";
        runtimeInputs = [ pkgs.git pkgs.nix ];
        text = ''
          root=$(git rev-parse --show-toplevel)
          nix flake update accelstore-src --flake "$root/modules/accelstore"
          nix flake update mod-accelstore --flake "$root"

          git -C "$root" add flake.lock modules/accelstore/flake.lock
          if ! git -C "$root" diff --cached --quiet; then
            git -C "$root" commit -m "update accelstore"
          fi
        '';
      };
    in
    rec {
      packages = core.packages.${system} // oma-arvio.modPkgs // variantImages // {
        inherit oma-arvio update-accelstore;
        inherit (guestSystem.packages) snp-guest-image direct-guest-image;
        default = guestSystem.packages.snp-guest-image;
      };

      apps.update-accelstore = flake-utils.lib.mkApp { drv = update-accelstore; };

      devShells.default = pkgs.mkShell {
        buildInputs = [ packages.oma-arvio update-accelstore ]
        ++ pre-commit-check.enabledPackages;
        shellHook = pre-commit-check.shellHook;
      };

      devShells.benchmarking = pkgs.mkShell {
        name = "benchmarking";
        packages = core.lib.hostPackages pkgs ++ oma-arvio.hostPkgs;
      };
    })) // {
      lib.nixpkgsRev = nixpkgs.shortRev;
      lib.kernelVariants = kernelVariants;
      lib.defaultKernel = defaultKernel;
      lib.mlperfVariants = mlperfVariants;

      nixosConfigurations = guestSystem.nixosConfigurations // (builtins.listToAttrs (builtins.concatMap
        (point: map
          (name: {
            name = "${name}${guestSuffix point}";
            value = guestSystems.${guestSuffix point}.nixosConfigurations.${name};
          }) [ "snp-guest" "direct-guest" ])
        guestPoints));
    };
}
