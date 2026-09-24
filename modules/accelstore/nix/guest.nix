{ accelstorePackages
, redis-cvm-store
, kernelPackages
, extraModulePackages ? [ ]
, extraKernelModules ? [ ]
, extraKernelParams ? [ ]
, mlperfImage ? null
}: { pkgs, lib, config, ... }:

let
  withMlperf = mlperfImage != null;
in
{
  imports = lib.optionals withMlperf [
    (import ./mlperf/mlperf-service.nix { dockerImagePkg = mlperfImage; })
    { services.mlperf.enable = true; }
  ];

  environment.systemPackages = accelstorePackages ++ [ redis-cvm-store ]
    ++ lib.optionals withMlperf [ pkgs.docker ];

  boot.kernelPackages = lib.mkForce kernelPackages;

  boot.extraModulePackages = extraModulePackages;
  boot.kernelModules = extraKernelModules;

  boot.kernelParams = lib.mkAfter extraKernelParams;

  # ttyS1 make it usable for kernel log only 
  systemd.services."serial-getty@ttyS1".enable = false;

  virtualisation.docker = lib.mkIf withMlperf {
    enable = true;
    enableOnBoot = true;
    storageDriver = "overlay2";
  };
}
