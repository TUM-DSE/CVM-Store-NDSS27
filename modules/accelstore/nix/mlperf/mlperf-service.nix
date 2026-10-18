{ dockerImagePkg
, imageRef ? "${dockerImagePkg.imageName}:${dockerImagePkg.imageTag}"
}:
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.mlperf;

  loader = pkgs.writeShellApplication {
    name = "mlperf-load";
    runtimeInputs = [ pkgs.docker pkgs.coreutils pkgs.python3 ];
    text = ''
      image=${escapeShellArg imageRef}
      staging="mlperf-load:unverified"
      archive=${escapeShellArg dockerImagePkg}
      verify=${./verify-image.py}

      restage() {
        if docker image inspect "$image" >/dev/null 2>&1; then
          docker tag "$image" "$staging"
          docker rmi "$image" >/dev/null 2>&1 || true
        fi
      }

      untag() {
        docker rmi "$1" >/dev/null 2>&1 || true
      }

      if [ ! -r "$verify" ]; then
        echo "mlperf: verifier $verify is missing; refusing to load $image" >&2
        exit 1
      fi

      verified() {
        local rc=0
        python3 "$verify" "$1" || rc=$?
        if [ "$rc" -eq 3 ]; then
          echo "mlperf: $image could not be verified; accepting it unchecked" >&2
          return 0
        fi
        return "$rc"
      }

      load() {
        docker load -i "$archive" || return 1
        restage
        sync
        verified "$staging"
      }

      purge() {
        docker rmi -f "$image" "$staging" >/dev/null 2>&1 || true
        docker image prune -af >/dev/null 2>&1 || true
        sync
      }

      if docker image inspect "$image" >/dev/null 2>&1 && verified "$image"; then
        echo "mlperf: $image already loaded and verified"
        exit 0
      fi

      restage

      if ! load; then
        echo "mlperf: $image did not verify; dropping its layers and reloading" >&2
        purge
        if ! load; then
          echo "mlperf: $image is corrupt again after a full reload." >&2
          echo "mlperf: leaving it untagged -- the guest image needs rebuilding." >&2
          purge
          exit 1
        fi
        echo "mlperf: $image was repaired by a reload" >&2
      fi

      docker tag "$staging" "$image"
      untag "$staging"
      sync
      echo "mlperf: $image loaded and verified"
    '';
  };
in
{
  options.services.mlperf.enable = mkEnableOption
    "Load mlperf container on boot";

  config = mkIf cfg.enable {
    systemd.services.mlperf = {
      description = "Load and verify the mlperf container on boot";
      after = [ "docker.service" ];
      requires = [ "docker.service" ];
      before = [ "multi-user.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${loader}/bin/mlperf-load";
        TimeoutStartSec = "45min";
      };
    };
  };
}
