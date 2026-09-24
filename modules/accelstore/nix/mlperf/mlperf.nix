{ runCommand
, dockerTools
, buildEnv
, callPackage
, accelstorePackages
, pyaccelstore
, imageName ? "mlperf"
, imageTag ? "nix"
, baseImage ? null
}:

if baseImage == null then
  runCommand "mlperf-image-missing" { passthru = { inherit imageName imageTag; }; } ''
    cat >&2 <<'EOF'
    The mlperf base image is not set.

    Build it with the host's docker:
        nix run .#build-mlperf-image
    then build this package against the tar it wrote:
        nix build .#mlperf --override-input mod-accelstore/mlperf-base-image "file+file://$PWD/build/mlperf-base-image.tar"

    Or let the harness do both against the guest image:
        oma-arvio accelstore.build-guest --mlperf
    EOF
    exit 1
  ''
else
  let
    accelstoreBin = buildEnv {
      name = "accelstore-image-bin";
      paths = accelstorePackages;
      pathsToLink = [ "/bin" ];
      ignoreCollisions = true;
    };

    accelstoreRoot = runCommand "mlperf-accelstore-root" { } ''
      mkdir -p $out/usr/local/bin $out/opt/accelstore
      for bin in ${accelstoreBin}/bin/*; do
        ln -s "$bin" "$out/usr/local/bin/$(basename "$bin")"
      done
      ln -s ${pyaccelstore} $out/opt/accelstore/python
    '';

    bridge = callPackage ./isolated-bridge.nix { inherit pyaccelstore; };
  in
  dockerTools.buildImage {
    name = imageName;
    tag = imageTag;
    fromImage = baseImage;

    copyToRoot = buildEnv {
      name = "mlperf-extra-contents";
      paths = [ accelstoreRoot bridge ];
    };

    config = {
      Entrypoint = [ "mlpstorage" ];
      WorkingDir = "/opt/mlcommons/storage";

      Env = [
        "PYACCELSTORE_LIB=${bridge.library}"
        "MLPS_ACCELSTORE_PYTHON_DIR=/opt/accelstore/python"
        "ACCELSTORE_PYTHON_DIR=/opt/accelstore/python"
        "PYTHONPATH=/opt/accelstore/python"
      ];
    };
  }
