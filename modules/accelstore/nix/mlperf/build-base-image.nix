{ writeShellApplication
, coreutils
, mlperf-src
, dlio-src
, imageName ? "mlperf"
, imageTag ? "nix-base"
}:

writeShellApplication {
  name = "build-mlperf-image";
  runtimeInputs = [ coreutils ];
  text = ''
    out=''${1:-$PWD/build/mlperf-base-image.tar}
    out_dir=$(dirname "$out")


    if ! command -v docker >/dev/null; then
      echo "build-mlperf-image: no 'docker' on PATH" >&2
      exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
      echo "build-mlperf-image: cannot reach the docker daemon" >&2
      echo "  try: sudo systemctl start docker" >&2
      exit 1
    fi

    ctx=$(mktemp -d)
    trap 'rm -rf "$ctx"' EXIT

    cp ${./Dockerfile} "$ctx/Dockerfile"
    cp -r --no-preserve=mode,ownership ${mlperf-src} "$ctx/mlperf-src"
    cp -r --no-preserve=mode,ownership ${dlio-src} "$ctx/dlio-src"

    docker build -t ${imageName}:${imageTag} \
      --build-arg SRC=mlperf-src --build-arg DLIO_SRC=dlio-src "$ctx"
    mkdir -p "$out_dir"
    docker save ${imageName}:${imageTag} -o "$out"

    out=$(realpath "$out")
    echo
    echo "wrote $out"
    echo "now build the nix-customized image with:"
    echo "  nix build .#mlperf --override-input mod-accelstore/mlperf-base-image \"file+file://$out\""
    echo "or put it in a guest image with:"
    echo "  oma-arvio accelstore.build-guest --mlperf --no-build-mlperf-base-image --mlperf-base-image $out"
  '';
}
