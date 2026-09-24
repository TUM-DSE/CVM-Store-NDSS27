{ lib, stdenv, fetchFromGitHub, kernel }:

let
  kbuild = "${kernel.dev}/lib/modules/${kernel.modDirVersion}/build";

  # required for server.c
  kcflags = "-Wno-error=int-conversion -Wno-error=incompatible-pointer-types";
in
stdenv.mkDerivation {
  pname = "dm-rollbaccine";
  version = "unstable-${kernel.modDirVersion}";

  src = fetchFromGitHub {
    owner = "davidchuyaya";
    repo = "rollbaccine";
    rev = "221c9aec60c35a4831ecd65f7715e871d5cc6744";
    hash = "sha256-NhYhKMJT2LMFm6LuRhtEbW6kipZ9UU7C0syxyeWidEE=";
  };

  nativeBuildInputs = kernel.moduleBuildDependencies;

  hardeningDisable = [ "all" ];

  makeFlags = kernel.makeFlags;

  postPatch = ''
    # Fix imports and functions for merkle.c
    sed -i '10i #include <linux/vmalloc.h>' src/merkle/merkle.c
    substituteInPlace src/merkle/merkle.c  --replace-fail "mem_is_zero" "mem_is_zero_internal"
  '';

  buildPhase = ''
    runHook preBuild

    # Build all kernel modules
    make -C ${kbuild} M=$(pwd)/src KCFLAGS="${kcflags}" modules
    for mod in encryption hash merkle network passthrough ping
    do
          make -C ${kbuild} M=$(pwd)/src/$mod KCFLAGS="${kcflags}" modules
    done

    # Build tools
    make -C src/tools all
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    for mod in $(find -iname '*.ko' 2>/dev/null)
    do
      echo "Installing: $mod"
      install -Dm444 $mod $out/lib/modules/${kernel.modDirVersion}/extra/$(basename $mod)
    done

    for tool in device_tester high_contention
    do
      install -Dm755 ./src/tools/$tool $out/bin/$tool
    done

    runHook postInstall
  '';

  meta = {
    description = "Rollbaccine device-mapper target for the 6.17 guest kernel";
    platforms = [ "x86_64-linux" ];
  };
}
