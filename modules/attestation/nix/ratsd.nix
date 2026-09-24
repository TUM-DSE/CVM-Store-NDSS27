# THS-on fork: carries the configfs-tsm (SEV-SNP) attester plugin.

{ lib
, buildGoModule
, fetchFromGitHub
}:

buildGoModule {
  pname = "ratsd";
  version = "unstable-2025-01-21";

  src = fetchFromGitHub {
    owner = "THS-on";
    repo = "ratsd";
    rev = "12a6fbcc86b49ffeab0b325c4aae44dc6818467a";
    hash = "sha256-Sv141cBKSRLpIFPC2Lz4hw/FfVrtQfe2TG5IPs86aCs=";
  };

  vendorHash = "sha256-orXohYYGz2rpWKvQr89usugRCBqUol10PSxM9eo5qfE=";

  # `make build` also builds the attester plugins.
  buildPhase = ''
    runHook preBuild
    make build
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 ratsd $out/bin/ratsd
    mkdir -p $out/lib/ratsd/attesters/bin
    find attesters -name "*.plugin" -type f -exec \
      install -Dm755 {} $out/lib/ratsd/attesters/bin/ \;
    runHook postInstall
  '';

  meta = with lib; {
    description = "Remote Attestation Token Service Daemon";
    homepage = "https://github.com/veraison/ratsd";
    license = licenses.asl20;
    platforms = platforms.linux;
    mainProgram = "ratsd";
  };
}
