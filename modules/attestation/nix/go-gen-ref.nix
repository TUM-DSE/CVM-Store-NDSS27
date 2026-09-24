{ lib
, buildGoModule
, fetchFromGitHub
}:

buildGoModule {
  pname = "go-gen-ref";
  version = "unstable-2026-02-11";

  src = fetchFromGitHub {
    owner = "THS-on";
    repo = "go-gen-ref";
    rev = "72123d91f4007e9f2f66ddcd26d55af41637790a";
    hash = "sha256-xV278A4aq4We1LN3AJCJCRE+IGOxdmEarCAPEocWxCM=";
  };

  vendorHash = "sha256-srNxYdy35gi8GiYplbZJBdWZ17BqJUBOhKVwGb0nwU4=";

  meta = with lib; {
    description = "Tool to generate reference values in CoRIM format for SEV-SNP";
    homepage = "https://github.com/THS-on/go-gen-ref";
    license = licenses.asl20;
    platforms = platforms.linux;
    mainProgram = "go-gen-ref";
  };
}
