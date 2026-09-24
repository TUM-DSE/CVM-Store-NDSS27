{ lib
, python3Packages
, fetchPypi
}:

let
  # Not in nixpkgs.
  eciespy = python3Packages.buildPythonPackage rec {
    pname = "eciespy";
    version = "0.4.6";
    pyproject = true;
    src = fetchPypi {
      inherit pname version;
      hash = "sha256-qpgb89pw3ozcpHerZenJ4tEz/WDshPqdMYeZm8uVj5w=";
    };
    build-system = [ python3Packages.poetry-core ];
    dependencies = with python3Packages; [
      eth-keys
      coincurve
      pycryptodome
    ];
    doCheck = false;
  };
in
python3Packages.buildPythonApplication {
  pname = "attestation-poc";
  version = "0.1.0";
  pyproject = true;

  src = ../attestation-poc;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    requests
    pydantic
    pyjwt
    urllib3
    fastapi
    uvicorn
    eciespy
  ];

  doCheck = false;

  meta = with lib; {
    description = "Attestation benchmark client and server for SEV-SNP";
    mainProgram = "attestation-client";
    platforms = platforms.linux;
  };
}
