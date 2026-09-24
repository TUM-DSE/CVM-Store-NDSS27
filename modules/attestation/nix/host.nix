# snpguest/snphost are core inv tools already; listed here for devShells.benchmarking.

{ pkgs, attestation-poc, go-gen-ref }:

[
  attestation-poc
  go-gen-ref
  pkgs.snpguest
  pkgs.snphost
]
