{
  description = "attestation module: SEV-SNP attestation PoC (ratsd in the guest, Veraison on the host)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    core = {
      url = "github:TUM-DSE/OmaArvio?dir=core";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, core }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };

      ratsd = pkgs.callPackage ./nix/ratsd.nix { };
      attestation-poc = pkgs.callPackage ./nix/attestation-poc.nix { };
      go-gen-ref = pkgs.callPackage ./nix/go-gen-ref.nix { };
    in
    {
      packages.${system} = {
        inherit ratsd attestation-poc go-gen-ref;
      };

      nixosModules.guest = import ./nix/guest.nix { inherit ratsd attestation-poc; };

      lib = {
        hostPackages = pkgs:
          import ./nix/host.nix { inherit pkgs attestation-poc go-gen-ref; };
      } // core.lib.mkModulePythonLib { name = "attestation"; inherit self; };
    };
}
