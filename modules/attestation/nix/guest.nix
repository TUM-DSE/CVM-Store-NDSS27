# snpguest already comes with core's guest base.

{ ratsd, attestation-poc }: { ... }:

{
  imports = [ (import ./ratsd-module.nix { inherit ratsd; }) ];

  services.ratsd.enable = true;

  environment.systemPackages = [ ratsd attestation-poc ];
}
