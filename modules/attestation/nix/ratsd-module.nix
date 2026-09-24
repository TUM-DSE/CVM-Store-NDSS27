{ ratsd }: { config, lib, pkgs, ... }:

let
  cfg = config.services.ratsd;
in
{
  options.services.ratsd = {
    enable = lib.mkEnableOption "Remote Attestation Token Service Daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = ratsd;
      description = "The ratsd package to use";
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0:8895";
      description = "Address and port for ratsd to listen on";
    };

    pluginDir = lib.mkOption {
      type = lib.types.str;
      default = "${cfg.package}/lib/ratsd/attesters/bin";
      description = "Directory containing attester plugins";
    };

    logLevel = lib.mkOption {
      type = lib.types.enum [ "error" "warning" "info" "debug" ];
      default = "info";
      description = "Logging level for ratsd";
    };

    configFile = lib.mkOption {
      type = lib.types.path;
      description = "ratsd configuration file";
      default = pkgs.writeText "ratsd-config.yaml" ''
        logging:
          level: ${cfg.logLevel}

        ratsd:
          listen-addr: ${cfg.listenAddress}
          protocol: http
          plugin-dir: ${cfg.pluginDir}
          list-options: selected
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.ratsd = {
      description = "Remote Attestation Token Service Daemon";
      documentation = [ "https://github.com/veraison/ratsd" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/ratsd -c ${cfg.configFile}";
        WorkingDirectory = "/var/lib/ratsd";
        StateDirectory = "ratsd";
        ConfigurationDirectory = "ratsd";
        # The configfs-tsm attester needs root.
        User = "root";
        PrivateTmp = true;
        ReadWritePaths = [ "/var/lib/ratsd" ];
        Restart = "on-failure";
        RestartSec = "5s";
      };
    };
  };
}
