flake:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.rycharger;

  settingsFormat = pkgs.formats.toml { };
  configFile = settingsFormat.generate "config.toml" cfg.settings;
in
{
  options.services.rycharger = {
    enable = lib.mkEnableOption "rycharger battery charge management daemon";

    package = lib.mkPackageOption flake.packages.${pkgs.system} "default" { };

    user = lib.mkOption {
      type = lib.types.str;
      description = "User account under which rycharger runs.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "rycharger";
      description = "Group under which rycharger runs.";
    };

    settings = lib.mkOption {
      type = settingsFormat.type;
      default = { };
      description = ''
        Configuration for rycharger, written to config.toml.
        See upstream documentation for available options.
      '';
      example = lib.literalExpression ''
        {
          battery = {
            device = "BAT0";
            hold_percent = 80;
            full_percent = 100;
          };
          daemon = {
            poll_interval_secs = 60;
          };
        }
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    users.groups.${cfg.group} = { };

    systemd.services.rycharger = {
      description = "Rycharger battery charge management daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "sys-subsystem-power_supply.target" ];

      environment = {
        XDG_CONFIG_HOME = "/var/lib/rycharger/config";
        XDG_DATA_HOME = "/var/lib/rycharger/data";
      };

      preStart = ''
        mkdir -p "$XDG_CONFIG_HOME/rycharger" "$XDG_DATA_HOME/rycharger"
        ln -sf ${configFile} "$XDG_CONFIG_HOME/rycharger/config.toml"
      '';

      serviceConfig = {
        Type = "simple";
        ExecStart = lib.getExe cfg.package;
        Restart = "on-failure";
        RestartSec = 10;

        User = cfg.user;
        Group = cfg.group;

        StateDirectory = "rycharger";

        AmbientCapabilities = [ "CAP_DAC_OVERRIDE" ];
        CapabilityBoundingSet = [ "CAP_DAC_OVERRIDE" ];

        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [
          "/sys/class/power_supply"
          "/var/lib/rycharger"
        ];
        PrivateTmp = true;
        ProtectKernelTunables = false; # needed for /sys/ writes
      };
    };
  };
}
