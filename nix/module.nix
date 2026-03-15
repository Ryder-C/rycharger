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
    environment.etc."rycharger/config.toml".source = configFile;

    systemd.services.rycharger = {
      description = "Rycharger battery charge management daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "sys-subsystem-power_supply.target" ];

      environment = {
        XDG_CONFIG_HOME = "/etc";
        XDG_DATA_HOME = "/var/lib";
      };

      serviceConfig = {
        Type = "simple";
        ExecStart = lib.getExe cfg.package;
        Restart = "on-failure";
        RestartSec = 10;
        StateDirectory = "rycharger";
      };
    };
  };
}
