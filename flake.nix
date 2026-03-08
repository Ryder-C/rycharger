{
  description = "Intelligent battery charge management daemon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "rycharger";
            version = "0.1.0";
            src = ./.;
            cargoHash = "sha256-gnb/V8utfZxMX/HssfKxnwl6dt4598Cb5ahbLMwdOv8=";
          };
        }
      );

      nixosModules.default = import ./nix/module.nix self;
    };
}
