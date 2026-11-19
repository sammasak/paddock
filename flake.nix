{
  description = "Paddock native Linux runner and Ampere kernel pack";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/6774f7bc253789b113a4f39285dc0fa100abeacc";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/b88764907c1b2de6c0e206d8d9d9634d91c81334";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
        config.allowUnfree = true;
      };
      # Build inputs exclude this fork's dev tooling and research notes.
      source = pkgs.lib.cleanSourceWith {
        src = self;
        name = "paddock-source";
        filter =
          path: _:
          let
            relative = pkgs.lib.removePrefix "${self}/" path;
          in
          !(builtins.elem relative [
            ".envrc"
            ".gitignore"
            "devenv.nix"
            "devenv.yaml"
            "devenv.lock"
            "deny.toml"
            "flake.nix"
            "flake.lock"
            "nix"
            "docs"
          ])
          && !(pkgs.lib.hasPrefix "nix/" relative || pkgs.lib.hasPrefix "docs/" relative);
      };
      support = import ./nix/support.nix { inherit pkgs; };
      rustPlatform = pkgs.makeRustPlatform {
        cargo = support.toolchain;
        rustc = support.toolchain;
      };
      runner = pkgs.callPackage ./nix/runner.nix {
        inherit rustPlatform support;
        src = source;
      };
      devLock = builtins.fromJSON (builtins.readFile ./devenv.lock);
    in
    {
      packages.${system} = {
        default = runner;
        paddock-runner = runner;
        cargo-vendor = runner.cargoDeps;
        paddock-cuda-sm86 = pkgs.callPackage ./nix/cuda-sm86.nix {
          src = source;
          cudaPackages = pkgs.cudaPackages_13_3;
        };
      };
      formatter.${system} = pkgs.nixfmt-tree;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ support.toolchain ] ++ support.nativeBuildInputs;
      };
      checks.${system} = {
        inherit runner;
        input-parity =
          assert nixpkgs.rev == devLock.nodes.nixpkgs.locked.rev;
          assert rust-overlay.rev == devLock.nodes.rust-overlay.locked.rev;
          pkgs.writeText "paddock-input-parity" "packaging and devenv input revisions match\n";
      };
    };
}
