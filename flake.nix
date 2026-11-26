{
  description = "Paddock native Linux runner and CUDA kernel pack";
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
      # Only x86_64-linux is built and supported (the pdfium prebuilt is linux-x64 only).
      # aarch64-linux is not declared: untested.
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
        config.allowUnfree = true;
      };
      # The Nix files and docs/ do not affect the build; leaving them out keeps
      # the source hash, and so the package, unchanged by edits to them.
      source = pkgs.lib.cleanSourceWith {
        src = self;
        name = "paddock-source";
        filter =
          path: _:
          let
            relative = pkgs.lib.removePrefix "${self}/" path;
          in
          !(builtins.elem relative [
            ".gitignore"
            "flake.nix"
            "flake.lock"
            "nix"
            "docs"
          ])
          && !(pkgs.lib.hasPrefix "nix/" relative || pkgs.lib.hasPrefix "docs/" relative);
      };
      support = import ./nix/support.nix { inherit pkgs; };
      cudaPackages = import ./nix/cuda-packages.nix { inherit pkgs; };
      rustPlatform = pkgs.makeRustPlatform {
        cargo = support.toolchain;
        rustc = support.toolchain;
      };
      # The runner is CPU-only and needs no CUDA toolkit; the CUDA kernel pack is a
      # separate output loaded at runtime over a C ABI, so each builds on its own.
      runner = pkgs.callPackage ./nix/runner.nix {
        inherit rustPlatform support;
        src = source;
      };
      # Other architectures: cudaPack.override { cudaArches = [ "86" "120" ]; }
      cudaPack = pkgs.callPackage ./nix/cuda-pack.nix {
        src = source;
        inherit cudaPackages;
      };
    in
    {
      packages.${system} = {
        default = runner;
        paddock-runner = runner;
        cargo-vendor = runner.cargoDeps;
        paddock-cuda-pack = cudaPack;
        # Kept as the name earlier users of this flake build.
        paddock-cuda-sm86 = cudaPack;
      };
      formatter.${system} = pkgs.nixfmt-tree;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ support.toolchain ] ++ support.nativeBuildInputs;
        # cudarc dlopens libcuda.so by name. On NixOS the driver lives only in
        # /run/opengl-driver/lib, and without it every GPU gate reports SKIP.
        shellHook = ''
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          fi
        '';
      };
      # Formatting plus evaluation only: building the runner or the pack is too heavy
      # for a check, and nix flake check already evaluates every output.
      checks.${system}.nix-format =
        pkgs.runCommand "paddock-nix-format" { nativeBuildInputs = [ pkgs.nixfmt ]; }
          ''
            cp ${./flake.nix} flake.nix
            cp -r ${./nix} nix && chmod -R u+w nix
            nixfmt --check flake.nix nix/*.nix
            touch $out
          '';
    };
}
