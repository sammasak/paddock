{ pkgs, ... }:
let
  support = import ./nix/support.nix { inherit pkgs; };
  nativeTools = support.nativeBuildInputs ++ [ pkgs.gnumake ];
  prepare = "install -CDm644 ${support.pdfium} packs/pdfium/linux-x64/libpdfium.a";
  check = "cargo check --workspace --all-targets --locked";
  clippy = "cargo clippy --workspace --all-targets --locked -- -D warnings";
  test = "cargo test --workspace --locked --no-fail-fast";
  nixCheck = "nix build .#checks.x86_64-linux.nix-lint .#checks.x86_64-linux.input-parity --no-link";
in
{
  name = "paddock";
  # stdenv supplies Make in derivations; interactive Cargo/CMake needs it explicitly.
  packages = nativeTools;
  platform.toolchains.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
    # TODO(adr-026): enable after gate is green — upstream workspace CI needs a measured baseline.
    git-hooks.enable = false;
    # Preserve upstream formatting/lints; optional all-feature CUDA/Metal builds are separate gates.
    policy.enable = false;
    strict.enable = false;
    tasks = {
      enable = true;
      ci = {
        # The CI wrapper clears inherited PATH; shell packages alone are insufficient.
        extraInputs = [ pkgs.nix ] ++ nativeTools;
        exec = ''
          ${prepare}
          ${nixCheck}
          cargo fmt --all -- --check
          ${check}
          ${clippy}
          cargo deny check bans licenses sources
          RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
          ${test}
        '';
      };
    };
  };
  tasks = {
    "paddock:nix-check" = {
      description = "Check Nix formatting, static correctness and development/package pin parity";
      exec = nixCheck;
    };
    "rust:check".exec = check;
    "rust:clippy".exec = clippy;
    "rust:test".exec = test;
    "paddock:prepare" = {
      description = "Stage the hash-verified upstream PDFium archive for native builds";
      exec = prepare;
      before = [
        "rust:check"
        "rust:clippy"
        "rust:test"
      ];
    };
  };
  platform.commit-conventions = {
    enable = true;
    requireWhy = true;
  };
}
