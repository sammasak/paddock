{ pkgs, ... }:
let
  support = import ./nix/support.nix { inherit pkgs; };
  check = "cargo check --workspace --all-targets --locked";
  clippy = "cargo clippy --workspace --all-targets --locked -- -D warnings";
  test = "cargo test --workspace --locked";
  nixCheck = "nix build .#checks.x86_64-linux.nix-lint .#checks.x86_64-linux.input-parity --no-link";
in
{
  name = "paddock";
  packages = support.nativeBuildInputs;
  platform.toolchains.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
    # TODO(adr-026): enable after gate is green — cargo deny rejects to_method 1.1.0 (CC0-1.0); full CI also pending.
    git-hooks.enable = false;
    # Preserve upstream formatting/lints; optional all-feature CUDA/Metal builds are separate gates.
    policy.enable = false;
    strict.enable = false;
    tasks = {
      enable = true;
      ci = {
        extraInputs = [ pkgs.nix ];
        exec = ''
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
      exec = "install -Dm644 ${support.pdfium} packs/pdfium/linux-x64/libpdfium.a";
      before = [
        "rust:ci"
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
