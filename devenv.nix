{ pkgs, ... }:
let
  support = import ./nix/support.nix { inherit pkgs; };
  check = "cargo check --workspace --all-targets --locked";
  clippy = "cargo clippy --workspace --all-targets --locked -- -D warnings";
  test = "cargo test --workspace --locked";
in
{
  name = "paddock";
  packages = support.nativeBuildInputs;
  platform.toolchains.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
    # TODO(adr-026): enable after gate is green — upstream workspace CI needs a measured baseline.
    git-hooks.enable = false;
    tasks.enable = true;
    # Preserve upstream formatting/lints; optional all-feature CUDA/Metal builds are separate gates.
    policy.enable = false;
    strict.enable = false;
    tasks.ci.exec = ''
      cargo fmt --all -- --check
      ${check}
      ${clippy}
      cargo deny check bans licenses sources
      RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
      ${test}
    '';
  };
  tasks."rust:check".exec = check;
  tasks."rust:clippy".exec = clippy;
  tasks."rust:test".exec = test;
  tasks."paddock:prepare" = {
    description = "Stage the hash-verified upstream PDFium archive for native builds";
    exec = "install -Dm644 ${support.pdfium} packs/pdfium/linux-x64/libpdfium.a";
    before = [
      "rust:ci"
      "rust:check"
      "rust:clippy"
      "rust:test"
    ];
  };
  platform.commit-conventions = {
    enable = true;
    requireWhy = true;
  };
}
