{ pkgs }:
let
  manifest = builtins.fromJSON (builtins.readFile ../packs/pdfium/prebuilt.json);
  asset = builtins.head (builtins.filter (file: file.platform == "linux-x64") manifest.files);
in
{
  toolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
  nativeBuildInputs = with pkgs; [
    cmake
    pkg-config
    nasm
  ];
  pdfium = pkgs.fetchurl {
    inherit (asset) url sha256;
  };
}
