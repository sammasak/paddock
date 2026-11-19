{
  lib,
  rustPlatform,
  support,
  src,
}:
rustPlatform.buildRustPackage {
  pname = "paddock-runner";
  version = "0.1.21";
  inherit src;
  cargoLock = {
    lockFile = ../Cargo.lock;
    outputHashes = {
      "rav1d-1.1.0" = "sha256-HTucSd0i+OVMHSBce6LoxjxpwaUSB0CaXan6Ik8qYQU=";
      "scriptor-crdt-0.1.0" = "sha256-zztBhts1nt3GL0+2hJQ1uJe+GioO6oxKE4ByLDRVT5I=";
      "truespar-siftx-0.1.0" = "sha256-ELVOece+gfmJ6R3Ug+mtM666Wy0YU99oih8KPCSu1iI=";
    };
  };
  nativeBuildInputs = support.nativeBuildInputs;
  postPatch = ''
    install -Dm644 ${support.pdfium} packs/pdfium/linux-x64/libpdfium.a
  '';
  cargoBuildFlags = [
    "-p"
    "paddock-runner"
  ];
  cargoTestFlags = [
    "-p"
    "paddock-runner"
  ];
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    $out/bin/paddock-runner --help > "$TMPDIR/runner-help"
    grep -F -- --kernel-pack "$TMPDIR/runner-help"
    runHook postInstallCheck
  '';
  meta = {
    description = "Native Paddock inference runner; CUDA kernel pack supplied separately";
    license = with lib.licenses; [
      mit
      asl20
    ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "paddock-runner";
  };
}
