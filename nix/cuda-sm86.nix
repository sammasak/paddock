{
  lib,
  stdenv,
  cudaPackages,
  binutils,
  src,
}:
assert lib.assertMsg (
  lib.versions.major cudaPackages.cuda_nvcc.version == "13"
) "Paddock requires a CUDA 13.x kernel compiler";
stdenv.mkDerivation {
  pname = "paddock-cuda-sm86";
  version = "0.1.21";
  inherit src;
  nativeBuildInputs = [
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cuobjdump
    binutils
  ];
  buildInputs = [ cudaPackages.cuda_cudart ];
  # Keep the build bounded; no GPU or display driver is needed to compile SASS.
  PD_BUILD_THREADS = "1";
  buildPhase = ''
    runHook preBuild
    bash packs/cuda/build.sh 86
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm755 packs/cuda/build/pd-cuda-sm120.so $out/lib/paddock/pd-cuda-sm86.so
    runHook postInstall
  '';
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    readelf -h $out/lib/paddock/pd-cuda-sm86.so
    cuobjdump --list-elf $out/lib/paddock/pd-cuda-sm86.so > "$TMPDIR/pack-elf"
    grep -F sm_86 "$TMPDIR/pack-elf"
    runHook postInstallCheck
  '';
  meta = {
    description = "Paddock CUDA kernel pack compiled only for compute capability 8.6";
    license = with lib.licenses; [
      mit
      asl20
    ];
    platforms = [ "x86_64-linux" ];
  };
}
