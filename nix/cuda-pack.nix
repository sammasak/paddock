{
  lib,
  stdenv,
  cudaPackages,
  binutils,
  src,
  # Compute capabilities to compile SASS for, without the "sm_" prefix.
  cudaArches ? [ "86" ],
}:
assert lib.assertMsg (
  lib.versions.major cudaPackages.cuda_nvcc.version == "13"
) "Paddock requires a CUDA 13.x kernel compiler";
assert lib.assertMsg (cudaArches != [ ]) "cudaArches must name at least one architecture";
let
  tag = "sm" + lib.concatStringsSep "-sm" cudaArches;
  pack = "$out/lib/paddock/pd-cuda-${tag}.so";
in
stdenv.mkDerivation {
  pname = "paddock-cuda-${tag}";
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
    bash packs/cuda/build.sh ${lib.concatStringsSep "," cudaArches}
    runHook postBuild
  '';
  # build.sh always names its output pd-cuda-sm120.so, whatever the arch list.
  installPhase = ''
    runHook preInstall
    install -Dm755 packs/cuda/build/pd-cuda-sm120.so ${pack}
    runHook postInstall
  '';
  # build.sh silently skips arches the toolkit does not know, so prove each one landed.
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    readelf -h ${pack}
    cuobjdump --list-elf ${pack} > "$TMPDIR/pack-elf"
    ${lib.concatMapStrings (arch: ''
      grep -F sm_${arch} "$TMPDIR/pack-elf"
    '') cudaArches}
    runHook postInstallCheck
  '';
  meta = {
    description = "Paddock CUDA kernel pack compiled for ${
      lib.concatMapStringsSep ", " (a: "sm_${a}") cudaArches
    }";
    license = with lib.licenses; [
      mit
      asl20
    ];
    platforms = [ "x86_64-linux" ];
  };
}
