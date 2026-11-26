{ pkgs }:
let
  inherit (pkgs) lib;
in
pkgs.cudaPackages_13_3.overrideScope (
  _: previous: {
    # This locked CCCL release already contains the patch still applied by Nixpkgs.
    cccl =
      assert previous.cccl.version == "13.3.3.4.1";
      previous.cccl.overrideAttrs (
        old:
        let
          stale = lib.filter (patch: (patch.name or "") == "fix-invalid-cpp-syntax") old.patches;
        in
        assert lib.length stale == 1;
        {
          patches = lib.subtractLists stale old.patches;
          postPatch = (old.postPatch or "") + ''
            # Reverse dry-run proves the source already contains both corrected hunks.
            patch --dry-run --reverse -p1 < ${lib.head stale}
          '';
        }
      );
  }
)
