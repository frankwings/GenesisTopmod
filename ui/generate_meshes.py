#!/usr/bin/env python3
"""Generate pre-computed OBJ meshes for the interactive UI.

Run from the GenesisTopmod root:
    python3 ui/generate_meshes.py [--out-dir ui/meshes]

Produces icosahedron_g{0..3}_cc{0..3}.obj files (16 total).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topmod import (
    make_icosahedron, add_handle, catmull_clark, check_all, to_obj,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=os.path.join(os.path.dirname(__file__), "meshes"),
        help="Output directory for OBJ files (default: ui/meshes/)",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for genus in range(4):
        for cc_level in range(4):
            key = f"icosahedron_g{genus}_cc{cc_level}"
            mesh = make_icosahedron()

            # Add handles
            for i in range(genus):
                faces = list(mesh.faces.values())
                n = len(faces)
                f1 = faces[i % n]
                f2 = faces[(n // 2 + i) % n]
                if f1.id == f2.id:
                    f2 = faces[(n // 2 + i + 1) % n]
                add_handle(mesh, f1, f2)

            # Catmull-Clark subdivision
            for _ in range(cc_level):
                mesh = catmull_clark(mesh)

            ok, errs = check_all(mesh)
            obj_path = os.path.join(args.out_dir, f"{key}.obj")
            to_obj(mesh, obj_path)
            status = "OK" if ok else f"MANIFOLD_ERR({len(errs)})"
            print(
                f"{key}: V={len(mesh.vertices)} E={len(mesh.edges)} "
                f"F={len(mesh.faces)} genus={mesh.genus()} [{status}]"
            )

    print(f"\nDone. Files in {args.out_dir}/")


if __name__ == "__main__":
    main()
