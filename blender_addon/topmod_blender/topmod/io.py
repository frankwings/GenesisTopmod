"""
OBJ import / export for DLFLMesh.

to_obj(mesh, path)  — write an OBJ file
from_obj(path)      — read an OBJ file and return a DLFLMesh
"""

from __future__ import annotations
import os
from typing import List, Tuple

from .dlfl import DLFLMesh
from .primitives import _build_mesh   # reuse the wiring helper


# ── export ────────────────────────────────────────────────────────────────────

def to_obj(mesh: DLFLMesh, path: str) -> None:
    """
    Export *mesh* to Wavefront OBJ format.

    Non-triangular faces are exported as-is (OBJ supports n-gons).
    """
    # Collect vertices in a stable order; assign 1-based indices
    vid_to_idx: dict[int, int] = {}
    vlist = list(mesh.vertices.values())
    for i, v in enumerate(vlist, start=1):
        vid_to_idx[v.id] = i

    lines: List[str] = [
        "# DLFLMesh OBJ export",
        f"# V={mesh.V()} E={mesh.E()} F={mesh.F()}",
        "",
    ]

    for v in vlist:
        lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")

    lines.append("")

    for face in mesh.faces.values():
        verts = face.vertices()
        if not verts:
            continue
        indices = " ".join(str(vid_to_idx[v.id]) for v in verts)
        lines.append(f"f {indices}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── import ────────────────────────────────────────────────────────────────────

def from_obj(path: str) -> DLFLMesh:
    """
    Import a Wavefront OBJ file and return a DLFLMesh.

    Limitations:
    - Only 'v' and 'f' lines are parsed (normals and UVs are ignored).
    - The OBJ must describe a closed, orientable 2-manifold (no boundaries).
    - Face indices may be 1-based (standard OBJ) or negative (relative).
    - Multi-object OBJ files: all objects merged into one mesh.
    """
    positions: List[Tuple[float, float, float]] = []
    face_indices: List[List[int]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            keyword = parts[0].lower()

            if keyword == "v" and len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append((x, y, z))

            elif keyword == "f":
                # OBJ face: "f v1 v2 v3" or "f v1/vt1/vn1 v2/vt2/vn2 ..."
                indices: List[int] = []
                n = len(positions)
                for token in parts[1:]:
                    raw = int(token.split("/")[0])
                    # Convert to 0-based
                    idx = raw - 1 if raw > 0 else n + raw
                    indices.append(idx)
                face_indices.append(indices)

    if not positions:
        raise ValueError(f"No vertices found in {path}")
    if not face_indices:
        raise ValueError(f"No faces found in {path}")

    return _build_mesh(positions, face_indices)


# ── triangle arrays (for rendering) ─────────────────────────────────────────

def to_triangle_arrays(mesh: DLFLMesh) -> Tuple[List[Tuple[float, float, float]],
                                                  List[Tuple[int, int, int]]]:
    """
    Convert a DLFLMesh to flat vertex-position and triangle-index arrays.

    Non-triangular faces are fan-triangulated from the first vertex.

    Returns:
        (positions, triangles) where:
        - positions: list of (x, y, z) tuples, one per vertex
        - triangles: list of (i, j, k) 0-based index tuples
    """
    # Stable vertex ordering with 0-based index map
    vid_to_idx: dict[int, int] = {}
    positions: List[Tuple[float, float, float]] = []
    for i, v in enumerate(mesh.vertices.values()):
        vid_to_idx[v.id] = i
        positions.append((v.x, v.y, v.z))

    triangles: List[Tuple[int, int, int]] = []
    for face in mesh.faces.values():
        verts = face.vertices()
        if len(verts) < 3:
            continue
        # Fan triangulation from vertex 0
        v0 = vid_to_idx[verts[0].id]
        for j in range(1, len(verts) - 1):
            v1 = vid_to_idx[verts[j].id]
            v2 = vid_to_idx[verts[j + 1].id]
            triangles.append((v0, v1, v2))

    return positions, triangles
