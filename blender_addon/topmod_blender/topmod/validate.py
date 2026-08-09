"""
Manifold invariant checks for DLFLMesh.

is_manifold(mesh)       -> bool   (comprehensive check)
euler_check(mesh)       -> bool   (χ = 2C - 2g is integer-consistent)
face_loop_check(mesh)   -> bool   (every face loop closes properly)
vertex_fan_check(mesh)  -> bool   (vertex fans are consistent)
"""

from __future__ import annotations
from typing import List, Tuple

from .dlfl import DLFLMesh, HalfEdge, Vertex, Face


# ─────────────────────────────────────────────────────────────────────────────

def face_loop_check(mesh: DLFLMesh) -> Tuple[bool, List[str]]:
    """
    Verify that every face's boundary loop:
    1. Closes (following .next returns to start).
    2. Has consistent .prev pointers.
    3. Each half-edge's .face pointer matches the face being checked.
    4. Each half-edge's origin vertex exists in the mesh.
    """
    errors: List[str] = []

    for face in mesh.iter_faces():
        if face.he is None:
            errors.append(f"Face {face.id} has no entry half-edge")
            continue

        visited = set()
        cur = face.he
        max_steps = len(mesh.halfedges) + 1
        step = 0

        while True:
            if cur.id in visited:
                if cur is not face.he:
                    errors.append(
                        f"Face {face.id} loop does not return to start "
                        f"(returned to HalfEdge {cur.id} instead)"
                    )
                break
            visited.add(cur.id)

            # face pointer
            if cur.face is not face:
                errors.append(
                    f"HalfEdge {cur.id} in face {face.id} has wrong face ptr "
                    f"(points to face {cur.face.id if cur.face else None})"
                )

            # .next / .prev consistency
            if cur.next is None:
                errors.append(f"HalfEdge {cur.id} in face {face.id} has None .next")
                break
            if cur.next.prev is not cur:
                errors.append(
                    f"HalfEdge {cur.id}.next.prev ≠ HalfEdge {cur.id} "
                    f"(face {face.id})"
                )

            # origin exists
            if cur.origin is None:
                errors.append(
                    f"HalfEdge {cur.id} in face {face.id} has None origin"
                )
            elif cur.origin.id not in mesh.vertices:
                errors.append(
                    f"HalfEdge {cur.id} in face {face.id}: origin vertex "
                    f"{cur.origin.id} not in mesh"
                )

            cur = cur.next
            step += 1
            if step > max_steps:
                errors.append(
                    f"Face {face.id} loop appears infinite (> {max_steps} steps)"
                )
                break

    ok = len(errors) == 0
    return ok, errors


def vertex_fan_check(mesh: DLFLMesh) -> Tuple[bool, List[str]]:
    """
    Verify that for every vertex v:
    1. v.he (if set) originates at v.
    2. The vertex fan (he, he.twin.next, ...) closes properly.
    3. Every outgoing half-edge's twin originates at the vertex's neighbours.
    """
    errors: List[str] = []

    for v in mesh.iter_vertices():
        if v.he is None:
            continue  # isolated; ok for degenerate vertices

        if v.he.origin is not v:
            errors.append(
                f"Vertex {v.id}.he does not originate at v "
                f"(originates at {v.he.origin.id if v.he.origin else None})"
            )
            continue

        visited = set()
        cur = v.he
        max_steps = len(mesh.halfedges) + 1
        step = 0

        while True:
            if cur.id in visited:
                if cur is not v.he:
                    errors.append(
                        f"Vertex {v.id} fan does not close (returns to "
                        f"HalfEdge {cur.id})"
                    )
                break
            visited.add(cur.id)

            if cur.origin is not v:
                errors.append(
                    f"Vertex {v.id} fan: HalfEdge {cur.id} origin is not v"
                )

            if cur.twin is None:
                errors.append(
                    f"Vertex {v.id} fan: HalfEdge {cur.id} has no twin"
                )
                break

            if cur.twin.twin is not cur:
                errors.append(
                    f"HalfEdge {cur.id}.twin.twin ≠ HalfEdge {cur.id} "
                    f"(vertex {v.id} fan)"
                )

            nxt = cur.twin.next
            if nxt is None:
                errors.append(
                    f"Vertex {v.id} fan: HalfEdge {cur.id}.twin.next is None"
                )
                break

            cur = nxt
            step += 1
            if step > max_steps:
                errors.append(
                    f"Vertex {v.id} fan appears infinite (> {max_steps} steps)"
                )
                break

    ok = len(errors) == 0
    return ok, errors


def twin_check(mesh: DLFLMesh) -> Tuple[bool, List[str]]:
    """
    Every half-edge must have a twin, and twin.twin must equal itself.
    Every edge's two half-edges must be mutual twins.
    """
    errors: List[str] = []

    for he in mesh.iter_halfedges():
        if he.twin is None:
            errors.append(f"HalfEdge {he.id} has no twin")
        elif he.twin.twin is not he:
            errors.append(
                f"HalfEdge {he.id}.twin.twin ≠ self "
                f"(twin={he.twin.id}, twin.twin={he.twin.twin.id if he.twin.twin else None})"
            )

    for e in mesh.iter_edges():
        if e.he0.twin is not e.he1:
            errors.append(
                f"Edge {e.id}: he0.twin ≠ he1 "
                f"(he0={e.he0.id}, he1={e.he1.id}, he0.twin={e.he0.twin.id if e.he0.twin else None})"
            )
        if e.he1.twin is not e.he0:
            errors.append(
                f"Edge {e.id}: he1.twin ≠ he0"
            )

    # Every half-edge in the mesh must belong to exactly one edge
    he_in_edges: set[int] = set()
    for e in mesh.iter_edges():
        for he in (e.he0, e.he1):
            if he.id in he_in_edges:
                errors.append(f"HalfEdge {he.id} appears in more than one edge")
            he_in_edges.add(he.id)

    for he in mesh.iter_halfedges():
        if he.id not in he_in_edges:
            errors.append(f"HalfEdge {he.id} is not referenced by any edge")

    ok = len(errors) == 0
    return ok, errors


def euler_check(mesh: DLFLMesh) -> Tuple[bool, List[str]]:
    """
    Euler characteristic must satisfy χ = 2C - 2g where:
        C = number of connected components
        g = genus (non-negative integer)

    Equivalently: (2C - χ) must be even and non-negative.
    """
    errors: List[str] = []

    V   = mesh.V()
    E   = mesh.E()
    F   = mesh.F()
    chi = V - E + F
    C   = mesh.component_count()

    remainder = 2 * C - chi
    if remainder < 0:
        errors.append(
            f"Euler: 2C - χ = {remainder} < 0  "
            f"(V={V}, E={E}, F={F}, χ={chi}, C={C})"
        )
    if remainder % 2 != 0:
        errors.append(
            f"Euler: 2C - χ = {remainder} is odd  "
            f"(V={V}, E={E}, F={F}, χ={chi}, C={C})"
        )

    ok = len(errors) == 0
    return ok, errors


def is_manifold(mesh: DLFLMesh) -> bool:
    """
    Full manifold invariant check.  Returns True if all sub-checks pass.
    """
    for check_fn in (twin_check, face_loop_check, vertex_fan_check, euler_check):
        ok, _ = check_fn(mesh)
        if not ok:
            return False
    return True


def check_all(mesh: DLFLMesh) -> Tuple[bool, List[str]]:
    """Run every check and collect all error messages."""
    all_errors: List[str] = []

    for check_fn in (twin_check, face_loop_check, vertex_fan_check, euler_check):
        _, errs = check_fn(mesh)
        all_errors.extend(errs)

    return len(all_errors) == 0, all_errors
