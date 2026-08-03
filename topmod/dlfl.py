"""
DLFL — Doubly-Linked Face List (half-edge mesh representation).

Every edge is represented by exactly two oriented HalfEdges (twins).
This structural invariant guarantees the mesh is always an orientable 2-manifold.

Traversal primitives
--------------------
Face loop  : he, he.next, he.next.next, ... until back to he
Vertex fan : he, he.twin.next, he.twin.next.twin.next, ... until back to he
"""

from __future__ import annotations
from typing import Iterator, List, Optional, Tuple
import itertools


# ── ID counters ────────────────────────────────────────────────────────────────

_vertex_id_counter  = itertools.count(1)
_halfedge_id_counter = itertools.count(1)
_face_id_counter    = itertools.count(1)
_edge_id_counter    = itertools.count(1)


# ── Core elements ──────────────────────────────────────────────────────────────

class Vertex:
    """A mesh vertex with a 3-D position and one outgoing half-edge."""

    __slots__ = ("id", "x", "y", "z", "he")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.id: int = next(_vertex_id_counter)
        self.x  = float(x)
        self.y  = float(y)
        self.z  = float(z)
        self.he: Optional[HalfEdge] = None   # any outgoing half-edge

    # ------------------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @position.setter
    def position(self, xyz: Tuple[float, float, float]) -> None:
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    # ------------------------------------------------------------------
    def outgoing_halfedges(self) -> List[HalfEdge]:
        """Return all outgoing half-edges in vertex fan order."""
        if self.he is None:
            return []
        result = []
        cur = self.he
        while True:
            result.append(cur)
            nxt = cur.twin.next if cur.twin is not None else None
            if nxt is None or nxt is self.he:
                break
            cur = nxt
        return result

    def degree(self) -> int:
        return len(self.outgoing_halfedges())

    def __repr__(self) -> str:
        return f"Vertex({self.id}, pos=({self.x:.3f},{self.y:.3f},{self.z:.3f}))"


class HalfEdge:
    """One directed half of an edge (also called a *corner* in TopMod)."""

    __slots__ = ("id", "origin", "face", "next", "prev", "twin", "edge")

    def __init__(self):
        self.id: int = next(_halfedge_id_counter)
        self.origin: Optional[Vertex]   = None
        self.face:   Optional[Face]     = None
        self.next:   Optional[HalfEdge] = None
        self.prev:   Optional[HalfEdge] = None
        self.twin:   Optional[HalfEdge] = None
        self.edge:   Optional[Edge]     = None

    # ------------------------------------------------------------------
    @property
    def destination(self) -> Optional[Vertex]:
        return self.twin.origin if self.twin else None

    def face_loop(self) -> List[HalfEdge]:
        """All half-edges in this face's boundary loop, starting from self."""
        result = [self]
        cur = self.next
        while cur is not None and cur is not self:
            result.append(cur)
            cur = cur.next
        return result

    def __repr__(self) -> str:
        o = self.origin.id if self.origin else "?"
        d = self.twin.origin.id if (self.twin and self.twin.origin) else "?"
        f = self.face.id if self.face else "?"
        return f"HalfEdge({self.id}, {o}→{d}, face={f})"


class Face:
    """A polygonal face, accessed via one representative half-edge."""

    __slots__ = ("id", "he")

    def __init__(self):
        self.id: int = next(_face_id_counter)
        self.he: Optional[HalfEdge] = None   # any half-edge on this face's boundary

    # ------------------------------------------------------------------
    def halfedges(self) -> List[HalfEdge]:
        if self.he is None:
            return []
        return self.he.face_loop()

    def vertices(self) -> List[Vertex]:
        return [he.origin for he in self.halfedges()]

    def degree(self) -> int:
        return len(self.halfedges())

    def normal(self) -> Tuple[float, float, float]:
        """Newell normal (averaged over all polygon edges)."""
        verts = self.vertices()
        n = [0.0, 0.0, 0.0]
        m = len(verts)
        for i in range(m):
            c  = verts[i]
            nx = verts[(i + 1) % m]
            n[0] += (c.y - nx.y) * (c.z + nx.z)
            n[1] += (c.z - nx.z) * (c.x + nx.x)
            n[2] += (c.x - nx.x) * (c.y + nx.y)
        length = (n[0]**2 + n[1]**2 + n[2]**2) ** 0.5
        if length < 1e-12:
            return (0.0, 0.0, 1.0)
        return (n[0] / length, n[1] / length, n[2] / length)

    def centroid(self) -> Tuple[float, float, float]:
        verts = self.vertices()
        m = len(verts)
        if m == 0:
            return (0.0, 0.0, 0.0)
        return (sum(v.x for v in verts) / m,
                sum(v.y for v in verts) / m,
                sum(v.z for v in verts) / m)

    def __repr__(self) -> str:
        vids = [v.id for v in self.vertices()]
        return f"Face({self.id}, verts={vids})"


class Edge:
    """An undirected edge backed by two twin half-edges."""

    __slots__ = ("id", "he0", "he1")

    def __init__(self, he0: HalfEdge, he1: HalfEdge):
        self.id:  int = next(_edge_id_counter)
        self.he0: HalfEdge = he0
        self.he1: HalfEdge = he1

    def other(self, he: HalfEdge) -> HalfEdge:
        return self.he1 if he is self.he0 else self.he0

    def vertices(self) -> Tuple[Vertex, Vertex]:
        return (self.he0.origin, self.he1.origin)

    def faces(self) -> Tuple[Optional[Face], Optional[Face]]:
        return (self.he0.face, self.he1.face)

    def __repr__(self) -> str:
        v0 = self.he0.origin.id if self.he0.origin else "?"
        v1 = self.he1.origin.id if self.he1.origin else "?"
        return f"Edge({self.id}, {v0}—{v1})"


# ── Mesh ───────────────────────────────────────────────────────────────────────

class DLFLMesh:
    """
    The DLFL mesh.

    Stores four disjoint sets:
        vertices   : dict[int, Vertex]
        halfedges  : dict[int, HalfEdge]
        faces      : dict[int, Face]
        edges      : dict[int, Edge]
    """

    def __init__(self):
        self.vertices:  dict[int, Vertex]   = {}
        self.halfedges: dict[int, HalfEdge] = {}
        self.faces:     dict[int, Face]     = {}
        self.edges:     dict[int, Edge]     = {}

    # ── factory helpers ────────────────────────────────────────────────

    def _new_vertex(self, x=0.0, y=0.0, z=0.0) -> Vertex:
        v = Vertex(x, y, z)
        self.vertices[v.id] = v
        return v

    def _new_halfedge(self) -> HalfEdge:
        he = HalfEdge()
        self.halfedges[he.id] = he
        return he

    def _new_face(self) -> Face:
        f = Face()
        self.faces[f.id] = f
        return f

    def _new_edge(self, he0: HalfEdge, he1: HalfEdge) -> Edge:
        e = Edge(he0, he1)
        he0.edge = e
        he1.edge = e
        he0.twin = he1
        he1.twin = he0
        self.edges[e.id] = e
        return e

    # ── deletion helpers ───────────────────────────────────────────────

    def _remove_vertex(self, v: Vertex) -> None:
        self.vertices.pop(v.id, None)

    def _remove_halfedge(self, he: HalfEdge) -> None:
        self.halfedges.pop(he.id, None)

    def _remove_face(self, f: Face) -> None:
        self.faces.pop(f.id, None)

    def _remove_edge(self, e: Edge) -> None:
        self.edges.pop(e.id, None)

    # ── topology queries ───────────────────────────────────────────────

    def V(self) -> int:
        return len(self.vertices)

    def E(self) -> int:
        return len(self.edges)

    def F(self) -> int:
        return len(self.faces)

    def euler_characteristic(self) -> int:
        return self.V() - self.E() + self.F()

    def genus(self) -> int:
        """
        For a closed orientable surface: χ = 2 - 2g  ⟹  g = (2 - χ) / 2.
        For multi-component: χ = 2C - 2g  ⟹  g = C - χ/2.
        """
        chi = self.euler_characteristic()
        C   = self.component_count()
        return C - chi // 2

    def component_count(self) -> int:
        """Connected components (union-find on vertices)."""
        if not self.vertices:
            return 0
        parent: dict[int, int] = {vid: vid for vid in self.vertices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in self.edges.values():
            if e.he0.origin and e.he1.origin:
                union(e.he0.origin.id, e.he1.origin.id)

        return len({find(vid) for vid in self.vertices})

    # ── iteration helpers ──────────────────────────────────────────────

    def iter_vertices(self) -> Iterator[Vertex]:
        return iter(list(self.vertices.values()))

    def iter_faces(self) -> Iterator[Face]:
        return iter(list(self.faces.values()))

    def iter_edges(self) -> Iterator[Edge]:
        return iter(list(self.edges.values()))

    def iter_halfedges(self) -> Iterator[HalfEdge]:
        return iter(list(self.halfedges.values()))

    # ── find helpers ───────────────────────────────────────────────────

    def find_edge(self, v0: Vertex, v1: Vertex) -> Optional[Edge]:
        """Return edge between v0 and v1, or None."""
        for he in v0.outgoing_halfedges():
            if he.destination is v1:
                return he.edge
        return None

    def find_halfedge(self, v_from: Vertex, v_to: Vertex) -> Optional[HalfEdge]:
        for he in v_from.outgoing_halfedges():
            if he.destination is v_to:
                return he
        return None

    # ── summary ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"DLFLMesh(V={self.V()}, E={self.E()}, F={self.F()}, "
                f"χ={self.euler_characteristic()}, genus={self.genus()})")
