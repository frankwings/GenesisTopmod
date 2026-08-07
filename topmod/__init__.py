"""
topmod — Pure-Python TopMod library (DLFL half-edge mesh + operators).

Based on Dr. Ergun Akleman's theory: 4 minimal & complete operators
for orientable 2-manifold meshes.
"""

from .dlfl import DLFLMesh, Vertex, HalfEdge, Face, Edge
from .operators import create_vertex, delete_vertex, insert_edge, delete_edge
from .validate import is_manifold, check_all, face_loop_check, vertex_fan_check, twin_check, euler_check
from .primitives import make_cube, make_tetrahedron, make_icosahedron, make_octahedron
from .io import to_obj, from_obj, to_triangle_arrays
from .high_level_ops import (extrude_face, add_handle, stellate, stellate_all,
                             subdivide_edge, subdivide_face)
from .subdivision import catmull_clark
from .remeshing import (dual, doo_sabin, simplest_subdivide,
                        vertex_cutting, loop_subdivide, sqrt3_subdivide,
                        honeycomb_subdivide, star_subdivide, corner_cutting,
                        loop_style_subdivide, fractal_subdivide)
from .tokenizer import (
    TopModToken,
    quantize_coord, dequantize_coord,
    tokenize, detokenize,
    build_vocabulary, encode_sequence, decode_sequence,
    token_stats, sequence_length,
)

__all__ = [
    # Data structures
    "DLFLMesh", "Vertex", "HalfEdge", "Face", "Edge",
    # Fundamental operators
    "create_vertex", "delete_vertex", "insert_edge", "delete_edge",
    # Validation
    "is_manifold", "check_all", "face_loop_check", "vertex_fan_check",
    "twin_check", "euler_check",
    # Primitives
    "make_cube", "make_tetrahedron", "make_icosahedron", "make_octahedron",
    # IO
    "to_obj", "from_obj", "to_triangle_arrays",
    # High-level ops
    "extrude_face", "add_handle", "stellate", "stellate_all",
    "subdivide_edge", "subdivide_face",
    # Subdivision / remeshing
    "catmull_clark", "dual", "doo_sabin", "simplest_subdivide",
    "vertex_cutting", "loop_subdivide", "sqrt3_subdivide",
    "honeycomb_subdivide", "star_subdivide", "corner_cutting",
    "loop_style_subdivide", "fractal_subdivide",
    # Tokenizer
    "TopModToken",
    "quantize_coord", "dequantize_coord",
    "tokenize", "detokenize",
    "build_vocabulary", "encode_sequence", "decode_sequence",
    "token_stats", "sequence_length",
]
