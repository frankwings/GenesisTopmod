#pragma once
/*
 * dlfl_core.h — C++ half-edge (DLFL) mesh kernel for GenesisTopmod.
 *
 * IDs are 1-based (0 = NULL_ID).  Slot 0 in each vector is a dummy.
 * Deleted items are marked alive=false and added to a free-list for reuse.
 */
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Id = uint32_t;
constexpr Id NULL_ID = 0;

struct Vertex  { double x, y, z; Id he;              bool alive; };
struct HalfEdge{ Id origin, face, next, prev, twin, edge; bool alive; };
struct Face    { Id he;                               bool alive; };
struct Edge    { Id he0, he1;                         bool alive; };

class Mesh {
public:
    std::vector<Vertex>   verts;    // slot 0 = dummy
    std::vector<HalfEdge> hes;
    std::vector<Face>     faces;
    std::vector<Edge>     edges;

    std::vector<Id> free_v, free_he, free_f, free_e;

    Mesh();

    // ── Allocation helpers ──────────────────────────────────────────────
    Id new_v(double x, double y, double z);
    Id new_he();
    Id new_f();
    Id new_e(Id ha, Id hb);  // creates edge, sets twins

    void del_v(Id id);
    void del_he(Id id);
    void del_f(Id id);
    void del_e(Id id);

    // ── Counts ─────────────────────────────────────────────────────────
    int V() const;
    int E() const;
    int F() const;

    // ── Traversal ──────────────────────────────────────────────────────
    std::vector<Id> fan_halfedges(Id v_id) const;
    std::vector<Id> face_halfedges(Id f_id) const;
    Id find_edge(Id va, Id vb) const;        // NULL_ID if not found
    bool adjacent(Id va, Id vb) const;       // any edge between va,vb?

    // ── Fundamental operators ───────────────────────────────────────────
    // insert_edge: inserts new edge between origins of he1 and he2
    //   returns new edge id (-same-face split or cross-face merge)
    Id insert_edge(Id he1, Id he2);
    // delete_edge: merges two adjacent faces, returns surviving face id
    Id delete_edge(Id edge_id);

    // ── High-level operations ───────────────────────────────────────────
    // Returns surviving vertex id, or NULL_ID if guard fails
    Id collapse_edge_tri(Id edge_id);
    // Returns new midpoint vertex id
    Id subdivide_edge(Id edge_id);
    // Stellate: returns new center vertex id
    Id stellate(Id face_id);
    // Edge flip: returns true if flipped
    bool try_flip(Id edge_id, double fold_cos = 0.0);

    // ── Validation ─────────────────────────────────────────────────────
    bool validate(std::vector<std::string>& errors) const;

    // ── I/O ────────────────────────────────────────────────────────────
    // Build from F[nf×3] triangle array (0-based indices)
    void build_from_arrays(const double* V, int64_t nv,
                           const int64_t* F, int64_t nf);
    // Build from general polygon list
    void build_from_polys(const double* V, int64_t nv,
                          const std::vector<std::vector<int64_t>>& polys);
    // Export triangle arrays (fan-triangulates non-tri faces)
    void to_arrays(std::vector<double>& V_out,
                   std::vector<int64_t>& F_out) const;
    // Export polygon lists (for catmull-clark output)
    void to_poly_arrays(std::vector<double>& V_out,
                        std::vector<std::vector<int64_t>>& polys_out) const;
};
