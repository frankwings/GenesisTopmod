#include <algorithm>
/*
 * dlfl_core.cpp — C++ half-edge DLFL mesh kernel.
 *
 * All operations match the semantics of the corresponding Python functions
 * in topmod/ (high_level_ops.py, operators.py, dlfl_untangle.py, etc.).
 */

#include "dlfl_core.h"
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <set>
#include <map>
#include <queue>

// ═══════════════════════════════════════════════════════════════════════════
// Mesh primitives
// ═══════════════════════════════════════════════════════════════════════════

Mesh::Mesh() {
    // Slot 0 = dummy
    Vertex  dv{}; dv.alive = false; verts.push_back(dv);
    HalfEdge dh{}; dh.alive = false; hes.push_back(dh);
    Face    df{}; df.alive = false; faces.push_back(df);
    Edge    de{}; de.alive = false; edges.push_back(de);
}

Id Mesh::new_v(double x, double y, double z) {
    Id id;
    if (!free_v.empty()) {
        id = free_v.back(); free_v.pop_back();
        verts[id] = Vertex{x, y, z, NULL_ID, true, ++seq_counter};
    } else {
        id = (Id)verts.size();
        verts.push_back(Vertex{x, y, z, NULL_ID, true, ++seq_counter});
    }
    return id;
}

Id Mesh::new_he() {
    Id id;
    if (!free_he.empty()) {
        id = free_he.back(); free_he.pop_back();
        hes[id] = HalfEdge{NULL_ID, NULL_ID, NULL_ID, NULL_ID, NULL_ID, NULL_ID, true};
    } else {
        id = (Id)hes.size();
        hes.push_back(HalfEdge{NULL_ID, NULL_ID, NULL_ID, NULL_ID, NULL_ID, NULL_ID, true});
    }
    return id;
}

Id Mesh::new_f() {
    Id id;
    if (!free_f.empty()) {
        id = free_f.back(); free_f.pop_back();
        faces[id] = Face{NULL_ID, true, ++seq_counter};
    } else {
        id = (Id)faces.size();
        faces.push_back(Face{NULL_ID, true, ++seq_counter});
    }
    return id;
}

Id Mesh::new_e(Id ha, Id hb) {
    Id id;
    if (!free_e.empty()) {
        id = free_e.back(); free_e.pop_back();
        edges[id] = Edge{ha, hb, true, ++seq_counter};
    } else {
        id = (Id)edges.size();
        edges.push_back(Edge{ha, hb, true, ++seq_counter});
    }
    hes[ha].twin = hb;
    hes[hb].twin = ha;
    hes[ha].edge = id;
    hes[hb].edge = id;
    return id;
}

void Mesh::del_v(Id id)  { if (id && id < verts.size())  { verts[id].alive  = false; free_v.push_back(id);  } }
void Mesh::del_he(Id id) { if (id && id < hes.size())    { hes[id].alive    = false; free_he.push_back(id); } }
void Mesh::del_f(Id id)  { if (id && id < faces.size())  { faces[id].alive  = false; free_f.push_back(id);  } }
void Mesh::del_e(Id id)  { if (id && id < edges.size())  { edges[id].alive  = false; free_e.push_back(id);  } }

int Mesh::V() const { int c=0; for (size_t i=1;i<verts.size();i++) if (verts[i].alive) c++; return c; }
int Mesh::E() const { int c=0; for (size_t i=1;i<edges.size();i++) if (edges[i].alive) c++; return c; }
int Mesh::F() const { int c=0; for (size_t i=1;i<faces.size();i++) if (faces[i].alive) c++; return c; }

// ═══════════════════════════════════════════════════════════════════════════
// Traversal
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Id> Mesh::fan_halfedges(Id v_id) const {
    std::vector<Id> result;
    if (!v_id || v_id >= verts.size() || !verts[v_id].alive) return result;
    Id start = verts[v_id].he;
    if (!start) return result;
    Id cur = start;
    int guard = 0;
    do {
        result.push_back(cur);
        Id tw = hes[cur].twin;
        if (!tw || tw >= hes.size() || !hes[tw].alive) break;
        cur = hes[tw].next;
        if (!cur || cur >= hes.size() || !hes[cur].alive) break;
        if (++guard > 100000) break; // safety
    } while (cur != start);
    return result;
}

std::vector<Id> Mesh::face_halfedges(Id f_id) const {
    std::vector<Id> result;
    if (!f_id || f_id >= faces.size() || !faces[f_id].alive) return result;
    Id start = faces[f_id].he;
    if (!start) return result;
    Id cur = start;
    int guard = 0;
    do {
        result.push_back(cur);
        cur = hes[cur].next;
        if (!cur || cur >= hes.size() || !hes[cur].alive) break;
        if (++guard > 100000) break;
    } while (cur != start);
    return result;
}

Id Mesh::find_edge(Id va, Id vb) const {
    auto fan = fan_halfedges(va);
    for (Id h : fan) {
        Id tw = hes[h].twin;
        if (tw && tw < hes.size() && hes[tw].origin == vb)
            return hes[h].edge;
    }
    return NULL_ID;
}

bool Mesh::adjacent(Id va, Id vb) const {
    return find_edge(va, vb) != NULL_ID;
}

// ═══════════════════════════════════════════════════════════════════════════
// Fundamental operator: insert_edge
// Inserts a new edge between origins of he1 and he2.
// ═══════════════════════════════════════════════════════════════════════════

Id Mesh::insert_edge(Id he1, Id he2) {
    Id f1 = hes[he1].face;
    Id f2 = hes[he2].face;

    Id new_he1 = new_he();
    Id new_he2 = new_he();
    Id new_edge = new_e(new_he1, new_he2);

    hes[new_he1].origin = hes[he1].origin;
    hes[new_he2].origin = hes[he2].origin;
    hes[new_he1].face   = f1;
    hes[new_he2].face   = f2;

    Id prev1 = hes[he1].prev;
    Id prev2 = hes[he2].prev;

    // Insert new_he1 before he1, leading into he2
    hes[prev1].next  = new_he1;
    hes[new_he1].prev = prev1;
    hes[new_he1].next = he2;
    hes[he2].prev    = new_he1;

    // Insert new_he2 before he2, leading into he1
    hes[prev2].next  = new_he2;
    hes[new_he2].prev = prev2;
    hes[new_he2].next = he1;
    hes[he1].prev    = new_he2;

    if (f1 == f2) {
        // Same face → split into two
        Id new_face = new_f();

        // new_he1's loop → new_face
        Id cur = new_he1;
        int guard = 0;
        do {
            hes[cur].face = new_face;
            cur = hes[cur].next;
            if (++guard > 100000) break;
        } while (cur != new_he1);
        faces[new_face].he = new_he1;

        // new_he2's loop → f1
        cur = new_he2;
        guard = 0;
        do {
            hes[cur].face = f1;
            cur = hes[cur].next;
            if (++guard > 100000) break;
        } while (cur != new_he2);
        faces[f1].he = new_he2;

        return new_edge;
    } else {
        // Different faces → merge into f1, remove f2
        Id cur = new_he1;
        int guard = 0;
        do {
            hes[cur].face = f1;
            cur = hes[cur].next;
            if (++guard > 100000) break;
        } while (cur != new_he1);
        faces[f1].he = new_he1;
        del_f(f2);

        return new_edge;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fundamental operator: delete_edge
// Merges two adjacent faces. Returns surviving face id.
// ═══════════════════════════════════════════════════════════════════════════

Id Mesh::delete_edge(Id edge_id) {
    Id he_a = edges[edge_id].he0;
    Id he_b = edges[edge_id].he1;

    Id f_a = hes[he_a].face;
    Id f_b = hes[he_b].face;

    Id prev_a = hes[he_a].prev;
    Id next_a = hes[he_a].next;
    Id prev_b = hes[he_b].prev;
    Id next_b = hes[he_b].next;

    hes[prev_a].next = next_b;
    hes[next_b].prev = prev_a;
    hes[prev_b].next = next_a;
    hes[next_a].prev = prev_b;

    if (f_a != f_b) {
        // Re-label f_b halfedges to f_a
        Id cur = next_b;
        int guard = 0;
        do {
            hes[cur].face = f_a;
            cur = hes[cur].next;
            if (++guard > 100000) break;
        } while (cur != next_b);
        del_f(f_b);
    }

    // f_a.he: pick a non-deleted halfedge
    faces[f_a].he = (next_a != he_a) ? next_a : next_b;

    // Fix vertex outgoing pointers
    Id v_a = hes[he_a].origin;
    Id v_b = hes[he_b].origin;
    if (verts[v_a].he == he_a) {
        verts[v_a].he = (next_b != he_a && hes[next_b].origin == v_a) ? next_b : NULL_ID;
        if (!verts[v_a].he) {
            // Search fan from next_a or next_b
            for (Id h : {next_a, next_b, prev_a, prev_b}) {
                if (h && h < hes.size() && hes[h].alive && hes[h].origin == v_a) {
                    verts[v_a].he = h; break;
                }
            }
        }
    }
    if (verts[v_b].he == he_b) {
        verts[v_b].he = (next_a != he_b && hes[next_a].origin == v_b) ? next_a : NULL_ID;
        if (!verts[v_b].he) {
            for (Id h : {next_a, next_b, prev_a, prev_b}) {
                if (h && h < hes.size() && hes[h].alive && hes[h].origin == v_b) {
                    verts[v_b].he = h; break;
                }
            }
        }
    }

    del_he(he_a);
    del_he(he_b);
    del_e(edge_id);

    return f_a;
}

// ═══════════════════════════════════════════════════════════════════════════
// collapse_edge_tri
// Collapses edge in a triangle mesh.  Uses fan traversal for re-origining.
// Returns surviving vertex id, or NULL_ID if guard fails.
// ═══════════════════════════════════════════════════════════════════════════

Id Mesh::collapse_edge_tri(Id edge_id) {
    if (!edge_id || edge_id >= edges.size() || !edges[edge_id].alive)
        return NULL_ID;

    Id he  = edges[edge_id].he0;  // v0→v1, face T0
    Id ht  = edges[edge_id].he1;  // v1→v0, face T1

    if (!he || !ht) return NULL_ID;

    Id n0 = hes[he].next;   // v1→a
    Id p0 = hes[he].prev;   // a→v0
    Id n1 = hes[ht].next;   // v0→b
    Id p1 = hes[ht].prev;   // b→v1

    if (!n0 || !p0 || !n1 || !p1) return NULL_ID;

    // Both faces must be triangles
    if (face_halfedges(hes[he].face).size() != 3) return NULL_ID;
    if (face_halfedges(hes[ht].face).size() != 3) return NULL_ID;

    Id v0 = hes[he].origin;
    Id v1 = hes[ht].origin;

    // a = destination of n0, b = destination of n1
    Id n0t = hes[n0].twin;
    Id p0t = hes[p0].twin;
    Id n1t = hes[n1].twin;
    Id p1t = hes[p1].twin;

    if (!n0t || !p0t || !n1t || !p1t) return NULL_ID;

    Id a = hes[n0t].origin;  // destination of n0
    Id b = hes[n1t].origin;  // destination of n1

    if (!a || !b) return NULL_ID;
    if (a == b) return NULL_ID;
    if (a == v0 || b == v1) return NULL_ID;

    // Link condition: common neighbours of v0 and v1 must be exactly {a, b}
    std::unordered_set<Id> nbrs0, nbrs1;
    for (Id h : fan_halfedges(v0)) {
        Id tw = hes[h].twin;
        if (tw) nbrs0.insert(hes[tw].origin);
    }
    for (Id h : fan_halfedges(v1)) {
        Id tw = hes[h].twin;
        if (tw) nbrs1.insert(hes[tw].origin);
    }
    std::unordered_set<Id> common;
    for (Id x : nbrs0) if (nbrs1.count(x)) common.insert(x);
    if (common.size() != 2 || !common.count(a) || !common.count(b))
        return NULL_ID;

    // Move v0 to midpoint
    verts[v0].x = (verts[v0].x + verts[v1].x) * 0.5;
    verts[v0].y = (verts[v0].y + verts[v1].y) * 0.5;
    verts[v0].z = (verts[v0].z + verts[v1].z) * 0.5;

    // Fan traversal: re-origin all of v1's outgoing halfedges to v0
    {
        Id start = verts[v1].he;
        if (start) {
            Id cur = start;
            int guard = 0;
            do {
                hes[cur].origin = v0;
                Id tw = hes[cur].twin;
                if (!tw || tw >= hes.size()) break;
                cur = hes[tw].next;
                if (!cur || cur >= hes.size()) break;
                if (++guard > 100000) break;
            } while (cur != start);
        }
    }

    // Merge duplicate edges
    Id e_keep0 = hes[p0].edge;   // edge (a↔v0)
    Id e_del0  = hes[n0].edge;   // edge (v1↔a → now v0↔a)
    Id e_keep1 = hes[n1].edge;   // edge (v0↔b)
    Id e_del1  = hes[p1].edge;   // edge (b↔v1 → now b↔v0)

    // Wire: p0t ↔ n0t (replace the two edges that go v0↔a)
    hes[p0t].twin = n0t;
    hes[n0t].twin = p0t;
    hes[n0t].edge = e_keep0;

    // Wire: n1t ↔ p1t (replace the two edges that go v0↔b)
    hes[n1t].twin = p1t;
    hes[p1t].twin = n1t;
    hes[p1t].edge = e_keep1;

    edges[e_keep0].he0 = p0t;
    edges[e_keep0].he1 = n0t;
    edges[e_keep1].he0 = n1t;
    edges[e_keep1].he1 = p1t;

    // Fix vertex anchors
    verts[v0].he = p0t;  // p0t.origin = v0 (v0→a after re-origin)
    verts[a].he  = n0t;  // n0t.origin = a
    verts[b].he  = n1t;  // n1t.origin = b

    // Validate v0.he (p0t should have origin v0 now)
    if (hes[p0t].origin != v0) {
        // Fallback: search
        for (Id h : fan_halfedges(v0)) { verts[v0].he = h; break; }
    }

    // Remove dead elements
    Id f_T0 = hes[he].face;
    Id f_T1 = hes[ht].face;
    del_f(f_T0);
    del_f(f_T1);
    del_he(he); del_he(ht);
    del_he(n0); del_he(p0);
    del_he(n1); del_he(p1);
    del_e(edge_id);
    del_e(e_del0);
    del_e(e_del1);
    del_v(v1);

    return v0;
}

// ═══════════════════════════════════════════════════════════════════════════
// subdivide_edge
// Splits edge at midpoint.  Returns new midpoint vertex id.
// ═══════════════════════════════════════════════════════════════════════════

Id Mesh::subdivide_edge(Id edge_id) {
    if (!edge_id || edge_id >= edges.size() || !edges[edge_id].alive)
        return NULL_ID;

    Id he_ab = edges[edge_id].he0;  // v0→v1
    Id he_ba = edges[edge_id].he1;  // v1→v0

    Id v0 = hes[he_ab].origin;
    Id v1 = hes[he_ba].origin;

    double mx = (verts[v0].x + verts[v1].x) * 0.5;
    double my = (verts[v0].y + verts[v1].y) * 0.5;
    double mz = (verts[v0].z + verts[v1].z) * 0.5;

    Id mid = new_v(mx, my, mz);

    Id new_fwd = new_he();  // mid→v1  on face of he_ab
    Id new_bwd = new_he();  // v1→mid  on face of he_ba

    hes[new_fwd].origin = mid;
    hes[new_bwd].origin = v1;
    hes[new_fwd].face   = hes[he_ab].face;
    hes[new_bwd].face   = hes[he_ba].face;

    new_e(new_fwd, new_bwd);  // new edge: mid—v1

    // Insert new_fwd after he_ab (in face of he_ab)
    Id next_ab = hes[he_ab].next;
    hes[he_ab].next   = new_fwd;
    hes[new_fwd].prev = he_ab;
    hes[new_fwd].next = next_ab;
    hes[next_ab].prev = new_fwd;

    // Change he_ba origin to mid; insert new_bwd before he_ba
    Id prev_ba = hes[he_ba].prev;
    hes[prev_ba].next = new_bwd;
    hes[new_bwd].prev = prev_ba;
    hes[new_bwd].next = he_ba;
    hes[he_ba].prev   = new_bwd;
    hes[he_ba].origin = mid;  // he_ba now goes mid→v0

    verts[mid].he = new_fwd;  // mid's outgoing: mid→v1

    // Fix v1.he if it was he_ba (which now goes mid→v0)
    if (verts[v1].he == he_ba)
        verts[v1].he = new_bwd;

    return mid;
}

// ═══════════════════════════════════════════════════════════════════════════
// stellate
// Stellation: add center vertex, split face into n triangles.
// Returns new center vertex id.
// ═══════════════════════════════════════════════════════════════════════════

Id Mesh::stellate(Id face_id) {
    auto fhes = face_halfedges(face_id);
    int n = (int)fhes.size();
    if (n < 3) return NULL_ID;

    // Centroid
    double cx=0, cy=0, cz=0;
    for (Id h : fhes) {
        cx += verts[hes[h].origin].x;
        cy += verts[hes[h].origin].y;
        cz += verts[hes[h].origin].z;
    }
    cx /= n; cy /= n; cz /= n;

    Id center = new_v(cx, cy, cz);

    // Snapshot: orig vertices and exterior twins
    std::vector<Id> orig_verts(n), ext_twins(n);
    for (int i = 0; i < n; i++) {
        orig_verts[i] = hes[fhes[i]].origin;
        ext_twins[i]  = hes[fhes[i]].twin;
    }

    // Remove original face and its halfedges/edges
    del_f(face_id);
    for (int i = 0; i < n; i++) {
        Id e = hes[fhes[i]].edge;
        if (e) del_e(e);
        del_he(fhes[i]);
    }

    // Build n triangles: (orig[i], orig[i+1], center)
    std::vector<Id> he_base(n), he_right(n), he_left(n);

    for (int i = 0; i < n; i++) {
        Id tf = new_f();
        Id b  = new_he();  // orig[i]   → orig[i+1]
        Id r  = new_he();  // orig[i+1] → center
        Id l  = new_he();  // center    → orig[i]

        hes[b].origin = orig_verts[i];
        hes[r].origin = orig_verts[(i+1)%n];
        hes[l].origin = center;

        hes[b].face = hes[r].face = hes[l].face = tf;
        faces[tf].he = b;

        hes[b].next = r; hes[r].prev = b;
        hes[r].next = l; hes[l].prev = r;
        hes[l].next = b; hes[b].prev = l;

        he_base[i]  = b;
        he_right[i] = r;
        he_left[i]  = l;
    }

    // Wire base ↔ exterior twins
    for (int i = 0; i < n; i++)
        new_e(he_base[i], ext_twins[i]);

    // Wire spokes: he_right[i] ↔ he_left[(i+1)%n]
    for (int i = 0; i < n; i++)
        new_e(he_right[i], he_left[(i+1)%n]);

    // Fix vertex pointers
    verts[center].he = he_left[0];
    for (int i = 0; i < n; i++) {
        Id v = orig_verts[i];
        if (!verts[v].he || !hes[verts[v].he].alive || hes[verts[v].he].origin != v)
            verts[v].he = he_base[i];
    }

    return center;
}

// ═══════════════════════════════════════════════════════════════════════════
// try_flip
// Flip an interior edge (both faces triangles).
// Returns true if the flip was performed.
// ═══════════════════════════════════════════════════════════════════════════

static void tri_normal(double ax, double ay, double az,
                       double bx, double by, double bz,
                       double cx, double cy, double cz,
                       double& nx, double& ny, double& nz, double& len) {
    double ux = bx-ax, uy = by-ay, uz = bz-az;
    double vx = cx-ax, vy = cy-ay, vz = cz-az;
    nx = uy*vz - uz*vy;
    ny = uz*vx - ux*vz;
    nz = ux*vy - uy*vx;
    len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-14) { nx/=len; ny/=len; nz/=len; }
}

bool Mesh::try_flip(Id edge_id, double fold_cos) {
    if (!edge_id || edge_id >= edges.size() || !edges[edge_id].alive)
        return false;

    Id he0 = edges[edge_id].he0;
    Id he1 = edges[edge_id].he1;

    Id fa = hes[he0].face;
    Id fb = hes[he1].face;

    if (!fa || !fb) return false;
    if (face_halfedges(fa).size() != 3) return false;
    if (face_halfedges(fb).size() != 3) return false;

    Id v0 = hes[he0].origin;
    Id v1 = hes[he1].origin;

    // c = third vertex of fa (prev of he0)
    Id c = hes[hes[he0].prev].origin;
    // d = third vertex of fb (prev of he1)
    Id d = hes[hes[he1].prev].origin;

    if (c == d || c == v0 || c == v1 || d == v0 || d == v1)
        return false;

    auto& vc0 = verts[v0]; auto& vc1 = verts[v1];
    auto& vcc = verts[c];  auto& vcd = verts[d];

    double na_x, na_y, na_z, la;
    double nb_x, nb_y, nb_z, lb;
    tri_normal(vc0.x,vc0.y,vc0.z, vc1.x,vc1.y,vc1.z, vcc.x,vcc.y,vcc.z, na_x,na_y,na_z,la);
    tri_normal(vc1.x,vc1.y,vc1.z, vc0.x,vc0.y,vc0.z, vcd.x,vcd.y,vcd.z, nb_x,nb_y,nb_z,lb);

    if (la < 1e-14 || lb < 1e-14) return false;

    double cur_dot = na_x*nb_x + na_y*nb_y + na_z*nb_z;
    if (cur_dot >= fold_cos) return false;

    // Check c-d not already adjacent
    if (adjacent(c, d)) return false;

    // New normals after flip: tri(c,v0,d) and tri(d,v1,c)
    double n1_x, n1_y, n1_z, l1;
    double n2_x, n2_y, n2_z, l2;
    tri_normal(vcc.x,vcc.y,vcc.z, vc0.x,vc0.y,vc0.z, vcd.x,vcd.y,vcd.z, n1_x,n1_y,n1_z,l1);
    tri_normal(vcd.x,vcd.y,vcd.z, vc1.x,vc1.y,vc1.z, vcc.x,vcc.y,vcc.z, n2_x,n2_y,n2_z,l2);

    if (l1 < 1e-3*(la+lb) || l2 < 1e-3*(la+lb)) return false;
    double new_dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
    if (new_dot <= cur_dot + 1e-6) return false;

    // Perform flip: delete edge, then insert along c-d diagonal
    Id merged_face = delete_edge(edge_id);

    // Find halfedges with origin c and d in merged_face
    auto mhes = face_halfedges(merged_face);
    Id hc = NULL_ID, hd = NULL_ID;
    for (Id h : mhes) {
        if (hes[h].origin == c) hc = h;
        if (hes[h].origin == d) hd = h;
    }
    if (!hc || !hd) return false;  // shouldn't happen

    insert_edge(hc, hd);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation
// ═══════════════════════════════════════════════════════════════════════════

bool Mesh::validate(std::vector<std::string>& errors) const {
    errors.clear();
    bool ok = true;

    // Twin check
    for (size_t i = 1; i < hes.size(); i++) {
        if (!hes[i].alive) continue;
        Id tw = hes[i].twin;
        if (!tw || tw >= hes.size() || !hes[tw].alive) {
            errors.push_back("HE " + std::to_string(i) + " has invalid twin");
            ok = false;
        } else if (hes[tw].twin != (Id)i) {
            errors.push_back("HE " + std::to_string(i) + " twin not mutual");
            ok = false;
        }
    }

    // Face loop check
    for (size_t fi = 1; fi < faces.size(); fi++) {
        if (!faces[fi].alive) continue;
        auto loop = face_halfedges((Id)fi);
        if (loop.empty()) {
            errors.push_back("Face " + std::to_string(fi) + " has empty loop");
            ok = false;
        }
        for (Id h : loop) {
            if (hes[h].face != (Id)fi) {
                errors.push_back("HE " + std::to_string(h) + " in face " +
                    std::to_string(fi) + " loop but face field = " +
                    std::to_string(hes[h].face));
                ok = false;
            }
        }
    }

    // Vertex fan check
    for (size_t vi = 1; vi < verts.size(); vi++) {
        if (!verts[vi].alive) continue;
        auto fan = fan_halfedges((Id)vi);
        for (Id h : fan) {
            if (hes[h].origin != (Id)vi) {
                errors.push_back("V " + std::to_string(vi) +
                    " fan has HE with wrong origin");
                ok = false;
            }
        }
    }

    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════
// Build from arrays
// ═══════════════════════════════════════════════════════════════════════════

void Mesh::build_from_arrays(const double* V, int64_t nv,
                             const int64_t* F, int64_t nf) {
    // Clear
    verts.clear(); hes.clear(); faces.clear(); edges.clear();
    free_v.clear(); free_he.clear(); free_f.clear(); free_e.clear();
    Vertex  dv{}; dv.alive=false;  verts.push_back(dv);
    HalfEdge dh{}; dh.alive=false; hes.push_back(dh);
    Face    df{}; df.alive=false;  faces.push_back(df);
    Edge    de{}; de.alive=false;  edges.push_back(de);

    // Create vertices (IDs 1..nv)
    std::vector<Id> vid(nv);
    for (int64_t i = 0; i < nv; i++) {
        vid[i] = new_v(V[3*i], V[3*i+1], V[3*i+2]);
    }

    // edge_map: key = vi * nv + vj (0-based vi,vj) → he_id
    std::unordered_map<uint64_t, Id> edge_map;
    edge_map.reserve(nf * 6);

    uint64_t NV = (uint64_t)nv;

    for (int64_t fi = 0; fi < nf; fi++) {
        Id fa = new_f();
        int n = 3;  // triangles
        std::vector<Id> he_ids(n);
        for (int k = 0; k < n; k++) {
            Id h = new_he();
            he_ids[k] = h;
            int64_t vi = F[fi*n + k];
            hes[h].origin = vid[vi];
            hes[h].face   = fa;
            int64_t vj = F[fi*n + (k+1)%n];
            edge_map[(uint64_t)vi * NV + (uint64_t)vj] = h;
        }
        // Wire next/prev
        for (int k = 0; k < n; k++) {
            hes[he_ids[k]].next = he_ids[(k+1)%n];
            hes[he_ids[k]].prev = he_ids[(k-1+n)%n];
        }
        faces[fa].he = he_ids[0];
        // Set vertex.he first occurrence
        for (int k = 0; k < n; k++) {
            int64_t vi = F[fi*n + k];
            if (!verts[vid[vi]].he)
                verts[vid[vi]].he = he_ids[k];
        }
    }

    // Pair twins
    for (int64_t fi = 0; fi < nf; fi++) {
        for (int k = 0; k < 3; k++) {
            int64_t vi = F[fi*3 + k];
            int64_t vj = F[fi*3 + (k+1)%3];
            uint64_t key_ij = (uint64_t)vi * NV + (uint64_t)vj;
            uint64_t key_ji = (uint64_t)vj * NV + (uint64_t)vi;
            auto it_ij = edge_map.find(key_ij);
            auto it_ji = edge_map.find(key_ji);
            if (it_ij != edge_map.end() && it_ji != edge_map.end()) {
                Id he_ij = it_ij->second;
                if (!hes[he_ij].twin) {
                    Id he_ji = it_ji->second;
                    new_e(he_ij, he_ji);
                }
            }
        }
    }
}

void Mesh::build_from_polys(const double* V, int64_t nv,
                             const std::vector<std::vector<int64_t>>& polys) {
    // Clear
    verts.clear(); hes.clear(); faces.clear(); edges.clear();
    free_v.clear(); free_he.clear(); free_f.clear(); free_e.clear();
    Vertex  dv{}; dv.alive=false;  verts.push_back(dv);
    HalfEdge dh{}; dh.alive=false; hes.push_back(dh);
    Face    df{}; df.alive=false;  faces.push_back(df);
    Edge    de{}; de.alive=false;  edges.push_back(de);

    std::vector<Id> vid(nv);
    for (int64_t i = 0; i < nv; i++)
        vid[i] = new_v(V[3*i], V[3*i+1], V[3*i+2]);

    uint64_t NV = (uint64_t)nv;
    std::unordered_map<uint64_t, Id> edge_map;

    int64_t nf = (int64_t)polys.size();
    for (int64_t fi = 0; fi < nf; fi++) {
        const auto& vindex_list = polys[fi];
        int n = (int)vindex_list.size();
        Id fa = new_f();
        std::vector<Id> he_ids(n);
        for (int k = 0; k < n; k++) {
            Id h = new_he();
            he_ids[k] = h;
            int64_t vi = vindex_list[k];
            hes[h].origin = vid[vi];
            hes[h].face   = fa;
            int64_t vj = vindex_list[(k+1)%n];
            edge_map[(uint64_t)vi * NV + (uint64_t)vj] = h;
        }
        for (int k = 0; k < n; k++) {
            hes[he_ids[k]].next = he_ids[(k+1)%n];
            hes[he_ids[k]].prev = he_ids[(k-1+n)%n];
        }
        faces[fa].he = he_ids[0];
        for (int k = 0; k < n; k++) {
            int64_t vi = vindex_list[k];
            if (!verts[vid[vi]].he)
                verts[vid[vi]].he = he_ids[k];
        }
    }

    // Pair twins
    for (int64_t fi = 0; fi < nf; fi++) {
        const auto& vindex_list = polys[fi];
        int n = (int)vindex_list.size();
        for (int k = 0; k < n; k++) {
            int64_t vi = vindex_list[k];
            int64_t vj = vindex_list[(k+1)%n];
            uint64_t key_ij = (uint64_t)vi * NV + (uint64_t)vj;
            uint64_t key_ji = (uint64_t)vj * NV + (uint64_t)vi;
            auto it_ij = edge_map.find(key_ij);
            auto it_ji = edge_map.find(key_ji);
            if (it_ij != edge_map.end() && it_ji != edge_map.end()) {
                Id he_ij = it_ij->second;
                if (!hes[he_ij].twin) {
                    Id he_ji = it_ji->second;
                    new_e(he_ij, he_ji);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Export
// ═══════════════════════════════════════════════════════════════════════════

void Mesh::to_arrays(std::vector<double>& V_out,
                     std::vector<int64_t>& F_out) const {
    V_out.clear(); F_out.clear();

    // Collect alive vertices sorted by ID (= creation order)
    std::vector<Id> alive_vids;
    alive_vids.reserve(verts.size());
    for (size_t i = 1; i < verts.size(); i++)
        if (verts[i].alive) alive_vids.push_back((Id)i);
    std::stable_sort(alive_vids.begin(), alive_vids.end(), [&](Id a, Id b){ return verts[a].seq < verts[b].seq; });
    // IDs are monotonically assigned → already sorted

    // old_id → new_index (0-based)
    std::vector<int64_t> remap(verts.size(), -1);
    V_out.reserve(alive_vids.size() * 3);
    for (int64_t idx = 0; idx < (int64_t)alive_vids.size(); idx++) {
        Id vid = alive_vids[idx];
        remap[vid] = idx;
        V_out.push_back(verts[vid].x);
        V_out.push_back(verts[vid].y);
        V_out.push_back(verts[vid].z);
    }

    // Export faces (fan-triangulate non-tri faces)
    std::vector<Id> ord_f;
    for (size_t i = 1; i < faces.size(); i++) if (faces[i].alive) ord_f.push_back((Id)i);
    std::stable_sort(ord_f.begin(), ord_f.end(), [&](Id a, Id b){ return faces[a].seq < faces[b].seq; });
    for (Id fi : ord_f) {
        auto fhes = face_halfedges((Id)fi);
        int fn = (int)fhes.size();
        if (fn < 3) continue;
        int64_t v0 = remap[hes[fhes[0]].origin];
        if (v0 < 0) continue;
        for (int k = 1; k < fn-1; k++) {
            int64_t vk   = remap[hes[fhes[k]].origin];
            int64_t vk1  = remap[hes[fhes[k+1]].origin];
            if (vk < 0 || vk1 < 0) continue;
            F_out.push_back(v0);
            F_out.push_back(vk);
            F_out.push_back(vk1);
        }
    }
}

void Mesh::to_poly_arrays(std::vector<double>& V_out,
                          std::vector<std::vector<int64_t>>& polys_out) const {
    V_out.clear(); polys_out.clear();

    std::vector<Id> alive_vids;
    alive_vids.reserve(verts.size());
    for (size_t i = 1; i < verts.size(); i++)
        if (verts[i].alive) alive_vids.push_back((Id)i);
    std::stable_sort(alive_vids.begin(), alive_vids.end(), [&](Id a, Id b){ return verts[a].seq < verts[b].seq; });

    std::vector<int64_t> remap(verts.size(), -1);
    V_out.reserve(alive_vids.size() * 3);
    for (int64_t idx = 0; idx < (int64_t)alive_vids.size(); idx++) {
        Id vid = alive_vids[idx];
        remap[vid] = idx;
        V_out.push_back(verts[vid].x);
        V_out.push_back(verts[vid].y);
        V_out.push_back(verts[vid].z);
    }

    std::vector<Id> ord_f;
    for (size_t i = 1; i < faces.size(); i++) if (faces[i].alive) ord_f.push_back((Id)i);
    std::stable_sort(ord_f.begin(), ord_f.end(), [&](Id a, Id b){ return faces[a].seq < faces[b].seq; });
    for (Id fi : ord_f) {
        auto fhes = face_halfedges((Id)fi);
        if (fhes.empty()) continue;
        std::vector<int64_t> poly;
        poly.reserve(fhes.size());
        for (Id h : fhes) {
            int64_t idx = remap[hes[h].origin];
            if (idx < 0) goto skip_face;
            poly.push_back(idx);
        }
        polys_out.push_back(std::move(poly));
        skip_face:;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Catmull-Clark subdivision (builds a new Mesh)
// ═══════════════════════════════════════════════════════════════════════════

static Mesh catmull_clark_subdivide(const double* V_ptr, int64_t nv,
                                     const std::vector<std::vector<int64_t>>& polys) {
    // Build helper mesh just for traversal
    Mesh src;
    src.build_from_polys(V_ptr, nv, polys);

    int64_t nf = (int64_t)polys.size();

    // Collect alive faces in order
    std::vector<Id> face_ids;
    face_ids.reserve(nf);
    for (size_t i = 1; i < src.faces.size(); i++)
        if (src.faces[i].alive) face_ids.push_back((Id)i);

    std::vector<Id> edge_ids;
    for (size_t i = 1; i < src.edges.size(); i++)
        if (src.edges[i].alive) edge_ids.push_back((Id)i);

    std::vector<Id> vert_ids;
    for (size_t i = 1; i < src.verts.size(); i++)
        if (src.verts[i].alive) vert_ids.push_back((Id)i);

    int64_t V = (int64_t)vert_ids.size();
    int64_t E = (int64_t)edge_ids.size();
    int64_t F = (int64_t)face_ids.size();

    // Index maps (by ID)
    std::unordered_map<Id,int64_t> vid_to_idx, eid_to_idx, fid_to_idx;
    for (int64_t i = 0; i < V; i++) vid_to_idx[vert_ids[i]] = i;
    for (int64_t i = 0; i < E; i++) eid_to_idx[edge_ids[i]] = V + i;
    for (int64_t i = 0; i < F; i++) fid_to_idx[face_ids[i]] = V + E + i;

    // Step 1: face points (centroids)
    std::unordered_map<Id,std::array<double,3>> face_pts;
    for (Id fi : face_ids) {
        auto fhes = src.face_halfedges(fi);
        int fn = (int)fhes.size();
        double cx=0,cy=0,cz=0;
        for (Id h : fhes) {
            auto& v = src.verts[src.hes[h].origin];
            cx+=v.x; cy+=v.y; cz+=v.z;
        }
        face_pts[fi] = {cx/fn, cy/fn, cz/fn};
    }

    // Step 2: edge points
    std::unordered_map<Id,std::array<double,3>> edge_pts;
    for (Id ei : edge_ids) {
        Id ha = src.edges[ei].he0, hb = src.edges[ei].he1;
        auto& va = src.verts[src.hes[ha].origin];
        auto& vb = src.verts[src.hes[hb].origin];
        double ex = va.x+vb.x, ey = va.y+vb.y, ez = va.z+vb.z;
        int cnt = 2;
        Id fa = src.hes[ha].face, fb = src.hes[hb].face;
        if (fa && face_pts.count(fa)) {
            auto& fp = face_pts[fa];
            ex+=fp[0]; ey+=fp[1]; ez+=fp[2]; cnt++;
        }
        if (fb && face_pts.count(fb)) {
            auto& fp = face_pts[fb];
            ex+=fp[0]; ey+=fp[1]; ez+=fp[2]; cnt++;
        }
        edge_pts[ei] = {ex/cnt, ey/cnt, ez/cnt};
    }

    // Step 3: new vertex positions (CC formula)
    std::unordered_map<Id,std::array<double,3>> new_vpos;
    for (Id vi : vert_ids) {
        auto& v = src.verts[vi];
        auto fan = src.fan_halfedges(vi);
        int val = (int)fan.size();
        if (val == 0) { new_vpos[vi] = {v.x,v.y,v.z}; continue; }

        double Qx=0,Qy=0,Qz=0, Rx=0,Ry=0,Rz=0;
        int nf_adj=0, ne_adj=0;
        for (Id h : fan) {
            Id fi = src.hes[h].face;
            if (fi && face_pts.count(fi)) {
                auto& fp = face_pts[fi];
                Qx+=fp[0]; Qy+=fp[1]; Qz+=fp[2]; nf_adj++;
            }
            // edge midpoint
            Id tw = src.hes[h].twin;
            if (tw) {
                auto& dst = src.verts[src.hes[tw].origin];
                Rx += (v.x+dst.x)*0.5;
                Ry += (v.y+dst.y)*0.5;
                Rz += (v.z+dst.z)*0.5;
                ne_adj++;
            }
        }
        if (nf_adj == 0 || ne_adj == 0) { new_vpos[vi] = {v.x,v.y,v.z}; continue; }
        Qx/=nf_adj; Qy/=nf_adj; Qz/=nf_adj;
        Rx/=ne_adj; Ry/=ne_adj; Rz/=ne_adj;
        double n = val;
        new_vpos[vi] = {
            (Qx + 2*Rx + (n-3)*v.x) / n,
            (Qy + 2*Ry + (n-3)*v.y) / n,
            (Qz + 2*Rz + (n-3)*v.z) / n
        };
    }

    // Step 4: build new mesh
    int64_t total_verts = V + E + F;
    std::vector<double> new_V(total_verts * 3);

    for (int64_t i = 0; i < V; i++) {
        auto& p = new_vpos[vert_ids[i]];
        new_V[3*i]   = p[0];
        new_V[3*i+1] = p[1];
        new_V[3*i+2] = p[2];
    }
    for (int64_t i = 0; i < E; i++) {
        auto& p = edge_pts[edge_ids[i]];
        int64_t base = V+i;
        new_V[3*base]   = p[0];
        new_V[3*base+1] = p[1];
        new_V[3*base+2] = p[2];
    }
    for (int64_t i = 0; i < F; i++) {
        auto& p = face_pts[face_ids[i]];
        int64_t base = V+E+i;
        new_V[3*base]   = p[0];
        new_V[3*base+1] = p[1];
        new_V[3*base+2] = p[2];
    }

    // Each n-gon face → n quads
    // Quad: [face_point, prev_edge_point, orig_vertex, curr_edge_point]
    std::vector<std::vector<int64_t>> new_polys;
    for (Id fi : face_ids) {
        auto fhes = src.face_halfedges(fi);
        int64_t fp_idx = fid_to_idx[fi];
        for (Id he : fhes) {
            int64_t v_idx  = vid_to_idx[src.hes[he].origin];
            int64_t ep_idx = eid_to_idx[src.hes[he].edge];
            int64_t pe_idx = eid_to_idx[src.hes[src.hes[he].prev].edge];
            new_polys.push_back({fp_idx, pe_idx, v_idx, ep_idx});
        }
    }

    Mesh result;
    result.build_from_polys(new_V.data(), total_verts, new_polys);
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch function helpers
// ═══════════════════════════════════════════════════════════════════════════

static py::tuple make_VF_tuple(const std::vector<double>& V_out,
                               const std::vector<int64_t>& F_out) {
    int64_t nv = (int64_t)V_out.size() / 3;
    int64_t nf = (int64_t)F_out.size() / 3;

    py::array_t<double>  V2({nv, (int64_t)3});
    py::array_t<int64_t> F2({nf, (int64_t)3});

    if (nv > 0) std::memcpy(V2.mutable_data(), V_out.data(), V_out.size()*sizeof(double));
    if (nf > 0) std::memcpy(F2.mutable_data(), F_out.data(), F_out.size()*sizeof(int64_t));

    return py::make_tuple(V2, F2);
}

// ═══════════════════════════════════════════════════════════════════════════
// flip_sweep batch
// ═══════════════════════════════════════════════════════════════════════════

py::tuple batch_flip_sweep(py::array_t<double>  V_arr,
                           py::array_t<int64_t> F_arr,
                           int passes, double fold_cos) {
    auto V_ = V_arr.unchecked<2>();
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_.shape(0), nf = F_.shape(0);

    Mesh mesh;
    mesh.build_from_arrays(V_arr.data(), nv, F_arr.data(), nf);

    int total_flips = 0;
    for (int p = 0; p < passes; p++) {
        // Snapshot edge ids
        std::vector<Id> eids;
        eids.reserve(mesh.edges.size());
        for (size_t i = 1; i < mesh.edges.size(); i++)
            if (mesh.edges[i].alive) eids.push_back((Id)i);
        std::stable_sort(eids.begin(), eids.end(), [&](Id a, Id b){ return mesh.edges[a].seq < mesh.edges[b].seq; });   // Python snapshot order = creation order
        int n_flip = 0;
        for (Id eid : eids) {
            if (!mesh.edges[eid].alive) continue;
            if (mesh.try_flip(eid, fold_cos)) n_flip++;
        }
        total_flips += n_flip;
        if (n_flip == 0) break;
    }

    if (total_flips == 0) {
        // Return original arrays unchanged
        return py::make_tuple(V_arr, F_arr, 0);
    }

    std::vector<double>  V_out;
    std::vector<int64_t> F_out;
    mesh.to_arrays(V_out, F_out);

    auto vf = make_VF_tuple(V_out, F_out);
    return py::make_tuple(vf[0], vf[1], total_flips);
}

// ═══════════════════════════════════════════════════════════════════════════
// collapse_short_edges batch
// ═══════════════════════════════════════════════════════════════════════════

py::tuple batch_collapse(py::array_t<double>  V_arr,
                         py::array_t<int64_t> F_arr,
                         double ratio, int max_n, double thr_abs,
                         py::array_t<double> vthr_arr) {
    auto V_ = V_arr.unchecked<2>();
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_.shape(0), nf = F_.shape(0);

    Mesh mesh;
    mesh.build_from_arrays(V_arr.data(), nv, F_arr.data(), nf);

    // Compute edge lengths and threshold
    // edge_len2: edge_id → length^2
    std::vector<std::pair<double,Id>> sorted_edges;
    sorted_edges.reserve(mesh.edges.size());

    double sum_len = 0.0;
    int ne_cnt = 0;
    for (size_t i = 1; i < mesh.edges.size(); i++) {
        if (!mesh.edges[i].alive) continue;
        Id ha = mesh.edges[i].he0;
        auto& va = mesh.verts[mesh.hes[ha].origin];
        Id hb = mesh.edges[i].he1;
        auto& vb = mesh.verts[mesh.hes[hb].origin];
        double dx = va.x-vb.x, dy = va.y-vb.y, dz = va.z-vb.z;
        double len = std::sqrt(dx*dx+dy*dy+dz*dz);
        sum_len += len;
        ne_cnt++;
        sorted_edges.push_back({len, (Id)i});
    }
    std::sort(sorted_edges.begin(), sorted_edges.end());

    double thr;
    bool use_vthr = (vthr_arr.size() > 0);
    const double* vthr_ptr = use_vthr ? vthr_arr.data() : nullptr;

    if (thr_abs >= 0.0) {
        thr = thr_abs;
    } else {
        double mean_len = (ne_cnt > 0) ? sum_len / ne_cnt : 0.0;
        thr = ratio * mean_len;
    }
    if (use_vthr) {
        // thr = max(vthr) as sort/break bound
        thr = 0.0;
        for (int64_t i = 0; i < vthr_arr.size(); i++)
            thr = std::max(thr, vthr_ptr[i]);
    }
    double thr2 = thr * thr;

    int n_collapsed = 0;
    for (auto& [len, eid] : sorted_edges) {
        if (!mesh.edges[eid].alive) continue;
        // Recompute current length (edge midpoint may have moved)
        Id ha = mesh.edges[eid].he0;
        Id hb = mesh.edges[eid].he1;
        auto& va = mesh.verts[mesh.hes[ha].origin];
        auto& vb = mesh.verts[mesh.hes[hb].origin];
        double dx = va.x-vb.x, dy = va.y-vb.y, dz = va.z-vb.z;
        double len2 = dx*dx + dy*dy + dz*dz;

        if (len2 > thr2) break;

        if (use_vthr) {
            // Per-edge vthr check: edge must be < min(vthr[v0_row], vthr[v1_row])
            // In C++: vertex at ID k was input row k-1
            Id v0_id = mesh.hes[ha].origin;
            Id v1_id = mesh.hes[hb].origin;
            int64_t row0 = (int64_t)v0_id - 1;
            int64_t row1 = (int64_t)v1_id - 1;
            double vt = thr;
            if (row0 >= 0 && row0 < vthr_arr.size()) vt = std::min(vt, vthr_ptr[row0]);
            if (row1 >= 0 && row1 < vthr_arr.size()) vt = std::min(vt, vthr_ptr[row1]);
            if (len2 >= vt*vt) continue;
        }

        if (mesh.collapse_edge_tri(eid) != NULL_ID) {
            n_collapsed++;
            if (n_collapsed >= max_n) break;
        }
    }

    if (n_collapsed == 0) {
        return py::make_tuple(V_arr, F_arr, 0);
    }

    std::vector<double>  V_out;
    std::vector<int64_t> F_out;
    mesh.to_arrays(V_out, F_out);

    auto vf = make_VF_tuple(V_out, F_out);
    return py::make_tuple(vf[0], vf[1], n_collapsed);
}

// ═══════════════════════════════════════════════════════════════════════════
// subdivide_faces batch
// ═══════════════════════════════════════════════════════════════════════════

py::tuple batch_subdivide_faces(py::array_t<double>  V_arr,
                                py::array_t<int64_t> F_arr,
                                py::list fids_list,
                                bool expand_ring) {
    auto V_ = V_arr.unchecked<2>();
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_.shape(0), nf = F_.shape(0);

    Mesh mesh;
    mesh.build_from_arrays(V_arr.data(), nv, F_arr.data(), nf);

    // Build face vertex sets for ring expansion
    // Face fi in input → face ID fi+1 in mesh (1-based, created in order)
    std::unordered_set<int64_t> tgt;
    for (auto item : fids_list) tgt.insert(item.cast<int64_t>());

    if (expand_ring) {
        // Expand via edge-twin adjacency (O(target_faces * face_degree))
        // For each half-edge on each target face, its twin's face is an edge-neighbor
        // Build face-ID → input-row map
        std::unordered_map<Id, int64_t> fid_to_row;
        fid_to_row.reserve(nf);
        for (int64_t fi = 0; fi < nf; fi++) {
            Id fid = (Id)(fi + 1);
            fid_to_row[fid] = fi;
        }
        std::vector<int64_t> tgt_vec(tgt.begin(), tgt.end());
        for (int64_t fi : tgt_vec) {
            Id fid = (Id)(fi + 1);
            if (fid >= mesh.faces.size() || !mesh.faces[fid].alive) continue;
            for (Id h : mesh.face_halfedges(fid)) {
                Id tw = mesh.hes[h].twin;
                if (!tw || tw >= mesh.hes.size() || !mesh.hes[tw].alive) continue;
                Id adj_fid = mesh.hes[tw].face;
                auto it = fid_to_row.find(adj_fid);
                if (it != fid_to_row.end()) tgt.insert(it->second);
            }
        }
    }

    // Collect edges of target faces: ascending face order, first-occurrence dedup (matches the Python
    // reference: `for fi in tgt` over a set of small ints iterates ascending; midpoints are created in that order)
    std::vector<int64_t> tgt_sorted(tgt.begin(), tgt.end());
    std::sort(tgt_sorted.begin(), tgt_sorted.end());
    std::unordered_set<Id> edge_set;
    std::vector<Id> edges_to_split;
    for (int64_t fi : tgt_sorted) {
        Id fid = (Id)(fi + 1);
        if (fid >= mesh.faces.size() || !mesh.faces[fid].alive) continue;
        for (Id h : mesh.face_halfedges(fid)) {
            Id eid = mesh.hes[h].edge;
            if (eid && edge_set.insert(eid).second) edges_to_split.push_back(eid);
        }
    }
    int n_split = (int)edges_to_split.size();
    for (Id eid : edges_to_split) {
        if (eid >= mesh.edges.size() || !mesh.edges[eid].alive) continue;
        mesh.subdivide_edge(eid);
    }

    // Stellate all non-tri faces
    std::vector<Id> non_tri_faces;
    for (size_t i = 1; i < mesh.faces.size(); i++) {
        if (!mesh.faces[i].alive) continue;
        if (mesh.face_halfedges((Id)i).size() > 3)
            non_tri_faces.push_back((Id)i);
    }
    for (Id fid : non_tri_faces) {
        if (fid >= mesh.faces.size() || !mesh.faces[fid].alive) continue;
        mesh.stellate(fid);
    }

    std::vector<double>  V_out;
    std::vector<int64_t> F_out;
    mesh.to_arrays(V_out, F_out);

    int64_t nv2 = (int64_t)V_out.size()/3;
    int64_t nf2 = (int64_t)F_out.size()/3;

    py::array_t<double>  V2({nv2,(int64_t)3});
    py::array_t<int64_t> F2({nf2,(int64_t)3});
    if (nv2>0) std::memcpy(V2.mutable_data(), V_out.data(), V_out.size()*sizeof(double));
    if (nf2>0) std::memcpy(F2.mutable_data(), F_out.data(), F_out.size()*sizeof(int64_t));

    return py::make_tuple(V2, F2, n_split);
}

// ═══════════════════════════════════════════════════════════════════════════
// catmull_clark batch
// ═══════════════════════════════════════════════════════════════════════════

py::tuple batch_catmull_clark(py::array_t<double> V_arr, py::list polys_list) {
    int64_t nv = V_arr.shape(0);
    std::vector<std::vector<int64_t>> polys;
    polys.reserve(polys_list.size());
    for (auto p : polys_list) {
        auto pl = p.cast<py::list>();
        std::vector<int64_t> face;
        face.reserve(pl.size());
        for (auto idx : pl) face.push_back(idx.cast<int64_t>());
        polys.push_back(std::move(face));
    }

    Mesh result = catmull_clark_subdivide(V_arr.data(), nv, polys);

    std::vector<double> V_out;
    std::vector<std::vector<int64_t>> polys_out;
    result.to_poly_arrays(V_out, polys_out);

    int64_t nv2 = (int64_t)V_out.size()/3;
    py::array_t<double> V2({nv2,(int64_t)3});
    if (nv2>0) std::memcpy(V2.mutable_data(), V_out.data(), V_out.size()*sizeof(double));

    py::list polys2;
    for (auto& p : polys_out) {
        py::list pl;
        for (int64_t idx : p) pl.append(idx);
        polys2.append(pl);
    }

    return py::make_tuple(V2, polys2);
}

// ═══════════════════════════════════════════════════════════════════════════
// triangulate_all batch
// Fan triangulation from vertex 0 of each polygon.
// ═══════════════════════════════════════════════════════════════════════════

py::array_t<int64_t> batch_triangulate_all(py::array_t<double> V_arr,
                                            py::list polys_list) {
    std::vector<int64_t> tris;
    for (auto p : polys_list) {
        auto pl = p.cast<py::list>();
        int n = (int)pl.size();
        if (n < 3) continue;
        int64_t v0 = pl[0].cast<int64_t>();
        for (int k = 1; k < n-1; k++) {
            tris.push_back(v0);
            tris.push_back(pl[k].cast<int64_t>());
            tris.push_back(pl[k+1].cast<int64_t>());
        }
    }
    int64_t nf = (int64_t)tris.size()/3;
    py::array_t<int64_t> F({nf,(int64_t)3});
    if (nf>0) std::memcpy(F.mutable_data(), tris.data(), tris.size()*sizeof(int64_t));
    return F;
}

// ═══════════════════════════════════════════════════════════════════════════
// add_handle batch
// ═══════════════════════════════════════════════════════════════════════════

static void do_add_handle(Mesh& mesh, Id fid1, Id fid2) {
    // Mirror of Python add_handle
    auto hes1 = mesh.face_halfedges(fid1);
    auto hes2 = mesh.face_halfedges(fid2);
    int n = (int)std::min(hes1.size(), hes2.size());
    if (n < 3) return;

    std::vector<Id> verts1(n), verts2(n);
    std::vector<Id> ext1(n), ext2(n);
    for (int i = 0; i < n; i++) {
        verts1[i] = mesh.hes[hes1[i]].origin;
        verts2[i] = mesh.hes[hes2[i]].origin;
        ext1[i]   = mesh.hes[hes1[i]].twin;
        ext2[i]   = mesh.hes[hes2[i]].twin;
    }

    // Remove both faces and their HEs/edges
    mesh.del_f(fid1);
    for (int i = 0; i < n; i++) {
        Id e = mesh.hes[hes1[i]].edge;
        if (e) mesh.del_e(e);
        mesh.del_he(hes1[i]);
    }
    mesh.del_f(fid2);
    for (int i = 0; i < n; i++) {
        Id e = mesh.hes[hes2[i]].edge;
        if (e) mesh.del_e(e);
        mesh.del_he(hes2[i]);
    }

    // Reverse verts2 for consistent winding
    std::vector<Id> verts2_rev(verts2.rbegin(), verts2.rend());
    // ext2_for_rev[i] = ext2[n-2-i]  (see Python logic)
    std::vector<Id> ext2_rev(n);
    for (int i = 0; i < n; i++) ext2_rev[i] = ext2[n-2-i];

    // Build n side quads
    std::vector<Id> quad_bot(n), quad_right(n), quad_top(n), quad_left(n);
    for (int i = 0; i < n; i++) {
        Id sf = mesh.new_f();
        Id bot = mesh.new_he();  // verts1[i]      → verts1[(i+1)%n]
        Id rt  = mesh.new_he();  // verts1[(i+1)%n] → verts2_rev[(i+1)%n]
        Id top = mesh.new_he();  // verts2_rev[(i+1)%n] → verts2_rev[i]
        Id lt  = mesh.new_he();  // verts2_rev[i]   → verts1[i]

        mesh.hes[bot].origin = verts1[i];
        mesh.hes[rt].origin  = verts1[(i+1)%n];
        mesh.hes[top].origin = verts2_rev[(i+1)%n];
        mesh.hes[lt].origin  = verts2_rev[i];

        mesh.hes[bot].face = mesh.hes[rt].face = mesh.hes[top].face = mesh.hes[lt].face = sf;
        mesh.faces[sf].he = bot;

        mesh.hes[bot].next = rt;  mesh.hes[rt].prev  = bot;
        mesh.hes[rt].next  = top; mesh.hes[top].prev = rt;
        mesh.hes[top].next = lt;  mesh.hes[lt].prev  = top;
        mesh.hes[lt].next  = bot; mesh.hes[bot].prev = lt;

        quad_bot[i]   = bot;
        quad_right[i] = rt;
        quad_top[i]   = top;
        quad_left[i]  = lt;
    }

    // Wire twins
    for (int i = 0; i < n; i++) mesh.new_e(quad_bot[i], ext1[i]);
    for (int i = 0; i < n; i++) mesh.new_e(quad_top[i], ext2_rev[i]);
    for (int i = 0; i < n; i++) mesh.new_e(quad_right[i], quad_left[(i+1)%n]);

    // Fix vertex.he
    for (int i = 0; i < n; i++) {
        Id v = verts1[i];
        if (!mesh.verts[v].he || !mesh.hes[mesh.verts[v].he].alive)
            mesh.verts[v].he = quad_bot[i];
    }
    for (int i = 0; i < n; i++) {
        Id v = verts2_rev[i];
        if (!mesh.verts[v].he || !mesh.hes[mesh.verts[v].he].alive)
            mesh.verts[v].he = quad_left[i];
    }
}

py::tuple batch_add_handle(py::array_t<double>  V_arr,
                           py::array_t<int64_t> F_arr,
                           int fi, int fj) {
    auto V_ = V_arr.unchecked<2>();
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_.shape(0), nf = F_.shape(0);

    Mesh mesh;
    mesh.build_from_arrays(V_arr.data(), nv, F_arr.data(), nf);

    // Face fi → ID fi+1 (faces created in order, 1-based)
    Id fid1 = (Id)(fi + 1);
    Id fid2 = (Id)(fj + 1);

    if (fid1 >= mesh.faces.size() || !mesh.faces[fid1].alive) return py::make_tuple(V_arr, F_arr);
    if (fid2 >= mesh.faces.size() || !mesh.faces[fid2].alive) return py::make_tuple(V_arr, F_arr);

    do_add_handle(mesh, fid1, fid2);

    // Stellate the side quads (spec: "single op incl. stellating the side quads")
    // After add_handle the mesh has some quad faces — stellate them
    {
        std::vector<Id> quad_faces;
        for (size_t i = 1; i < mesh.faces.size(); i++) {
            if (!mesh.faces[i].alive) continue;
            if (mesh.face_halfedges((Id)i).size() == 4)
                quad_faces.push_back((Id)i);
        }
        for (Id qf : quad_faces) {
            if (qf < mesh.faces.size() && mesh.faces[qf].alive)
                mesh.stellate(qf);
        }
    }

    std::vector<double>  V_out;
    std::vector<int64_t> F_out;
    mesh.to_arrays(V_out, F_out);

    int64_t nv2 = (int64_t)V_out.size()/3;
    int64_t nf2 = (int64_t)F_out.size()/3;
    py::array_t<double>  V2({nv2,(int64_t)3});
    py::array_t<int64_t> F2({nf2,(int64_t)3});
    if (nv2>0) std::memcpy(V2.mutable_data(), V_out.data(), V_out.size()*sizeof(double));
    if (nf2>0) std::memcpy(F2.mutable_data(), F_out.data(), F_out.size()*sizeof(int64_t));

    return py::make_tuple(V2, F2);
}

// ═══════════════════════════════════════════════════════════════════════════
// check_watertight
// ═══════════════════════════════════════════════════════════════════════════

py::tuple batch_check_watertight(py::array_t<int64_t> F_arr) {
    auto F_ = F_arr.unchecked<2>();
    int64_t nf = F_.shape(0);
    int64_t nv = 0;
    for (int64_t i = 0; i < nf; i++)
        for (int k = 0; k < 3; k++) nv = std::max(nv, F_(i,k)+1);

    std::unordered_map<uint64_t,int> directed;
    directed.reserve(nf * 6);
    uint64_t NV = (uint64_t)nv;
    for (int64_t i = 0; i < nf; i++) {
        int64_t a=F_(i,0), b=F_(i,1), c=F_(i,2);
        directed[(uint64_t)a*NV+(uint64_t)b]++;
        directed[(uint64_t)b*NV+(uint64_t)c]++;
        directed[(uint64_t)c*NV+(uint64_t)a]++;
    }
    int n_bad = 0;
    for (auto& [key, cnt] : directed) {
        uint64_t a = key / NV, b = key % NV;
        auto it = directed.find((uint64_t)b*NV+(uint64_t)a);
        if (it == directed.end() || it->second != cnt) n_bad++;
    }
    return py::make_tuple(n_bad == 0, n_bad);
}

// ═══════════════════════════════════════════════════════════════════════════
// euler_genus
// ═══════════════════════════════════════════════════════════════════════════

int batch_euler_genus(py::array_t<double>  V_arr,
                      py::array_t<int64_t> F_arr) {
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_arr.shape(0);
    int64_t nf = F_.shape(0);

    // Count unique edges
    std::set<std::pair<int64_t,int64_t>> edge_set;
    for (int64_t i = 0; i < nf; i++) {
        int64_t a=F_(i,0), b=F_(i,1), c=F_(i,2);
        edge_set.insert({std::min(a,b), std::max(a,b)});
        edge_set.insert({std::min(b,c), std::max(b,c)});
        edge_set.insert({std::min(c,a), std::max(c,a)});
    }
    int64_t ne = (int64_t)edge_set.size();

    // Connected components (union-find on vertices referenced by faces)
    std::vector<int64_t> parent(nv);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int64_t(int64_t)> find = [&](int64_t x) -> int64_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](int64_t a, int64_t b) {
        a = find(a); b = find(b); if (a != b) parent[a] = b;
    };

    std::unordered_set<int64_t> used_verts;
    for (int64_t i = 0; i < nf; i++) {
        int64_t a=F_(i,0), b=F_(i,1), c=F_(i,2);
        used_verts.insert(a); used_verts.insert(b); used_verts.insert(c);
        unite(a, b); unite(b, c);
    }
    std::unordered_set<int64_t> roots;
    for (int64_t v : used_verts) roots.insert(find(v));
    int64_t C = (int64_t)roots.size();

    int64_t V = (int64_t)used_verts.size();
    int64_t chi = V - ne + nf;
    int genus = (int)(C - chi / 2);
    return genus;
}

// ═══════════════════════════════════════════════════════════════════════════
// validate batch
// ═══════════════════════════════════════════════════════════════════════════

py::list batch_validate(py::array_t<double>  V_arr,
                        py::array_t<int64_t> F_arr) {
    auto V_ = V_arr.unchecked<2>();
    auto F_ = F_arr.unchecked<2>();
    int64_t nv = V_.shape(0), nf = F_.shape(0);

    Mesh mesh;
    mesh.build_from_arrays(V_arr.data(), nv, F_arr.data(), nf);

    std::vector<std::string> errors;
    mesh.validate(errors);

    py::list result;
    for (auto& e : errors) result.append(e);
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// pybind11 module
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(topmod_core, m) {
    m.doc() = "C++ DLFL half-edge mesh kernel for GenesisTopmod";

    m.def("flip_sweep", &batch_flip_sweep,
          py::arg("V"), py::arg("F"), py::arg("passes")=4, py::arg("fold_cos")=0.0,
          "Flip edges to reduce fold. Returns (V, F2, n_flips).");

    m.def("collapse_short_edges", &batch_collapse,
          py::arg("V"), py::arg("F"),
          py::arg("ratio")=0.3, py::arg("max_n")=400,
          py::arg("thr_abs")=-1.0, py::arg("vthr")=py::array_t<double>(),
          "Collapse short edges. Returns (V2, F2, n_collapsed).");

    m.def("subdivide_faces", &batch_subdivide_faces,
          py::arg("V"), py::arg("F"), py::arg("fids"), py::arg("expand_ring")=true,
          "Subdivide faces (edge split + stellate). Returns (V2, F2, n_split_edges).");

    m.def("catmull_clark", &batch_catmull_clark,
          py::arg("V"), py::arg("polys"),
          "Catmull-Clark subdivision. Returns (V2, polys2).");

    m.def("triangulate_all", &batch_triangulate_all,
          py::arg("V"), py::arg("polys"),
          "Fan-triangulate all polygons. Returns F[Mx3].");

    m.def("add_handle", &batch_add_handle,
          py::arg("V"), py::arg("F"), py::arg("fi"), py::arg("fj"),
          "Add topological handle between faces fi and fj. Returns (V2, F2).");

    m.def("check_watertight", &batch_check_watertight,
          py::arg("F"),
          "Check if mesh is watertight. Returns (ok, n_bad_edges).");

    m.def("euler_genus", &batch_euler_genus,
          py::arg("V"), py::arg("F"),
          "Compute genus via Euler characteristic. Returns genus.");

    m.def("validate", &batch_validate,
          py::arg("V"), py::arg("F"),
          "Validate half-edge mesh. Returns list of error strings.");
}
