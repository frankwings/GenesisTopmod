"""
app.py — GenesisTopmod Interactive Demo (Gradio)

Four-tab Gradio application covering all four phases of GenesisTopmod:
  Tab 1: Topology Explorer  — Phase 1 (DLFL + operators)
  Tab 2: Geometry Optimizer — Phase 2 (nvdiffrast silhouette fitting)
  Tab 3: Tokenizer          — Phase 3 (MeshGPT-style token sequences)
  Tab 4: Manifold Loss      — Phase 4 (differentiable topology constraints)

Usage:
  cd /home/kingy/Projects/Genesis/GenesisTopmod
  python3 app.py
"""

from __future__ import annotations

import os
import sys
import io
import math
import tempfile
import traceback
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── output dir ────────────────────────────────────────────────────────────────
TMP = "/tmp/topmod_gradio"
os.makedirs(TMP, exist_ok=True)

import gradio as gr
import torch
import numpy as np

# ── topmod imports ────────────────────────────────────────────────────────────
from topmod import (
    make_icosahedron, make_cube, make_tetrahedron, make_octahedron,
    add_handle, catmull_clark, check_all, to_obj, to_triangle_arrays,
)
from topmod.tokenizer import (
    tokenize, detokenize, token_stats, sequence_length,
    DEFAULT_COORD_LO, DEFAULT_COORD_HI,
)
from pipeline.manifold_loss import (
    edge_manifold_loss, euler_loss, orientation_consistency_loss,
    manifold_loss, manifold_loss_breakdown,
)

# ── optional nvdiffrast ───────────────────────────────────────────────────────
try:
    import nvdiffrast.torch as dr
    _NVDIFFRAST = True
except ImportError:
    _NVDIFFRAST = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _tmp(name: str) -> str:
    return os.path.join(TMP, name)


def _build_genus_mesh(genus: int, cc_level: int):
    """Build a mesh of the requested genus and Catmull-Clark level."""
    mesh = make_icosahedron()
    excluded = set()
    for _ in range(genus):
        faces_list = list(mesh.faces.values())
        n = len(faces_list)
        f1 = f2 = None
        for i, fa in enumerate(faces_list):
            va = {v.id for v in fa.vertices()}
            if va & excluded:
                continue
            for fb in faces_list[i + 1:]:
                vb = {v.id for v in fb.vertices()}
                if vb & excluded or va & vb:
                    continue
                f1, f2 = fa, fb
                break
            if f1 is not None:
                break
        if f1 is None:
            f1, f2 = faces_list[0], faces_list[len(faces_list) // 2]
        excluded |= {v.id for v in f1.vertices()} | {v.id for v in f2.vertices()}
        add_handle(mesh, f1, f2)
    for _ in range(cc_level):
        mesh = catmull_clark(mesh)
    return mesh


def _compute_normals(positions, triangles):
    """Compute per-vertex normals from positions and triangle indices."""
    verts = np.array(positions, dtype=np.float32)
    norms = np.zeros_like(verts)
    for tri in triangles:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(n)
        if ln > 1e-12:
            n /= ln
        for idx in tri:
            norms[idx] += n
    # Normalize
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    norms /= lengths
    return norms


def _write_glb(verts_np, faces_np, path, color=(70, 130, 255, 255)):
    """Write a GLB file with a PBR material.

    NOTE: Gradio 6 Model3D (Babylon.js) renders material-less OBJ files as
    solid black. GLB with an explicit PBR material renders correctly.
    """
    import trimesh
    tm = trimesh.Trimesh(vertices=np.asarray(verts_np, dtype=np.float64),
                         faces=np.asarray(faces_np, dtype=np.int64),
                         process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=list(color), metallicFactor=0.1, roughnessFactor=0.6))
    tm.export(path)


def _mesh_to_obj(mesh, name: str) -> str:
    """Save mesh to a temp GLB file for proper 3D rendering (name .obj → .glb)."""
    if name.endswith(".obj"):
        name = name[:-4] + ".glb"
    path = _tmp(name)
    positions, triangles = to_triangle_arrays(mesh)
    _write_glb(positions, triangles, path)
    return path


def _stats_markdown(mesh) -> str:
    """Return a markdown table with mesh statistics."""
    ok, errs = check_all(mesh)
    chi = mesh.euler_characteristic() if hasattr(mesh, "euler_characteristic") else (mesh.V() - mesh.E() + mesh.F())
    status = "✅ Valid manifold" if ok else f"❌ Non-manifold ({len(errs)} error{'s' if len(errs) != 1 else ''})"
    rows = [
        ("Vertices (V)", mesh.V()),
        ("Edges (E)", mesh.E()),
        ("Faces (F)", mesh.F()),
        ("Euler χ = V−E+F", chi),
        ("Genus", mesh.genus()),
        ("Manifold", status),
    ]
    lines = ["| Property | Value |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Topology Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def tab1_generate(genus: int, cc_level: int) -> tuple:
    """Generate mesh → OBJ path + stats markdown."""
    try:
        mesh = _build_genus_mesh(int(genus), int(cc_level))
        obj_path = _mesh_to_obj(mesh, f"tab1_g{genus}_cc{cc_level}.obj")
        stats = _stats_markdown(mesh)
        return obj_path, stats
    except Exception as exc:
        tb = traceback.format_exc()
        return None, f"**Error:** {exc}\n\n```\n{tb}\n```"


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Geometry Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _silhouette_from_image(img_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to a binary silhouette (float32, [H, W])."""
    if img_array is None:
        return None
    # Greyscale, then threshold at 128
    if img_array.ndim == 3:
        grey = img_array.mean(axis=2)
    else:
        grey = img_array.astype(float)
    # Mark dark pixels as the object (dark object on white background)
    binary = (grey < 128).astype(np.float32)
    # The *object* should be the minority region. If the mask covers more
    # than half the image, we grabbed the background — flip it.
    # (Handles bright object on dark background automatically.)
    if binary.mean() > 0.5:
        binary = 1.0 - binary
    return binary


def _write_obj_with_normals(verts_np, faces_np, path):
    """Write mesh for the 3D viewer (GLB; see _write_glb for why not OBJ)."""
    _write_glb(verts_np, faces_np, path)


def _prep_target(img_array: np.ndarray, res: int) -> np.ndarray:
    """
    Image → normalized binary silhouette target [res, res] float32.

    Steps: binarize → crop to object bbox → pad square (aspect preserved) →
    scale object to ~55% of frame (matches the seed sphere's projected
    extent: scale 0.6 @ radius 3, fov 40°) → resize → vertical flip
    (image row 0 = top, raster row 0 = bottom in OpenGL y-up clip space).
    """
    sil = _silhouette_from_image(img_array)
    ys, xs = np.where(sil > 0.5)
    if len(ys) > 0:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = sil[y0:y1, x0:x1]
        h, w = crop.shape
        side = int(max(h, w) / 0.55)  # object occupies 55% of frame
        canvas = np.zeros((side, side), dtype=np.float32)
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = crop
        sil = canvas
    from PIL import Image as PILImage
    sil_img = PILImage.fromarray((sil * 255).astype(np.uint8))
    sil_img = sil_img.resize((res, res), PILImage.BILINEAR)
    sil_np = np.array(sil_img).astype(np.float32) / 255.0
    return np.flipud(sil_np).copy()


def tab2_run(
    files,
    example_shape: str,
    genus: int,
    n_steps: int,
    n_views: int,
    progress=gr.Progress(),
) -> tuple:
    """Run geometry optimization and return before-mesh, after-mesh, loss plot, status."""
    if not _NVDIFFRAST:
        msg = (
            "## ⚠️ nvdiffrast not available\n\n"
            "Phase 2 geometry optimization requires CUDA + nvdiffrast.\n\n"
            "**To install:**\n"
            "```bash\npip install git+https://github.com/NVlabs/nvdiffrast\n```\n\n"
            "The other tabs (1, 3, 4) work without GPU."
        )
        return None, None, None, msg

    try:
        device = torch.device("cuda")
        import nvdiffrast.torch as dr
        from pipeline.topology_builder import build_topology
        from pipeline.geometry_optimizer import optimize, render_silhouette
        from pipeline.cameras import orbit_cameras

        n_steps = int(n_steps)
        n_views = int(n_views)
        genus = int(genus)
        res = 256

        # ── Load uploaded silhouette image(s). Each file = ONE camera view. ──
        images: list = []
        if files:
            from PIL import Image as PILImage
            if not isinstance(files, (list, tuple)):
                files = [files]
            for f in files:
                fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
                images.append(np.array(PILImage.open(fpath).convert("RGB")))
            # Uploads define the number of views — the slider is ignored.
            n_views = len(images)

        progress(0.05, desc="Initialising nvdiffrast...")
        ctx = dr.RasterizeCudaContext()

        progress(0.1, desc="Building topology seed...")
        init_verts, init_faces = build_topology(
            genus=genus, boundaries=0,
            # uploads need more vertex capacity for irregular outlines
            subdivisions=3 if images else 2,
            scale=0.6, device=device,
        )

        # Camera rig. Uploaded views get explicit azimuths:
        #   1 image  → front (0°)
        #   2 images → front + side (0°, 90°) — best depth constraint
        #   N images → evenly spaced over 360°
        # An uploaded silhouette is treated as an eye-level view (elevation 0).
        azimuths = None
        if images:
            if n_views == 1:
                azimuths = [0.0]
            elif n_views == 2:
                azimuths = [0.0, 90.0]
            else:
                azimuths = [360.0 * i / n_views for i in range(n_views)]
        mvps, _ = orbit_cameras(
            n=n_views,
            elevation_deg=0.0 if images else 25.0,
            radius=3.0,
            fov_deg=40.0, device=device,
            azimuths_deg=azimuths,
        )

        # Build target silhouettes
        progress(0.15, desc="Rendering target silhouettes...")
        if images:
            sils = [_prep_target(img, res) for img in images]
            target_images = torch.tensor(
                np.stack(sils, axis=0), device=device
            ).unsqueeze(-1).contiguous()          # [N, res, res, 1]
        else:
            # Use a reference shape to render targets
            import trimesh
            if example_shape == "Torus":
                tm = trimesh.creation.torus(major_radius=1.0, minor_radius=0.30,
                                            major_sections=48, minor_sections=24)
            else:
                tm = trimesh.creation.icosphere(subdivisions=4)
            ref_np = np.array(tm.vertices, dtype=np.float32)
            ref_np -= ref_np.mean(axis=0, keepdims=True)
            max_ext = np.abs(ref_np).max()
            if max_ext > 1e-6:
                ref_np = ref_np / max_ext * 0.85
            ref_verts = torch.tensor(ref_np, dtype=torch.float32, device=device)
            ref_faces = torch.tensor(np.array(tm.faces, dtype=np.int32), device=device)
            imgs = []
            for i in range(n_views):
                sil = render_silhouette(ctx, ref_verts, ref_faces, mvps[i], (res, res))
                imgs.append(sil)
            target_images = torch.cat(imgs, dim=0).clamp(0, 1)

        # Save before mesh
        before_path = _tmp("tab2_before.glb")
        v_np = init_verts.detach().cpu().numpy()
        f_np = init_faces.cpu().numpy()
        _write_obj_with_normals(v_np, f_np, before_path)

        # Optimize
        progress(0.2, desc=f"Optimizing ({n_steps} steps)...")
        final_verts, history = optimize(
            ctx=ctx,
            verts_init=init_verts,
            faces=init_faces,
            target_images=target_images,
            mvps=mvps,
            num_steps=n_steps,
            lr=5e-3,
            lambda_lap=0.05,
            lambda_edge=0.01,
            # Anti-flattening (active with < 3 views, where depth is weakly
            # constrained and the mesh degenerates into paper-thin sheets):
            #  - normal consistency: folded/creased sheets have adjacent faces
            #    with opposing normals → directly penalises local thin fins
            #  - volume hinge: keeps at least 40% of the seed volume
            lambda_normal=0.02 if n_views < 3 else 0.0,
            lambda_vol=2.0 if n_views < 3 else 0.0,
            vol_min_ratio=0.4,
            resolution=(res, res),
            log_every=max(1, n_steps // 20),
            scheduler=True,
        )

        # Save after mesh.
        # Rotate -90° about Y so the optimised camera view (camera on +X axis)
        # faces the Model3D viewer's default front (+Z) — the user immediately
        # sees the view that was fitted to the silhouette.
        rot = np.array([[0.0, 0.0, -1.0],
                        [0.0, 1.0,  0.0],
                        [1.0, 0.0,  0.0]])
        after_path = _tmp("tab2_after.glb")
        vout_np = final_verts.detach().cpu().numpy() @ rot.T
        _write_obj_with_normals(vout_np, f_np, after_path)
        # re-export the before mesh with the same rotation for fair comparison
        _write_obj_with_normals(v_np @ rot.T, f_np, before_path)

        # Fitted-view comparison: target silhouette vs final rendered silhouette
        with torch.no_grad():
            fitted_sil = render_silhouette(
                ctx, final_verts, init_faces, mvps[0], (res, res)
            ).squeeze().detach().cpu().numpy()
        # raster row 0 = bottom; flip both for display in image orientation
        target_np = np.flipud(target_images[0, :, :, 0].detach().cpu().numpy())
        fitted_sil = np.flipud(fitted_sil)

        # Plot: target | fitted | overlay + loss curve
        plot_path = None
        if _MPL and history:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
            axes[0].imshow(target_np, cmap="gray"); axes[0].set_title("Target silhouette")
            axes[1].imshow(fitted_sil, cmap="gray"); axes[1].set_title("Fitted (view 0)")
            overlay = np.stack([target_np, fitted_sil, np.zeros_like(target_np)], axis=-1)
            axes[2].imshow(overlay); axes[2].set_title("Overlay (yellow = match)")
            for a in axes[:3]:
                a.set_xticks([]); a.set_yticks([])
            steps_list = [h["step"] for h in history]
            total_list = [h["loss"] for h in history]
            sil_list = [h["sil_loss"] for h in history]
            axes[3].semilogy(steps_list, total_list, label="total", linewidth=2, color="#4f8ef7")
            axes[3].semilogy(steps_list, sil_list, label="silhouette", linewidth=1.5,
                             linestyle="--", color="#e74c3c")
            axes[3].set_xlabel("Step"); axes[3].set_ylabel("Loss (log)")
            axes[3].set_title("Loss"); axes[3].legend(); axes[3].grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = _tmp("tab2_loss.png")
            plt.savefig(plot_path, dpi=120)
            plt.close()

        initial_loss = history[0]["sil_loss"] if history else 0
        final_loss = history[-1]["sil_loss"] if history else 0
        improvement = (initial_loss - final_loss) / max(initial_loss, 1e-8) * 100

        status = (
            f"## ✅ Optimization complete\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Steps | {n_steps} |\n"
            f"| Views | {n_views} |\n"
            f"| Vertices | {final_verts.shape[0]} |\n"
            f"| Faces | {init_faces.shape[0]} |\n"
            f"| Genus | {genus} |\n"
            f"| Initial loss | {initial_loss:.5f} |\n"
            f"| Final loss | {final_loss:.5f} |\n"
            f"| Improvement | {improvement:.1f}% |\n"
        )
        return before_path, after_path, plot_path, status

    except Exception as exc:
        tb = traceback.format_exc()
        return None, None, None, f"**Error:** {exc}\n\n```\n{tb}\n```"


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

_TOKEN_COLORS = {
    "EOS":  "#888888",
    "CC":   "#e67e22",
    "CV":   "#27ae60",
    "IE":   "#2980b9",
    "DE":   "#c0392b",
    "HDL":  "#8e44ad",
}


def _tokens_to_html(tokens) -> str:
    """Render token list as colored HTML spans."""
    parts = []
    for tok in tokens:
        color = _TOKEN_COLORS.get(tok.op, "#444")
        if tok.op == "CV" and tok.pos is not None:
            label = f"CV({tok.pos[0]},{tok.pos[1]},{tok.pos[2]})"
        elif tok.op == "HDL" and tok.corner1 is not None:
            label = f"HDL(f{tok.corner1[0]},f{tok.corner2[0]})"
        elif tok.op == "IE" and tok.corner1 is not None:
            label = f"IE(f{tok.corner1[0]}:h{tok.corner1[1]},f{tok.corner2[0]}:h{tok.corner2[1]})"
        elif tok.op == "DE" and tok.edge_ord is not None:
            label = f"DE(e{tok.edge_ord})"
        else:
            label = tok.op
        parts.append(
            f'<span style="background:{color};color:white;padding:2px 6px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-family:monospace;'
            f'font-size:12px">{label}</span>'
        )
    return '<div style="line-height:2.2;padding:8px">' + " ".join(parts) + "</div>"


def tab3_run(
    genus: int,
    n_bins: int,
    max_cc: int,
    normalize: bool,
) -> tuple:
    """Tokenize a genus-g mesh and detokenize it back."""
    try:
        genus = int(genus)
        n_bins = int(n_bins)
        max_cc = int(max_cc)

        # Build source mesh
        mesh = _build_genus_mesh(genus, cc_level=0)

        # Tokenize
        tokens = tokenize(
            mesh,
            n_position_bins=n_bins,
            max_cc_rounds=max_cc,
            normalize=normalize,
        )
        stats = token_stats(tokens)
        total_ids = sequence_length(tokens)

        # Detokenize
        reco = detokenize(tokens, n_position_bins=n_bins)

        # Compute roundtrip stats
        reco_genus = reco.genus()
        genus_match = reco_genus == genus
        ok, errs = check_all(reco)

        # Save OBJs
        src_path  = _mesh_to_obj(mesh, f"tab3_src_g{genus}.obj")
        reco_path = _mesh_to_obj(reco, f"tab3_reco_g{genus}_b{n_bins}.obj")

        # Stats markdown
        stat_lines = [
            "| Metric | Value |", "|---|---|",
            f"| Tokens | {len(tokens)} |",
            f"| Integer IDs | {total_ids} |",
        ]
        for op, cnt in sorted(stats.items()):
            stat_lines.append(f"| `{op}` tokens | {cnt} |")
        stat_lines += [
            f"| Source genus | {genus} |",
            f"| Roundtrip genus | {reco_genus} |",
            f"| Genus match | {'✅' if genus_match else '❌'} |",
            f"| Manifold valid | {'✅' if ok else '❌'} |",
            f"| n_position_bins | {n_bins} |",
        ]

        token_html = _tokens_to_html(tokens)
        stats_md   = "\n".join(stat_lines)

        return token_html, stats_md, src_path, reco_path

    except Exception as exc:
        tb = traceback.format_exc()
        err = f"**Error:** {exc}\n\n```\n{tb}\n```"
        return err, err, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Manifold Loss
# ═══════════════════════════════════════════════════════════════════════════════

def _mesh_tensors(name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (verts, faces, n_rogue) torch tensors for the named demo mesh."""
    if name == "Tetrahedron (valid)":
        verts = torch.tensor([
            [ 1.,  1.,  1.], [ 1., -1., -1.],
            [-1.,  1., -1.], [-1., -1.,  1.],
        ])
        faces = torch.tensor([[0,1,2],[0,2,3],[0,3,1],[1,3,2]], dtype=torch.int64)
        return verts, faces, 0

    elif name == "Octahedron (valid)":
        verts = torch.tensor([
            [1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],
            [0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.],
        ])
        faces = torch.tensor([
            [0,2,4],[0,4,3],[0,3,5],[0,5,2],
            [1,4,2],[1,3,4],[1,5,3],[1,2,5],
        ], dtype=torch.int64)
        return verts, faces, 0

    elif name == "Cube (valid)":
        import sys; sys.path.insert(0, ROOT)
        from topmod import make_cube, to_triangle_arrays
        cube = make_cube()
        positions, tris = to_triangle_arrays(cube)
        verts = torch.tensor(positions, dtype=torch.float32)
        faces = torch.tensor(tris, dtype=torch.int64)
        return verts, faces, 0

    else:  # Non-manifold (tet + extra face)
        verts = torch.tensor([
            [ 1.,  1.,  1.], [ 1., -1., -1.],
            [-1.,  1., -1.], [-1., -1.,  1.],
        ])
        base_faces = torch.tensor([[0,1,2],[0,2,3],[0,3,1],[1,3,2]], dtype=torch.int64)
        rogue      = torch.tensor([[0,1,3]], dtype=torch.int64)
        faces = torch.cat([base_faces, rogue], dim=0)
        return verts, faces, 1


def tab4_compute(
    mesh_name: str,
    rogue_prob: float,
) -> tuple:
    """Compute manifold losses and produce the breakdown table + plot."""
    try:
        verts, faces, n_rogue = _mesh_tensors(mesh_name)
        F = faces.shape[0]

        # Build face_probs: base faces = 1.0, rogue face = slider value
        fp = torch.ones(F)
        if n_rogue > 0:
            fp[-n_rogue:] = float(rogue_prob)
        fp.requires_grad_(True)

        # Compute losses
        l_edge  = edge_manifold_loss(verts, faces, fp)
        l_euler = euler_loss(verts, faces, target_genus=0, face_probs=fp)
        l_ori   = orientation_consistency_loss(verts, faces, fp)
        l_total = manifold_loss(verts, faces, fp, target_genus=0)

        # Gradient info
        l_total.backward()
        grad = fp.grad
        grad_info = ""
        if grad is not None:
            grad_info = "| Face | Prob | ∂Loss/∂p |\n|---|---|---|\n"
            for i in range(F):
                tag = "BASE" if i < F - n_rogue else "ROGUE"
                grad_info += f"| {i} [{tag}] | {fp[i].item():.3f} | {grad[i].item():+.4f} |\n"

        # Stats table
        breakdown = manifold_loss_breakdown(verts, faces, fp.detach())
        stats_md = (
            "| Loss | Value |\n|---|---|\n"
            f"| Edge Manifold | {breakdown['edge_manifold']:.5f} |\n"
            f"| Euler χ | {breakdown['euler']:.5f} |\n"
            f"| Orientation | {breakdown['orientation']:.5f} |\n"
            f"| **Total** | **{breakdown['total']:.5f}** |\n"
        )

        # Sweep plot: loss vs rogue_prob (only if there's a rogue face)
        plot_path = None
        if _MPL and n_rogue > 0:
            probs = torch.linspace(0, 1, 50)
            edge_vals, euler_vals, ori_vals = [], [], []
            for p in probs:
                fp_s = torch.ones(F)
                fp_s[-n_rogue:] = p.item()
                edge_vals.append(edge_manifold_loss(verts, faces, fp_s).item())
                euler_vals.append(euler_loss(verts, faces, face_probs=fp_s).item())
                ori_vals.append(orientation_consistency_loss(verts, faces, fp_s).item())

            fig, ax = plt.subplots(figsize=(7, 4))
            xs = probs.numpy()
            ax.plot(xs, edge_vals,  label="Edge Manifold",  linewidth=2, color="#e74c3c")
            ax.plot(xs, euler_vals, label="Euler χ",        linewidth=2, color="#3498db")
            ax.plot(xs, ori_vals,   label="Orientation",    linewidth=2, color="#2ecc71")
            ax.axvline(float(rogue_prob), color="black", linestyle="--",
                       linewidth=1.5, label=f"Current prob={rogue_prob:.2f}")
            ax.set_xlabel("Rogue face probability")
            ax.set_ylabel("Loss value")
            ax.set_title("Manifold Loss vs Rogue Face Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = _tmp("tab4_sweep.png")
            plt.savefig(plot_path, dpi=130)
            plt.close()
        elif _MPL and n_rogue == 0:
            # Valid mesh: bar chart of near-zero losses
            labels = ["Edge", "Euler", "Orientation"]
            vals = [breakdown["edge_manifold"], breakdown["euler"], breakdown["orientation"]]
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(labels, vals, color=["#e74c3c", "#3498db", "#2ecc71"])
            ax.set_ylabel("Loss value")
            ax.set_title("Manifold Losses (all ≈ 0 for valid mesh)")
            ax.set_ylim(0, max(max(vals) * 1.5, 0.01))
            plt.tight_layout()
            plot_path = _tmp("tab4_bars.png")
            plt.savefig(plot_path, dpi=130)
            plt.close()

        # 3D OBJ for viewer (filter by face_probs ≥ 0.5)
        fp_np    = fp.detach().numpy()
        verts_np = verts.numpy()
        faces_np = faces.numpy()
        active_faces = np.array([f for i, f in enumerate(faces_np) if fp_np[i] >= 0.5])
        obj_path = _tmp("tab4_mesh.glb")
        if len(active_faces) > 0:
            _write_obj_with_normals(verts_np, active_faces, obj_path)
        else:
            obj_path = None  # no faces survive the threshold — nothing to render

        return stats_md, grad_info or "_Gradient info only shown for meshes with rogue faces._", plot_path, obj_path

    except Exception as exc:
        tb = traceback.format_exc()
        err = f"**Error:** {exc}\n\n```\n{tb}\n```"
        return err, err, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Build Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

HEADER_MD = """
# 🔺 GenesisTopmod Interactive Demo

**A pure-Python TopMod library for provably-manifold mesh generation**

> Based on Akleman & Chen (2003): *4 minimal operators (CV, DV, IE, DE) that are
> necessary and sufficient to build any orientable 2-manifold.*

Each tab demonstrates one phase of the system — explore topology, optimize geometry,
tokenize meshes for autoregressive generation, and apply differentiable manifold losses.
"""

with gr.Blocks(title="GenesisTopmod Demo") as demo:
    gr.Markdown(HEADER_MD)

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🔺 Tab 1: Topology Explorer"):
        gr.Markdown("""
### Phase 1 — DLFL Half-Edge Mesh + Operators

Generate any orientable 2-manifold by selecting a **genus** (number of handles / holes)
and a **Catmull-Clark subdivision level**. Every intermediate state is guaranteed valid
by the TopMod invariants (twin check, face-loop check, Euler characteristic).

- **Genus 0** → sphere-like (icosahedron base)
- **Genus 1** → torus-like (one handle added)
- **Genus 2+** → higher-genus surfaces
""")
        with gr.Row():
            with gr.Column(scale=1):
                t1_genus  = gr.Slider(0, 3, value=0, step=1, label="Genus (handles)")
                t1_cc     = gr.Slider(0, 3, value=1, step=1, label="Catmull-Clark level")
                t1_btn    = gr.Button("Generate Mesh", variant="primary")
            with gr.Column(scale=2):
                t1_model  = gr.Model3D(label="3D Mesh Viewer")
                t1_stats  = gr.Markdown(label="Mesh Statistics")

        t1_btn.click(
            fn=tab1_generate,
            inputs=[t1_genus, t1_cc],
            outputs=[t1_model, t1_stats],
        )
        # Auto-generate on load
        demo.load(fn=tab1_generate, inputs=[t1_genus, t1_cc], outputs=[t1_model, t1_stats])

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🎯 Tab 2: Geometry Optimizer"):
        gr.Markdown(f"""
### Phase 2 — Differentiable Silhouette Fitting (nvdiffrast)

Upload **one or more target silhouettes** (each image = one camera view) or pick a
built-in example shape.  The optimizer builds a mesh with the requested topology, then
adjusts vertex positions to match the silhouettes using differentiable rasterization.

**View assignment for uploads:** 1 image → front view · 2 images → front + side (90°,
best depth constraint) · N ≥ 3 → evenly spaced around 360°.  With fewer than 3 views,
anti-flattening regularizers (normal consistency + volume hinge) are enabled, since a
single silhouette cannot constrain depth — a paper-thin sheet has the same silhouette
as a solid object.

**GPU status:** {"✅ CUDA + nvdiffrast available" if _NVDIFFRAST else "⚠️ nvdiffrast **not** available — install with `pip install git+https://github.com/NVlabs/nvdiffrast`"}
""")
        if not _NVDIFFRAST:
            gr.Markdown(
                "> **Note:** This tab requires CUDA + nvdiffrast. "
                "Tabs 1, 3, 4 work without GPU."
            )
        with gr.Row():
            with gr.Column(scale=1):
                t2_image   = gr.File(
                    label="Target silhouette(s) — upload 1..N images, each = one view "
                          "(1: front · 2: front+side 90° · N≥3: evenly spaced 360°)",
                    file_count="multiple", file_types=["image"],
                )
                t2_example = gr.Dropdown(
                    ["Sphere", "Torus"], value="Sphere", label="Built-in example shape"
                )
                t2_genus   = gr.Slider(0, 2, value=0, step=1, label="Target genus")
                t2_steps   = gr.Slider(100, 500, value=200, step=50, label="Optimization steps")
                t2_views   = gr.Slider(1, 16, value=8, step=1, label="Camera views (1 = single-view silhouette fit)")
                t2_btn     = gr.Button("Run Optimization", variant="primary", interactive=_NVDIFFRAST)
            with gr.Column(scale=2):
                with gr.Row():
                    t2_before = gr.Model3D(label="Before (topology seed)")
                    t2_after  = gr.Model3D(label="After (optimized)")
                t2_plot   = gr.Image(label="Loss curve")
                t2_status = gr.Markdown()

        t2_btn.click(
            fn=tab2_run,
            inputs=[t2_image, t2_example, t2_genus, t2_steps, t2_views],
            outputs=[t2_before, t2_after, t2_plot, t2_status],
        )

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🔤 Tab 3: Tokenizer"):
        gr.Markdown("""
### Phase 3 — MeshGPT-Style Topology Tokenizer

Decompose an arbitrary mesh into a **sequence of TopMod operators** (HDL, CC, CV, IE, DE).
This is the *reverse decomposition* problem: given a mesh, find the operator sequence
that reconstructs it from a standard template (icosahedron).

The roundtrip test verifies that `detokenize(tokenize(mesh))` recovers the correct genus
and a valid manifold.

**Token types:**
- 🟠 `CC` — Catmull-Clark subdivision (doubles resolution)
- 🟢 `CV` — Set vertex coordinate (quantized to N bins)
- 🟣 `HDL` — Add handle (genus +1)
- 🔵 `IE` — Insert edge (face split)
- 🔴 `DE` — Delete edge (face merge)
- ⬛ `EOS` — End of sequence
""")
        with gr.Row():
            with gr.Column(scale=1):
                t3_genus    = gr.Slider(0, 2, value=0, step=1, label="Genus")
                t3_bins     = gr.Slider(32, 512, value=128, step=32, label="Position bins")
                t3_max_cc   = gr.Slider(0, 5, value=3, step=1, label="Max CC rounds")
                t3_normalize = gr.Checkbox(value=True, label="Normalize coordinates")
                t3_btn      = gr.Button("Tokenize → Detokenize", variant="primary")
            with gr.Column(scale=2):
                t3_tokens   = gr.HTML(label="Token sequence")
                t3_stats    = gr.Markdown(label="Roundtrip Statistics")
                with gr.Row():
                    t3_src  = gr.Model3D(label="Source mesh")
                    t3_reco = gr.Model3D(label="Reconstructed mesh")

        t3_btn.click(
            fn=tab3_run,
            inputs=[t3_genus, t3_bins, t3_max_cc, t3_normalize],
            outputs=[t3_tokens, t3_stats, t3_src, t3_reco],
        )

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with gr.Tab("📐 Tab 4: Manifold Loss"):
        gr.Markdown("""
### Phase 4 — Differentiable Manifold Constraint Losses

Three PyTorch loss functions that penalise topological violations in (verts, faces) pairs:

| Loss | Invariant | Formula |
|------|-----------|---------|
| **Edge Manifold** | Every edge must be adjacent to exactly 2 faces | mean((count_e − 2)²) |
| **Euler χ** | V − E + F = 2C − 2g | (χ_eff − χ_target)² |
| **Orientation** | Adjacent faces must share edge in opposite directions | mean(signed_count_e²) |

Use `face_probs` ∈ [0,1] per face to softly weight which faces "exist" — enables
gradient-based topology optimisation (DMesh / LATO.2 style).
""")
        with gr.Row():
            with gr.Column(scale=1):
                t4_mesh = gr.Dropdown(
                    ["Tetrahedron (valid)", "Octahedron (valid)", "Cube (valid)",
                     "Non-manifold (tet + extra face)"],
                    value="Tetrahedron (valid)",
                    label="Test mesh",
                )
                t4_rogue_prob = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Rogue face probability (non-manifold mesh only)",
                )
                t4_btn = gr.Button("Compute Losses", variant="primary")
            with gr.Column(scale=2):
                t4_stats = gr.Markdown(label="Loss Breakdown")
                t4_grad  = gr.Markdown(label="Gradient Info")
                t4_plot  = gr.Image(label="Loss vs probability sweep")
                t4_model = gr.Model3D(label="Mesh (faces with prob ≥ 0.5)")

        t4_btn.click(
            fn=tab4_compute,
            inputs=[t4_mesh, t4_rogue_prob],
            outputs=[t4_stats, t4_grad, t4_plot, t4_model],
        )
        # NOTE: do NOT auto-compute via demo.load here.
        # Gradio 6.12 has a Svelte "effect_update_depth_exceeded" infinite-loop bug:
        # a demo.load that writes outputs into a *hidden* tab (with 3+ tabs present)
        # freezes the whole page on any tab switch. User clicks the button instead.

    gr.Markdown("""
---
*GenesisTopmod — Pure-Python DLFL mesh library + differentiable topology pipeline.*
*Based on Akleman & Chen (2003), "A minimal and complete set of operators for manifold mesh modelers."*
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GenesisTopmod Gradio demo")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--root-path", default=None,
                        help="Root path when served behind a reverse proxy (e.g. /topmod)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GenesisTopmod Interactive Demo")
    print(f"  nvdiffrast: {'available' if _NVDIFFRAST else 'not available'}")
    print(f"  matplotlib:  {'available' if _MPL else 'not available'}")
    print(f"  torch:       {torch.__version__}")
    print(f"  gradio:      {gr.__version__}")
    print("=" * 60)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
