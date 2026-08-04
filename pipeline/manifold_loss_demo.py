"""
manifold_loss_demo.py — Demonstrate differentiable manifold constraint optimization.

This demo shows how face_probs can be optimized via gradient descent to turn a
non-manifold mesh (with surplus faces causing edge-adjacency violations) into a
manifold-compliant one, without any hard combinatorial search.

Setup
-----
  • Base mesh: a tetrahedron (4 faces, 4 vertices, 6 edges).  Valid manifold.
  • Perturbation: 4 extra "rogue" triangles sharing existing edges, creating
    edges with 3 adjacent faces instead of 2.  This violates the edge-manifold
    invariant.
  • face_probs: [8] probabilities, all initialised to 0.9.  The base 4 faces
    should converge to prob ≈ 1.0 and the rogue 4 faces to prob ≈ 0.0.
  • Optimizer: Adam, 300 steps.

Outputs (saved to /tmp/topmod_manifold_demo/)
-----
  • loss_curve.png    — per-step total + component losses.
  • before_mesh.obj   — tetrahedron + all rogue faces (face_probs = 0.9 each).
  • after_mesh.obj    — only faces with final prob > 0.5 (should be the 4 base
                         tetrahedron faces).
  • face_probs.txt    — final face probabilities.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifold_loss import manifold_loss, manifold_loss_breakdown

# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = "/tmp/topmod_manifold_demo"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh helpers
# ─────────────────────────────────────────────────────────────────────────────

def tetrahedron_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (verts [4,3], faces [4,3]) of a regular tetrahedron."""
    # Unit regular tetrahedron centred at origin
    verts = torch.tensor([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ], dtype=torch.float32)
    # CCW-oriented faces (consistent outward normals)
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=torch.int64)
    return verts, faces


def add_rogue_faces(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Append 4 extra faces that duplicate edges of the tetrahedron, creating
    edges with 3 adjacent faces.  A new central vertex is added.
    """
    # Add a fifth vertex at the centroid (inside the tet)
    centroid = verts.mean(dim=0, keepdim=True)   # [1, 3]
    verts_aug = torch.cat([verts, centroid], dim=0)   # [5, 3]

    # Rogue faces: each uses vertex 4 (centroid) and one edge of face 0
    # Face 0 has edges (0,1), (1,2), (2,0).  We add triangles that share each.
    rogue = torch.tensor([
        [0, 1, 4],   # shares edge (0,1) of base face 0 and face 2
        [1, 2, 4],   # shares edge (1,2) of base face 0 and face 3
        [0, 2, 4],   # shares edge (0,2) of base face 0 and face 1
        [0, 3, 4],   # shares edge (0,3) of base faces 1 and 2
    ], dtype=torch.int64)

    faces_aug = torch.cat([faces, rogue], dim=0)   # [8, 3]
    return verts_aug, faces_aug


def save_obj(
    path:       str,
    verts:      torch.Tensor,
    faces:      torch.Tensor,
    face_probs: torch.Tensor,
    threshold:  float = 0.5,
) -> None:
    """Write an OBJ file keeping only faces with probability ≥ threshold."""
    verts_np = verts.detach().cpu().numpy()
    fp_np    = face_probs.detach().cpu().numpy()
    faces_np = faces.cpu().numpy()

    with open(path, "w") as fh:
        fh.write("# TopMod manifold loss demo\n")
        for x, y, z in verts_np:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, (a, b, c) in enumerate(faces_np):
            if fp_np[i] >= threshold:
                # OBJ is 1-indexed
                fh.write(f"f {a+1} {b+1} {c+1}\n")


def save_loss_curve(history: dict[str, list[float]]) -> str:
    """Try to plot with matplotlib; fall back to a plain text file."""
    path_txt = os.path.join(OUT_DIR, "loss_curve.txt")
    with open(path_txt, "w") as fh:
        fh.write("step\ttotal\tedge\teuler\torient\n")
        for i, (t, e, eu, o) in enumerate(
            zip(history["total"], history["edge"],
                history["euler"], history["orient"])
        ):
            fh.write(f"{i}\t{t:.6f}\t{e:.6f}\t{eu:.6f}\t{o:.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(history["total"])))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.semilogy(steps, history["total"], "k-", linewidth=2, label="total")
        ax1.set_ylabel("Loss (log scale)")
        ax1.set_title("Manifold Constraint Loss — Optimization Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, history["edge"],   label="edge manifold")
        ax2.plot(steps, history["euler"],  label="euler")
        ax2.plot(steps, history["orient"], label="orientation")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path_png = os.path.join(OUT_DIR, "loss_curve.png")
        plt.savefig(path_png, dpi=150)
        plt.close()
        print(f"  Saved loss curve → {path_png}")
        return path_png

    except ImportError:
        print("  matplotlib not available; saved loss_curve.txt instead")
        return path_txt


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(
    n_steps:      int   = 300,
    lr:           float = 0.05,
    lambda_edge:  float = 1.0,
    lambda_euler: float = 0.5,
    lambda_orient: float = 0.3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  TopMod Manifold Loss Demo")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Steps  : {n_steps}, LR: {lr}")

    # ── 1. Build non-manifold mesh ─────────────────────────────────────────
    verts, faces_base = tetrahedron_mesh()
    verts, faces      = add_rogue_faces(verts, faces_base)
    verts = verts.to(device)
    faces = faces.to(device)

    F_base  = faces_base.shape[0]   # 4  (valid tetrahedron faces)
    F_total = faces.shape[0]        # 8  (4 base + 4 rogue)

    print(f"\n  Mesh: {verts.shape[0]} vertices, {F_total} faces")
    print(f"  Base (valid) : {F_base} faces  (indices 0-{F_base-1})")
    print(f"  Rogue        : {F_total - F_base} faces  (indices {F_base}-{F_total-1})")

    # Initial breakdown (all probs = 0.9)
    fp_init = torch.full((F_total,), 0.9, device=device)
    init_bd = manifold_loss_breakdown(verts, faces, fp_init,
                                      target_genus=0,
                                      lambda_edge=lambda_edge,
                                      lambda_euler=lambda_euler,
                                      lambda_orient=lambda_orient)
    print(f"\n  Initial losses:")
    for k, v in init_bd.items():
        print(f"    {k:20s}: {v:.4f}")

    # Save before mesh
    before_path = os.path.join(OUT_DIR, "before_mesh.obj")
    save_obj(before_path, verts, faces, fp_init, threshold=0.5)
    print(f"\n  Saved before mesh → {before_path}")

    # ── 2. Optimise face_probs ─────────────────────────────────────────────
    # Parameterise as logits so we can optimise unconstrained
    logits = torch.zeros(F_total, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_steps, eta_min=lr * 0.01
    )

    history: dict[str, list[float]] = {
        "total": [], "edge": [], "euler": [], "orient": []
    }

    print(f"\n  Optimising...\n")
    print(f"  {'Step':>5}  {'Total':>10}  {'Edge':>10}  {'Euler':>10}  {'Orient':>10}")
    print(f"  {'-'*55}")

    for step in range(n_steps):
        optimiser.zero_grad()
        fp = torch.sigmoid(logits)
        loss = manifold_loss(
            verts, faces, face_probs=fp,
            target_genus=0,
            lambda_edge=lambda_edge,
            lambda_euler=lambda_euler,
            lambda_orient=lambda_orient,
        )
        loss.backward()
        optimiser.step()
        scheduler.step()

        bd = manifold_loss_breakdown(verts, faces, fp.detach(),
                                     target_genus=0,
                                     lambda_edge=lambda_edge,
                                     lambda_euler=lambda_euler,
                                     lambda_orient=lambda_orient)
        history["total"].append(bd["total"])
        history["edge"].append(bd["edge_manifold"])
        history["euler"].append(bd["euler"])
        history["orient"].append(bd["orientation"])

        if step % 50 == 0 or step == n_steps - 1:
            print(
                f"  {step:>5}  {bd['total']:>10.5f}  "
                f"{bd['edge_manifold']:>10.5f}  "
                f"{bd['euler']:>10.5f}  "
                f"{bd['orientation']:>10.5f}"
            )

    # ── 3. Final results ───────────────────────────────────────────────────
    final_fp = torch.sigmoid(logits).detach()
    print(f"\n  Final face probabilities:")
    for i, p in enumerate(final_fp.cpu().tolist()):
        tag = "BASE " if i < F_base else "ROGUE"
        print(f"    face {i:2d} [{tag}]: {p:.4f}")

    # Save after mesh (threshold 0.5: only high-prob faces kept)
    after_path = os.path.join(OUT_DIR, "after_mesh.obj")
    save_obj(after_path, verts, faces, final_fp, threshold=0.5)
    print(f"\n  Saved after mesh → {after_path}")

    # Save probabilities text
    prob_path = os.path.join(OUT_DIR, "face_probs.txt")
    with open(prob_path, "w") as fh:
        for i, p in enumerate(final_fp.cpu().tolist()):
            tag = "BASE" if i < F_base else "ROGUE"
            fh.write(f"face_{i:02d}\t{tag}\t{p:.6f}\n")
    print(f"  Saved face probs  → {prob_path}")

    # Save loss curve
    save_loss_curve(history)

    # ── 4. Convergence check ───────────────────────────────────────────────
    final_bd = manifold_loss_breakdown(verts, faces, final_fp,
                                       target_genus=0,
                                       lambda_edge=lambda_edge,
                                       lambda_euler=lambda_euler,
                                       lambda_orient=lambda_orient)
    print(f"\n  Final losses:")
    for k, v in final_bd.items():
        print(f"    {k:20s}: {v:.6f}")

    converged = final_bd["total"] < init_bd["total"] * 0.05
    print(f"\n  Convergence: {'✓ PASS' if converged else '✗ FAIL'}")
    print(f"    ({init_bd['total']:.4f} → {final_bd['total']:.6f},"
          f" reduction: {100*(1-final_bd['total']/init_bd['total']):.1f}%)")

    # Check that base faces have high prob and rogue faces have low prob
    base_min  = final_fp[:F_base].min().item()
    rogue_max = final_fp[F_base:].max().item()
    good_assignment = base_min > 0.8 and rogue_max < 0.2
    print(f"\n  Face assignment:")
    print(f"    Base min prob  : {base_min:.4f}  (want > 0.8): {'✓' if base_min > 0.8 else '?'}")
    print(f"    Rogue max prob : {rogue_max:.4f}  (want < 0.2): {'✓' if rogue_max < 0.2 else '?'}")

    print(f"\n{'='*60}\n")

    return converged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TopMod manifold loss demo")
    parser.add_argument("--steps",  type=int,   default=300)
    parser.add_argument("--lr",     type=float, default=0.05)
    args = parser.parse_args()

    success = run_demo(n_steps=args.steps, lr=args.lr)
    sys.exit(0 if success else 1)
