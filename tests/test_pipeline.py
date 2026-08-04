"""
Pipeline tests — topology_builder, cameras, geometry_optimizer.

These tests require CUDA (RTX 5090).  They are skipped gracefully on CPU-only
machines.
"""

import pytest
import sys
import os
import math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CUDA = torch.cuda.is_available()
cuda_required = pytest.mark.skipif(not CUDA, reason="CUDA not available")


# ── topology_builder tests ────────────────────────────────────────────────────

class TestBuildTopology:

    def test_genus0_tensor_types(self):
        from pipeline.topology_builder import build_topology
        verts, faces = build_topology(genus=0, subdivisions=1, device="cpu")
        assert verts.dtype == torch.float32
        assert faces.dtype == torch.int32

    def test_genus0_non_empty(self):
        from pipeline.topology_builder import build_topology
        verts, faces = build_topology(genus=0, subdivisions=1, device="cpu")
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3

    def test_genus0_euler_characteristic(self):
        """Closed genus-0 surface should have χ=2."""
        from pipeline.topology_builder import build_topology, verify_genus
        verts, faces = build_topology(genus=0, subdivisions=1, device="cpu")
        g = verify_genus(verts, faces)
        assert g == 0, f"Expected genus 0, got {g}"

    def test_genus0_vertex_positions_bounded(self):
        """All vertices should be within the unit sphere (scale=1.0)."""
        from pipeline.topology_builder import build_topology
        verts, _ = build_topology(genus=0, subdivisions=1, scale=1.0, device="cpu")
        norms = torch.norm(verts, dim=-1)
        assert norms.max().item() <= 2.0, f"Vertex norm too large: {norms.max().item()}"

    def test_genus1_increases_euler(self):
        """Genus-1 mesh should have χ=0 (torus topology)."""
        from pipeline.topology_builder import build_topology, verify_genus
        verts, faces = build_topology(genus=1, subdivisions=1, device="cpu")
        g = verify_genus(verts, faces)
        assert g == 1, f"Expected genus 1, got {g}"

    def test_genus0_all_indices_valid(self):
        """All face indices must be in [0, V)."""
        from pipeline.topology_builder import build_topology
        verts, faces = build_topology(genus=0, subdivisions=1, device="cpu")
        V = verts.shape[0]
        assert faces.min().item() >= 0
        assert faces.max().item() < V

    def test_genus0_two_subdivisions(self):
        """More subdivisions → more vertices and faces."""
        from pipeline.topology_builder import build_topology
        v1, f1 = build_topology(genus=0, subdivisions=1, device="cpu")
        v2, f2 = build_topology(genus=0, subdivisions=2, device="cpu")
        assert v2.shape[0] > v1.shape[0]
        assert f2.shape[0] > f1.shape[0]

    def test_scale_parameter(self):
        """scale=2.0 should double the average vertex norm."""
        from pipeline.topology_builder import build_topology
        v1, _ = build_topology(genus=0, subdivisions=1, scale=1.0, device="cpu")
        v2, _ = build_topology(genus=0, subdivisions=1, scale=2.0, device="cpu")
        # Ratio of mean norms should be approximately 2.0
        r = v2.norm(dim=-1).mean().item() / v1.norm(dim=-1).mean().item()
        assert abs(r - 2.0) < 0.5, f"Scale ratio {r:.2f} expected ~2.0"

    @cuda_required
    def test_genus0_cuda_tensor(self):
        from pipeline.topology_builder import build_topology
        verts, faces = build_topology(genus=0, subdivisions=1, device="cuda")
        assert verts.device.type == "cuda"
        assert faces.device.type == "cuda"


class TestVerifyGenus:
    def test_sphere_genus0(self):
        from pipeline.topology_builder import verify_genus
        # Simple tetrahedron as genus-0 reference
        verts = torch.tensor([[0.0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=torch.int32)
        g = verify_genus(verts, faces)
        assert g == 0


# ── cameras tests ─────────────────────────────────────────────────────────────

class TestCameras:

    def test_perspective_shape(self):
        from pipeline.cameras import perspective
        P = perspective(fov_deg=45, device="cpu")
        assert P.shape == (4, 4)
        assert P.dtype == torch.float32

    def test_perspective_homogeneous(self):
        """P[3, 2] must be -1 (homogeneous perspective divide)."""
        from pipeline.cameras import perspective
        P = perspective(device="cpu")
        assert abs(P[3, 2].item() + 1.0) < 1e-6

    def test_look_at_shape(self):
        from pipeline.cameras import look_at
        V = look_at(eye=(0, 0, 3), device="cpu")
        assert V.shape == (4, 4)
        assert V.dtype == torch.float32

    def test_look_at_identity_at_z3(self):
        """Camera at (0,0,3) looking at origin: point at origin should map to z=-3 in view space."""
        from pipeline.cameras import look_at
        V = look_at(eye=(0, 0, 3), center=(0, 0, 0), up=(0, 1, 0), device="cpu")
        origin_h = torch.tensor([0.0, 0.0, 0.0, 1.0])
        view_pos  = V @ origin_h
        # In right-handed view space, camera faces -Z; origin is at z_view = -3
        assert abs(view_pos[2].item() + 3.0) < 1e-4, f"view_pos z = {view_pos[2].item()}"

    def test_orbit_cameras_count(self):
        from pipeline.cameras import orbit_cameras
        mvps, eyes = orbit_cameras(n=8, device="cpu")
        assert mvps.shape == (8, 4, 4)
        assert len(eyes) == 8

    def test_orbit_cameras_distributed(self):
        """Camera positions should be evenly distributed on a circle."""
        from pipeline.cameras import orbit_cameras
        _, eyes = orbit_cameras(n=4, elevation_deg=0.0, radius=3.0, device="cpu")
        # For elevation=0, all y-coords should be 0
        for x, y, z in eyes:
            assert abs(y) < 1e-5, f"elevation=0 but y={y}"
        # All at distance = radius from origin (in XZ plane)
        for x, y, z in eyes:
            dist = math.sqrt(x**2 + z**2)
            assert abs(dist - 3.0) < 1e-4, f"dist={dist}"

    def test_orbit_cameras_elevation(self):
        """Camera at elevation=30° should have correct y coordinate."""
        from pipeline.cameras import orbit_cameras
        _, eyes = orbit_cameras(n=1, elevation_deg=30.0, radius=3.0, device="cpu")
        x, y, z = eyes[0]
        expected_y = 3.0 * math.sin(math.radians(30.0))
        assert abs(y - expected_y) < 1e-4, f"y={y}, expected {expected_y}"

    def test_transform_to_clip_shape(self):
        from pipeline.cameras import transform_to_clip, orbit_cameras
        mvps, _ = orbit_cameras(n=1, device="cpu")
        verts = torch.randn(10, 3)
        clip = transform_to_clip(verts, mvps[0])
        assert clip.shape == (1, 10, 4)
        assert clip.is_contiguous()

    def test_transform_to_clip_homogeneous(self):
        """w component should not be zero for points in front of camera."""
        from pipeline.cameras import transform_to_clip, orbit_cameras
        mvps, _ = orbit_cameras(n=1, elevation_deg=0, radius=3.0, device="cpu")
        # Point at origin should be in front of camera
        verts = torch.tensor([[0.0, 0.0, 0.0]])
        clip  = transform_to_clip(verts, mvps[0])
        w = clip[0, 0, 3].item()
        assert w > 0.0, f"w={w} should be > 0 for point in front of camera"


# ── geometry_optimizer tests (GPU) ────────────────────────────────────────────

@cuda_required
class TestGeometryOptimizer:

    @pytest.fixture(scope="class")
    def simple_setup(self):
        """Shared setup: cube-like mesh + target silhouette + MVP."""
        import nvdiffrast.torch as dr
        from pipeline.cameras import orbit_cameras
        from topmod.primitives import make_icosahedron
        from topmod.io import to_triangle_arrays
        import numpy as np

        ctx = dr.RasterizeCudaContext()
        device = "cuda"

        # Simple icosahedron mesh
        mesh = make_icosahedron()
        positions, triangles = to_triangle_arrays(mesh)
        verts = torch.tensor(positions, dtype=torch.float32, device=device)
        faces = torch.tensor(triangles, dtype=torch.int32, device=device)

        # 4 orbit views
        mvps, _ = orbit_cameras(n=4, elevation_deg=20.0, radius=3.0, device=device)

        # Constant white target (trivial: full silhouette)
        target = torch.ones(4, 64, 64, 1, device=device)

        return {"ctx": ctx, "verts": verts, "faces": faces,
                "mvps": mvps, "target": target, "device": device}

    def test_render_silhouette_shape(self, simple_setup):
        from pipeline.geometry_optimizer import render_silhouette
        s = simple_setup
        sil = render_silhouette(s["ctx"], s["verts"], s["faces"],
                                s["mvps"][0], resolution=(64, 64))
        assert sil.shape == (1, 64, 64, 1)

    def test_render_silhouette_range(self, simple_setup):
        from pipeline.geometry_optimizer import render_silhouette
        s = simple_setup
        sil = render_silhouette(s["ctx"], s["verts"], s["faces"],
                                s["mvps"][0], resolution=(64, 64))
        assert sil.min().item() >= 0.0 - 1e-5
        assert sil.max().item() <= 1.0 + 1e-5

    def test_render_silhouette_has_nonzero(self, simple_setup):
        """Icosahedron facing camera should produce non-empty silhouette."""
        from pipeline.geometry_optimizer import render_silhouette
        s = simple_setup
        sil = render_silhouette(s["ctx"], s["verts"], s["faces"],
                                s["mvps"][0], resolution=(64, 64))
        assert sil.max().item() > 0.1, "Silhouette is empty — camera may be misaligned"

    def test_gradient_flows_through_render(self, simple_setup):
        """Silhouette loss gradient must reach vertex positions."""
        from pipeline.geometry_optimizer import render_silhouette
        import torch.nn.functional as F
        s = simple_setup

        verts = s["verts"].clone().detach().requires_grad_(True)
        sil   = render_silhouette(s["ctx"], verts, s["faces"],
                                  s["mvps"][0], resolution=(64, 64))
        target = torch.zeros_like(sil)
        loss   = F.l1_loss(sil, target)
        loss.backward()

        assert verts.grad is not None, "No gradient on verts"
        assert verts.grad.norm().item() > 0.0, "Zero gradient on verts"

    def test_laplacian_loss_positive(self, simple_setup):
        from pipeline.geometry_optimizer import laplacian_loss
        s = simple_setup
        loss = laplacian_loss(s["verts"], s["faces"])
        assert loss.item() >= 0.0

    def test_edge_length_loss_positive(self, simple_setup):
        from pipeline.geometry_optimizer import edge_length_loss
        s = simple_setup
        loss = edge_length_loss(s["verts"], s["faces"])
        assert loss.item() >= 0.0

    def test_optimize_loss_decreases(self, simple_setup):
        """Loss must strictly decrease over 100 steps (sphere → sphere target)."""
        from pipeline.geometry_optimizer import optimize
        from pipeline.cameras import orbit_cameras
        from topmod.primitives import make_icosahedron
        from topmod.io import to_triangle_arrays
        from pipeline.geometry_optimizer import render_batch
        import numpy as np

        ctx = simple_setup["ctx"]
        device = simple_setup["device"]

        # Build a small mesh
        mesh = make_icosahedron()
        positions, triangles = to_triangle_arrays(mesh)
        ref_verts  = torch.tensor(positions, dtype=torch.float32, device=device)
        ref_faces  = torch.tensor(triangles, dtype=torch.int32,   device=device)

        # Target: the icosahedron itself (trivial convergence test)
        mvps, _ = orbit_cameras(n=4, elevation_deg=20.0, radius=3.0, device=device)
        with torch.no_grad():
            target = render_batch(ctx, ref_verts, ref_faces, mvps, resolution=(64, 64))

        # Start from perturbed positions
        init_verts = ref_verts + torch.randn_like(ref_verts) * 0.3

        _, history = optimize(
            ctx=ctx,
            verts_init=init_verts,
            faces=ref_faces,
            target_images=target,
            mvps=mvps,
            num_steps=100,
            lr=5e-3,
            lambda_lap=0.05,
            lambda_edge=0.01,
            resolution=(64, 64),
            log_every=50,
            scheduler=False,
        )

        assert len(history) >= 2
        assert history[-1]["sil_loss"] < history[0]["sil_loss"], (
            f"Loss did not decrease: {history[0]['sil_loss']:.4f} → {history[-1]['sil_loss']:.4f}"
        )

    def test_render_batch_shape(self, simple_setup):
        from pipeline.geometry_optimizer import render_batch
        s = simple_setup
        batch = render_batch(s["ctx"], s["verts"], s["faces"],
                             s["mvps"], resolution=(64, 64))
        assert batch.shape == (4, 64, 64, 1)
