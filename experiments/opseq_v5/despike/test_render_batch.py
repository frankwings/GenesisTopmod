"""test_render_batch.py — validate that render_sdd_batch produces outputs
equivalent to the per-view render_sdd loop.

Tests (acceptance criteria):
  1. fg: boundary-pixel count <= NV (1 pixel per view from fp matmul ordering)
  2. sil/ndc/diff: per-pixel info (max/mean) — fp edge diffs are expected
  3. Losses: sil/depth/diff match loop with rtol=1e-4 (training-signal identity)
  4. Grads: atol=2e-4, rtol=1e-2 (1 boundary pixel can cause small local grad diffs)
  5. Timing: forward+backward speedup >= 2x

Acceptance definitions:
  - PASS = losses match to rtol=1e-4, grads match to atol=2e-4, speedup >= 2x
  - Individual pixel value differences at silhouette edges are EXPECTED and OK

Run:
  cd /home/kingy/Projects/Genesis/GenesisTopmod-wt-batch/experiments/opseq_v5
  python3 despike/test_render_batch.py
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod-wt-batch/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod-wt-batch/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod-wt-batch")
sys.path.append("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.append("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.append("/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

os.environ["RENDER_BATCH"] = "1"

import run_64v
from run_64v import render_sdd, render_sdd_batch, star_cameras, TRAIN_RES, NV
from eval_local_refine import depth_loss_masked, normalize_to_range, load_obj, BUNNY_PATH
from batch_losses import sil_loss_batch, depth_loss_batch, diff_loss_batch

DEVICE = "cuda"
results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    results.append((name, cond))
    return cond


def _load_mesh(npz_path):
    z = np.load(npz_path)
    return z["verts"].astype(np.float32), z["tris"].astype(np.int32)


def _build_scene(npz_path, shape="armadillo"):
    V, F = _load_mesh(npz_path)
    verts_t = torch.tensor(V, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(F, dtype=torch.int32, device=DEVICE)
    ctx = dr.RasterizeCudaContext()
    gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj"))
    gvn = normalize_to_range(gv)
    mvps, views = star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
    gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, shape)
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gtd_t   = [torch.from_numpy(np.asarray(gtd[i], np.float32)).to(DEVICE) for i in range(NV)]
    gtfg_t  = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(NV)]
    gtdf_t  = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]
    return ctx, verts_t, faces_t, mvps, views, targets, gtd_t, gtfg_t, gtdf_t


# ─────────────────────────────────────────────────────────────────────────────
_WT_DESPIKE = "/home/kingy/Projects/Genesis/GenesisTopmod-wt-batch/experiments/opseq_v5/despike"
NPZ1 = "/tmp/liou_cow_viz/cow_armadillo_v4cpp2_early.npz"
NPZ2 = f"{_WT_DESPIKE}/results_genus/armadillo_v4_auto.npz"

for npz_tag, npz_path in [("early", NPZ1), ("v4auto_49k", NPZ2)]:
    if not os.path.exists(npz_path):
        print(f"[SKIP] {npz_path} not found", flush=True)
        continue

    print(f"\n=== mesh: {npz_tag} ({npz_path}) ===", flush=True)
    ctx, verts_t, faces_t, mvps, views, targets, gtd_t, gtfg_t, gtdf_t = _build_scene(npz_path)
    gtd_stack  = torch.stack(gtd_t)
    gtfg_stack = torch.stack(gtfg_t)
    gtdf_stack = torch.stack(gtdf_t)
    tgt_batch  = targets[..., 0]   # [N,H,W]

    # ── Build per-view references ──────────────────────────────────────────────
    with torch.no_grad():
        sil_loop_list, ndc_loop_list, fg_loop_list, diff_loop_list = [], [], [], []
        for i in range(NV):
            sil, ndc, fg, diff = render_sdd(ctx, verts_t, faces_t, mvps[i], views[i])
            sil_loop_list.append(sil[0, :, :, 0])   # [H,W]
            ndc_loop_list.append(ndc)
            fg_loop_list.append(fg)
            diff_loop_list.append(diff)
        sil_stack_ref  = torch.stack(sil_loop_list)   # [N,H,W]
        ndc_stack_ref  = torch.stack(ndc_loop_list)
        fg_stack_ref   = torch.stack(fg_loop_list)
        diff_stack_ref = torch.stack(diff_loop_list)

        sil_b, ndc_b, fg_b, diff_b = render_sdd_batch(ctx, verts_t, faces_t, mvps, views)

    # Test 1: fg boundary pixel count
    n_fg_diff = int((fg_b != fg_stack_ref).sum())
    check(f"{npz_tag}/fg_boundary_pixels",
          n_fg_diff <= NV,
          f"boundary_pixels={n_fg_diff} (allow<={NV})")

    # Informational: per-output buffer stats (not acceptance tests)
    same_fg = fg_b == fg_stack_ref
    for name, a, b, mask in [
        ("sil",  sil_b,  sil_stack_ref,  same_fg),
        ("ndc",  ndc_b,  ndc_stack_ref,  same_fg & fg_stack_ref),
        ("diff", diff_b, diff_stack_ref, same_fg),
    ]:
        d = (a - b).abs()
        dm = d[mask] if mask.sum() > 0 else d.new_zeros(0)
        print(f"  [{name}] max={float(d.max()):.2e}  "
              f"mean_interior={float(dm.mean()) if dm.numel()>0 else 0:.2e}  "
              f"boundary_pixels={n_fg_diff}", flush=True)

    # Test 2: loss equivalence (PRIMARY acceptance criterion)
    with torch.no_grad():
        sl_loop = dl_loop = fl_loop = torch.tensor(0.0, device=DEVICE)
        for i in range(NV):
            sl_loop = sl_loop + F.l1_loss(sil_loop_list[i], tgt_batch[i])
            dl_loop = dl_loop + depth_loss_masked(ndc_loop_list[i], fg_loop_list[i],
                                                  gtd_t[i], gtfg_t[i])
            fl_loop = fl_loop + F.l1_loss(diff_loop_list[i], gtdf_t[i])
        sl_loop = sl_loop / NV; dl_loop = dl_loop / NV; fl_loop = fl_loop / NV

        sl_b = sil_loss_batch(sil_b, tgt_batch)
        dl_b = depth_loss_batch(ndc_b, fg_b, gtd_stack, gtfg_stack)
        fl_b = diff_loss_batch(diff_b, gtdf_stack)

    check(f"{npz_tag}/sil_loss_eq",
          torch.allclose(sl_b, sl_loop, rtol=1e-4),
          f"loop={sl_loop.item():.8f} batch={sl_b.item():.8f}")
    check(f"{npz_tag}/depth_loss_eq",
          torch.allclose(dl_b, dl_loop, rtol=1e-4),
          f"loop={dl_loop.item():.8f} batch={dl_b.item():.8f}")
    check(f"{npz_tag}/diff_loss_eq",
          torch.allclose(fl_b, fl_loop, rtol=1e-4),
          f"loop={fl_loop.item():.8f} batch={fl_b.item():.8f}")

    # Test 3: gradient equivalence
    # atol=2e-4: 1 boundary pixel antialias grad can cause small local differences;
    # rtol=1e-2: 1% relative tolerance on individual vertex gradients
    v_loop = verts_t.detach().clone().requires_grad_(True)
    sl_g = dl_g = fl_g = torch.tensor(0.0, device=DEVICE)
    for i in range(NV):
        sil, ndc, fg, diff = render_sdd(ctx, v_loop, faces_t, mvps[i], views[i])
        sl_g = sl_g + F.l1_loss(sil[0, :, :, 0], tgt_batch[i])
        dl_g = dl_g + depth_loss_masked(ndc, fg, gtd_t[i], gtfg_t[i])
        fl_g = fl_g + F.l1_loss(diff, gtdf_t[i])
    (sl_g / NV + dl_g / NV + fl_g / NV).backward()
    grad_loop = v_loop.grad.clone()

    v_batch = verts_t.detach().clone().requires_grad_(True)
    sil_b2, ndc_b2, fg_b2, diff_b2 = render_sdd_batch(ctx, v_batch, faces_t, mvps, views)
    loss_b2 = (sil_loss_batch(sil_b2, tgt_batch)
               + depth_loss_batch(ndc_b2, fg_b2, gtd_stack, gtfg_stack)
               + diff_loss_batch(diff_b2, gtdf_stack))
    loss_b2.backward()
    grad_batch = v_batch.grad.clone()

    max_ge  = float((grad_loop - grad_batch).abs().max())
    ref_max = float(grad_loop.abs().max())
    ref_mean = float(grad_loop.abs().mean())
    check(f"{npz_tag}/grad_atol_2e4",
          torch.allclose(grad_batch, grad_loop, atol=2e-4, rtol=1e-2),
          f"max_diff={max_ge:.2e} ref_max={ref_max:.2e} ref_mean={ref_mean:.2e}")

    # Test 4: timing speedup
    WARMUP = 3; REPS = 5

    def _loop_step():
        v = verts_t.detach().clone().requires_grad_(True)
        sl = dl = fl = torch.tensor(0.0, device=DEVICE)
        for i in range(NV):
            sil, ndc, fg, diff = render_sdd(ctx, v, faces_t, mvps[i], views[i])
            sl = sl + F.l1_loss(sil[0, :, :, 0], tgt_batch[i])
            dl = dl + depth_loss_masked(ndc, fg, gtd_t[i], gtfg_t[i])
            fl = fl + F.l1_loss(diff, gtdf_t[i])
        (sl / NV + dl / NV + fl / NV).backward()
        torch.cuda.synchronize()

    def _batch_step():
        v = verts_t.detach().clone().requires_grad_(True)
        sil_b, ndc_b, fg_b, diff_b = render_sdd_batch(ctx, v, faces_t, mvps, views)
        loss_b = (sil_loss_batch(sil_b, tgt_batch)
                  + depth_loss_batch(ndc_b, fg_b, gtd_stack, gtfg_stack)
                  + diff_loss_batch(diff_b, gtdf_stack))
        loss_b.backward()
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        _loop_step(); _batch_step()

    t0 = time.perf_counter()
    for _ in range(REPS): _loop_step()
    ms_loop = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS): _batch_step()
    ms_batch = (time.perf_counter() - t0) / REPS * 1000

    speedup = ms_loop / ms_batch
    print(f"[timing] {npz_tag}: loop={ms_loop:.1f}ms  batch={ms_batch:.1f}ms  speedup={speedup:.2f}x",
          flush=True)
    check(f"{npz_tag}/speedup_ge_2x", speedup >= 2.0, f"{speedup:.2f}x")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===", flush=True)
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
for name, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{n_pass}/{len(results)} tests passed", flush=True)
if n_fail:
    sys.exit(1)
