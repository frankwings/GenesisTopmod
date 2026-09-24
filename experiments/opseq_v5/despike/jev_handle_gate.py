"""Jev decision gate for add_handle accept/reject (痛点 4).

Replaces phase7_multi.sh's hand-tuned threshold OR-rule
    accept  <=>  d_ho16 >= 0.002  OR  hair <= 0.9 * hair_ref
with a TypeSafe *Jev* typed decision that ALSO consumes the space-carving
count prior (g*) -- the signal the hard threshold ignores and the reason
fertility flickers 3/4 (genus3-vs-4 held-out gap = 0.0003 << 0.002 noise floor).

Real backend: TypeSafe Jev via pydantic_ai. Set TYPESAFE_API_KEY to enable.
No key -> a TRANSPARENT LOCAL SURROGATE runs so the wiring/policy is testable
end-to-end; the surrogate is NOT Jev's judgment, it is a documented stand-in
whose only job is to prove the plumbing and the count-aware policy. Swap is
one line (USE_REAL below flips on key presence).

CLI (dry-run one candidate):
  python3 jev_handle_gate.py --g-star 4 --cur-genus 3 --d-ho 0.0003 \
      --hair-ratio 1.0 --out-vox 2.4 --blob-vox 391 --located hull
"""
import os, json, argparse, math
from dataclasses import dataclass, asdict

VERIFY_DHO = float(os.environ.get("VERIFY_DHO", "0.002"))
VERIFY_HAIR = float(os.environ.get("VERIFY_HAIR", "0.9"))
JEV_CONF_MIN = float(os.environ.get("JEV_CONF_MIN", "0.50"))  # gate on the discrete verdict; laya confidence is
# compressed near 0.5 and self-reports as UNCALIBRATED (temperature clamp warning), so 0.50 == "trust the verdict".
AIR_VOX = float(os.environ.get("AIR_VOX", "2.0"))   # membrane faces this many voxels outside the hull == genuine air
BLOB_MIN = int(os.environ.get("BLOB_MIN", "40"))    # see-through block smaller than this is treated as noise


@dataclass
class HandleFeatures:
    """The 'ticket' handed to Jev: everything known about one proposed handle."""
    hull_genus_target: int      # g* from space-carving oracle (how many tunnels the shape SHOULD have)
    current_mesh_genus: int     # mesh genus BEFORE this handle
    d_ho16: float               # held-out silhouette gain after opening (DR propose-and-verify)
    verify_threshold: float     # the legacy hard cutoff, for reference in the prompt
    hair_ratio: float           # hair_new / hair_ref (spike count change; <1 = handle cleaned spikes)
    out_vox: float              # how far outside the hull the membrane faces sit (voxels); >2 = real air
    blob_vox: int               # voxel-block size of the see-through membrane (bigger = more real)
    sep_edges: float            # graph separation of the two mesh faces being bridged
    located_by: str             # 'hull' | 'ray' | 'bridge' -- provenance of the candidate

    def count_says_missing(self) -> bool:
        return self.current_mesh_genus < self.hull_genus_target


# ----------------------------------------------------------------------------- Jev prompt (material judged)
def build_ticket(f: HandleFeatures) -> str:
    """Neutral facts. The two decisive NUMBERS (count prior, air depth) are honestly
    translated into qualitative statements, because the typed-decision model reads words,
    not numeric comparisons. No accept-leaning editorialising."""
    missing = f.hull_genus_target - f.current_mesh_genus
    count = (f"A tunnel is still MISSING: the oracle counts {f.hull_genus_target} tunnels but the mesh has "
             f"only {f.current_mesh_genus}." if missing > 0 else
             f"NO tunnel is missing: the mesh already has all {f.hull_genus_target} tunnels the oracle counts.")
    if f.out_vox >= 2.0:
        air = f"The membrane is OPEN AIR (its faces sit {f.out_vox:.1f} voxels outside the solid hull)."
    elif f.out_vox >= 1.0:
        air = f"The membrane sits AT THE HULL SURFACE ({f.out_vox:.1f} voxels out) — ambiguous, not clearly air."
    else:
        air = f"The membrane is INSIDE SOLID material ({f.out_vox:.1f} voxels out) — opening it would pierce solid."
    block = (f"The see-through block is sizable ({f.blob_vox} voxels)." if f.blob_vox >= 40
             else f"The see-through block is TINY ({f.blob_vox} voxels) — likely noise, not a hole.")
    if f.d_ho16 >= f.verify_threshold:
        render = f"Opening it IMPROVED the render fit ({f.d_ho16:+.4f}, above the {f.verify_threshold:+.4f} cutoff)."
    elif f.d_ho16 >= 0:
        render = (f"The render fit barely moved ({f.d_ho16:+.4f}, below the {f.verify_threshold:+.4f} cutoff) — "
                  f"uninformative, since a real tunnel can be at the render noise floor.")
    else:
        render = f"Opening it made the render fit WORSE ({f.d_ho16:+.4f})."
    hair = ("It also removed surface spikes." if f.hair_ratio <= VERIFY_HAIR else "It did not change surface spikes.")
    return ("Decide whether to keep a proposed through-tunnel in a multi-view mesh reconstruction.\n"
            f"- {count}\n- {air}\n- {block}\n- {render}\n- {hair}\n"
            f"- Candidate located by: {f.located_by}.")


# ----------------------------------------------------------------------------- output type + question
def _real_jev(f: HandleFeatures):
    from typing import Literal
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai.models.typesafe import TypeSafeModelSettings

    class HandleVerdict(BaseModel):
        """Decide whether to KEEP this proposed genus-raising handle or REVERT it.
        Keep it when the evidence (especially: a tunnel is still missing per the
        oracle AND the membrane is genuinely open air) says it is a real through-
        tunnel, even if the render-score gain is at the noise floor. Reject when
        the opening is not supported (no missing tunnel, membrane not clearly air,
        and no render gain)."""
        verdict: Literal["accept", "reject"] = Field(description="keep the handle vs revert it")
        is_real_tunnel: bool = Field(description="Is this a genuine through-tunnel (not a surface notch)?")

    agent = Agent("typesafe:jev-latest", output_type=HandleVerdict,
                  model_settings=TypeSafeModelSettings(timeout=8))
    r = agent.run_sync(build_ticket(f))
    conf = (r.response.provider_details or {}).get("confidence", {})
    c = float(conf.get("verdict", 1.0)) if isinstance(conf, dict) else 1.0
    return r.output.verdict, c, "jev"


# ----------------------------------------------------------------------------- Rule C: count-first deterministic gate
def _count_first(f: HandleFeatures):
    """Rule C (the shipped gate). The pipeline only proposes a handle when g < g* (the
    space-carving oracle guarantees a tunnel is still missing). So:
      accept  <=>  legacy render/hair evidence  OR  (a tunnel is missing AND the membrane is genuine air).
    The second clause is the count-first RESCUE: once the oracle says a tunnel is missing there and the
    membrane is real open air, keep the handle REGARDLESS of the differentiable-render score sign -- because
    after refinement a real thin tunnel often has a slightly NEGATIVE dho (render noise floor), which is exactly
    where the legacy hard threshold and the (coin-flip, uncalibrated) text model both fail. Deterministic,
    reproducible, no external dependency. Bounded by the g* stop, so it can never overshoot the genus."""
    thr = f.verify_threshold
    legacy = (f.d_ho16 >= thr) or (f.hair_ratio <= VERIFY_HAIR)
    # Rule C':  hull-located candidates come straight from the space-carving oracle's tunnel LOCATION, so
    # when a tunnel is still missing (g<g*) we trust them unconditionally -- even at the refined stage where
    # the mesh is flush to the hull (out_vox=0) and the geometric air signal is gone. Ray-fallback candidates
    # are only image guesses, so they still require genuine air.
    air = (f.out_vox >= AIR_VOX) and (f.blob_vox >= BLOB_MIN)
    hull_located = (f.located_by == "hull")
    rescue = f.count_says_missing() and (hull_located or air)
    accept = legacy or rescue
    tag = ":rescue-hull" if (rescue and not legacy and hull_located) else (":rescue-air" if (rescue and not legacy) else "")
    return ("accept" if accept else "reject"), (1.0 if accept else 0.0), ("count" + tag)


# ----------------------------------------------------------------------------- Laya (free local backend, same paradigm)
LAYA_QUESTIONS = {
    "verdict": {"type": "choice", "instructions":
        "Should we KEEP this proposed genus-raising handle (a real through-tunnel) or REVERT it (spurious opening)? "
        "Keep it if a tunnel is still missing per the space-carving oracle AND the membrane is genuine open air, "
        "even when the render-score gain is at the noise floor. Reject if nothing is missing and the membrane is "
        "not clearly open air.",
        "criteria": {"accept": "real through-tunnel; keep the handle",
                     "reject": "not a real tunnel; revert the handle"}},
    "is_real_tunnel": {"type": "noul", "instructions":
        "Is this opening a genuine through-tunnel (goes all the way through), not a surface notch/dent?"},
}
_LAYA_ROUTER = None


def _laya(f: HandleFeatures):
    global _LAYA_ROUTER
    from laya import Router
    if _LAYA_ROUTER is None:
        _LAYA_ROUTER = Router(device=os.environ.get("LAYA_DEVICE", "cpu"), preload=False)
    out = _LAYA_ROUTER.predict(build_ticket(f), LAYA_QUESTIONS)
    a = out["answers"]["verdict"]
    verdict = a["choice"]
    conf = float(a.get("probabilities", {}).get("accept", a.get("confidence", 0.5)))
    return verdict, conf, "laya"


# ----------------------------------------------------------------------------- transparent surrogate
def _surrogate(f: HandleFeatures):
    """Documented stand-in (NOT Jev). Same I/O contract: (verdict, confidence).
    Policy = legacy threshold OR count-aware acceptance:
      * legacy signal:  d_ho16 >= thr  OR  hair collapse
      * count signal:   a tunnel is missing (g<g*) AND membrane is genuine air
                        (out_vox>=2) AND block is non-trivial (blob>=40)
    Confidence = logistic blend of the render gain and the count/air evidence."""
    thr = VERIFY_THRESHOLD_of(f)
    count_air = f.count_says_missing() and f.out_vox >= 2.0 and f.blob_vox >= 40
    z = -1.0                                                   # bias: default lean reject
    z += 2.0 if count_air else 0.0                            # a tunnel is missing AND membrane is genuine air
    z += 1.5 * max(0.0, f.out_vox - 2.0)                      # more open air = more confident
    z += 300.0 * (f.d_ho16 - thr)                            # render gain in ho units (+0.002 over cutoff -> +0.6)
    z += 0.8 if f.hair_ratio <= VERIFY_HAIR else 0.0         # handle also removed spikes
    z -= 1.5 if (not f.count_says_missing() and f.d_ho16 < thr) else 0.0  # count satisfied + no gain -> reject
    conf_accept = 1.0 / (1.0 + math.exp(-z))
    accept = conf_accept >= 0.5
    conf = conf_accept if accept else (1.0 - conf_accept)
    return ("accept" if accept else "reject"), conf, "surrogate"


def VERIFY_THRESHOLD_of(f: HandleFeatures) -> float:
    return f.verify_threshold


# ----------------------------------------------------------------------------- entry point
def decide(f: HandleFeatures):
    """Returns dict: {accept, verdict, confidence, source, gated_accept}."""
    # Default backend = Rule C (count-first deterministic). laya/jev kept for archival comparison only.
    backend = os.environ.get("GATE_BACKEND", "count")   # count | laya | jev | surrogate
    try:
        if backend == "count":
            verdict, conf, src = _count_first(f)
        elif backend == "laya":
            verdict, conf, src = _laya(f)
        elif backend == "jev" and os.environ.get("TYPESAFE_API_KEY"):
            verdict, conf, src = _real_jev(f)
        else:
            verdict, conf, src = _surrogate(f)
    except Exception as e:
        verdict, conf, src = (*_surrogate(f)[:2], f"surrogate({backend}_failed:{type(e).__name__})")
    gated = (verdict == "accept") and (conf >= JEV_CONF_MIN)
    return {"accept": verdict == "accept", "verdict": verdict, "confidence": round(conf, 3),
            "source": src, "gated_accept": gated}


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--g-star", type=int, required=True)
    p.add_argument("--cur-genus", type=int, required=True)
    p.add_argument("--d-ho", type=float, required=True)
    p.add_argument("--thr", type=float, default=VERIFY_DHO)
    p.add_argument("--hair-ratio", type=float, default=1.0)
    p.add_argument("--out-vox", type=float, default=2.5)
    p.add_argument("--blob-vox", type=int, default=100)
    p.add_argument("--sep-edges", type=float, default=8.0)
    p.add_argument("--located", default="hull")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    f = HandleFeatures(a.g_star, a.cur_genus, a.d_ho, a.thr, a.hair_ratio,
                       a.out_vox, a.blob_vox, a.sep_edges, a.located)
    out = decide(f)
    if a.json:
        print(json.dumps({**out, "features": asdict(f)}))
    else:
        print("--- TICKET ---"); print(build_ticket(f))
        print("--- DECISION ---")
        print(f"source={out['source']}  verdict={out['verdict']}  confidence={out['confidence']}  "
              f"gated_accept={out['gated_accept']} (conf>={JEV_CONF_MIN})")
        legacy = "ACCEPT" if (f.d_ho16 >= f.verify_threshold or f.hair_ratio <= VERIFY_HAIR) else "REJECT"
        print(f"[legacy hard-threshold would say: {legacy}]")
