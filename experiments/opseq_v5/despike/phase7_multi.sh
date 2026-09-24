#!/bin/bash
# Multi-hole driver: repeat {detect tunnel evidence -> 1 add_handle -> short Stage-4 loop} until no evidence,
# then Stage 5 + Taubin. One handle per round avoids tube-adjacency conflicts (the opened hole leaves no membrane).
# Usage: SHAPE=threeholes ROUNDS=6 bash despike/phase7_multi.sh <base_npz_after_stage4>
set -u
cd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5
S=$SHAPE; R=${ROUNDS:-6}; cur=$1; LOOP=${LOOP_STEPS:-400}
export HANDLES_JSON=/tmp/liou_cow_viz/handles_${S}.json; rm -f $HANDLES_JSON /tmp/liou_cow_viz/cow_${S}_${S}_h[0-9]*.npz   # no stale round outputs
COMMON="MEMB_EXEMPT=2 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_FRAC=${COLLAPSE_FRAC:-0.02} SI_PUSH=0.15"
for r in $(seq 1 $R); do
  [ "${SKIP_ROUNDS:-0}" = "1" ] && break
  echo "##### $S round $r: detect + add_handle on $cur"
  out=$(MODE=64v SHAPE=$S TAG=${S}_h$r MAX_HANDLES=1 BASE_NPZ=$cur python3 despike/phase7_handle.py 2>&1 | grep "\[p7\]\|\[memb\]\|\[after\|\[base\]\|Traceback\|Error")
  echo "$out"
  if echo "$out" | grep -q "handles added: 0"; then echo "##### no more tunnel evidence after $((r-1)) handles"; break; fi
  if [ ! -f /tmp/liou_cow_viz/cow_${S}_${S}_h$r.npz ]; then echo "##### handle stage failed in round $r (see above); stopping with $((r-1)) handles"; break; fi
  prev_cur=$cur; cur=/tmp/liou_cow_viz/cow_${S}_${S}_h$r.npz
  echo "##### $S round $r: Stage-4 loop $LOOP steps"
  fin=$(env MODE=64v SHAPE=$S TAG=${S}_h${r}b STEPS=$LOOP $COMMON BASE_NPZ=$cur python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|Traceback\|Error"); echo "$fin"
  # VERIFY (propose-and-verify): the handle must open something. Compare the post-DR held-out score with the
  # reference (previous accepted round's post-DR score; round 1: the [base] score of the input mesh).
  ho_new=$(echo "$fin" | sed -nE 's/.*ho16=([0-9.]+) hair=([0-9]+).*/\1/p' | tail -1); hair_new=$(echo "$fin" | sed -nE 's/.*ho16=([0-9.]+) hair=([0-9]+).*/\2/p' | tail -1)
  if [ -z "${ho_ref:-}" ]; then ho_ref=$(echo "$out" | sed -nE 's/^\[base\].*ho16=([0-9.]+) hair=([0-9]+).*/\1/p' | head -1); hair_ref=$(echo "$out" | sed -nE 's/^\[base\].*ho16=([0-9.]+) hair=([0-9]+).*/\2/p' | head -1); fi
  ok=$(python3 -c "import sys; ho,hr,h0,r0=map(float,sys.argv[1:]); print(1 if (ho>=h0+${VERIFY_DHO:-0.002} or hr<=${VERIFY_HAIR:-0.9}*r0) else 0)" "${ho_new:-0}" "${hair_new:-0}" "${ho_ref:-0}" "${hair_ref:-0}")
  # --- Jev decision gate (痛点 4): count-aware typed decision replaces the hard threshold. USE_JEV=1 to enable.
  # Feeds the space-carving prior (g*) + geometric air evidence, so borderline missing tunnels (dho at noise floor)
  # are accepted when the oracle says a tunnel is missing AND the membrane is genuine air. Needs TYPESAFE_API_KEY
  # for the real backend; otherwise runs the documented surrogate. Legacy path unchanged when USE_JEV unset.
  if [ "${USE_JEV:-0}" = "1" ]; then
    gstar=$(echo "$out" | grep -oE 'g\*=[0-9]+' | grep -oE '[0-9]+' | head -1)
    curg=$(python3 -c "import numpy as np; z=np.load('$prev_cur'); V,F=z['verts'],z['tris'].astype('int64'); E=len(np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),1),axis=0)); print((2-(len(V)-E+len(F)))//2)" 2>/dev/null)
    outvox=$(echo "$out" | sed -nE 's/.*add_handle between faces [0-9]+,[0-9]+: out ([0-9.]+)\/([0-9.]+) vox.*/\1 \2/p' | tail -1 | python3 -c "import sys; a=sys.stdin.read().split(); print(min(map(float,a)) if a else 2.5)" 2>/dev/null)
    blob=$(python3 -c "import json; h=json.load(open('$HANDLES_JSON')); b=h[-1].get('blob'); print(int(min(b)) if isinstance(b,list) and b else 100)" 2>/dev/null)
    prov=hull; echo "$out" | grep -q "(ray" && prov=ray   # real provenance: ray-fallback candidates are image guesses, not oracle-located
    jev=$(VERIFY_DHO=${VERIFY_DHO:-0.002} python3 despike/jev_handle_gate.py --json \
      --g-star "${gstar:-0}" --cur-genus "${curg:-0}" \
      --d-ho "$(python3 -c "print(${ho_new:-0}-${ho_ref:-0})")" --hair-ratio "$(python3 -c "print(${hair_new:-1}/max(${hair_ref:-1},1e-9))")" \
      --out-vox "${outvox:-2.5}" --blob-vox "${blob:-100}" --located "$prov" 2>/dev/null)
    ok=$(echo "$jev" | python3 -c "import sys,json; print(1 if json.load(sys.stdin)['gated_accept'] else 0)" 2>/dev/null)
    echo "##### $S round $r: JEV gate (g$curg<g*$gstar, dho=$(python3 -c "print(round(${ho_new:-0}-${ho_ref:-0},4))"), out=${outvox} blob=${blob} prov=${prov}) -> $jev"
  fi
  if [ "$ok" = "1" ]; then
    echo "##### $S round $r: handle VERIFIED (ho16 $ho_ref -> $ho_new, hair $hair_ref -> $hair_new)"; ho_ref=$ho_new; hair_ref=$hair_new
    cur=/tmp/liou_cow_viz/cow_${S}_${S}_h${r}b.npz
  else
    echo "##### $S round $r: handle REJECTED (ho16 $ho_ref -> $ho_new, hair $hair_ref -> $hair_new): reverting; plug marked rejected (hull will not re-propose it; rays may)"
    python3 -c "
import json; p='$HANDLES_JSON'; h=json.load(open(p)); h[-1]['rejected']=True; h[-1]['mid']=None; json.dump(h, open(p,'w'))"
    cur=$prev_cur
  fi
done
echo "##### $S: last round mesh: $cur"
[ "${SKIP_FINAL:-0}" = "1" ] && { cp $cur /tmp/liou_cow_viz/cow_${S}_${S}_hlast.npz; echo "##### SKIP_FINAL: saved cow_${S}_${S}_hlast.npz"; exit 0; }
echo "##### $S: final Stage-4 (1200) -> Stage 5 -> Taubin from $cur"
env MODE=64v SHAPE=$S TAG=${S}_hfin STEPS=1200 $COMMON BASE_NPZ=$cur python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|Traceback\|Error"
env MODE=64v SHAPE=$S TAG=${S}_p5 SUBDIV_ALL=1 LAP_MULT=3 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.4 COLLAPSE_FRAC=${COLLAPSE_FRAC:-0.02} SI_PUSH=0.15 BASE_NPZ=/tmp/liou_cow_viz/cow_${S}_${S}_hfin.npz python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|subdivision\|Traceback\|Error"
MODE=64v SHAPE=$S ITERS=5 TAG=${S}_taubin5 BASE_NPZ=/tmp/liou_cow_viz/cow_${S}_${S}_p5.npz python3 despike/phase5_taubin.py 2>&1 | grep "taubin\|Traceback"
python3 - <<PY
import numpy as np
for t in ["${S}_hfin", "${S}_p5", "${S}_taubin5"]:
    z = np.load(f"/tmp/liou_cow_viz/cow_${S}_{t}.npz"); V, F = z["verts"], z["tris"].astype(np.int64)
    E = len(np.unique(np.sort(np.concatenate([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]), axis=1), axis=0)); print(t, "genus", (2 - (len(V) - E + len(F))) // 2)
PY
echo "##### $S multi-handle done"
