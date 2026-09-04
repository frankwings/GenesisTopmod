#!/bin/bash
# Multi-hole driver: repeat {detect tunnel evidence -> 1 add_handle -> short Stage-4 loop} until no evidence,
# then Stage 5 + Taubin. One handle per round avoids tube-adjacency conflicts (the opened hole leaves no membrane).
# Usage: SHAPE=threeholes ROUNDS=6 bash despike/phase7_multi.sh <base_npz_after_stage4>
set -u
cd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5
S=$SHAPE; R=${ROUNDS:-6}; cur=$1; LOOP=${LOOP_STEPS:-400}
export HANDLES_JSON=/tmp/liou_cow_viz/handles_${S}.json; rm -f $HANDLES_JSON /tmp/liou_cow_viz/cow_${S}_${S}_h[0-9]*.npz   # no stale round outputs
COMMON="FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_MAX=300 SI_PUSH=0.15"
for r in $(seq 1 $R); do
  [ "${SKIP_ROUNDS:-0}" = "1" ] && break
  echo "##### $S round $r: detect + add_handle on $cur"
  out=$(MODE=64v SHAPE=$S TAG=${S}_h$r MAX_HANDLES=1 BASE_NPZ=$cur python3 despike/phase7_handle.py 2>&1 | grep "\[p7\]\|\[after\|Traceback\|Error")
  echo "$out"
  if echo "$out" | grep -q "handles added: 0"; then echo "##### no more tunnel evidence after $((r-1)) handles"; break; fi
  if [ ! -f /tmp/liou_cow_viz/cow_${S}_${S}_h$r.npz ]; then echo "##### handle stage failed in round $r (see above); stopping with $((r-1)) handles"; break; fi
  cur=/tmp/liou_cow_viz/cow_${S}_${S}_h$r.npz
  echo "##### $S round $r: Stage-4 loop $LOOP steps"
  env MODE=64v SHAPE=$S TAG=${S}_h${r}b STEPS=$LOOP $COMMON BASE_NPZ=$cur python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|Traceback\|Error"
  cur=/tmp/liou_cow_viz/cow_${S}_${S}_h${r}b.npz
done
echo "##### $S: final Stage-4 (1200) -> Stage 5 -> Taubin from $cur"
env MODE=64v SHAPE=$S TAG=${S}_hfin STEPS=1200 $COMMON BASE_NPZ=$cur python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|Traceback\|Error"
env MODE=64v SHAPE=$S TAG=${S}_p5 SUBDIV_ALL=1 LAP_MULT=3 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.4 COLLAPSE_MAX=600 SI_PUSH=0.15 BASE_NPZ=/tmp/liou_cow_viz/cow_${S}_${S}_hfin.npz python3 despike/phase4_inloop.py 2>&1 | grep "\[final\]\|subdivision\|Traceback\|Error"
MODE=64v SHAPE=$S ITERS=5 TAG=${S}_taubin5 BASE_NPZ=/tmp/liou_cow_viz/cow_${S}_${S}_p5.npz python3 despike/phase5_taubin.py 2>&1 | grep "taubin\|Traceback"
python3 - <<PY
import numpy as np
for t in ["${S}_hfin", "${S}_p5", "${S}_taubin5"]:
    z = np.load(f"/tmp/liou_cow_viz/cow_${S}_{t}.npz"); V, F = z["verts"], z["tris"].astype(np.int64)
    E = len(np.unique(np.sort(np.concatenate([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]), axis=1), axis=0)); print(t, "genus", (2 - (len(V) - E + len(F))) // 2)
PY
echo "##### $S multi-handle done"
