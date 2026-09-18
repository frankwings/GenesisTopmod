#!/bin/bash
# GOLDEN v3 chain, complete (2026-09-12): all topology changes = TopMod operators, genus DISCOVERED with the
# space-carving genus target g* (LESSONS 24) at the coarse stage AND re-checked after Stage 5 (contact join, LESSONS 25b).
#   1 run_64v cc2->CC->cc3 | 2 phase4 clean 400 | 3 phase7_multi (rays + g* stop/continue) | 4 run_64v CC->cc4 + despike
#   5 phase4 v3 1200 | 5b phase7 late pass (rays -> contact join) only if genus < g* | 6 phase4 v3 LAP x3 1200 | 7 AUTO Taubin | exam
# SERIAL, GPU guard, idempotent. env: SHAPES="a b c" SEED=0 TAGP=v4 SNAPSHOT_EVERY=0 (>0 -> 64-view frames + GIF)
set -u
cd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5; ln -sfn $PWD/out_liou /tmp/liou_cow_viz; O=/tmp/liou_cow_viz
NS=/usr/lib/wsl/lib/nvidia-smi; export PATH=$PATH:/usr/lib/wsl/lib
GLIM=${GUARD_MIB:-28000}
guard() { for n in $(seq 1 180); do u=$($NS --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le $GLIM ] && { echo "[guard] GPU used ${u} MiB -> go"; return; }; echo "[guard] GPU used ${u} MiB > $GLIM, waiting 60s ($n/180)"; sleep 60; done; }
PALF="ADAM_BETAS=0.8,0.8 PALF_LAP=0.02 PALF_CLIP=10 LR_EDGE=0.3 ADAPT_REMESH=1 ADAPT_MODE=velocity ADAPT_NU_GAIN=0.2 ADAPT_LMIN_PX=${ADAPT_LMIN_PX:-1.3} ADAPT_MAX_F=100000 ADAPT_SI_GATE=0.3 FLIP_EVERY=25 COLLAPSE_EVERY=50 COLLAPSE_RATIO=0.4 SI_PUSH=0.15 STEPS=1200"
export SEED=${SEED:-0}; P=${TAGP:-v4}; [ "$SEED" != "0" ] && P=${P}s${SEED}
export SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-0} SNAPSHOT_MODE=64
F4="\[adapt\] step.*->\|\[final\]\|\[vram\]\|Traceback\|Error"
declare -A GT=([armadillo]=0 [kitten]=1 [rockerarm]=1 [fertility]=4 [threeholes]=3)
_gt_of() { local g="${GT[$1]:-?}"; echo "$g"; }
genus_of() { python3 -c "
import numpy as np,sys; z=np.load(sys.argv[1]); V,F=z['verts'],z['tris'].astype(np.int64)
E=len(np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),1),axis=0)); print((2-(len(V)-E+len(F)))//2)" $1; }
chain() { S=$1; T0=$(date +%s); T=${S}_$P
  if [ "$SNAPSHOT_EVERY" != "0" ]; then FD=$PWD/out_liou/frames_${P}_$S; export SNAPSHOT_DIR=$FD; mkdir -p $FD; else unset SNAPSHOT_DIR; fi
  echo "##### $S: golden v3 chain ($P, seed $SEED), GT genus $(_gt_of $S)"
  if [ -f $O/cow_${T}_cc3.npz ]; then echo "[$S 1] skip"; else guard; env SNAPSHOT_TITLE="Stage 1" MODE=64v SHAPE=$S TAG=${T}_cc3 STOP_AFTER=cc3 python3 -u despike/run_64v.py 2>&1 | grep --line-buffered "STOP_AFTER\|Traceback\|Error" | sed "s/^/[$S 1] /"; fi
  if [ -f $O/cow_${S}_${T}_cc3p4.npz ]; then echo "[$S 2] skip"; else guard; env SNAPSHOT_TITLE="Stage 2 [DLFL clean loop, coarse]" MODE=64v SHAPE=$S TAG=${T}_cc3p4 STEPS=400 MEMB_EXEMPT=2 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_FRAC=0.02 SI_PUSH=0.15 BASE_NPZ=$O/cow_${T}_cc3.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "\[final\]\|Traceback\|Error" | sed "s/^/[$S 2] /"; fi
  if [ -f $O/cow_${T}_hlast.npz ]; then echo "[$S 3] skip"; else guard; rm -f $O/cow_${S}_${S}_hlast.npz
    env SNAPSHOT_TITLE="Stage 3 [loop after add_handle]" SHAPE=$S ROUNDS=8 COLLAPSE_FRAC=0.02 SKIP_FINAL=1 GENUS_TARGET=hull bash despike/phase7_multi.sh $O/cow_${S}_${T}_cc3p4.npz 2>&1 | grep --line-buffered "genus target\|after handle\|relaxation\|contact\|no more\|UNREACHED\|handles added\|failed\|VERIFIED\|REJECTED\|Traceback\|Error" | sed "s/^/[$S 3] /"
    cp $O/cow_${S}_${S}_hlast.npz $O/cow_${T}_hlast.npz; cp $O/handles_${S}.json despike/results_genus/handles_${T}.json 2>/dev/null; fi
  echo "[$S 3] genus after coarse discovery: $(genus_of $O/cow_${T}_hlast.npz) (GT $(_gt_of $S))"
  if [ -f $O/cow_${T}_early.npz ]; then echo "[$S 4] skip"; else guard; env SNAPSHOT_TITLE="Stage 4" MODE=64v SHAPE=$S TAG=${T}_early RESUME_FROM=$O/cow_${T}_hlast.npz python3 -u despike/run_64v.py 2>&1 | grep --line-buffered "heldout\|\[train\]\|Traceback\|Error" | sed "s/^/[$S 4] /"; fi
  if [ -f $O/cow_${S}_${T}_p4.npz ]; then echo "[$S 5] skip"; else guard; env SNAPSHOT_TITLE="Stage 5 [golden v3 loop]" MODE=64v SHAPE=$S TAG=${T}_p4 $PALF MEMB_EXEMPT=2 BASE_NPZ=$O/cow_${T}_early.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "$F4" | sed "s/^/[$S 5] /"; fi
  # 5b late genus pass: only if the fine mesh is still below g* (rays first, then contact join)
  if [ -f $O/cow_${T}_p4g.npz ]; then echo "[$S 5b] skip"; else guard
    # late genus pass through the same propose-and-verify loop as Stage 3 (one handle per round, DR-verified, reverted if ineffective)
    env SNAPSHOT_TITLE="Stage 5b [late genus pass]" SHAPE=$S ROUNDS=4 COLLAPSE_FRAC=0.02 SKIP_FINAL=1 GENUS_TARGET=hull bash despike/phase7_multi.sh $O/cow_${S}_${T}_p4.npz 2>&1 | grep --line-buffered "genus target\|after handle\|contact\|no more\|UNREACHED\|handles added\|broke\|VERIFIED\|REJECTED\|failed\|Traceback\|Error" | sed "s/^/[$S 5b] /"
    cp /tmp/liou_cow_viz/cow_${S}_${S}_hlast.npz $O/cow_${T}_p4g.npz; fi
  echo "[$S 5b] genus after late pass: $(genus_of $O/cow_${T}_p4g.npz) (GT $(_gt_of $S))"
  if [ -f $O/cow_${S}_${T}_p5.npz ]; then echo "[$S 6] skip"; else guard; env SNAPSHOT_TITLE="Stage 6 [golden v3 refine, LAP x3]" MODE=64v SHAPE=$S TAG=${T}_p5 $PALF LAP_MULT=3 BASE_NPZ=$O/cow_${T}_p4g.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "$F4" | sed "s/^/[$S 6] /"; fi
  if [ -f $O/cow_${S}_${T}_auto.npz ]; then echo "[$S 7] skip"; else guard; env SNAPSHOT_TITLE="Stage 7 Taubin" MODE=64v SHAPE=$S AUTO=1 TAG=${T}_auto BASE_NPZ=$O/cow_${S}_${T}_p5.npz python3 -u despike/phase5_taubin.py 2>&1 | grep --line-buffered "\[taubin x\|\[vram\]\|Traceback\|Error" | sed "s/^/[$S 7] /"; fi
  cp $O/cow_${S}_${T}_p5.npz despike/results_genus/${T}_raw.npz; cp $O/cow_${S}_${T}_auto.npz despike/results_genus/${T}_auto.npz
  G=$(genus_of despike/results_genus/${T}_auto.npz); echo "[RESULT] $S $P: final genus $G (GT $(_gt_of $S)) $([ "$G" = "$(_gt_of $S)" ] && echo OK || echo MISMATCH)"
  guard; if [ -n "${REAL_DATA:-}" ]; then
    REAL_DATA=$REAL_DATA SHAPE=$S python3 despike/exam_real.py raw=despike/results_genus/${T}_raw.npz taubin=despike/results_genus/${T}_auto.npz 2>&1 | grep -E "exam_real|Error" | sed "s/^/[$S exam] /"
  else
    SHAPE=$S python3 despike/eval_cd_iou.py raw=despike/results_genus/${T}_raw.npz taubin=despike/results_genus/${T}_auto.npz 2>&1 | grep -E "cd_iou|Error" | sed "s/^/[$S exam] /"
  fi
  if [ "$SNAPSHOT_EVERY" != "0" ]; then G2=despike/results_genus/gifs/${T}; mkdir -p despike/results_genus/gifs
    ffmpeg -y -loglevel error -framerate 15 -pattern_type glob -i "$FD/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 ${G2}.mp4
    ffmpeg -y -loglevel error -framerate 15 -pattern_type glob -i "$FD/frame_*.png" -vf "select='not(mod(n\,5))',scale=560:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=32[p];[b][p]paletteuse=dither=none" -vsync vfr ${G2}_discord.gif; fi
  echo "[$S] wall $(( $(date +%s) - T0 ))s"; }
for S in ${SHAPES:-fertility threeholes kitten rockerarm armadillo}; do chain $S; done
echo "##### golden_chain done (chain version: golden v6 structure = v3 stages + cpp kernel + batched render + hull-located handles; TAGP=$P)"
