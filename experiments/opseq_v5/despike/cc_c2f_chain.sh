#!/bin/bash
# Golden v3 (= chain v9, Palfinger optimizer params) on ALL shapes from the icosphere, SERIAL, GPU guard,
# idempotent (skips stages whose output exists), with per-step 64-view mosaic frames -> GIF per shape.
# Uniform pipeline for every shape (genus discovered, never given):
#   1 run_64v (cc2 800 -> cc3 800, stop)  2 phase4 400 (clean)  3 phase7_multi (tunnel evidence -> DLFL add_handle, <=8 rounds)
#   4 run_64v RESUME (cc4 800 + despike surgery + settle)  5 phase4 v9 1200  6 phase4 v9 LAP_MULT=3 1200  7 AUTO Taubin
set -u
# SERIAL: wait until the gif_v9_all task has finished (never share the GPU with our own chains)
L=/home/kingy/Foundation/EdenGateway/agents/hani/bg_tasks/logs/gif_v9_all.log
for n in $(seq 1 200); do grep -q "##### gif_v9_all done" $L 2>/dev/null && break; pgrep -f gif_v9_chain.sh >/dev/null || break; echo "[wait] gif_v9_all still running ($n/200, 5 min)"; sleep 300; done
export C2F_SUBDIV=cc   # Stage 1 / 4 refinement = TopMod Catmull-Clark instead of numpy midpoint
cd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5; ln -sfn $PWD/out_liou /tmp/liou_cow_viz; O=/tmp/liou_cow_viz
NS=/usr/lib/wsl/lib/nvidia-smi; export PATH=$PATH:/usr/lib/wsl/lib
guard() { for n in $(seq 1 180); do u=$($NS --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 24000 ] && { echo "[guard] GPU used ${u} MiB -> go"; return; }; echo "[guard] GPU used ${u} MiB > 24000, waiting 60s ($n/180)"; sleep 60; done; }
PALF="ADAM_BETAS=0.8,0.8 PALF_LAP=0.02 PALF_CLIP=10 LR_EDGE=0.3 ADAPT_REMESH=1 ADAPT_MODE=velocity ADAPT_NU_GAIN=0.2 ADAPT_LMIN_PX=1.3 ADAPT_MAX_F=100000 ADAPT_SI_GATE=0.3 FLIP_EVERY=25 COLLAPSE_EVERY=50 COLLAPSE_RATIO=0.4 SI_PUSH=0.15 STEPS=1200"
export SNAPSHOT_EVERY=10 SNAPSHOT_MODE=64
F4="\[step [0-9]*00/\|\[adapt\] step.*->\|\[final\]\|\[vram\]\|Traceback\|Error"
chain() { S=$1; T0=$(date +%s); FD=$PWD/out_liou/frames_cc_$S; export SNAPSHOT_DIR=$FD; mkdir -p $FD despike/results_genus/gifs
  echo "##### $S: golden v3 chain from icosphere, C2F_SUBDIV=cc (TopMod Catmull-Clark), frames -> $FD"
  # 1 sphere -> cc3
  if [ -f $O/cow_${S}_g3cccc3.npz ]; then echo "[$S 1] skip"; else guard; env SNAPSHOT_TITLE="Stage 1" MODE=64v SHAPE=$S TAG=${S}_g3cccc3 STOP_AFTER=cc3 python3 -u despike/run_64v.py 2>&1 | grep --line-buffered "STOP_AFTER\|Traceback\|Error" | sed "s/^/[$S 1] /"; fi
  # 2 short clean loop on the coarse mesh
  if [ -f $O/cow_${S}_${S}_g3cccc3p4.npz ]; then echo "[$S 2] skip"; else guard; env SNAPSHOT_TITLE="Stage 2 [DLFL clean loop, coarse]" MODE=64v SHAPE=$S TAG=${S}_g3cccc3p4 STEPS=400 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_FRAC=0.02 SI_PUSH=0.15 BASE_NPZ=$O/cow_${S}_g3cccc3.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "\[final\]\|Traceback\|Error" | sed "s/^/[$S 2] /"; fi
  # 3 genus discovery: tunnel evidence -> DLFL add_handle, one per round, short loop between
  if [ -f $O/cow_${S}_g3cchlast.npz ]; then echo "[$S 3] skip"; else guard; rm -f $O/cow_${S}_${S}_hlast.npz
    env SNAPSHOT_TITLE="Stage 3 [loop after add_handle]" SHAPE=$S ROUNDS=8 COLLAPSE_FRAC=0.02 SKIP_FINAL=1 bash despike/phase7_multi.sh $O/cow_${S}_${S}_g3cccc3p4.npz 2>&1 | grep --line-buffered "after handle\|no more\|handles added\|failed\|Traceback\|Error" | sed "s/^/[$S 3] /"
    cp $O/cow_${S}_${S}_hlast.npz $O/cow_${S}_g3cchlast.npz; cp $O/handles_${S}.json despike/results_genus/handles_${S}_g3ccchain.json 2>/dev/null; fi
  # 4 continue coarse-to-fine (cc4) + despike surgery
  if [ -f $O/cow_${S}_g3ccearly.npz ]; then echo "[$S 4] skip"; else guard; env SNAPSHOT_TITLE="Stage 4" MODE=64v SHAPE=$S TAG=${S}_g3ccearly RESUME_FROM=$O/cow_${S}_g3cchlast.npz python3 -u despike/run_64v.py 2>&1 | grep --line-buffered "heldout\|\[train\]\|Traceback\|Error" | sed "s/^/[$S 4] /"; fi
  # 5, 6 golden v3 loops
  if [ -f $O/cow_${S}_${S}_g3ccp4.npz ]; then echo "[$S 5] skip"; else guard; env SNAPSHOT_TITLE="Stage 5 [golden v3: DLFL in-loop remesh + Palfinger optimizer]" MODE=64v SHAPE=$S TAG=${S}_g3ccp4 $PALF BASE_NPZ=$O/cow_${S}_g3ccearly.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "$F4" | sed "s/^/[$S 5] /"; fi
  if [ -f $O/cow_${S}_${S}_g3ccp5.npz ]; then echo "[$S 6] skip"; else guard; env SNAPSHOT_TITLE="Stage 6 [golden v3 refine, LAP x3]" MODE=64v SHAPE=$S TAG=${S}_g3ccp5 $PALF LAP_MULT=3 BASE_NPZ=$O/cow_${S}_${S}_g3ccp4.npz python3 -u despike/phase4_inloop.py 2>&1 | grep --line-buffered "$F4" | sed "s/^/[$S 6] /"; fi
  # 7 Taubin
  if [ -f $O/cow_${S}_${S}_g3ccauto.npz ]; then echo "[$S 7] skip"; else guard; env SNAPSHOT_TITLE="Stage 7 Taubin" MODE=64v SHAPE=$S AUTO=1 TAG=${S}_g3ccauto BASE_NPZ=$O/cow_${S}_${S}_g3ccp5.npz python3 -u despike/phase5_taubin.py 2>&1 | grep --line-buffered "\[taubin x\|\[vram\]\|Traceback\|Error" | sed "s/^/[$S 7] /"; fi
  cp $O/cow_${S}_${S}_g3ccp5.npz despike/results_genus/${S}_g3ccchain_raw.npz; cp $O/cow_${S}_${S}_g3ccauto.npz despike/results_genus/${S}_g3ccchain_auto.npz
  python3 - <<PY
import numpy as np
for t in ["${S}_g3ccchain_raw", "${S}_g3ccchain_auto"]:
    z = np.load(f"despike/results_genus/{t}.npz"); V, F = z["verts"], z["tris"].astype(np.int64)
    E = len(np.unique(np.sort(np.concatenate([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]), axis=1), axis=0)); print(f"[$S] {t}: V={len(V)} F={len(F)} genus={(2 - (len(V) - E + len(F))) // 2}")
PY
  guard; SHAPE=$S python3 despike/eval_cd_iou.py raw=despike/results_genus/${S}_g3ccchain_raw.npz taubin=despike/results_genus/${S}_g3ccchain_auto.npz 2>&1 | grep -E "cd_iou|Error" | sed "s/^/[$S exam] /"
  # GIF + MP4 from the frame sequence
  NF=$(ls $FD/frame_*.png | wc -l); echo "[$S gif] $NF frames"
  G=despike/results_genus/gifs/${S}_golden_v3
  ffmpeg -y -loglevel error -framerate 15 -pattern_type glob -i "$FD/frame_*.png" -vf "scale=1024:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" ${G}.gif
  ffmpeg -y -loglevel error -framerate 15 -pattern_type glob -i "$FD/frame_*.png" -vf "select='not(mod(n\,2))',scale=640:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" -vsync vfr ${G}_small.gif
  ffmpeg -y -loglevel error -framerate 15 -pattern_type glob -i "$FD/frame_*.png" -c:v libx264 -pix_fmt yuv420p -crf 22 ${G}.mp4
  ls -la ${G}.gif ${G}_small.gif ${G}.mp4 | sed "s/^/[$S gif] /"
  echo "[$S] wall $(( $(date +%s) - T0 ))s"; }
for S in ${SHAPES:-armadillo}; do chain $S; done
echo "##### cc_c2f done"
