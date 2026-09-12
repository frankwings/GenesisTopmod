#!/bin/bash
# Stage-3 only test of the space-carving genus target (LESSONS 24): run phase7_multi (detect -> add_handle -> 400-step loop,
# <=8 rounds) on every archived coarse base (different Stage-1/2 runs = different "seeds") and compare final genus with GT.
set -u
cd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5; ln -sfn $PWD/out_liou /tmp/liou_cow_viz; O=/tmp/liou_cow_viz
NS=/usr/lib/wsl/lib/nvidia-smi; export PATH=$PATH:/usr/lib/wsl/lib
guard() { for n in $(seq 1 180); do u=$($NS --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 28000 ] && { echo "[guard] GPU used ${u} MiB -> go"; return; }; echo "[guard] GPU used ${u} MiB > 28000, waiting 60s ($n/180)"; sleep 60; done; }
declare -A GT=([armadillo]=0 [kitten]=1 [rockerarm]=1 [fertility]=4)
run() { S=$1; B=$2; T0=$(date +%s); base=$O/cow_${S}_${S}_${B}.npz; [ -f $base ] || { echo "[$S $B] base missing"; return; }
  if [ -f $O/cow_${S}_gtest_${B}.npz ]; then echo "[$S $B] done before, skip"; else guard; rm -f $O/cow_${S}_${S}_hlast.npz
    env SHAPE=$S ROUNDS=8 COLLAPSE_FRAC=0.02 SKIP_FINAL=1 GENUS_TARGET=hull bash despike/phase7_multi.sh $base 2>&1 | grep --line-buffered "genus target\|after handle\|relaxation\|no more\|UNREACHED\|handles added\|failed\|Traceback\|Error" | sed "s/^/[$S $B] /"
    cp $O/cow_${S}_${S}_hlast.npz $O/cow_${S}_gtest_${B}.npz 2>/dev/null; fi
  python3 - <<PY
import numpy as np
z=np.load("$O/cow_${S}_gtest_${B}.npz"); V,F=z["verts"],z["tris"].astype(np.int64)
E=len(np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),1),axis=0)); g=(2-(len(V)-E+len(F)))//2
print(f"[RESULT] $S base=$B final genus={g} GT=${GT[$S]} {'OK' if g==${GT[$S]} else 'MISMATCH'} V={len(V)}")
PY
  echo "[$S $B] wall $(( $(date +%s) - T0 ))s"; }
for S in fertility kitten rockerarm armadillo; do for B in g3cccc3p4 g3cc3p4 cc3p4; do run $S $B; done; done
echo "##### genus_target_test done"
