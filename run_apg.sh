#!/bin/bash
# Sequential APG runs on GPU 0. APG args: r=10 (median ||delta||), beta=-0.5.
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
R=10; B=-0.5
$PY generate_all.py --out_dir outputs/_repro_s7 --scale 7 --start 0 --end 1 && echo "SENTINEL repro done"
$PY generate_all.py --out_dir outputs/apg_s7 --scale 7 --method apg --eta 0 --norm_threshold $R --momentum $B && echo "SENTINEL apg_s7 done"
$PY generate_all.py --out_dir outputs/apg_s10 --scale 10 --method apg --eta 0 --norm_threshold $R --momentum $B && echo "SENTINEL apg_s10 done"
$PY generate_all.py --out_dir outputs/cfg_s15 --scale 15 && echo "SENTINEL cfg_s15 done"
$PY generate_all.py --out_dir outputs/apg_s15 --scale 15 --method apg --eta 0 --norm_threshold $R --momentum $B && echo "SENTINEL apg_s15 done"
$PY generate_all.py --out_dir outputs/apg_s10_eta1 --scale 10 --method apg --eta 1 --norm_threshold $R --momentum $B && echo "SENTINEL apg_s10_eta1 done"
echo "SENTINEL all done"
