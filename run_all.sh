#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")"
.venv/bin/python generate_all.py --out_dir outputs/baseline_s7 --scale 7.0
for s in 3 10 1 5; do .venv/bin/python generate_all.py --out_dir outputs/cfg_s$s --scale $s.0; done
