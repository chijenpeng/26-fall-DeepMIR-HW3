#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=1 ./run_eval_all.sh run1 run2 ...   (dirs under outputs/)
cd "$(dirname "$0")"; mkdir -p outputs/eval
REF=outputs/retrieval/retrieved_audio
for r in "$@"; do
  .venv-eval/bin/python evaluate.py --gen_dir outputs/$r --ref_dir $REF --csv test.csv --out outputs/eval/$r.json --workers 4 2>&1 | grep -v "it/s\|B/s"
  .venv-eval/bin/python evaluate.py --gen_dir outputs/$r --ref_dir data/audio_mp3 --csv test.csv --out outputs/eval/${r}__fullmusiccaps.json --workers 4 --skip_clap 2>&1 | grep -v "it/s\|B/s"
done
