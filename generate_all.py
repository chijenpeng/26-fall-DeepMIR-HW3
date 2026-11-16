"""Generate one clip per caption in test.csv and save as <out_dir>/<id>.mp3.

    python generate_all.py --out_dir outputs/baseline_s7 --scale 7.0

Seed rule: clip `id` uses seed = --seed + id (same noise for the same id across guidance scales).
Existing mp3 files are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import csv
import json
import os
import time

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from generate import generate
from generate_apg import generate_apg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0, help="clip seed = seed + id")
    parser.add_argument("--start", type=int, default=0, help="first id (inclusive)")
    parser.add_argument("--end", type=int, default=99, help="last id (inclusive)")
    parser.add_argument("--method", choices=["cfg", "apg"], default="cfg")
    parser.add_argument("--eta", type=float, default=0.0, help="APG: weight of the parallel component")
    parser.add_argument("--norm_threshold", type=float, default=0.0, help="APG: rescale radius r in denoised space (0 = off)")
    parser.add_argument("--momentum", type=float, default=0.0, help="APG: momentum beta (negative; 0 = off)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [(int(r["id"]), r["caption"]) for r in csv.DictReader(f)]
    rows = [(i, c) for i, c in rows if args.start <= i <= args.end]

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    t0 = time.time()
    done = 0
    for clip_id, caption in rows:
        path = os.path.join(args.out_dir, f"{clip_id}.mp3")
        if os.path.exists(path):
            continue
        if args.method == "cfg":  # unchanged baseline path
            audio = generate(pipe, caption, args.scale, args.steps, args.seconds, args.seed + clip_id)
        else:
            audio = generate_apg(pipe, caption, args.scale, args.steps, args.seconds, args.seed + clip_id,
                                 args.eta, args.norm_threshold, args.momentum)
        sf.write(path + ".tmp", audio.T.float().clamp(-1, 1).cpu().numpy(), sr, format="MP3")
        os.replace(path + ".tmp", path)  # only complete files carry the final name
        done += 1
        print(f"[{clip_id}] {(time.time() - t0) / done:.1f}s/clip avg  {caption}", flush=True)


if __name__ == "__main__":
    main()
