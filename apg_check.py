"""(1) eta=1/no-rescale/no-momentum round trip vs plain CFG; (2) ||delta|| statistics in denoised space.
Uses non-test prompts only."""
import sys, json
import numpy as np, torch
from diffusers import StableAudioPipeline
from generate import generate
from generate_apg import generate_apg

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
P = ["jazz, electric guitar, piano, relaxing", "energetic EDM with heavy bass", "solo classical violin, sad"]
tr = []
a = generate(pipe, P[0], 7.0, 100, 10.0, 0)
b = generate_apg(pipe, P[0], 7.0, 100, 10.0, 0, eta=1.0, norm_threshold=0.0, momentum=0.0, trace=tr)
print("ROUNDTRIP final audio max|cfg-apg(eta=1)| =", (a.float() - b.float()).abs().max().item(), " audio max|x| =", a.abs().max().item())
print("ROUNDTRIP per-step max |v_apg - v_cfg| =", max(t["v_diff_vs_cfg"] for t in tr))
res = {}
for w in (7.0, 10.0, 15.0):
    allt = []
    for k, p in enumerate(P):
        t = []
        generate_apg(pipe, p, w, 100, 10.0, 100 + k, eta=1.0, trace=t)  # CFG trajectory (eta=1)
        allt.append([x["delta_norm"] for x in t])
    arr = np.array(allt)  # (prompts, steps)
    res[w] = arr
    print(f"scale {w}: ||delta|| (denoised space; trajectory itself is guided at this scale)")
    print("step sigma mn med mx")
    sig = [x["sigma"] for x in t]
    for i in range(0, 100, 5):
        print(i, f"{sig[i]:.3f}", f"{arr[:,i].min():.2f} {np.median(arr[:,i]):.2f} {arr[:,i].max():.2f}")
    print("overall percentiles 10/25/50/75/90:", np.percentile(arr, [10, 25, 50, 75, 90]).round(2))
np.save("outputs/_apg_norms.npy", np.stack([res[7.0], res[10.0], res[15.0]]))
