"""Per-dir check: 100 files, 10 s, finite, RMS min/med/max, fraction with max|x|>=0.99 (clipping proxy).
    python verify_outputs.py outputs/apg_s7 ...    (--cmp A B compares decoded ids 0,1)"""
import sys, glob, os
import numpy as np, soundfile as sf

if sys.argv[1] == "--cmp":
    for i in (0, 1):
        a, _ = sf.read(f"{sys.argv[2]}/{i}.mp3"); b, _ = sf.read(f"{sys.argv[3]}/{i}.mp3")
        print(f"id {i}: shapes {a.shape} {b.shape} max|diff| = {np.abs(a-b).max():.3e}")
    sys.exit()
print("dir n dur(min,max) finite rms(min/med/max) frac_clip(|x|>=0.99 any) frac_samples_clipped(mean)")
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{d}/*.mp3")); rms = []; dur = []; fin = True; clip = []; fr = []
    for f in fs:
        x, sr = sf.read(f); dur.append(len(x) / sr); fin &= bool(np.isfinite(x).all())
        rms.append(np.sqrt((x**2).mean())); clip.append(np.abs(x).max() >= 0.99); fr.append((np.abs(x) >= 0.99).mean())
    print(f"{os.path.basename(d)} n={len(fs)} dur=({min(dur):.2f},{max(dur):.2f}) finite={fin} "
          f"rms={min(rms):.3f}/{np.median(rms):.3f}/{max(rms):.3f} clipped_clips={np.mean(clip):.2f} clipped_samples={np.mean(fr)*100:.4f}%")
