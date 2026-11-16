"""CLAP text->audio retrieval over MusicCaps (DeepMIR HW3).

Run:  CUDA_VISIBLE_DEVICES=1 .venv-clap/bin/python retrieve.py

laion_clap (enable_fusion=False) always feeds the HTSAT encoder exactly 10 s @ 48 kHz:
clips < 10 s are repeat-padded; clips > 10 s are *randomly cropped* to 10 s
(data_truncating='rand_trunc'). To stay deterministic we crop/pad ourselves
(first 10 s, repeat-pad) before calling get_audio_embedding_from_data, so the
library never random-crops. Audio is int16-quantised by the library as usual.
"""
import json, os, sys, time, csv
import numpy as np, torch, librosa
import laion_clap

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(ROOT, "data/audio_mp3")
OUT = os.path.join(ROOT, "outputs/retrieval")
CKPT = os.path.join(ROOT, "checkpoints/music_audioset_epoch_15_esc_90.14.pt")
SR, N = 48000, 480000
BS = 32
os.makedirs(OUT, exist_ok=True)


def fix_len(y):
    if len(y) >= N:
        return y[:N]
    reps = N // len(y)
    y = np.tile(y, reps)
    return np.pad(y, (0, N - len(y)))


def load_model():
    # torch.load default weights_only changed in torch>=2.6; env pins torch 2.5.1 so default is fine.
    m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="cuda")
    m.load_ckpt(CKPT)
    m.eval()
    return m


def audio_embeddings(model):
    ids_p, emb_p = os.path.join(OUT, "audio_ids.json"), os.path.join(OUT, "audio_emb.npy")
    if os.path.exists(ids_p) and os.path.exists(emb_p):
        print("using cached audio embeddings")
        return json.load(open(ids_p)), np.load(emb_p)
    files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3"))
    ids, embs, bad, buf, bufids = [], [], [], [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            e = model.get_audio_embedding_from_data(x=np.stack(buf).astype(np.float32), use_tensor=False)
        embs.append(e); ids.extend(bufids); buf.clear(); bufids.clear()

    t0 = time.time()
    for i, f in enumerate(files):
        try:
            y, _ = librosa.load(os.path.join(AUDIO_DIR, f), sr=SR, mono=True)
            if len(y) == 0:
                raise ValueError("empty")
            buf.append(fix_len(y)); bufids.append(f[:-4])
        except Exception as ex:
            print(f"SKIP unreadable {f}: {ex}"); bad.append(f)
        if len(buf) == BS:
            flush()
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    flush()
    emb = np.concatenate(embs).astype(np.float32)
    np.save(emb_p, emb); json.dump(ids, open(ids_p, "w"))
    json.dump(bad, open(os.path.join(OUT, "unreadable.json"), "w"))
    print(f"audio emb {emb.shape}, skipped {len(bad)}")
    return ids, emb


def main():
    model = load_model()
    ids, A = audio_embeddings(model)
    rows = list(csv.DictReader(open(os.path.join(ROOT, "test.csv"))))
    caps = [r["caption"] for r in rows]; tids = [r["id"] for r in rows]
    with torch.no_grad():
        T = model.get_text_embedding(caps, use_tensor=False)
        T2 = model.get_text_embedding(caps, use_tensor=False)
    print("text determinism max abs diff:", float(np.abs(T - T2).max()))
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    S = T @ A.T
    top = np.argsort(-S, axis=1)[:, :5]
    ret = {tid: ids[top[i, 0]] for i, tid in enumerate(tids)}
    json.dump(ret, open(os.path.join(OUT, "retrieval.json"), "w"), indent=1)
    with open(os.path.join(OUT, "retrieval_top5.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "caption", "rank", "musiccaps_id", "similarity"])
        for i, tid in enumerate(tids):
            for r, j in enumerate(top[i], 1):
                w.writerow([tid, caps[i], r, ids[j], f"{S[i, j]:.4f}"])
    # sanity
    s1 = np.array([S[i, top[i, 0]] for i in range(len(tids))])
    print("keys", len(ret), "all exist:", all(os.path.exists(f"{AUDIO_DIR}/{v}.mp3") for v in ret.values()))
    print(f"top1 sim mean {s1.mean():.4f} min {s1.min():.4f} max {s1.max():.4f}; distinct {len(set(ret.values()))}")
    for i in (0, 1, 2):
        print(f"[{tids[i]}] {caps[i][:80]}... -> {ret[tids[i]]} ({s1[i]:.4f})")
    d = os.path.join(OUT, "retrieved_audio"); os.makedirs(d, exist_ok=True)
    for tid, v in ret.items():
        dst = os.path.join(d, f"{tid}.mp3")
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.abspath(f"{AUDIO_DIR}/{v}.mp3"), dst)


if __name__ == "__main__":
    main()
