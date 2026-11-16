"""One-command evaluation for DeepMIR HW3: FD-VGG, KL-PaSST, LAION-CLAP score.

Built on hkchengrex/av-benchmark (patched, see patches/av-benchmark-hw3.patch). Run with .venv-eval:

  CUDA_VISIBLE_DEVICES=1 .venv-eval/bin/python evaluate.py \
      --gen_dir outputs/baseline_s7 \
      --ref_dir outputs/retrieval/retrieved_audio \
      --csv test.csv --out outputs/baseline_s7/results.json

Semantics (mirrors av_bench/evaluate.py, but pairing is done here by file stem):
  FD-VGG     set level: Frechet distance between Gaussians fitted to all VGGish frame embeddings
             (128-d, one per 0.96 s) of gen_dir vs ref_dir.  Needs no pairing.
  KL-PaSST   paired: for every stem present in both dirs, KL(softmax(ref logits) || softmax(gen
             logits)) over the 527 AudioSet classes (PaSST, 32 kHz, 10 s), averaged over pairs.
  CLAP       paired with the CSV: mean cosine(LAION-CLAP audio emb of gen/<id>.mp3,
             text emb of caption <id>), music checkpoint music_audioset_epoch_15_esc_90.14.pt.
             Does not need ref_dir.
If ref_dir does not contain every generated stem (e.g. --ref_dir data/audio_mp3, full MusicCaps),
KL is skipped (null) and FD-VGG is computed against the whole ref set.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('AVB_CLAP_CKPT', str(ROOT / 'checkpoints' / 'music_audioset_epoch_15_esc_90.14.pt'))
os.environ.setdefault('AVB_VGGISH_CKPT', str(ROOT / 'checkpoints' / 'vggish-10086976.pth'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'checkpoints' / 'torch'))  # PaSST weights land here
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))  # roberta tokenizer for CLAP text tower
sys.path.insert(0, str(ROOT / 'third_party' / 'av-benchmark'))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

AUDIO_EXT = ('.wav', '.flac', '.mp3')


def list_audio(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT)


def dir_signature(files, audio_length):
    h = hashlib.sha1(f'len={audio_length}'.encode())
    for p in files:
        st = p.stat()
        h.update(f'{p.name}:{st.st_size}:{int(st.st_mtime)}'.encode())
    return h.hexdigest()


def get_features(audio_dir: Path, cache_root: Path, audio_length, want_clap, recompute, device, workers):
    """Run av_bench.extract on audio_dir unless a matching cache exists. Returns the cache dir."""
    from av_bench.extract import extract
    files = list_audio(audio_dir)
    if not files:
        raise SystemExit(f'no audio in {audio_dir}')
    sig = dir_signature(files, audio_length)
    cdir = cache_root / (audio_dir.resolve().name + '_' + hashlib.sha1(str(audio_dir.resolve()).encode()).hexdigest()[:8])
    sig_file = cdir / 'signature.json'
    needed = ['vggish_features.pth', 'passt_features_embed.pth', 'passt_logits.pth'] + (
        ['clap_laion_audio.pth'] if want_clap else [])
    ok = (not recompute and sig_file.exists() and json.loads(sig_file.read_text()).get('sig') == sig
          and all((cdir / n).exists() for n in needed))
    if ok:
        print(f'[cache hit] {audio_dir} -> {cdir}', flush=True)
    else:
        print(f'[extract] {len(files)} files from {audio_dir} -> {cdir}', flush=True)
        t = time.time()
        # batch_size=1: supports variable-length clips (upstream: "only bs=1 supports variable length")
        extract(audio_path=audio_dir, output_path=cdir, audio_length=audio_length, device=device,
                batch_size=1, num_workers=workers, skip_video_related=True, skip_clap=not want_clap)
        sig_file.write_text(json.dumps({'sig': sig, 'n': len(files), 'clap': want_clap}))
        print(f'[extract] done in {time.time() - t:.0f}s', flush=True)
    return cdir


def text_embeddings(csv_path: Path, cache_root: Path, device):
    df = pd.read_csv(csv_path)
    idcol = 'id' if 'id' in df.columns else 'name'
    sig = hashlib.sha1(csv_path.read_bytes()).hexdigest()[:12]
    f = cache_root / f'clap_text_{sig}.pth'
    if f.exists():
        return torch.load(f, weights_only=True)
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(device).eval()
    m.load_ckpt(os.environ['AVB_CLAP_CKPT'], verbose=False)
    out = {}
    with torch.inference_mode():
        for _, r in df.iterrows():  # one by one, like upstream extract_text.py
            out[str(r[idcol])] = m.get_text_embedding([r['caption']], use_tensor=True).cpu().squeeze()
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.save(out, f)
    del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', type=Path, required=True)
    ap.add_argument('--ref_dir', type=Path, required=True)
    ap.add_argument('--csv', type=Path, default=ROOT / 'test.csv')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache_dir', type=Path, default=ROOT / 'outputs' / '_eval_cache')
    ap.add_argument('--audio_length', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip_clap', action='store_true')
    ap.add_argument('--recompute', action='store_true', help='ignore feature caches')
    args = ap.parse_args()

    from av_bench.metrics import compute_fd, compute_kl
    gen_cache = get_features(args.gen_dir, args.cache_dir, args.audio_length, not args.skip_clap,
                             args.recompute, args.device, args.workers)
    ref_cache = get_features(args.ref_dir, args.cache_dir, args.audio_length, False,
                             args.recompute, args.device, args.workers)
    L = lambda c, n: torch.load(c / n, weights_only=True)  # noqa: E731

    res = {'gen_dir': str(args.gen_dir), 'ref_dir': str(args.ref_dir), 'audio_length': args.audio_length,
           'clap_ckpt': Path(os.environ['AVB_CLAP_CKPT']).name}
    gv, rv = L(gen_cache, 'vggish_features.pth'), L(ref_cache, 'vggish_features.pth')
    res['n_gen'], res['n_ref'] = len(gv), len(rv)
    cat = lambda d: torch.cat(list(d.values()), 0).numpy()  # noqa: E731  (all frames of all clips)
    # compute_fd(background=generated, eval=reference), as in upstream evaluate.py
    res['FD-VGG'] = float(compute_fd(cat(gv), cat(rv)))
    gp, rp = L(gen_cache, 'passt_features_embed.pth'), L(ref_cache, 'passt_features_embed.pth')
    res['FD-PaSST'] = float(compute_fd(torch.stack(list(gp.values())).numpy(), torch.stack(list(rp.values())).numpy()))

    gl, rl = L(gen_cache, 'passt_logits.pth'), L(ref_cache, 'passt_logits.pth')
    common = sorted(set(gl) & set(rl), key=lambda s: (len(s), s))
    res['paired'] = len(common) == len(gl)  # every generated stem has a reference
    res['n_pairs'] = len(common)
    if res['paired']:
        kl = compute_kl([torch.stack([gl[k] for k in common])], torch.stack([rl[k] for k in common]))
        res['KL-PaSST'] = kl['kl_softmax']
        res['KL-PaSST-sigmoid'] = kl['kl_sigmoid']
    else:
        res['KL-PaSST'] = None
        print(f'[warn] only {len(common)}/{len(gl)} generated stems found in ref_dir -> KL skipped', flush=True)

    if not args.skip_clap:
        ga = L(gen_cache, 'clap_laion_audio.pth')
        txt = text_embeddings(args.csv, args.cache_dir, args.device)
        ids = [k for k in ga if k in txt]
        if len(ids) != len(ga):
            print(f'[warn] {len(ga) - len(ids)} generated files have no caption in {args.csv}', flush=True)
        cs = torch.cosine_similarity(torch.stack([txt[k] for k in ids]), torch.stack([ga[k] for k in ids]), dim=-1)
        res['CLAP'] = float(cs.mean())
        res['n_clap'] = len(ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
