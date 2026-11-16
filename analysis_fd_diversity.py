"""FD-VGG decomposition + diversity analysis (DeepMIR HW3).
Stage 1 (compute): run with .venv-eval from the project root:  python analysis_fd_diversity.py compute
Stage 2 (plot):    run with .venv-eval (needs matplotlib, pandas): python analysis_fd_diversity.py plot
FD unit = per-frame VGGish embeddings (128-d, 0.96 s), all frames of all clips pooled (av-benchmark compute_fd).
"""
import sys, json, os, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
A = ROOT / 'results'
FIGS = ROOT / 'outputs' / 'figs'; FIGS.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'outputs' / '_eval_cache'
RUNS = ['cfg_s1', 'cfg_s3', 'cfg_s5', 'baseline_s7', 'cfg_s10', 'cfg_s15', 'apg_s7', 'apg_s10', 'apg_s15', 'apg_s10_eta1']
W = {'cfg_s1': 1, 'cfg_s3': 3, 'cfg_s5': 5, 'baseline_s7': 7, 'cfg_s10': 10, 'cfg_s15': 15}
REPORTED = {  # (retrieved, full) as given in the task
    'cfg_s1': (2.988, 6.359), 'cfg_s3': (2.052, 7.302), 'cfg_s5': (1.952, 7.674), 'baseline_s7': (1.979, 7.719),
    'cfg_s10': (1.930, 7.768), 'cfg_s15': (1.965, 7.559), 'apg_s7': (1.954, 7.477), 'apg_s10': (1.958, 7.653),
    'apg_s15': (2.025, 7.593), 'apg_s10_eta1': (1.995, 7.506)}


def cache_dir(name):
    c = [p for p in CACHE.iterdir() if p.name.rsplit('_', 1)[0] == name]
    assert len(c) == 1, (name, c)
    return c[0]


def compute():
    import numpy as np, torch
    sys.path.insert(0, str(ROOT / 'third_party' / 'av-benchmark'))
    from av_bench.metrics.fad import calculate_embd_statistics, calculate_frechet_distance
    from scipy import linalg

    def load(name):
        return torch.load(cache_dir(name) / 'vggish_features.pth', weights_only=True)

    def stats(d):
        frames = torch.cat(list(d.values()), 0).numpy().astype(np.float64)
        mu, sig = calculate_embd_statistics(frames)
        return mu, sig

    def clipvecs(d, keys=None):
        keys = keys if keys is not None else list(d)
        return torch.stack([d[k].float().mean(0) for k in keys]).numpy().astype(np.float64)

    def pairwise(X):
        n = len(X)
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = 1 - Xn @ Xn.T
        iu = np.triu_indices(n, 1)
        return float(D[iu].mean()), float(C[iu].mean())

    def decomp(m1, s1, m2, s2):
        mean = float(((m1 - m2) ** 2).sum())
        sq = linalg.sqrtm(s1 @ s2)
        sq = sq.real
        cov = float(np.trace(s1) + np.trace(s2) - 2 * np.trace(sq))
        return mean, cov

    refs = {}
    rd = load('retrieved_audio')
    full = load('audio_mp3')
    # distinct retrieved clips via symlink targets
    rdir = ROOT / 'outputs' / 'retrieval' / 'retrieved_audio'
    tgt = {k: os.path.basename(os.readlink(rdir / f'{k}.mp3')) if (rdir / f'{k}.mp3').is_symlink() else k for k in rd}
    distinct = {}
    for k, t in tgt.items():
        distinct.setdefault(t, k)
    rd59 = {t: rd[k] for t, k in distinct.items()}
    print('retrieved entries', len(rd), 'distinct', len(rd59))
    refs['retrieved'] = rd
    refs['full'] = full
    rstat = {n: stats(d) for n, d in [('retrieved', rd), ('full', full), ('retrieved_distinct59', rd59)]}

    rows = []
    for r in RUNS:
        g = load(r)
        mg, sg = stats(g)
        row = {'name': r, 'kind': 'run', 'cfg_w': W.get(r), 'n_clips': len(g),
               'n_frames': int(sum(len(v) for v in g.values())), 'trace_sigma': float(np.trace(sg))}
        row['clip_euclid'], row['clip_cosine'] = pairwise(clipvecs(g))
        for tag, key in [('ret', 'retrieved'), ('full', 'full')]:
            mr, sr = rstat[key]
            fd_av = float(calculate_frechet_distance(mg, sg, mr, sr))
            mean, cov = decomp(mg, sg, mr, sr)
            row[f'fd_{tag}'] = fd_av
            row[f'fd_{tag}_mean'] = mean
            row[f'fd_{tag}_cov'] = cov
            row[f'fd_{tag}_ours'] = mean + cov
            row[f'fd_{tag}_reported'] = REPORTED[r][0 if tag == 'ret' else 1]
        rows.append(row)
    for name, d in [('ref_retrieved_100', rd), ('ref_retrieved_distinct59', rd59), ('ref_fullmusiccaps', full)]:
        key = {'ref_retrieved_100': 'retrieved', 'ref_retrieved_distinct59': 'retrieved_distinct59', 'ref_fullmusiccaps': 'full'}[name]
        row = {'name': name, 'kind': 'reference', 'cfg_w': None, 'n_clips': len(d),
               'n_frames': int(sum(len(v) for v in d.values())), 'trace_sigma': float(np.trace(rstat[key][1]))}
        row['clip_euclid'], row['clip_cosine'] = pairwise(clipvecs(d))  # full: ALL 5367 clips, no subsampling
        rows.append(row)
    # extra: FD between the two references
    m, c = decomp(*rstat['retrieved'], *rstat['full'])
    extra = {'FD_retrieved_vs_full': m + c, 'mean_term': m, 'cov_term': c}
    keys = sorted({k for r in rows for k in r})
    with open(A / 'fd_diversity.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    (A / 'fd_diversity.json').write_text(json.dumps({'rows': rows, 'extra': extra}, indent=1))
    print('sanity (ours vs reported / av-benchmark fn):')
    for r in rows:
        if r['kind'] == 'run':
            print(f"{r['name']:13s} ret ours {r['fd_ret_ours']:.4f} av {r['fd_ret']:.4f} rep {r['fd_ret_reported']:.3f} | "
                  f"full ours {r['fd_full_ours']:.4f} av {r['fd_full']:.4f} rep {r['fd_full_reported']:.3f}")


def plot():
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 12, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    d = json.loads((A / 'fd_diversity.json').read_text())['rows']
    R = {r['name']: r for r in d}
    cfg = [r for r in RUNS if r in W]
    ws = [W[r] for r in cfg]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.33, 5.2), gridspec_kw={'width_ratios': [1, 1.15]})
    # left: Tr(Sigma)
    a1.plot(ws, [R[r]['trace_sigma'] for r in cfg], 'o-', color='#1f4e9c', lw=2.5, ms=8, label='CFG generated')
    mk = {'apg_s7': ('s', 7), 'apg_s10': ('s', 10), 'apg_s15': ('s', 15), 'apg_s10_eta1': ('D', 10)}
    lab = {'apg_s7': 'APG', 'apg_s10_eta1': r'APG ($\eta$=1)'}
    seen = set()
    for r, (m, x) in mk.items():
        l = lab.get(r, '_')
        a1.scatter([x], [R[r]['trace_sigma']], marker=m, s=90, color='#d95f02', zorder=5, label=l if l not in seen else None)
        seen.add(l)
    a1.axhline(R['ref_fullmusiccaps']['trace_sigma'], color='#555', ls='--', lw=2, label='Full MusicCaps (5,367)')
    a1.axhline(R['ref_retrieved_100']['trace_sigma'], color='#2a9d57', ls=':', lw=2.5, label='Retrieved ref (100)')
    a1.set_xlabel('Guidance scale $w$'); a1.set_ylabel(r'Frame-level diversity  Tr($\Sigma$)')
    a1.set_xticks(ws); a1.set_ylim(7, 20); a1.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, 0.55))
    # right: FD mean vs cov
    x = np.arange(len(cfg)); bw = 0.38
    for off, tag, name, c_mean, c_cov in [(-bw / 2 - 0.02, 'ret', 'Retrieved', '#74c69d', '#2a9d57'), (bw / 2 + 0.02, 'full', 'Full MusicCaps', '#9bb7e0', '#1f4e9c')]:
        mean = np.array([R[r][f'fd_{tag}_mean'] for r in cfg]); cov = np.array([R[r][f'fd_{tag}_cov'] for r in cfg])
        a2.bar(x + off, mean, bw, color=c_mean, edgecolor='k', lw=0.6, label=f'{name}: mean term')
        a2.bar(x + off, cov, bw, bottom=mean, color=c_cov, edgecolor='k', lw=0.6, label=f'{name}: covariance term')
    a2.set_xticks(x); a2.set_xticklabels(ws); a2.set_xlabel('Guidance scale $w$ (CFG)'); a2.set_ylabel('FD-VGG (stacked terms)')
    a2.legend(frameon=False, loc='upper left', ncol=1, fontsize=11)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.25)
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(FIGS / f'why_diversity.{e}', dpi=200)


if __name__ == '__main__':
    {'compute': compute, 'plot': plot}[sys.argv[1]]()
