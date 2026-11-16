"""Per-clip CFG vs APG comparison (DeepMIR HW3).
Stage 1: .venv-eval/bin/python analysis_perclip.py compute   (stats, CSV, paired tests)
Stage 2: .venv-eval/bin/python analysis_perclip.py plot    (figures; needs matplotlib, pandas)
Run from project root. Clip = |x|>=0.99 over all channels of decoded mp3 (can exceed 1.0 after decode).
HF ratio = STFT power (n_fft 2048, hop 512, mono mean) above 5 kHz / total.
"""
import sys, csv, json
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parent
A = ROOT / 'results'
FIGS = ROOT / 'outputs' / 'figs'; FIGS.mkdir(parents=True, exist_ok=True)
RUNS = ['cfg_s10', 'cfg_s15', 'apg_s10', 'apg_s15']
NF, HOP = 2048, 512


def load(run, i):
    x, sr = sf.read(ROOT / 'outputs' / run / f'{i}.mp3', always_2d=True)
    return x.astype(np.float64), sr


def stft(m, fs, nperseg=NF, noverlap=NF - HOP):
    # numpy-only Hann STFT (scipy is not in the plotting venv); returns f, t, complex Z
    hop = nperseg - noverlap; w = np.hanning(nperseg)
    n = 1 + (len(m) - nperseg) // hop
    fr = np.stack([m[i * hop:i * hop + nperseg] * w for i in range(n)], 1)
    Z = np.fft.rfft(fr, axis=0) / w.sum()
    return np.fft.rfftfreq(nperseg, 1 / fs), (np.arange(n) * hop + nperseg / 2) / fs, Z


def stft_pow(x, sr):
    f, t, Z = stft(x.mean(1), sr)
    return f, t, np.abs(Z) ** 2


def metrics(x, sr):
    a = np.abs(x)
    peak = a.max(); rms = np.sqrt((x ** 2).mean())
    f, t, P = stft_pow(x, sr)
    hf = P[f >= 5000].sum() / P.sum()
    return dict(clip_frac=(a >= 0.99).mean(), peak=peak, rms_db=20 * np.log10(rms + 1e-12),
                crest_db=20 * np.log10(peak / (rms + 1e-12)), hf_ratio=hf)


def compute():
    from scipy.stats import wilcoxon
    caps = {int(r['id']): r['caption'] for r in csv.DictReader(open(ROOT / 'test.csv'))}
    rows = []; tot = {r: [0, 0] for r in RUNS}
    for i in range(100):
        row = {'id': i, 'caption': caps[i][:60]}
        for r in RUNS:
            x, sr = load(r, i)
            m = metrics(x, sr)
            tot[r][0] += (np.abs(x) >= 0.99).sum(); tot[r][1] += x.size
            for k, v in m.items(): row[f'{r}_{k}'] = v
        rows.append(row)
    with open(A / 'perclip_cfg_vs_apg.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = ['pooled clipped-sample %: ' + ', '.join(f'{r}={100*tot[r][0]/tot[r][1]:.3f}' for r in RUNS)]
    out.append(f"{'w':>3} {'metric':10} {'N(APG<CFG)':>10} {'N(APG>CFG)':>10} {'ties':>5} {'med(APG-CFG)':>13} {'wilcoxon_p':>11}")
    for w_ in (15, 10):
        for k in ['clip_frac', 'peak', 'rms_db', 'crest_db', 'hf_ratio']:
            c = np.array([r[f'cfg_s{w_}_{k}'] for r in rows]); a = np.array([r[f'apg_s{w_}_{k}'] for r in rows])
            d = a - c
            try: p = wilcoxon(a, c).pvalue
            except ValueError: p = float('nan')
            out.append(f'{w_:>3} {k:10} {(d<0).sum():>10} {(d>0).sum():>10} {(d==0).sum():>5} {np.median(d):>13.5g} {p:>11.3g}')
    for w_ in (15, 10):
        c = np.array([r[f'cfg_s{w_}_clip_frac'] for r in rows]); a = np.array([r[f'apg_s{w_}_clip_frac'] for r in rows])
        out.append(f'w={w_}: zero clipping in both: {((c==0)&(a==0)).sum()}; CFG>0: {(c>0).sum()}; APG>0: {(a>0).sum()}; '
                   f'only CFG clips: {((c>0)&(a==0)).sum()}; only APG clips: {((c==0)&(a>0)).sum()}')
    c = np.array([r['cfg_s15_clip_frac'] for r in rows]); a = np.array([r['apg_s15_clip_frac'] for r in rows])
    d = c - a; order = np.argsort(-d, kind='stable'); rank = {int(i): n + 1 for n, i in enumerate(order)}
    out.append('\nTop-10 ids by CFG-APG clip fraction at w=15:')
    for i in order[:10]:
        out.append(f'  id {i:3d} cfg={c[i]*100:.3f}% apg={a[i]*100:.3f}% diff={d[i]*100:.3f}pp  {rows[i]["caption"]}')
    for i in (42, 17, 0):
        out.append(f'id {i}: rank {rank[i]}/100 (by CFG-APG) cfg={c[i]*100:.3f}% apg={a[i]*100:.3f}% '
                   f'peak cfg={rows[i]["cfg_s15_peak"]:.3f} apg={rows[i]["apg_s15_peak"]:.3f} '
                   f'hf cfg={rows[i]["cfg_s15_hf_ratio"]:.4f} apg={rows[i]["apg_s15_hf_ratio"]:.4f} '
                   f'crest cfg={rows[i]["cfg_s15_crest_db"]:.2f} apg={rows[i]["apg_s15_crest_db"]:.2f}')
    out.append(f'share of total CFG-APG clipped-sample excess from top-10 ids: {d[order[:10]].sum()/d.sum():.2%}')
    txt = '\n'.join(out); print(txt); (A / 'perclip_summary.txt').write_text(txt)


def plot():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14})
    df = pd.read_csv(A / 'perclip_cfg_vs_apg.csv')
    c = df.cfg_s15_clip_frac.values * 100; a = df.apg_s15_clip_frac.values * 100
    n = int((a < c).sum()); nt = int((a == c).sum())
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    lim = 10 ** np.ceil(np.log10(max(c.max(), a.max()) * 1.2))
    ax.set_xscale('symlog', linthresh=1e-2); ax.set_yscale('symlog', linthresh=1e-2)
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
    ax.scatter(c, a, s=22, color='#4C78A8', alpha=0.7, edgecolor='none')
    for i, col, dx, dy in ((42, '#D62728', 8, -16), (17, '#E68613', 10, 12)):
        ax.scatter(c[i], a[i], s=70, color=col, edgecolor='k', zorder=5)
        ax.annotate(f'id {i}', (c[i], a[i]), xytext=(dx, dy), textcoords='offset points', color=col, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.002, lim); ax.set_ylim(-0.002, lim)
    ax.set_xlabel('CFG w=15 clipped samples (%)'); ax.set_ylabel('APG w=15 clipped samples (%)')
    ax.text(0.03, 0.97, f'APG lower on {n}/100 clips\n(tie, mostly both 0: {nt})', transform=ax.transAxes, va='top', fontsize=14,
            bbox=dict(fc='white', ec='0.6'))
    ax.legend(loc='lower right', fontsize=12); ax.grid(alpha=0.3)
    fig.tight_layout()
    for e in ('pdf', 'png'): fig.savefig(FIGS / f'why_perclip_scatter.{e}')
    plt.close(fig)
    # fig 2
    import soundfile as sf
    from matplotlib.colors import Normalize
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=150, gridspec_kw={'height_ratios': [1, 1.2]})
    specs = []
    for j, (run, name) in enumerate((('cfg_s15', 'CFG w=15'), ('apg_s15', 'APG w=15'))):
        x, sr = load(run, 42)
        t = np.arange(len(x)) / sr; m = x.mean(1)
        ax = axs[0, j]; ax.plot(t, x[:, 0], lw=0.3, color='#4C78A8')
        cl = np.abs(x).max(1) >= 0.99
        ax.plot(t[cl], np.sign(x[cl, 0]) * np.maximum(np.abs(x[cl, 0]), 0.99), '.', ms=2, color='#D62728', label=f'|x|>=0.99 ({(np.abs(x)>=0.99).mean()*100:.3f}% of samples)')
        for y in (-1, 1): ax.axhline(y, color='k', ls='--', lw=1)
        ax.set_ylim(-1.5, 1.5); ax.set_xlim(0, 10); ax.set_title(f'id 42, {name} (peak {np.abs(x).max():.2f})')
        ax.set_ylabel('amplitude'); ax.legend(loc='upper right', fontsize=12)
        f, tt, Z = stft(m, fs=sr, nperseg=NF, noverlap=NF - HOP)
        specs.append((f, tt, 20 * np.log10(np.abs(Z) + 1e-6)))
    vmax = max(s[2].max() for s in specs); vmin = vmax - 90
    for j in range(2):
        f, tt, S = specs[j]; ax = axs[1, j]
        im = ax.pcolormesh(tt, f / 1000, S, vmin=vmin, vmax=vmax, cmap='magma', shading='auto', rasterized=True)
        ax.axhline(5, color='w', ls=':', lw=1); ax.set_ylim(0, 22.05); ax.set_xlabel('time (s)'); ax.set_ylabel('kHz')
    fig.subplots_adjust(left=0.06, right=0.91, top=0.94, bottom=0.08, hspace=0.25, wspace=0.12)
    cax = fig.add_axes([0.93, 0.08, 0.015, axs[1, 1].get_position().height])
    fig.colorbar(im, cax=cax, label='dB (shared scale)')
    for e in ('pdf', 'png'): fig.savefig(FIGS / f'id42_wave_spec.{e}')


if __name__ == '__main__':
    {'compute': compute, 'plot': plot}[sys.argv[1]]()
