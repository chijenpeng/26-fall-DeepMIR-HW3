# DeepMIR HW3: text-to-music with Stable Audio Open (CFG baseline + APG)

Release [`submission-v1`](https://github.com/chijenpeng/26-fall-DeepMIR-HW3/releases/tag/submission-v1) holds the generated audio (submitted run and all ten evaluated runs), `retrieval.json` and `results.zip`. MusicCaps audio is not redistributed.

Only public pretrained models are used; there is no trained checkpoint. All code here reproduces the submitted
`generated_samples/` and `retrieval.json`. Do not put the dataset, model caches or checkpoints into this folder.

## Environments
Three separate virtualenvs are needed because of version conflicts. Run everything from this folder.

| Env | Used by | Key versions | Install |
| --- | --- | --- | --- |
| `.venv` (Python 3.11) | generation | torch 2.11.0+cu128, diffusers 0.41.0, transformers 5.20.1 | `pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt` |
| `.venv-clap` | retrieval | laion-clap 1.1.7, transformers 4.30.2, numpy 1.26.4, torch 2.11.0+cu128, torchvision 0.26.0 | `pip install -r eval_tools/requirements-clap.txt` (add `--extra-index-url https://download.pytorch.org/whl/cu128` for the `+cu128` wheels) |
| `.venv-eval` | evaluation | transformers 4.57.6, numpy 2.2.6, av-benchmark + patch, laion-clap and passt (hkchengrex forks) | see below |

Evaluation env (`eval_tools/requirements-eval.in` is the input list; `requirements-eval.txt` is the frozen result and
has the laion-clap and passt lines commented out; install them from the checkouts below (`pip install ./third_party/CLAP ./third_party/passt_hear21`):
```bash
mkdir -p third_party && cd third_party
git clone https://github.com/hkchengrex/av-benchmark
git clone https://github.com/hkchengrex/CLAP            # laion-clap fork
git clone https://github.com/hkchengrex/passt_hear21    # passt fork
cd av-benchmark && patch -p2 < ../../eval_tools/av-benchmark-hw3.patch && cd ../..
python3.11 -m venv .venv-eval
.venv-eval/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r eval_tools/requirements-eval.in
```
(The patch paths start with `third_party/av-benchmark/`; `-p2` run inside `third_party/av-benchmark` strips them.
The fork URLs and the patch command are not re-tested after cleanup.)

## Data and weights
- MusicCaps mp3s (5,367 files, course Drive link): `data/audio_mp3/<ytid>.mp3`.
- Stable Audio Open 1.0 (`stabilityai/stable-audio-open-1.0`, gated): accept the license, then `hf auth login`. Downloaded automatically into the HF cache.
- CLAP music checkpoint `music_audioset_epoch_15_esc_90.14.pt` from Hugging Face `lukewys/laion_clap`, saved to `checkpoints/`. Used for BOTH retrieval and the CLAP score.
- VGGish `vggish-10086976.pth` (torchvggish weights) saved to `checkpoints/`. `evaluate.py` sets `AVB_CLAP_CKPT` and `AVB_VGGISH_CKPT` to these paths by default; PaSST weights download automatically (into `checkpoints/torch`).

## Reproduce (in order)
1. Retrieval (writes `outputs/retrieval/retrieval.json`, `retrieval_top5.csv`, `retrieved_audio/`; the submitted `retrieval.json` is a copy):
   ```bash
   .venv-clap/bin/python retrieve.py
   ```
   First 10 s of each clip (repeat-pad if shorter), cosine similarity between CLAP text and audio embeddings, top-1.
2. Baseline (plain CFG, scale 7):
   ```bash
   .venv/bin/python generate_all.py --out_dir outputs/baseline_s7 --scale 7.0
   ```
3. APG (scale 10, eta 0, r = 10, momentum -0.5):
   ```bash
   .venv/bin/python generate_all.py --out_dir outputs/apg_s10 --scale 10 --method apg --eta 0 --norm_threshold 10 --momentum -0.5
   ```
   100 steps, 10 s, fp16, seed = id (clip seed = `--seed` + id, default 0), about 10.5 s/clip on an RTX 4070 SUPER. APG is applied in denoised (x0) space; scheduler CosineDPMSolverMultistep (v-prediction). Output is `outputs/<run>/<id>.mp3`.
4. Evaluation (FD-VGG, KL-PaSST, CLAP; reference = retrieved clip per test id, plus FD against full MusicCaps; mp3 decoded with soundfile, no ffmpeg):
   ```bash
   bash eval_tools/run_eval_all.sh baseline_s7 apg_s10
   ```
   The script `cd`s to its own directory, so copy it next to `evaluate.py` first (`cp eval_tools/run_eval_all.sh .`).
   Results go to `outputs/eval/<run>.json` and `<run>__fullmusiccaps.json`; copies of all runs are in `results/`.

The CFG sweep (scales 3, 10, 1, 5, plus 15) is in `run_all.sh` and `run_apg.sh` (hard-coded to the author's machine paths; edit the `cd`).

## Extra analysis (FD decomposition, per-clip CFG vs APG)
Both scripts run from this folder after the evaluation runs exist and use the `.venv-eval` env (scipy is already in `eval_tools/requirements-eval.in`; the `plot` stage also needs `pip install matplotlib`). Each has a `compute` stage (writes to `results/`) and a `plot` stage (figures to `outputs/figs/`).
- `analysis_fd_diversity.py` splits FD-VGG into a mean term and a covariance term, and reports Tr(Sigma) as a diversity measure, from the cached VGGish features in `outputs/_eval_cache/` (created by `evaluate.py`) and `third_party/av-benchmark`. Output: `results/fd_diversity.csv`, `results/fd_diversity.json`.
  ```bash
  .venv-eval/bin/python analysis_fd_diversity.py compute
  .venv-eval/bin/python analysis_fd_diversity.py plot
  ```
- `analysis_perclip.py` computes per-clip clipping fraction, peak, RMS, crest factor and high-frequency (>5 kHz) energy ratio for `cfg_s10`, `cfg_s15`, `apg_s10`, `apg_s15` (decoded mp3 in `outputs/<run>/`), with paired Wilcoxon tests. Output: `results/perclip_cfg_vs_apg.csv` and `results/perclip_summary.txt`.
  ```bash
  .venv-eval/bin/python analysis_perclip.py compute
  .venv-eval/bin/python analysis_perclip.py plot
  ```

## Files
| File | Description |
| --- | --- |
| `cfg.py` | Submitted `cfg()` (classifier-free guidance) |
| `generate.py` | TA sampling loop using `cfg` (unchanged) |
| `generate_all.py` | Batch driver over `test.csv`; `--method {cfg,apg}`, `--eta`, `--norm_threshold`, `--momentum` |
| `apg.py`, `generate_apg.py` | Adaptive Projected Guidance and its sampling loop |
| `apg_check.py` | Round-trip test (eta=1, no rescale/momentum equals CFG) and measurement of the delta norm used to choose r; run only on non-test prompts |
| `retrieve.py` | LAION-CLAP text-to-audio retrieval over MusicCaps |
| `evaluate.py` | FD-VGG / KL-PaSST / CLAP score on top of patched av-benchmark |
| `verify_outputs.py` | Sanity check of generated dirs (count, duration, finite, clipping) |
| `analysis_fd_diversity.py`, `analysis_perclip.py` | FD decomposition / diversity and per-clip CFG vs APG statistics |
| `run_all.sh`, `run_apg.sh` | Sweep scripts that produced `results/` |
| `eval_tools/` | `av-benchmark-hw3.patch`, `requirements-eval.in/.txt`, `requirements-clap.txt`, `run_eval_all.sh` |
| `results/` | Metric JSONs, `combined.csv`, `retrieval.json`, `retrieval_top5.csv`, `fd_diversity.csv/.json`, `perclip_cfg_vs_apg.csv` |
| `test.csv` | The 100 test captions (`id,caption`) |
| `requirements.txt` | Generation env |

Note: `apg_check.py` was used only with non-test prompts to choose the rescale radius r (about the median delta norm, r = 10).
