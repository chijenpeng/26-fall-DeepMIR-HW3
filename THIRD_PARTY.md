# Data, code and citations

- **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`, Stability AI community license; Evans et al., 2024). Used as a pretrained, unmodified model. Gated; not redistributed.
- **Hugging Face diffusers** (Apache-2.0): `StableAudioPipeline` and `CosineDPMSolverMultistepScheduler`.
- **TA sample code** (course-provided): `generate.py` unchanged and the `cfg.py` skeleton; `cfg()` is our implementation.
- **LAION-CLAP** (Apache-2.0, https://github.com/LAION-AI/CLAP; Wu et al., ICASSP 2023) and the checkpoint `music_audioset_epoch_15_esc_90.14.pt` from `lukewys/laion_clap`. Used for retrieval and the CLAP score. Not redistributed.
- **av-benchmark** (hkchengrex, https://github.com/hkchengrex/av-benchmark), used for FD-VGG, KL-PaSST and CLAP score; our changes are in `eval_tools/av-benchmark-hw3.patch`.
- **hkchengrex forks** of CLAP (`hkchengrex/CLAP`) and `passt_hear21` (`hkchengrex/passt_hear21`).
- **VGGish** (torchvggish weights `vggish-10086976.pth`) and **PaSST** (Koutini et al., 2021) weights, downloaded by the evaluation tools.
- **MusicCaps** (Agostinelli et al., *MusicLM*, 2023); audio supplied through the course Drive link, not redistributed.
- **APG**: Sadat et al., *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*, ICLR 2025; `apg.py` is our implementation.

Code written with AI assistance (Anthropic Claude).
