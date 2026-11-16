# Data, code and citations

- **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`, Stability AI community license; Evans et al., ICASSP 2025). Used as a pretrained, unmodified model. Gated; not redistributed.
- **Hugging Face diffusers** (Apache-2.0): `StableAudioPipeline` and `CosineDPMSolverMultistepScheduler`.
- **TA sample code** (course-provided): `generate.py` unchanged and the `cfg.py` skeleton; `cfg()` is our implementation.
- **LAION-CLAP** (Apache-2.0, https://github.com/LAION-AI/CLAP; Wu et al., ICASSP 2023) and the checkpoint `music_audioset_epoch_15_esc_90.14.pt` from `lukewys/laion_clap`. Used for retrieval and the CLAP score. Not redistributed.
- **av-benchmark** (hkchengrex, https://github.com/hkchengrex/av-benchmark), used for FD-VGG, KL-PaSST and CLAP score; our changes are in `eval_tools/av-benchmark-hw3.patch`. Its README asks to cite MMAudio: H. K. Cheng et al., *MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis*, CVPR 2025.
- **hkchengrex forks** of CLAP (`hkchengrex/CLAP`) and `passt_hear21` (`hkchengrex/passt_hear21`).
- **VGGish** (torchvggish weights `vggish-10086976.pth`) and **PaSST** (Koutini et al., Interspeech 2022) weights, downloaded by the evaluation tools.
- **CLAP** (Elizalde et al., *CLAP: Learning Audio Concepts from Natural Language Supervision*, ICASSP 2023), the model family behind the CLAP score.
- **VGGish** (Hershey et al., *CNN Architectures for Large-Scale Audio Classification*, ICASSP 2017), the embedding used for FD-VGG.
- **Fréchet Audio Distance** (Kilgour et al., *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms*, Interspeech 2019).
- **Classifier-free guidance**: Ho and Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (arXiv:2207.12598).
- **EDM**: Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS 2022.
- **MusicCaps** (Agostinelli et al., *MusicLM*, 2023); audio supplied through the course Drive link, not redistributed.
- **APG**: Sadat et al., *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*, ICLR 2025; `apg.py` is our implementation.

Code written with AI assistance (Anthropic Claude).
