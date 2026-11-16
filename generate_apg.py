"""Sampling loop identical to generate.py except that the CFG step is replaced by APG (apg.py).

    python generate_apg.py --prompt "jazz, electric guitar, piano, relaxing" --scale 7 --out out.wav
"""

import argparse

import soundfile as sf
import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from apg import MomentumBuffer, apg_v_prediction
from cfg import cfg


@torch.no_grad()
def generate_apg(pipe, prompt, guidance_scale=7.0, num_inference_steps=100, audio_end_in_s=10.0, seed=0,
                 eta=0.0, norm_threshold=0.0, momentum=0.0, trace=None):
    """trace: optional list; one dict per step (step, sigma, delta_norm, max |v_apg - v_cfg|)."""
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate

    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, audio_end_in_s, device, False, batch_size=1)
    cond = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    encoder_hidden_states = torch.cat([torch.zeros_like(cond), cond])
    global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)

    generator = torch.Generator(device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
        cond.dtype, device, generator, audio_channels=pipe.vae.config.audio_channels,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim, latents.shape[2] + global_hidden_states.shape[1],
        use_real=True, repeat_interleave_real=False,
    )
    buffer = MomentumBuffer(momentum) if momentum != 0 else None
    sigmas = pipe.scheduler.sigmas

    for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = pipe.transformer(
            latent_model_input, t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        v_uncond, v_cond = noise_pred.chunk(2)
        tr = {} if trace is not None else None
        # sigmas[i] is the sigma the scheduler uses for this step (sigmas has steps+1 entries, last is 0).
        noise_pred = apg_v_prediction(float(sigmas[i]), latents, v_uncond, v_cond, guidance_scale,
                                      eta, norm_threshold, buffer, tr)
        if trace is not None:
            tr.update(step=i, sigma=float(sigmas[i]),
                      v_diff_vs_cfg=(noise_pred.float() - cfg(v_uncond, v_cond, guidance_scale).float()).abs().max().item())
            trace.append(tr)
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(audio_end_in_s * sampling_rate)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--scale", type=float, default=7.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--norm_threshold", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--out", default="out_apg.wav")
    a = p.parse_args()
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    audio = generate_apg(pipe, a.prompt, a.scale, a.steps, a.seconds, a.seed, a.eta, a.norm_threshold, a.momentum)
    sf.write(a.out, audio.T.float().cpu().numpy(), pipe.vae.config.sampling_rate)


if __name__ == "__main__":
    main()
