# Generative Models — Cheat Sheet

## One line each

| Model | One line |
|---|---|
| **VAE** | Encode to a smooth latent, decode back; stable but soft. |
| **GAN** | Generator fools discriminator; sharp and instant, unstable and narrow. |
| **Autoregressive** | Predict the next token/pixel; exact likelihood, slow serial sampling, great text. |
| **Normalizing Flow** | Invertible noise→data map; exact likelihood, parallel, constrained. |
| **Diffusion** | Noise data, learn to denoise; best quality + diversity, many steps. |
| **VQ-VAE / tokenizer** | Discrete image codes; the bridge to language-model generation. |
| **Flow matching / consistency** | Straight noise→data path, distilled to 1–4 steps. |

## Master comparison

| Axis | GAN | VAE | AR | Flow | Diffusion | Flow matching / consistency |
|---|---|---|---|---|---|---|
| Objective | Minimax game | ELBO | Cross-entropy | Exact log-lik | Denoising regression | Vector-field regression / distillation |
| Training stability | Low | High | High | Medium | High | High |
| Sample sharpness | Very high | Low–med | High | Medium | Very high | Very high |
| Mode coverage | Low (collapse) | Good | Good | Good | Excellent | Excellent |
| Sampling cost | 1 pass | 1 pass | Sequential | Parallel | T steps (20–1000) | 1–4 steps |
| Encoder / latent | No | Yes, cont. | Tokenizer | Invertible | No (add VAE) | No (add VAE) |
| Exact likelihood | No | No (bound) | Yes | Yes | No (approx. ODE) | No (approx.) |
| Conditioning | Moderate | Moderate | Excellent | Moderate | Excellent | Excellent |
| Text in images | Weak | Weak | Strong | Weak | Historically weak | Improving |
| maturity / ecosystem | Mature | Mature | Fast-growing | Specialized | Dominant | Fast-moving |

## The tradeoff triangle

- **GAN** → fidelity + speed, sacrifices diversity
- **VAE** → diversity + speed, sacrifices fidelity
- **Diffusion** → fidelity + diversity, sacrifices speed
- **AR / Flow** → likelihood + stability, sacrifices speed / expressivity
- **Distillation & two-stage designs** → the attempt to get all three

## Decision tree

```
Exact density? ─────────────────► Flow or Autoregressive
Need latent codes for real data? ► VAE (or VQ-VAE for discrete)
Hard latency budget? ───────────► GAN / consistency / distilled diffusion
Readable text in image? ───────► Autoregressive multimodal
Max realism + variety? ────────► Diffusion (latent / DiT)
Unified text+image+audio? ─────► Autoregressive over tokens
Best of everything? ───────────► VAE/VQ compressor + diffusion/AR prior + distillation
```

## Failure modes to name in conversation

- **GAN:** mode collapse, vanishing gradients, oscillation, no validation loss
- **VAE:** posterior collapse, blur from MSE + Gaussian decoder, ELBO looseness
- **AR:** exposure bias, O(n²) attention, tokenizer ceiling, slow serial decode
- **Flow:** invertibility limits expressivity, equal dimensions (no compression)
- **Diffusion:** many-step latency, guidance diversity tradeoff, weak text rendering
- **Distillation:** quality/diversity loss vs teacher

## Metrics that actually get used

- **FID** — distribution distance (lower better)
- **Precision / Recall** — fidelity vs coverage for generative models
- **IS (Inception Score)** — quality + diversity, older
- **CLIP score** — text-image alignment
- **Bits/dim or bits/token** — likelihood-based models
- **Human preference studies** — the real ground truth; automatic metrics correlate weakly with judgment

## Flagship names to drop

- **GAN:** DCGAN, WGAN-GP, BigGAN, StyleGAN2/3, CycleGAN, HiFi-GAN
- **VAE:** β-VAE, VQ-VAE, diffusion autoencoders
- **AR:** PixelCNN, iGPT, DALL·E, LlamaGen, Emu3, Chameleon, GPT Image, VAR
- **Flow:** NICE, RealNVP, Glow, MAF, Neural Spline Flows
- **Diffusion:** DDPM, DDIM, GLIDE, Imagen, DALL·E 2, Stable Diffusion, SDXL, SD3, Flux, Midjourney, Sora
- **Fast variants:** Consistency Models, LCM, progressive distillation, Rectified Flow

## Myth-busters

- "GANs are dead" → displaced for general T2I, alive for real-time + as loss component
- "VAE ≈ diffusion" → VAE gives latents; diffusion gives quality; latent diffusion uses both
- "Higher likelihood = better images" → weak correlation; judge with FID/precision-recall/humans
- "Flow matching is a separate universe" → same transport continuum, straighter path
- "Diffusion blurs like VAE" → iterative denoising doesn't average competing modes

## 2026 in three sentences

Diffusion/flow transformers dominate quality; autoregressive multimodal models are the serious challenger wherever text and instructions matter; and production systems are hybrids of VAE/VQ compression + diffusion/AR priors + adversarial sharpening + RL alignment, with consistency distillation collapsing the sampling cost.
