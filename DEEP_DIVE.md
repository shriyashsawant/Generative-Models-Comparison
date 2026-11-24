# The Generative Model Landscape — A Deep Comparison

**Diffusion · GANs · VAEs · Autoregressive · Normalizing Flows · VQ/Tokenizers · Flow Matching & Consistency Models**

A technical reference comparing every major generative model family. Read `CHEAT_SHEET.md` for the one-page summary, `GENERATIVE_MODELS_LAB.ipynb` for the executable PyTorch benchmark, and `RESULTS.md` for measured numbers.

---

## 0. The frame: what are we actually comparing?

Every generative model is trying to learn the same thing: the data distribution `p_data(x)`, and then give you a procedure for drawing new samples from it. They differ on **what information they learn** and **how generation happens at inference time**.

Seven questions expose almost every meaningful difference:

1. **Objective** — What loss is being minimized? Is it a likelihood, a lower bound, a regression, or a game?
2. **Sampling cost** — One forward pass, or many sequential steps?
3. **Mode coverage** — Does it represent the whole distribution, or only the "safe" parts?
4. **Sample fidelity** — Sharp, high-frequency detail, or smoothed averages?
5. **Latent space** — Is there an encoder? Is the latent continuous, discrete, or nonexistent?
6. **Likelihood** — Can you compute `p(x)` exactly, approximately, or not at all?
7. **Conditioning** — How easily can you steer the output with text, masks, reference images, or control signals?

Keep these seven in mind. The rest of this document is essentially filling in the grid.

### The central tradeoff triangle

```
                    FIDELITY / SHARPNESS
                            ▲
                           ╱ ╲
                          ╱   ╲
                         ╱     ╲
        GANs ●──────────╱       ╲──────────● Diffusion
              (fast,    ╱         ╲    (sharp, diverse,
               narrow) ╱           ╲    slow)
                      ╱             ╲
                     ╱               ╲
          ● VAE ────────────────────────● Autoregressive / Flows
     (fast, diverse,                (exact likelihood,
      soft/blurry)                   stable, sequential)
```

No family dominates all three axes. **GANs** buy sharpness + speed with diversity. **VAEs** buy speed + coverage with sharpness. **Diffusion** buys sharpness + coverage with speed. **Autoregressive models and flows** buy exact likelihood + stability with sampling speed. Most of the progress since 2023 is about **breaking this triangle** — distillation, flow matching, and two-stage designs that borrow the best property of each family.

---

## 1. VAEs — the compression workhorse

### How it works
Two networks: an **encoder** `qφ(z|x)` maps data to a distribution over a compact latent code, and a **decoder** `pθ(x|z)` maps a code back to data. You train by maximizing the ELBO:

```
L = E_q[ log pθ(x|z) ]  −  KL( qφ(z|x) ‖ p(z) )
     └── reconstruction ──┘  └── regularization ──┘
```

The **reparameterization trick** (`z = μ + σ·ε`, `ε ~ N(0,I)`) makes the sampling differentiable, so you can use ordinary SGD. At generation time you sample `z ~ p(z)` and run the decoder once.

### Strengths
- **Stable training.** One differentiable objective; no adversarial min–max game, no delicate balance.
- **One-pass sampling.** Fast inference, comparable to GANs.
- **A real latent space.** You get an encoder, so you can encode real images, interpolate smoothly, do attribute manipulation, anomaly detection, retrieval, and downstream tasks. Neither GANs nor vanilla diffusion give you this for free.
- **Good mode coverage.** The likelihood-style objective pushes mass across the data distribution instead of collapsing onto a few modes.
- **Excellent compressor.** This is why VAEs did not disappear: modern latent diffusion systems use a VAE-style autoencoder as a *first stage*. Stable Diffusion's autoencoder even adds adversarial and perceptual losses to sharpen reconstructions — a VAE wearing a GAN jacket.

### Weaknesses
- **Blurry samples.** A Gaussian decoder with MSE loss is optimal at predicting the *conditional mean*. When several plausible outputs share a latent code, the model averages them — the classic "blurry face" effect. A discriminator or an iterative denoiser does not have this problem.
- **Posterior collapse.** With a strong decoder (especially in text), the KL term can drive `qφ(z|x) ≈ p(z)`, making the latent code ignored. β-VAE and free-bits are partial fixes.
- **Bound, not exact likelihood.** The ELBO is a lower bound; density estimates are approximate and often pessimistic.
- **Prior mismatch.** A simple isotropic Gaussian prior may not match the aggregated posterior, producing holes when sampling.

### Use it when
You need a compact, structured representation: compression, interpolation, anomaly detection, scientific/tabular generation, or a first-stage tokenizer/encoder for a larger generative system.

### Flagships / lineage
Auto-Encoding Variational Bayes (Kingma & Welling, 2013) · β-VAE · VQ-VAE · VAEs inside Stable Diffusion, Imagen, and Latent Consistency models.

---

## 2. GANs — the speed demo that changed everything

### How it works
A **generator** `G(z)` maps noise to a sample; a **discriminator** `D(x)` outputs "real or fake." They play a minimax game:

```
min_G max_D  V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1 − D(G(z)))]
```

At equilibrium, the generator's distribution matches the data distribution and the discriminator can do no better than chance. Crucially, this learns an **implicit** distribution — no explicit likelihood is ever computed.

### Strengths
- **Razor-sharp outputs.** The adversarial loss directly penalizes blur; the discriminator hunts for telltale smoothing.
- **Single forward pass.** Once trained, generation is one network evaluation. This is a real systems advantage for real-time graphics, mobile, games, and interactive tools.
- **No likelihood needed.** You avoid the blurring induced by pixel-wise MSE on high-frequency detail.
- **Rich control in narrow domains.** StyleGAN's style space supports mixing, interpolation, and disentangled attribute editing in a way that is hard to beat for faces/objects.
- **Still a workhorse component.** Adversarial losses survive inside super-resolution, neural vocoders (HiFi-GAN), image-to-image translation, and the autoencoders used by latent diffusion.

### Weaknesses
- **Training instability.** This is a saddle-point optimization, not minimization. Loss curves oscillate and do not reliably indicate sample quality; the generator and discriminator must improve at a similar pace.
- **Mode collapse.** The generator can find a few outputs that reliably fool the discriminator and ignore the rest of the data. Samples look great individually and terrible as a set.
- **Vanishing gradients.** As `D` gets good, `log(1−D(G(z)))` saturates and `G` stops learning. Non-saturating loss, WGAN-GP, hinge loss, spectral normalization, and TTUR were all invented to patch this.
- **No encoder, no likelihood.** You cannot encode a real image into `z`, and you cannot report a meaningful `p(x)` or validation loss. Evaluating "when to stop" is notoriously hard.
- **Struggles with discrete data** (text, tokens) and catastrophic forgetting between modes.
- **Scales poorly to huge, diverse distributions** compared with likelihood-based methods.

### Use it when
Latency is the hard constraint and the domain is narrow and well-curated: real-time avatars, game assets, super-resolution, vocoders, style editing — or when you want an adversarial loss as a *component* of a bigger system.

### Flagships / lineage
DCGAN · WGAN / WGAN-GP · BigGAN · StyleGAN / StyleGAN2 / StyleGAN3 · CycleGAN · HiFi-GAN. For general text-to-image, GANs have largely been displaced since ~2022.

---

## 3. Autoregressive models — next-token prediction, applied to pixels

### How it works
Factorize the joint distribution into a product of conditionals:

```
p(x) = ∏_{i=1}^{N} pθ(x_i | x_{<i})
```

PixelRNN/PixelCNN did this at the pixel level; modern systems do it over **discrete tokens** from a learned tokenizer (VQGAN, VQ-VAE) — the same machinery as language models. DALL·E (2021) and iGPT were early text-to-image examples; LlamaGen, Emu3, Chameleon, and GPT-Image-style systems are the 2024–2026 lineage.

### Strengths
- **Exact likelihood + stable training.** Cross-entropy is a well-understood, single, tractable objective. You get a real validation metric (bits/token) and clean scaling laws.
- **Natural conditioning.** Text, previous image tokens, and other modalities live in the same token stream, so instruction following, long prompts, and multi-element scenes come naturally.
- **Text rendering.** This is the famous differentiator: because written language is sequential, autoregressive models are much better at putting *legible words* inside images. Diffusion models have historically garbled letters.
- **Unified multimodality.** One transformer can speak text, images, and audio — no separate diffusion prior required.
- **Parallel-ish alternatives exist.** Visual Autoregressive (VAR) models predict tokens scale-by-scale rather than strictly one-by-one, narrowing the speed gap.

### Weaknesses
- **Sequential generation = slow.** Sampling is inherently serial, so latency grows with token count.
- **O(n²) attention cost** in sequence length; high-resolution images mean very long token sequences.
- **Tokenizer ceiling.** Quality is bounded by the discrete tokenizer's reconstruction fidelity; quantization artifacts are inherited.
- **Exposure bias.** Training sees ground-truth prefixes; inference sees the model's own mistakes.
- **Texture/detail.** Historically weaker than diffusion on pure photorealism and fine texture, though the gap is closing fast.

### Use it when
Instruction following, readable text-in-image, complex multi-object scenes, or a single unified multimodal pipeline matters more than raw texture fidelity.

### Flagships / lineage
PixelCNN / PixelRNN · iGPT · DALL·E · VQGAN+Transformer · LlamaGen · Emu3 · Chameleon · GPT Image · VAR.

---

## 4. Normalizing flows — exact likelihood, parallel both ways

### How it works
Learn an **invertible** transformation `f` from a simple base distribution to the data:

```
x = f(z),  z ~ p(z)          log p_X(x) = log p_Z(z) − log |det J_f(z)|
```

Architectures (coupling layers, autoregressive flows, invertible 1×1 convolutions) are designed so both `f` and `f⁻¹` and the Jacobian determinant are cheap.

### Strengths
- **Exact, tractable likelihood** — like autoregressive models, but with **parallel** density evaluation and sampling.
- **Invertible latent mapping** — encoding and decoding are the same learned function; useful for density estimation and scientific applications.
- **Memory-efficient training** with reversible architectures.
- **Clean mathematical foundation** and a genuine validation metric.

### Weaknesses
- **Invertibility is a straitjacket.** `f` must be bijective and differentiable, with input and output dimensions equal — no compression, and expressivity is constrained by the architectural tricks needed to keep the Jacobian tractable.
- **Harder to train on complex, high-dimensional data** than VAEs/diffusion; convergence can be slow.
- **Not competitive on high-resolution image fidelity** — the family is smaller and more specialized today.
- Adding stochasticity improves expressivity but destroys the exact-likelihood property.

### Use it when
You need exact density estimation, invertible inference, or a principled latent mapping — e.g., scientific ML, tabular data, density estimation, or as a component inside other models.

### Flagships / lineage
NICE · Real NVP · Glow · MAF / IAF · Neural Spline Flows · stochastic / relaxed flows.

---

## 5. Diffusion models — the current default for fidelity

### How it works
**Forward process (fixed, no learning):** gradually corrupt data with Gaussian noise over `T` steps.

```
q(x_t | x_{t−1}) = N( √(1 − β_t) · x_{t−1},  β_t I )
```

**Reverse process (learned):** train a network to predict the noise added at each step (or the score, or `x₀`):

```
L = E_{t, x₀, ε} [ ‖ ε − εθ(x_t, t) ‖² ]
```

Generation starts from pure noise and iteratively denoises. Mathematically this unifies with score-based models and SDEs: noise prediction and score estimation are algebraically equivalent, and samplers like DDIM, DPM-Solver, and probability-flow ODEs are discretizations of the same underlying flow.

**Latent diffusion** (Stable Diffusion) compresses images with a VAE-style autoencoder and runs the denoising in that smaller latent space, cutting compute dramatically. **DiT** replaces the convolutional U-Net with a transformer over latent patches — it scales more predictably with parameters and compute (SD3, Flux).

### Strengths
- **Best-in-class fidelity and diversity.** Strong state-of-the-art FID/recall, no mode collapse, covers the full data distribution.
- **Stable objective.** A simple regression loss — no adversarial dynamics, no min–max game.
- **Flexible conditioning.** Classifier-free guidance, ControlNet, inpainting, masks, reference images, and cross-attention text conditioning all drop in naturally.
- **Scales well** with data, parameters, and compute; DiT made that scaling especially predictable.
- **Rich ecosystem:** LoRA, adapters, editors, video extensions, open weights.

### Weaknesses
- **Slow sampling.** Tens to hundreds of sequential network evaluations per sample; this is a step-change in latency and memory traffic versus GANs/VAEs.
- **Training cost** is substantial (large datasets, large backbones).
- **Guidance trades off diversity.** High classifier-free guidance improves prompt adherence but reduces variety and can oversaturate.
- **Text rendering** has historically been weak because diffusion reasons globally over pixels, not character-by-character — one of the main reasons autoregressive/image-language models are gaining ground.
- **No natural encoder/latent code** unless you bolt on a separate autoencoder.

### Use it when
Quality and variety are the priority and latency is not the bottleneck: text-to-image, image editing, video, audio, scientific generation, offline/batch pipelines.

### Flagships / lineage
DDPM · DDIM · GLIDE · Imagen · DALL·E 2 · Latent Diffusion / Stable Diffusion · SDXL · SD3 · Flux · Sora / Open-Sora · Midjourney.

---

## 6. VQ-VAE & neural tokenizers — the bridge between families

### How it works
An encoder maps an image to a grid of **discrete codes** from a learned codebook; a decoder reconstructs from those codes. Training uses a commitment loss plus the straight-through estimator; VQGAN adds perceptual and adversarial losses for sharp reconstruction.

### Why it matters
Discrete tokens let you reuse **language-model tooling**: a transformer can now predict image tokens exactly as it predicts words. That is the foundation of both **autoregressive image generators** (VQGAN + Transformer, LlamaGen) and the **tokenizer/autoencoder stage** of many latent diffusion systems.

### Strengths
- Strong perceptual reconstruction at high compression.
- Discrete tokens unify modalities in one vocabulary.
- Compatible with standard LM infrastructure (tokenizers, attention, sampling tricks).

### Weaknesses
- **Codebook collapse** — unused codes waste capacity.
- **Quantization artifacts** impose a reconstruction ceiling on everything downstream.
- Tokenizer must usually be trained separately first; errors compound.

---

## 7. Flow matching, rectified flow & consistency models — breaking the speed penalty

This is the most important recent chapter, and the one most "diffusion vs GAN" posts miss.

- **Flow Matching / Rectified Flow:** instead of a noisy Gaussian-forward path, learn a vector field that transports noise to data along (ideally straight) paths. Simpler objective, faster sampling, strong quality. Used in Flux, SD3, Qwen-Image and others.
- **Consistency Models / progressive distillation / LCM / Turbo / lightning:** distill the iterative teacher into a student that maps *any* point on the path directly to the endpoint — enabling **1–4 step** generation instead of 20–1000.

### Strengths
- Near-diffusion quality with a fraction of the inference steps.
- Simpler, more direct training objectives than noise prediction.
- Straight paths are friendly to fast ODE solvers.

### Weaknesses
- Still iterative unless distilled; distillation can cost some fidelity/diversity.
- Newer, faster-moving tooling and less mature ecosystem than classic diffusion.
- The theoretical and practical boundary between "diffusion" and "flow" is now genuinely blurry — which is why modern comparisons should treat them as one continuum.

---

## 8. Side-by-side summary

| Family | Objective | Sampling | Mode coverage | Fidelity | Encoder / latent | Likelihood | Training stability | Conditioning | Typical use today |
|---|---|---|---|---|---|---|---|---|---|
| **VAE** | ELBO (recon + KL) | 1 pass | Good | Softer / blurry | Yes, continuous | Approx. lower bound | High | Moderate via latent | Compression, interpolation, first stage |
| **GAN** | Adversarial min–max | 1 pass | Poor (collapse) | Very sharp | No | None | Low | Moderate | Real-time, narrow domains, adversarial loss component |
| **Autoregressive** | Cross-entropy / exact likelihood | Sequential | Good | Strong, great text | Via tokenizer | Exact | High | Excellent | Unified multimodal, text-in-image, instruction following |
| **Normalizing Flow** | Exact log-likelihood | Parallel | Good | Moderate | Invertible | Exact | Medium | Moderate | Density estimation, scientific ML |
| **Diffusion** | Denoising regression / score matching | T steps | Excellent | Best overall | No (bolt on VAE) | Approx. (ODE) | High | Excellent (CFG, ControlNet) | Text-to-image, video, editing |
| **VQ / tokenizer** | Commitment + perceptual/adv. | 1 pass decode | Good | Strong reconstruction | Yes, discrete | None | Medium | Via downstream model | Tokenization for AR + diffusion |
| **Flow matching / consistency** | Vector-field regression / distillation | 1–few steps | Excellent | Near-diffusion | No | Approx. | High | Excellent | Fast high-quality generation |

---

## 9. How to choose — a decision guide

```
Need exact likelihood / density estimation? ──► Normalizing Flow or Autoregressive
Need a reusable latent code for real data? ───► VAE (or VQ-VAE for discrete codes)
Hard real-time / single-pass constraint? ─────► GAN, distilled diffusion, consistency model
Readable text inside the image? ─────────────► Autoregressive / unified multimodal model
Max photorealism + variety, latency OK? ─────► Diffusion (DiT / latent diffusion)
Unified text+image+audio pipeline? ──────────► Autoregressive over discrete tokens
Want the best of both worlds? ───────────────► Two-stage: VAE/VQ compressor + diffusion/AR prior
```

**Rule of thumb by constraint:**

| Your hard constraint | Reach for |
|---|---|
| Latency / cost per image | GAN, consistency model, distilled diffusion |
| Diversity / mode coverage | Diffusion, VAE, flows |
| Sharpness / fidelity | Diffusion, GAN |
| Interpretability / latent structure | VAE, flows |
| Likelihood / eval metric | Autoregressive, flows |
| Prompt adherence & text rendering | Autoregressive multimodal models |
| Fast fine-tuning / community ecosystem | Latent diffusion (LoRA, ControlNet) |

---

## 10. What actually happened in the landscape (2023 → 2026)

1. **Diffusion won the quality war.** Stable regression objective + mode coverage + flexible conditioning beat adversarial training for general text-to-image and video.
2. **The backbone changed.** U-Net → Diffusion Transformer (DiT). Transformers scale more predictably, which is why SD3/Flux-class systems lean on them.
3. **The objective changed.** Noise prediction → flow matching / rectified flow, giving straighter paths and fewer steps.
4. **The speed penalty is being erased.** Consistency models, LCM/LoRA-style distillations, and few-step samplers push diffusion toward single-digit steps.
5. **Autoregressive made a comeback.** Token-based, unified multimodal models are competitive again — especially where text rendering and instruction following matter.
6. **Nobody uses a single family anymore.** The dominant architecture is a **pipeline**: a VAE/VQ first stage compresses, a diffusion or AR model models the latents, adversarial/perceptual losses sharpen the decoder, and guidance/RL post-training aligns outputs. The "which family wins" question is increasingly the wrong question — the winners *combine* them.
7. **Post-training matters as much as the base paradigm.** GRPO-style RL, on-policy distillation, preference alignment, and in-context editing are now first-class levers (e.g., 2025–2026 Qwen-Image / FLUX.1 Kontext lineage).

---

## 11. Common misconceptions

- **"VAE and diffusion solve the same problem."** No. VAEs give you a structured latent code; diffusion gives you sample quality. Latent diffusion literally uses both.
- **"GANs are dead."** For general text-to-image, mostly displaced. For real-time/narrow tasks and as an adversarial loss component, very much alive.
- **"Diffusion = blurry like VAE."** No — iterative denoising does not average competing modes the way a Gaussian decoder does.
- **"Diffusion cannot render text, AR can."** Historically true, increasingly a spectrum — hybrid designs are closing it.
- **"Flow matching is a totally different family."** It is better understood as a different (straighter) path within the same diffusion/transport continuum.
- **"Higher likelihood = better images."** Not reliably. Likelihood and perceptual quality correlate weakly; FID, precision/recall, CLIP score, and human preference studies are the practical metrics.

---

## 12. Reference trail

- *Text-to-Image Synthesis: A Decade Survey* — https://arxiv.org/html/2411.16164
- *Diffusion models in deep generative learning: a unified review* (Springer, 2026) — https://link.springer.com/article/10.1007/s41060-026-01257-9
- *Image Generation Models: A Technical History* — https://arxiv.org/html/2603.07455
- *Vision-centric generative AI models: a software–hardware perspective* — https://arxiv.org/html/2608.27199
- *How AI Image Generation Models Work, and How to Choose One* — https://www.derar.dev/blog/comprehensive-guide-ai-image-generation-models-landscape
- *Diffusion vs Autoregressive Image Models* — https://ai-tldr.dev/learn/multimodal-ai/image-generation/diffusion-vs-autoregressive
- *Comparing Diffusion, GAN, and VAE Techniques* — https://generativeailab.org/l/generative-ai/a-tale-of-three-generative-models-comparing-diffusion-gan-and-vae-techniques/569
- *Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models* (IEEE TPAMI) — https://ieeexplore.ieee.org/document/9555209
- *Variations and Relaxations of Normalizing Flows* — https://arxiv.org/html/2309.04433v1
- *How Generative Models Generate: A Concise Guide to the Generative AI Landscape* — https://arxiv.org/pdf/2611.24728

*Prepared 2026-11-25. Model names and versions move fast — verify current specs on vendor sites before publishing hard claims.*
