# Real Results — Generative Models Lab

Produced by executing `GENERATIVE_MODELS_LAB.ipynb` end-to-end on 2026-11-25.
Environment: Python 3.15, PyTorch 2.15 (CPU), NumPy 2.5, scikit-learn 1.9. Shared dataset: 6,000 points from 8 Gaussian modes on a ring. Fixed seeds, identical metrics for every model.

## Measured results

| Model | MMD² ↓ | Precision ↑ | Recall ↑ | Mode coverage ↑ | Train seconds | Note |
|---|---:|---:|---:|---:|---:|---|
| VAE | 0.0011 | 0.7027 | 0.6527 | 1.00 | 0.75 | ELBO (lower bound) |
| GAN (WGAN-GP) | 0.0099 | 0.1373 | 0.5080 | 1.00 | 40.41 | implicit density, no likelihood |
| Autoregressive | 0.0018 | 0.9100 | 0.9653 | 1.00 | 0.58 | exact likelihood: 3.416 bits/dim |
| Normalizing Flow | 0.0009 | 0.6813 | 0.9467 | 1.00 | 6.54 | exact NLL: −0.487 nats |
| Diffusion (DDIM 100s) | 0.0012 | 0.7627 | 0.9300 | 1.00 | 1.41 | 100 reverse steps, MSE 0.5508 |
| VQ-VAE | 0.0005 | 0.9947 | 0.6927 | 1.00 | 1.52 | 127/128 codes used (empirical prior) |
| Flow Matching (10s) | 0.0014 | 0.4800 | 0.8787 | 1.00 | 0.72 | velocity MSE 2.2013 |
| Consistency (1 step) | 0.0016 | 0.4793 | 0.8380 | 1.00 | 1.14 | distilled from 10-step teacher |

Lower MMD² is better; precision, recall and mode coverage are higher-is-better. `Seconds` is training time only on this CPU — sampling cost is discussed in the notebook.

## What the numbers actually show

- **Mode coverage = 1.00 for every family** on this toy target, including WGAN-GP. Mode collapse is a risk of adversarial training, not an inevitability — the gradient penalty and 10,000 training steps fixed it here.
- **GAN is the slowest to train (40.4s)** and has the weakest precision (0.137): its samples cover the ring but are noticeably more diffuse than the real modes. This is the adversarial game in practice — no monotone loss, no likelihood, no clean stopping criterion.
- **VAE (0.75s) and Autoregressive (0.58s)** are the cheapest to train. The VAE's lower precision (0.70 vs 0.91 for AR) is the classic MSE-decoder softening. The AR model's 48-bin tokenizer gives the sharpest precision/recall tradeoff but a coarse grid.
- **Normalizing Flow** achieves an exact NLL of −0.487 nats and strong coverage (0.947). It needed 8 coupling layers and 3,000 steps; an incorrect coupling implementation initially produced divergent samples — the corrected version is in the notebook.
- **Diffusion** gets near-best MMD² (0.0012) with a simple regression objective, but inference needs 100 sequential network evaluations (the training time shown excludes sampling).
- **VQ-VAE** has the best MMD² (0.0005) and precision (0.995) once codes are initialized from data and sampled from the empirical code prior, with 127 of 128 codes live. A uniform prior over a codebook wastes capacity and produces garbage.
- **Flow Matching (10 steps)** and its **1-step distilled student** land at MMD² 0.0014 vs 0.0016 — the distillation retains nearly all fidelity while removing 9 of 10 ODE evaluations. The pairing matters: the student must be trained on the teacher output for the *same* noise vector.

## Interpretation caveat

These are 2D demonstrations. The objectives and training loops transfer to images, but absolute numbers do not predict FID or CLIP scores at image scale. Replace the toy MLPs with conv/DiT backbones and the 2D data with latent patches to reproduce the modern systems described in `DEEP_DIVE.md`.

## Files

`DEEP_DIVE.md` · `CHEAT_SHEET.md` · `GENERATIVE_MODELS_LAB.ipynb` (executed, with plots) · `RESULTS.csv` · `RESULTS.md` · `images/`
