# PolyCrysDiff

This repository accompanies the paper [PolyCrysDiff: Controllable Generation of Three-Dimensional Computable Polycrystalline Material Structures](https://arxiv.org/abs/2603.11695). The code is provided here; the training datasets and typical results are linked in the Download assets section.

## Download assets

The four datasets used in this study are available separately: the [training set](https://drive.google.com/file/d/1wIIhjub3V68Hl_z4UvMFP3djjw3wIjwN/view?usp=drive_link), [validation set](https://drive.google.com/file/d/1EPTpswBEXCA55RRrB6dyWcGIVEOh3WdA/view?usp=drive_link), [sphericity training set](https://drive.google.com/file/d/1mnspCoOLVf4aNn3U6lqUgrJU6sM0hbNQ/view?usp=drive_link), and [enhanced-phase training set](https://drive.google.com/file/d/1bQETWO8UXGq2CkXzWJjMOo8u6mVGWTgE/view?usp=drive_link).

The [typical results](https://drive.google.com/file/d/1xNECG1OfPgbmbJMhYmG6P6lQ2L7QpaRc/view?usp=drive_link) and [pretrained VGG weights](https://drive.google.com/file/d/1dEQwUQUgBg5AMWIUQOrNV8rQA_zNF_Fn/view?usp=drive_link) used for evaluation are also available separately.

## Setup

Use Python 3.8 with a PyTorch/CUDA build appropriate for your machine. Install the remaining dependencies with `pip install -r requirements.txt`.

```bash
git clone https://github.com/kanchi-c/PolyCrysDiff.git
cd PolyCrysDiff
pip install -r requirements.txt
```

## Configurations and commands

| Configuration | Purpose | Output directory |
| --- | --- | --- |
| `config/var_grain.yaml` | Main autoencoder and unconditional diffusion | `out/var_grain_more/` |
| `config/var_grain_cond.yaml` | Grain-count conditioning | `out/var_grain_more/` |
| `config/var_enhance.yaml` | Enhanced-phase example | `out/var_enhance/` |
| `config/var_grain_vae_config.yaml` | KL-regularized VAE baseline | `out/vae_var_grain/` |
| `config/var_grain_sph_cond.yaml` | **Joint** grain-count/sphericity experiment | `out/var_grain_sph/` |

Train the 3D VAE first, followed by the unconditional or conditional model:

The diffusion configurations below resume from existing checkpoints (`ldm_ckpt_epoch` is 400 or 600). To train either diffusion model from scratch, set its `ldm_ckpt_epoch` to `0` in the corresponding configuration. Sampling requires the checkpoints named in the configuration.

```bash
# Train the 3D VAE.
python -m tools.train_vae3d_KL --config config/var_grain.yaml

# Train the unconditional model.
python -m tools.train_ddpm_vae3d --config config/var_grain.yaml

# Train the conditional model.
python -m tools.train_ddpm_vae3d_cond --config config/var_grain_cond.yaml

# Sampling requires the corresponding downloaded checkpoints.
# Sample from the unconditional model.
python -m tools.sample_ddpm_vae3d --config config/var_grain.yaml --num 1

# Generate two grain-count-conditioned samples and timing data (CUDA required).
python -m tools.benchmark_conditional_generation --config config/var_grain_cond.yaml --num 2
```


## Acknowledgement

This implementation builds on [explainingai-code/StableDiffusion-PyTorch](https://github.com/explainingai-code/StableDiffusion-PyTorch). Its license and copyright notice are retained in `LICENSE`.
