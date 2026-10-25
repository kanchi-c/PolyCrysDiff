import argparse
import csv
import json
import os
import platform
import time

import numpy as np
import torch
import yaml

from models.unet3D import Unet3D
from models.vae3d import VAE3d
from scheduler.linear_noise_scheduler import LinearNoiseScheduler


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_models(config, device):
    diffusion_model_config = config["ldm_params"]
    autoencoder_model_config = config["autoencoder_params"]
    dataset_config = config["dataset_params"]
    train_config = config["train_params"]

    model = Unet3D(
        im_channels=autoencoder_model_config["z_channels"],
        model_config=diffusion_model_config,
    ).to(device)
    model_path = os.path.join(
        train_config["task_name"],
        f'{train_config["ldm_ckpt_name"]}{train_config["ldm_ckpt_epoch"]}.pth',
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    autoencoder = VAE3d(
        im_channels=dataset_config["im_channels"],
        model_config=autoencoder_model_config,
    ).to(device)
    autoencoder_path = os.path.join(
        train_config["task_name"],
        f'{train_config["vae_autoencoder_ckpt_name"]}{train_config["vae_ckpt_epoch"]}.pth',
    )
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    autoencoder.eval()

    return model, autoencoder, model_path, autoencoder_path


@torch.inference_mode()
def generate_one(
    model,
    autoencoder,
    noise_scheduler,
    latent_shape,
    condition_value,
    num_timesteps,
    device,
):
    latent = torch.randn(latent_shape, device=device)
    condition = {
        "value": torch.tensor([[condition_value]], dtype=torch.float32, device=device)
    }

    for timestep in reversed(range(num_timesteps)):
        t_batch = torch.tensor([timestep], device=device)
        predicted_noise = model(latent, t_batch, condition)
        latent, _ = noise_scheduler.sample_prev_timestep(
            latent, predicted_noise, torch.tensor(timestep, device=device)
        )

    generated = autoencoder.decode(latent)
    generated = torch.clamp(generated[0], -1.0, 1.0)
    generated = ((generated + 1.0) / 2.0).permute(1, 2, 3, 0)
    return generated.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark conditional PolyCrysDiff generation time."
    )
    parser.add_argument("--config", default="config/var_grain_cond.yaml")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="out/var_grain_more/conditional_generation_timing",
    )
    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("--num must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark")

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda")
    seed = int(config["train_params"].get("seed", 1111))
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model, autoencoder, model_path, autoencoder_path = load_models(config, device)
    diffusion_config = config["diffusion_params"]
    autoencoder_config = config["autoencoder_params"]
    dataset_config = config["dataset_params"]
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )

    downsample_factor = 2 ** sum(bool(x) for x in autoencoder_config["down_sample"])
    latent_size = dataset_config["im_size"] // downsample_factor
    latent_shape = (
        1,
        autoencoder_config["z_channels"],
        latent_size,
        latent_size,
        latent_size,
    )

    value_config = config["ldm_params"]["condition_config"]["value_condition_config"]
    condition_key = value_config["keys"][0]
    configured_values = value_config.get("infer", [])
    if len(configured_values) >= args.num:
        condition_values = configured_values[: args.num]
    else:
        condition_values = [150.0] * args.num

    # Warm up CUDA kernels without producing or timing an additional sample.
    warmup_latent = torch.randn(latent_shape, device=device)
    warmup_condition = {
        "value": torch.tensor(
            [[float(condition_values[0])]], dtype=torch.float32, device=device
        )
    }
    with torch.inference_mode():
        _ = model(
            warmup_latent,
            torch.tensor([diffusion_config["num_timesteps"] - 1], device=device),
            warmup_condition,
        )
        _ = autoencoder.decode(warmup_latent)
    synchronize(device)

    records = []
    for index, condition_value in enumerate(condition_values):
        synchronize(device)
        start = time.perf_counter()
        sample = generate_one(
            model=model,
            autoencoder=autoencoder,
            noise_scheduler=noise_scheduler,
            latent_shape=latent_shape,
            condition_value=float(condition_value),
            num_timesteps=diffusion_config["num_timesteps"],
            device=device,
        )
        synchronize(device)
        inference_seconds = time.perf_counter() - start

        output_path = os.path.join(
            output_dir, f"model{index}_{condition_key}_{condition_value}.npy"
        )
        save_start = time.perf_counter()
        np.save(output_path, sample)
        save_seconds = time.perf_counter() - save_start

        record = {
            "sample": index,
            "condition_name": condition_key,
            "condition_value": condition_value,
            "inference_seconds": inference_seconds,
            "save_seconds": save_seconds,
            "output_file": os.path.basename(output_path),
        }
        records.append(record)
        print(
            f"Sample {index + 1}/{args.num}: {inference_seconds:.3f} s "
            f"(condition {condition_key}={condition_value})",
            flush=True,
        )

    csv_path = os.path.join(output_dir, "generation_times.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    inference_times = np.array(
        [record["inference_seconds"] for record in records], dtype=np.float64
    )
    summary = {
        "num_samples": args.num,
        "mean_seconds": float(inference_times.mean()),
        "standard_deviation_seconds": float(inference_times.std(ddof=1))
        if args.num > 1
        else 0.0,
        "minimum_seconds": float(inference_times.min()),
        "maximum_seconds": float(inference_times.max()),
        "median_seconds": float(np.median(inference_times)),
        "timing_scope": (
            "1000 reverse-diffusion steps, final autoencoder decoding, "
            "and CPU transfer; model loading and file writing excluded"
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": seed,
        "config": os.path.abspath(args.config),
        "diffusion_checkpoint": os.path.abspath(model_path),
        "autoencoder_checkpoint": os.path.abspath(autoencoder_path),
    }
    summary_path = os.path.join(output_dir, "timing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
