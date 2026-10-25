import argparse
import json
import os
import tempfile
import time

import numpy as np
import torch
import yaml

from models.vae3d import VAE3d


def main():
    parser = argparse.ArgumentParser(
        description="Measure the extra decode/CPU-transfer/NPY-write overhead in sampling."
    )
    parser.add_argument("--config", default="config/var_grain_cond.yaml")
    parser.add_argument("--num-extra-saves", type=int, default=20)
    parser.add_argument(
        "--report",
        default="out/var_grain_more/conditional_generation_timing/io_overhead_estimate.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark")

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda")
    dataset_config = config["dataset_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_params"]

    autoencoder = VAE3d(
        im_channels=dataset_config["im_channels"],
        model_config=autoencoder_config,
    ).to(device)
    checkpoint = os.path.join(
        train_config["task_name"],
        f'{train_config["vae_autoencoder_ckpt_name"]}{train_config["vae_ckpt_epoch"]}.pth',
    )
    autoencoder.load_state_dict(torch.load(checkpoint, map_location=device))
    autoencoder.eval()

    downsample_factor = 2 ** sum(bool(x) for x in autoencoder_config["down_sample"])
    latent_size = dataset_config["im_size"] // downsample_factor
    latent = torch.randn(
        (1, autoencoder_config["z_channels"], latent_size, latent_size, latent_size),
        device=device,
    )

    with torch.inference_mode():
        _ = autoencoder.decode(latent)
    torch.cuda.synchronize()

    individual_seconds = []
    report_dir = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(report_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=report_dir) as temp_dir:
        with torch.inference_mode():
            for index in range(args.num_extra_saves):
                torch.cuda.synchronize()
                start = time.perf_counter()
                decoded = autoencoder.decode(latent)
                decoded = torch.clamp(decoded[0], -1.0, 1.0).detach().cpu()
                decoded = ((decoded + 1.0) / 2.0).permute(1, 2, 3, 0).numpy()
                np.save(os.path.join(temp_dir, f"intermediate_{index}.npy"), decoded)
                elapsed = time.perf_counter() - start
                individual_seconds.append(elapsed)

    values = np.asarray(individual_seconds, dtype=np.float64)
    report = {
        "device": torch.cuda.get_device_name(device),
        "num_extra_saves": args.num_extra_saves,
        "total_extra_seconds": float(values.sum()),
        "mean_seconds_per_decode_and_save": float(values.mean()),
        "standard_deviation_seconds": float(values.std(ddof=1))
        if len(values) > 1
        else 0.0,
        "minimum_seconds": float(values.min()),
        "maximum_seconds": float(values.max()),
        "operations_included": "autoencoder decode, clamp/rescale, GPU-to-CPU transfer, and NPY write",
        "checkpoint": os.path.abspath(checkpoint),
    }
    with open(args.report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
