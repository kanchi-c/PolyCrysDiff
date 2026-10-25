import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vae3d import VAE3d


def parse_args():
    parser = argparse.ArgumentParser(description='Sample a trained 3D VAE prior.')
    parser.add_argument('--config', dest='config_path',
                        default='config/var_grain_vae_config.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output-dir', default='out/var_grain_vae_samples')
    parser.add_argument('--num', type=int, default=200)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def choose_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return device


def main():
    args = parse_args()
    if args.num <= 0 or args.batch_size <= 0 or args.start_index < 0:
        raise ValueError(
            '--num and --batch-size must be positive; '
            '--start-index must be non-negative.')

    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_name = (
            train_config['vae_autoencoder_ckpt_name']
            + str(train_config['vae_ckpt_epoch'])
            + '.pth'
        )
        checkpoint_path = os.path.join(train_config['task_name'], checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('VAE checkpoint not found: ' + checkpoint_path)

    model = VAE3d(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_config,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    downsample_factor = 2 ** sum(bool(value) for value in autoencoder_config['down_sample'])
    latent_size = dataset_config['im_size'] // downsample_factor
    latent_channels = autoencoder_config['z_channels']
    os.makedirs(args.output_dir, exist_ok=True)

    written = 0
    with torch.no_grad():
        while written < args.num:
            current_batch = min(args.batch_size, args.num - written)
            latent = torch.randn(
                current_batch,
                latent_channels,
                latent_size,
                latent_size,
                latent_size,
                device=device,
            )
            generated = model.decode(latent).clamp(-1, 1)
            generated = generated.add(1).div(2).cpu().numpy()
            for batch_index in range(current_batch):
                sample = generated[batch_index].transpose(1, 2, 3, 0).astype(np.float32)
                sample_index = args.start_index + written + batch_index
                np.save(os.path.join(args.output_dir, f'model{sample_index}.npy'), sample)
            written += current_batch

    final_index = args.start_index + written - 1
    print(
        f'Saved {written} raw VAE samples '
        f'(model{args.start_index}.npy--model{final_index}.npy) '
        f'to {args.output_dir}'
    )


if __name__ == '__main__':
    main()
