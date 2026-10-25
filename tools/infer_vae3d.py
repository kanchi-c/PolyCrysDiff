import argparse
import glob
import numpy  as np
import os
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from utils.post_utils import visual

from dataset.class_dataset import Class_Dataset
from dataset.poly_dataset import Poly_Dataset
from models.vae3d import VAE3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the dataset
    im_dataset_cls = {
        'class': Class_Dataset,
        'poly': Poly_Dataset,
    }.get(dataset_config['name'])

    if im_dataset_cls is None:
        raise ValueError(
            f"Unknown dataset name: {dataset_config['name']}. "
            "Supported: class, poly."
        )

    dataset = im_dataset_cls(split='train',
                                root_dir=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                debug=False)

    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)

    model = VAE3d(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)


    vae_file = train_config['vae_autoencoder_ckpt_name'] + str(train_config['vae_ckpt_epoch']) + ".pth"
    vae_path = os.path.join(train_config['task_name'], vae_file)
    if os.path.exists(vae_path):
        print('Loaded vae checkpoint: ' + vae_path)
        state_dict = torch.load(vae_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Can not find the vae checkpoint file.")
    model.eval()

    save_dir = os.path.join(train_config['task_name'], "vae_reconstruction_test")
    os.makedirs(save_dir, exist_ok=True)

    dataset_size = len(dataset)
    random_indices = np.random.choice(dataset_size, args.num, replace=False)

    with torch.no_grad():
        for i, index in enumerate(random_indices):
            # Get original data
            original_data = dataset[index][None, ...].to(device)
            # (b, c, h, w, d)

            # Save original data
            original_data_np = original_data.squeeze(0).cpu().numpy()
            original_data_np = (original_data_np + 1) / 2  # Rescale to [0, 1]
            np.save(os.path.join(save_dir, f"original_{i}.npy"), original_data_np.transpose(1,2,3,0))
            # (h, w, d, c)


            # Get reconstructed data
            reconstructed_data, _, _, _ = model(original_data)
            print(reconstructed_data.shape)
            # Save reconstructed data
            reconstructed_data_np = reconstructed_data.squeeze(0).cpu().numpy()
            reconstructed_data_np = (reconstructed_data_np + 1) / 2  # Rescale to [0, 1]
            np.save(os.path.join(save_dir, f"reconstructed_{i}.npy"), reconstructed_data_np.clip(0, 1).transpose(1,2,3,0))

    print(f"Saved {args.num} original and reconstructed samples to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/var_grain.yaml', type=str)
    parser.add_argument('--num', dest='num',
                        default=2, type=int)
    args = parser.parse_args()
    infer(args)
