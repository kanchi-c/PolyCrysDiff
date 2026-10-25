import matplotlib.pyplot as plt
import pandas as pd
import torch
import argparse
import yaml
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.unet3D import Unet3D as Unet3D
from models.vae3d import VAE3d
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, name, cond_input=None):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size, im_size, im_size)).to(device)
    save_count = 0
    save_steps = diffusion_config['sample_save_steps']
    if not os.path.exists(os.path.join(train_config['task_name'], f'samples_{name}')):
        os.mkdir(os.path.join(train_config['task_name'], f'samples_{name}'))


    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), cond_input)

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        if i%save_steps == 0 or i==diffusion_config['num_timesteps']-1:
            # np.save(os.path.join(train_config['task_name'], f'samples_{name}/latent_{i}.npy'), xt.squeeze().permute(1, 2, 3, 0).cpu())
            ims = vae.decode(xt)
            ims = torch.clamp(ims[0], -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            ims = ims.permute(1, 2, 3, 0).cpu()
            np.save(os.path.join(train_config['task_name'], f'samples_{name}/test_{i}.npy'), ims)


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    model = Unet3D(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()

    ckpt_file = train_config['ldm_ckpt_name'] + str(train_config["ldm_ckpt_epoch"]) + ".pth"
    ckpt_path = os.path.join(train_config['task_name'], ckpt_file)
    if os.path.exists(ckpt_path):
        print(f'Loaded pretrained unet checkpoint: {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Can not find the pretrained ldm checkpoint file: {ckpt_path}.")

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    vae = VAE3d(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()

    # Load vae if found
    vae_file = train_config['vae_autoencoder_ckpt_name'] + str(train_config['vae_ckpt_epoch']) + ".pth"
    vae_path = os.path.join(train_config['task_name'], vae_file)
    if os.path.exists(vae_path):
        print('Loaded vae checkpoint: ' + vae_path)
        state_dict = torch.load(vae_path, map_location=device)
        vae.load_state_dict(state_dict)
    else:
        raise ValueError("Can not find the vae checkpoint file.")
    vae.eval()

    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    cond_inputs = None
    nums = args.infer_number

    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

        if 'image' in condition_types:
            im_condition_config = condition_config['image_condition_config']
            transform = transforms.Compose([
                transforms.ToTensor(),  # 将图片转换为 [0, 1] 的 Tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将 [0, 1] 转换为 [-1, 1]
            ])

            im = Image.open(im_condition_config['infer_im_dir'])
            im_tensor = transform(im)[None, ...].to(device)
            cond_input = im_tensor.to(device)

        if 'class' in condition_types:
            class_condition_config = condition_config['class_condition_config']
            temp = torch.zeros(class_condition_config['num_classes'])
            infer_class = class_condition_config['infer_class']
            temp[infer_class] = 1
            cond_input = torch.tensor(temp).unsqueeze(0).to(device)

        if 'value' in condition_types:
            value_condition_config = condition_config['value_condition_config']

            # value condition arr are given be config file
            # info_arr = np.array(value_condition_config['infer'])
            # info_arr = info_arr.astype(np.float32) # (n, cond_input_dim)

            # sph_mean = 0.7684357651064376
            # sph_std = 0.018839151612943705
            # ratios_mean = 1.828678747187789
            # ratios_std = 0.1663598292187173
            # v_mean = 0
            # v_std = 1

            attr_means = {"sph": 0.7684357651064376, "seeds_num": 0, "ratio": 1.828678747187789}
            attr_stds = {"sph": 0.018839151612943705, "seeds_num": 1, "ratio": 0.1663598292187173}

            info_keys = value_condition_config['keys']
            assert len(info_keys) == value_condition_config['len_vector']

            info_arr = np.array([[np.random.randint(50, 300), 0.65 + 0.2*np.random.rand()] for i in range(nums)])
            info_means = np.array([attr_means[key] for key in info_keys])
            info_stds = np.array([attr_stds[key] for key in info_keys])

            print(info_means.shape)
            print(info_stds.shape)

            scaled_info_arr = (info_arr - info_means) / info_stds
            info_arr = info_arr.astype(np.float32)
            scaled_info_arr = scaled_info_arr.astype(np.float32)
            print(info_arr.shape)
            print(info_arr)
            print(scaled_info_arr)
            cond_inputs = [{'value': torch.tensor(info_item).unsqueeze(0).to(device)} for info_item in scaled_info_arr]


    if condition_config is None:
        assert cond_inputs is None
        for i in range(nums):
            with torch.no_grad():
                sample(model, scheduler, train_config, diffusion_model_config,
                       autoencoder_model_config, diffusion_config, dataset_config, vae, i)
    else:
        assert len(cond_inputs) == nums
        for i, info in enumerate(info_arr):
            name = ""
            if 'value' in condition_config['condition_types']:
                name = f"{i}"
                for item in info:
                    if isinstance(item, int):
                        name += f"_{item}"
                    else:
                        name += f"_{item:.4f}"

            else:
                name = str(i)

            with torch.no_grad():
                sample(model, scheduler, train_config, diffusion_model_config,
                       autoencoder_model_config, diffusion_config, dataset_config, vae, name, cond_inputs[i])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/var_grain.yaml', type=str)
    parser.add_argument('--num', dest='infer_number',
                        default=1, type=int)
    args = parser.parse_args()
    infer(args)
