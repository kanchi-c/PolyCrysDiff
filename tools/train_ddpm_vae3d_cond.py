import os
import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.class_dataset import Class_Dataset
from dataset.poly_dataset import Poly_Dataset
from torch.utils.data import DataLoader
from models.unet3D import Unet3D as Unet3D
from models.vae3d import VAE3d
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
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

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################

    # Instantiate Condition related components
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    im_dataset_cls = {
        'class': Class_Dataset,
        'poly': Poly_Dataset,
    }.get(dataset_config['name'])

    if im_dataset_cls is None:
        raise ValueError(
            f"Unknown dataset name: {dataset_config['name']}. "
            "Supported: class, poly."
        )

    im_dataset = im_dataset_cls(split='train',
                                root_dir=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                debug=train_config['debug'],
                                condition_config=condition_config,
                                )

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)

    # Instantiate the unet model
    model = Unet3D(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)

    # load a pretrained ldm model if existed
    if train_config["ldm_ckpt_epoch"] > 0:
        ckpt_file = train_config['ldm_ckpt_name'] + str(train_config["ldm_ckpt_epoch"]) + ".pth"
        ckpt_path = os.path.join(train_config['task_name'], ckpt_file)
        if os.path.exists(ckpt_path):
            print(f'Loaded pretrained unet checkpoint: {ckpt_path}')
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Can not find the pretrained ldm checkpoint file: {ckpt_path}.")
    else:
        print("Start a new training.")
    model.train()


    # Load VAE
    vae_file = train_config['vae_autoencoder_ckpt_name'] + str(train_config['vae_ckpt_epoch']) + ".pth"
    vae_path = os.path.join(train_config['task_name'], vae_file)
    vae = VAE3d(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M parameters in total for Unet Model")
    print(f"{sum(p.numel() for p in vae.parameters()) / 1e6} M parameters in total for VAE Model")



    # Load vae if found
    if os.path.exists(vae_path):
        print('Loaded vae checkpoint: ' + vae_path)
        state_dict = torch.load(vae_path, map_location=device)
        vae.load_state_dict(state_dict)
    else:
        raise ValueError("Can not find the vae checkpoint file.")
    vae.eval()

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    save_steps = train_config['ldm_ckpt_save_epochs']
    start_epoch = train_config["ldm_ckpt_epoch"]
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Freeze vae parameters
    for param in vae.parameters():
        param.requires_grad = False

    # Run training
    losses = []
    if start_epoch <= 0:
        start_epoch = 0
    for epoch_idx in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        epoch_losses = []
        for data in tqdm(data_loader):
            im, cond_input = data

            optimizer.zero_grad()
            im = im.float().to(device)

            with torch.no_grad():
                mean, logvar = vae.encode(im)
                im = vae.reparameterize(mean, logvar)

            ########### Handling Conditional Input ###########
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                cond_input['image'] = cond_input['image'].to(device)
            if 'class' in condition_types:
                assert 'class' in cond_input
                cond_input['class'] = cond_input['class'].to(device)
            if "value" in condition_types:
                assert 'value' in cond_input, 'Conditioning Type Value but no value conditioning input present'
                cond_input['value'] = cond_input['value'].to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx,
            np.mean(epoch_losses)))
        losses.append([epoch_idx, np.mean(epoch_losses)])

        if epoch_idx%save_steps == 0:
            ckpt_file = train_config['ldm_ckpt_name'] + str(epoch_idx) + ".pth"
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], ckpt_file))

    loss_file = f"cond_diffusion_loss_{start_epoch+1}_{start_epoch+num_epochs}.npy"
    loss_path = os.path.join(train_config['task_name'], loss_file)
    np.save(loss_path, np.array(losses))
    print(f'Loss save in {loss_file} \nDone Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/var_grain_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)
