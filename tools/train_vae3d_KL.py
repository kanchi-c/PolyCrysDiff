import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm

from models.vae3d import VAE3d
from torch.utils.data.dataloader import DataLoader
from dataset.class_dataset import Class_Dataset
from dataset.poly_dataset import Poly_Dataset
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def KL_divergence(mean, logvar, reduction='sum'):
    loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError("kl_reduction must be either 'mean' or 'sum'.")


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    #############################

    # Create the model and dataset #
    model = VAE3d(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
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

    im_dataset = im_dataset_cls(split='train',
                                root_dir=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                debug=train_config['debug'])
    print(f"{len(im_dataset)} is loaded.")
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # KL Loss to control latent space


    # optimizer
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    disc_step_start = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    ckpt_save_epochs = train_config['autoencoder_ckpt_save_epochs']

    KL_coeff = train_config["kl_weight"]
    KL_reduction = train_config.get("kl_reduction", "sum")
    KL_warmup_epochs = train_config.get("kl_warmup_epochs", 0)
    all_losses = []
    losses1 = [] # for reconstruction loss
    losses2 = [] # for KL divergence loss
    for epoch_idx in range(1, num_epochs + 1):
        recon_losses = []
        kl_losses = []

        if KL_warmup_epochs > 0:
            current_KL_coeff = KL_coeff * min(1.0, epoch_idx / KL_warmup_epochs)
        else:
            current_KL_coeff = KL_coeff

        optimizer_g.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, mean, logvar, z = model_output

            # L2 Loss
            recon_loss = recon_criterion(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps

            # KL Loss
            kl_loss = current_KL_coeff * KL_divergence(mean, logvar, KL_reduction)
            kl_losses.append(kl_loss.item())
            kl_loss = kl_loss / acc_steps

            all_loss = kl_loss + recon_loss
            all_loss.backward()


            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        # this is for the last several steps that cannot be evenly divided in this epoch
        optimizer_g.step()
        optimizer_g.zero_grad()

        loss1 = np.mean(recon_losses)
        loss2 = np.mean(kl_losses)
        losses1.append(loss1)
        losses2.append(loss2)
        all_losses.append(loss1 + loss2)

        print('Finished epoch: {} | Recon Loss : {:.4f} | KL Loss : {:.4f} | KL Weight : {:.6g}'.
              format(epoch_idx, loss1, loss2, current_KL_coeff))

        if epoch_idx % ckpt_save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['vae_autoencoder_ckpt_name']
                                                        + str(epoch_idx) + ".pth"))

    np.save(os.path.join(train_config['task_name'], "recon_loss.npy"), np.array(losses1))
    np.save(os.path.join(train_config['task_name'], "kl_loss.npy"), np.array(losses2))
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/var_grain.yaml', type=str)
    args = parser.parse_args()
    train(args)
