import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.ticker import PercentFormatter
# import seaborn as sns
from scipy.stats import gaussian_kde, ks_2samp, linregress, wasserstein_distance, entropy

from scipy.io import savemat
from utils.post_utils import *


import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from scipy.stats import entropy
import random

def compare_distributions(a, b, bins=200):
    # KS
    ks_stat, ks_p = ks_2samp(a, b)

    # KL using histograms
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    epsilon = 1e-12
    hist_a = hist_a + epsilon
    hist_b = hist_b + epsilon
    kl = entropy(hist_a, hist_b)

    # raw EMD
    emd_raw = wasserstein_distance(a, b)

    # normalized EMD in [0,1]
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    a_norm = (a - lo) / (hi - lo)
    b_norm = (b - lo) / (hi - lo)
    emd_norm = wasserstein_distance(a_norm, b_norm)

    return {
        "KS": ks_stat,
        # "KS_p": ks_p,
        # "KL": kl,
        # "EMD_raw": emd_raw,
        "EMD_norm": emd_norm
    }


def plot_vae_training_losses(
        recon_file="./out/var_grain_vae_prior/recon_loss.npy",
        kl_file="./out/var_grain_vae_prior/kl_loss.npy",
        output_dir="./figs/vae-var-grain"):
    """Plot the epoch-level reconstruction and weighted KL losses."""
    recon_loss = np.load(recon_file)
    kl_loss = np.load(kl_file)
    if recon_loss.ndim != 1 or kl_loss.ndim != 1:
        raise ValueError("VAE loss histories must be one-dimensional arrays.")
    if len(recon_loss) != len(kl_loss):
        raise ValueError("VAE reconstruction and KL histories have different lengths.")

    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    epochs = np.arange(1, len(recon_loss) + 1)
    total_loss = recon_loss + kl_loss

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    # Match the standard 6.4 x 4.8 inch panel used by the revision-enhance
    # loss figure.  Two such panels are arranged horizontally.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    axes[0].plot(epochs, total_loss, color="#315F8C", linewidth=2,
                 label="Total loss")
    axes[0].plot(epochs, recon_loss, color="#D9822B", linewidth=2,
                 label="Reconstruction loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)
    axes[0].text(-0.13, 1.02, "(a)", transform=axes[0].transAxes,
                 fontsize=16, fontweight="bold")

    axes[1].plot(epochs, kl_loss, color="#4C956C", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted KL loss")
    axes[1].text(-0.13, 1.02, "(b)", transform=axes[1].transAxes,
                 fontsize=16, fontweight="bold")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)
    fig.tight_layout()
    png_file = os.path.join(output_dir, "training_loss_curve.png")
    pdf_file = os.path.join(output_dir, "training_loss_curve.pdf")
    fig.savefig(png_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_file, pdf_file


def plot_training_loss_file(
        loss_file,
        output_dir,
        output_stem,
        ylabel,
        color="#315F8C"):
    """Plot a one-series epoch-level training loss with project styling.

    ``loss_file`` may contain either a one-dimensional loss history or an
    ``(n, 2)`` array whose columns are epoch and loss, respectively.
    """
    loss = np.load(loss_file)
    if loss.ndim == 1:
        epochs = np.arange(1, len(loss) + 1)
        values = loss
    elif loss.ndim == 2 and loss.shape[1] == 2:
        epochs = loss[:, 0]
        values = loss[:, 1]
    else:
        raise ValueError(
            f"Unsupported loss shape {loss.shape} in {loss_file}")

    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(epochs, values, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.tight_layout()

    png_file = os.path.join(output_dir, output_stem + ".png")
    pdf_file = os.path.join(output_dir, output_stem + ".pdf")
    fig.savefig(png_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_file, pdf_file


def plot_gan_training_losses(
        metrics_file="./out/gan_var_grain/metrics.csv",
        output_dir="./figs/gan-var-grain",
        smooth_window=100):
    """Plot raw and rolling-average WGAN-GP generator/critic losses."""
    metrics = pd.read_csv(metrics_file)
    required = {"step", "generator_loss", "critic_loss"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"GAN metrics are missing columns: {sorted(missing)}")
    metrics = metrics.dropna(subset=list(required))
    # Resumed or concatenated logs may contain repeated early steps. Preserve
    # the first complete training trajectory and avoid double weighting them.
    metrics = metrics.drop_duplicates(subset="step", keep="first").sort_values("step")
    if metrics.empty:
        raise ValueError("GAN metrics contain no valid loss records.")

    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    steps = metrics["step"].to_numpy() / 1000.0
    losses = [
        ("generator_loss", "Generator loss", "#315F8C"),
        ("critic_loss", "Critic loss", "#C44E52"),
    ]

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for panel, (axis, (column, label, color)) in enumerate(zip(axes, losses)):
        raw = metrics[column]
        smooth = raw.rolling(
            window=smooth_window, min_periods=max(2, smooth_window // 5),
            center=True).mean()
        axis.plot(steps, raw, color=color, linewidth=0.5, alpha=0.12,
                  label="Recorded loss")
        axis.plot(steps, smooth, color=color, linewidth=2,
                  label=f"Rolling mean ({smooth_window} records)")
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel(label)
        axis.legend(frameon=False)
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.text(-0.13, 1.02, f"({chr(97 + panel)})",
                  transform=axis.transAxes, fontsize=16, fontweight="bold")
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)
    axes[1].set_xlabel(r"Training step ($\times 10^3$)")
    fig.tight_layout()
    png_file = os.path.join(output_dir, "training_loss_curve.png")
    pdf_file = os.path.join(output_dir, "training_loss_curve.pdf")
    fig.savefig(png_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_file, pdf_file


def get_shared_bins(arr1, arr2, bins=30, give_min=None, give_max=None):
    """Compute common bin edges based on min/max of combined data."""
    combined_min = give_min
    combined_max = give_max
    if not give_min:
        combined_min = min(np.min(arr1), np.min(arr2))
    if not give_max:
        combined_max = max(np.max(arr1), np.max(arr2))
    return np.linspace(combined_min, combined_max, bins + 1)

# for fi1 1 plot
def visual_slice(model_file):
    # file = "./out/var_grain_more/diffusion400/samples_2/test_0.npy"
    # file = "./data/var_grain/model792.npy"
    data = np.load(model_file)

    ix, iy ,iz = np.random.randint(low=0, high=data.shape[0], size=3)

    x_slice = data[ix, :, :, :]
    y_slice = data[:, iy, :, :]
    z_slice = data[:, :, iz, :]
    slices = [x_slice, y_slice, z_slice]
    names = ["x", "y", "z"]

    for slice, name in zip(slices, names):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(slice)

        ax.set_position([0, 0, 1, 1])
        ax.set_xticks([])  # 隐藏 x 轴刻度
        ax.set_yticks([])  # 隐藏 y 轴刻度
        ax.spines['top'].set_visible(False)  # 隐藏上边框
        ax.spines['right'].set_visible(False)  # 隐藏右边框
        ax.spines['left'].set_visible(False)  # 隐藏左边框
        ax.spines['bottom'].set_visible(False)  # 隐藏下边框

        plt.savefig(f"./figs/fig1/{name}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f"./figs/fig1/{name}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

    visual(data, save_dir="./figs/fig1/gnerated_3d.pdf")

def plot_random_generation(
        reference_grain_file="./figs/dataset_grain_info.csv",
        generated_grain_file="./figs/generated_grain_info.csv",
        reference_sample_file="./data/var_grain/info.txt",
        generated_sample_file="./figs/generated_sample_info.csv",
        out_dir="./figs/uncondition",
        reference_label="Dataset"):
    """Plot grain statistics using the original manuscript figure style."""
    os.makedirs(out_dir, exist_ok=True)

    def select_column(frame, candidates):
        for candidate in candidates:
            if candidate in frame.columns:
                return np.asarray(frame[candidate])
        raise KeyError(f"None of {candidates} found in {list(frame.columns)}")

    # for per-model mean info
    data = pd.read_csv(reference_sample_file)
    grain_nums_dataset = select_column(data, ["seeds_num", "grain_number"])
    # mean_aspect_ratios_dataset = np.array(data.loc[:, 'aspect_ratio'])
    # mean_sphs_dataset = np.array(data.loc[:, 'sph'])

    # for per-model mean info
    data = pd.read_csv(generated_sample_file)
    grain_nums_generated = select_column(data, ["grain_nums_generated", "grain_number"])
    # mean_aspect_ratios_generated = np.array(data.loc[:, 'aspect_ratio'])
    # mean_sphs_generated = np.array(data.loc[:, 'sph'])


    # load grain info of dataset examples
    data = pd.read_csv(reference_grain_file)
    grain_sizes_dataset = select_column(data, ["grain_sizes_dataset", "grain_size"])
    aspect_ratios_dataset = select_column(data, ["aspect_ratios_dataset", "aspect_ratio"])
    sphs_dataset = select_column(data, ["sphs_dataset", "sphericity"])
    etas_111_dataset = select_column(data, ["etas_111_dataset", "eta_111"])

    # load grain info of generated examples
    data = pd.read_csv(generated_grain_file)
    grain_sizes_generated = select_column(data, ["grain_sizes_generated", "grain_size"])
    aspect_ratios_generated = select_column(data, ["aspect_ratios_generated", "aspect_ratio"])
    sphs_generated = select_column(data, ["sphs_generated", "sphericity"])
    etas_111_generated = select_column(data, ["etas_111_generated", "eta_111"])

    # do KS-test
    # print(compare_distributions(grain_nums_dataset, grain_nums_generated))
    print(compare_distributions(grain_sizes_dataset, grain_sizes_generated))
    print(compare_distributions(aspect_ratios_dataset, aspect_ratios_generated))
    print(compare_distributions(sphs_dataset, sphs_generated))
    print(compare_distributions(etas_111_dataset, etas_111_generated))

    plt.rcParams.update({
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 16,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2
    width = 0.8
    color1 = '#394E9D'
    color2 = '#C6857A'
    alpha = 0.5

    # --- Plot 1: Grain Size Distribution ---
    bins_size = get_shared_bins(grain_sizes_dataset, grain_sizes_generated, bins=30, give_max=8000)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax.hist(grain_sizes_generated, bins=bins_size, density=True, color=color2,
             alpha=alpha, label="Generated", width=width*(bins_size[1] - bins_size[0]))
    ax.hist(grain_sizes_dataset, bins=bins_size, density=True, color=color1,
            alpha=alpha, label=reference_label, width=width * (bins_size[1] - bins_size[0]))
    ax.set_xlabel("Grain Size")
    ax.set_ylabel("Probability Density")
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)
    ax.legend()

    plt.savefig(os.path.join(out_dir, f"grain_size.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"grain_size.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # --- Plot 2: Aspect Ratio Distribution ---
    bins_size = get_shared_bins(aspect_ratios_dataset, aspect_ratios_generated, bins=30)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(aspect_ratios_generated, bins=bins_size, density=True, color=color2,
            alpha=alpha, label="Generated", width=width * (bins_size[1] - bins_size[0]))
    ax.hist(aspect_ratios_dataset, bins=bins_size, density=True, color=color1,
            alpha=alpha, label=reference_label, width=width * (bins_size[1] - bins_size[0]))

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Probability Density")
    ax.legend()
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig(os.path.join(out_dir, f"aspect_ratio.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"aspect_ratio.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    # --- Plot 3: sphericity Distribution ---
    bins_size = get_shared_bins(sphs_dataset, sphs_generated, bins=30)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(sphs_generated, bins=bins_size, density=True, color=color2,
            alpha=alpha, label="Generated", width=width * (bins_size[1] - bins_size[0]))
    ax.hist(sphs_dataset, bins=bins_size, density=True, color=color1,
            alpha=alpha, label=reference_label, width=width * (bins_size[1] - bins_size[0]))

    ax.set_xlabel("Sphericity")
    ax.set_ylabel("Probability Density")
    ax.legend()
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig(os.path.join(out_dir, f"sphericity.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"sphericity.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # --- Plot 4: eta_111 Distribution ---
    bins_size = get_shared_bins(etas_111_dataset, etas_111_generated, bins=30, give_min=-0.02, give_max=0.02)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(etas_111_generated, bins=bins_size, density=True, color=color2,
            alpha=alpha, label="Generated", width=width * (bins_size[1] - bins_size[0]))
    ax.hist(etas_111_dataset, bins=bins_size, density=True, color=color1,
            alpha=alpha, label=reference_label, width=width * (bins_size[1] - bins_size[0]))

    ax.set_xlabel(r"$\eta_{111}$")
    ax.set_ylabel("Probability Density")
    ax.legend()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig(os.path.join(out_dir, f"eta_111.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"eta_111.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # --- Plot 5: Grain Number Distribution ---
    bins_size = get_shared_bins(grain_nums_dataset, grain_nums_generated, bins=15)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(grain_nums_generated, bins=bins_size, density=True, color=color2,
            alpha=alpha, label="Generated", width=width * (bins_size[1] - bins_size[0]))
    ax.hist(grain_nums_dataset, bins=bins_size, density=True, color=color1,
            alpha=alpha, label=reference_label, width=width * (bins_size[1] - bins_size[0]))

    ax.set_xlabel("Grain Number")
    ax.set_ylabel("Probability Density")
    ax.legend()
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig(os.path.join(out_dir, f"grain_num.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"grain_num.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    # --- Plot 5: Mean sph Distribution ---
    # bins_num = get_shared_bins(mean_sphs_dataset, mean_sphs_generated, bins=20)
    # plt.figure(figsize=(6, 4))
    # plt.hist(mean_sphs_dataset, bins=bins_num, density=True, color='tab:blue',
    #          alpha=0.5, label="Dataset", edgecolor='black')
    # plt.hist(mean_sphs_generated, bins=bins_num, density=True, color='tab:orange',
    #          alpha=0.5, label="Generated", edgecolor='black')
    # plt.xlabel("Mean sphericity")
    # plt.ylabel("Probability")
    # plt.title("Mean Sphericity Distribution Comparison")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"MeanSph.png"), dpi=300)
    # plt.close()

    # # --- Plot 6: Mean aspect_ratio Distribution ---
    # bins_num = get_shared_bins(mean_aspect_ratios_dataset, mean_aspect_ratios_generated, bins=20)
    # plt.figure(figsize=(6, 4))
    # plt.hist(mean_aspect_ratios_dataset, bins=bins_num, density=True, color='tab:blue',
    #          alpha=0.5, label="Dataset", edgecolor='black')
    # plt.hist(mean_aspect_ratios_generated, bins=bins_num, density=True, color='tab:orange',
    #          alpha=0.5, label="Generated", edgecolor='black')
    # plt.xlabel("Mean Aspect Ratio")
    # plt.ylabel("Probability")
    # plt.title("Mean Aspect Ratio Comparison")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"MeanAspectRatio.png"), dpi=300)
    # plt.close()

def VGG_plot(
        reference_file="./figs/training_vgg_scores.csv",
        generated_file="./figs/generated_vgg_scores.csv",
        out_dir="./figs/uncondition",
        reference_label="Dataset",
        output_stem="VGG_Score"):
    # plot VGG score
    scores1 = pd.read_csv(reference_file).loc[:, 'score']
    scores2 = pd.read_csv(generated_file).loc[:, 'score']
    bins_size = get_shared_bins(scores1, scores2, bins=25)

    print(compare_distributions(scores1, scores2))

    plt.rcParams.update({
        'font.family': 'Arial',
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 18,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2

    fig, ax = plt.subplots()
    width = 0.9 * (bins_size[1] - bins_size[0])

    ax.hist([scores1, scores2], bins=bins_size, density=True,  color=['#CCE1B0', '#8AAFD4'],
            label=[reference_label, "Generated"])
    # ax.hist(scores2, bins=bins_size, density=True,  color='#94A0D1', width=width,
    #         label="Generated")

    kde1 = gaussian_kde(scores1)
    kde2 = gaussian_kde(scores2)
    x_range = np.linspace(bins_size[0], bins_size[-1], 100)
    # plt.plot(x_range, kde1(x_range), c='#00ff00')
    # plt.plot(x_range, kde2(x_range), c='#3D80C2')


    ax.set_xlabel("VGG Score")
    ax.set_ylabel("Probability Density")
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.8)

    # plt.show()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{output_stem}.png"), dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"{output_stem}.pdf"), dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def S2_plot(
        file="./figs/S2_data.csv",
        out_dir="./figs/fig1",
        reference_label="Dataset",
        output_stem="S2",
        save_eps=True):
    data = pd.read_csv(file)

    r = data.loc[:, 'r']
    reference_suffix = "validation" if "mean_validation" in data.columns else "dataset"
    mean_dataset = data.loc[:, f'mean_{reference_suffix}']
    min_dataset = data.loc[:, f'min_{reference_suffix}']
    max_dataset = data.loc[:, f'max_{reference_suffix}']
    mean_generated = data.loc[:, 'mean_generated']
    min_generated = data.loc[:, 'min_generated']
    max_generated = data.loc[:, 'max_generated']

    plt.rcParams.update({
        'font.family': 'Arial',
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 18,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks
    })

    fig, ax = plt.subplots()
    border_width = 2

    # Dataset: plot mean and min–max band
    ax.plot(r, mean_dataset, c='yellow', label=f"{reference_label} (mean)")
    ax.fill_between(r, min_dataset, max_dataset, color='#fcf3a1', label=f"{reference_label} range")

    # Generated: plot mean and min–max band
    ax.plot(r, mean_generated, c='#e52421', label="Generated (mean)")
    ax.fill_between(r, min_generated, max_generated, color='#f0a498', label="Generated range")

    ax.set_xlabel("Distance (voxels)")
    ax.set_ylabel("S$_2$(r)")
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.8)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{output_stem}.png"), dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(out_dir, f"{output_stem}.pdf"), dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    if save_eps:
        plt.savefig(os.path.join(out_dir, f"{output_stem}.eps"), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)

def RGB_plot():
    info_file = "./figs/RGB_data.csv"
    # 2 for dataset, 1 for generated
    data = pd.read_csv(info_file)
    R_cdf1 = data['R_cdf1']
    G_cdf1 = data['G_cdf1']
    B_cdf1 = data['B_cdf1']

    R_cdf2 = data['R_cdf2']
    G_cdf2 = data['G_cdf2']
    B_cdf2 = data['B_cdf2']

    plt.rcParams.update({
        'font.family': 'Arial',
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 18,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2


    fig = plt.figure(figsize=(6.4*3, 4.8))

    ax1 = fig.add_subplot(131)
    xs = np.arange(1, 256)
    ax1.fill_between(xs, R_cdf2, R_cdf2.min()-1, color='#F7B49D', label='Dataset')
    ax1.scatter(xs[::2], R_cdf1[::2], c='red', label='Generated', s=2)
    ax1.set_xlabel("R channel")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_xlim(0, 255.5)
    ax1.legend()

    for spine in ax1.spines.values():  # 四条一起改
        spine.set_color('red')
        spine.set_linewidth(border_width)

    ax2 = fig.add_subplot(132)
    ax2.fill_between(xs, G_cdf2, G_cdf2.min()-1, color='#AED184', label='Dataset')
    ax2.scatter(xs[::2], G_cdf1[::2], c='green', label='Generated', s=2)
    ax2.set_xlabel("G channel")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.01, 1.01)
    ax2.set_xlim(0, 255.5)
    ax2.legend()

    for spine in ax2.spines.values():  # 四条一起改
        spine.set_color('green')
        spine.set_linewidth(border_width)

    ax3 = fig.add_subplot(133)
    ax3.fill_between(xs, B_cdf2, B_cdf2.min()-1, color='#6D83C0', label='Dataset')
    ax3.scatter(xs[::2], B_cdf1[::2], c='blue', label='Generated', s=2)
    ax3.set_xlabel("B channel")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(-0.01, 1.01)
    ax3.set_xlim(0, 255.5)
    ax3.legend()

    for spine in ax3.spines.values():  # 四条一起改
        spine.set_color('blue')
        spine.set_linewidth(border_width)

    plt.savefig("./figs/fig1/RGB_comparison.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig("./figs/fig1/RGB_comparison.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)


def orientation_fraction_plot(
        reference_csv="./figs/orientation_distribution/validation_orientation_fractions.csv",
        generated_csv="./figs/orientation_distribution/generated_orientation_fractions.csv",
        output_dir="./figs/orientation_distribution",
        reference_label="Validation",
        output_stem="orientation_fraction_comparison"):
    """Plot mean orientation fractions for a reference and generated set."""
    reference = pd.read_csv(reference_csv)
    generated = pd.read_csv(generated_csv)
    expected_columns = [f"orientation_{i}" for i in range(1, 11)]

    if list(reference.columns) != expected_columns:
        raise ValueError(f"Unexpected reference columns: {list(reference.columns)}")
    if list(generated.columns) != expected_columns:
        raise ValueError(f"Unexpected generated columns: {list(generated.columns)}")

    reference_mean = reference.mean(axis=0).to_numpy() * 100.0
    generated_mean = generated.mean(axis=0).to_numpy() * 100.0
    colors = get_colors(10)

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    x = np.arange(10)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x - width / 2, reference_mean, width,
           color=colors, edgecolor="black", linewidth=0.8)
    ax.bar(x + width / 2, generated_mean, width,
           color=colors, edgecolor="black", linewidth=0.8, hatch="///")

    ax.set_xlabel("Orientation Class")
    ax.set_ylabel("Mean Voxel Fraction")
    ax.set_xticks(x, [str(i) for i in range(1, 11)])
    ax.set_xlim(-0.65, 9.65)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    y_max = max(reference_mean.max(), generated_mean.max())
    for index, (reference_value, generated_value) in enumerate(
            zip(reference_mean, generated_mean)):
        difference = generated_value - reference_value
        ax.text(index, max(reference_value, generated_value) + 0.22,
                rf"$\Delta${difference:+.2f}%", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0, y_max * 1.16)
    ax.legend(handles=[
        Patch(facecolor="0.7", edgecolor="black", label=reference_label),
        Patch(facecolor="0.7", edgecolor="black", hatch="///", label="Generated"),
    ], frameon=True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)

    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{output_stem}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(output_dir, f"{output_stem}.pdf"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return reference_mean, generated_mean


def plot_evaluation(evaluation_dir="./figs/evaluation"):
    """Redraw validation-versus-generated panels from evaluation CSV files."""
    plot_random_generation(
        reference_grain_file=os.path.join(evaluation_dir, "validation_grain_info.csv"),
        generated_grain_file=os.path.join(evaluation_dir, "generated_grain_info.csv"),
        reference_sample_file=os.path.join(evaluation_dir, "validation_sample_info.csv"),
        generated_sample_file=os.path.join(evaluation_dir, "generated_sample_info.csv"),
        out_dir=evaluation_dir,
        reference_label="Validation",
    )
    VGG_plot(
        reference_file=os.path.join(evaluation_dir, "validation_vgg_scores.csv"),
        generated_file=os.path.join(evaluation_dir, "generated_vgg_scores.csv"),
        out_dir=evaluation_dir,
        reference_label="Validation",
        output_stem="vgg_score",
    )
    S2_plot(
        file=os.path.join(evaluation_dir, "S2_data.csv"),
        out_dir=evaluation_dir,
        reference_label="Validation",
        output_stem="s2",
        save_eps=False,
    )
    orientation_fraction_plot(
        reference_csv=os.path.join(evaluation_dir, "validation_orientation_fractions.csv"),
        generated_csv=os.path.join(evaluation_dir, "generated_orientation_fractions.csv"),
        output_dir=evaluation_dir,
        reference_label="Validation",
        output_stem="orientation_fraction_comparison",
    )

def conditional_plot():
    sph_file = "./figs/sph_cond_info_200_6.csv"
    num_file = "./figs/num_cond_info_200_6.csv"

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 18,
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 16,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2
    linewidth = 3


    sph_data = pd.read_csv(sph_file)
    num_data = pd.read_csv(num_file)

    num_cond = num_data['cond_nums'].to_numpy()
    num_pred = num_data['nums'].to_numpy()
    size_cond = 32**3 / num_cond # voxel scale 0.5um, so 32um in total
    size_pred = 32**3 / num_pred

    sph_cond = sph_data['cond_sphs'].to_numpy()
    sph_pred = sph_data['sphs'].to_numpy()

    out_names = ["size", "num", "sph"]
    infos = [[size_cond, size_pred], [num_cond, num_pred], [sph_cond, sph_pred]]

    for i in range(3):
        cond, pred = infos[i]
        out_name = out_names[i]
        # ---- 1. Compute basic metrics ----
        slope, intercept, r_value, p_value, std_err = linregress(cond, pred)
        r2 = r_value**2
        mae = np.mean(np.abs(pred - cond))
        rmse = np.sqrt(np.mean((pred - cond)**2))

        # ---- 2. Prepare figure ----
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

        # Panel 1: Parity plot with regression and identity line
        ax = axes[0]
        lims = [min(cond.min(), pred.min()), max(cond.max(), pred.max())]
        # ax.plot(lims, lims, 'k--', label='Identity line')
        xs = np.linspace(lims[0], lims[1], 100)
        ax.plot(xs, slope*xs + intercept, c='#94a8b7', linewidth=linewidth, label=f'Fitted line')
        ax.scatter(cond, pred, c='#ffcc00', s=50, label='Samples')
        ax.set_xlabel(f"Condition {out_name}s")
        ax.set_ylabel(f"Generated {out_name}s")

        # ax.text(0.95, 0.05, f"$R^2$={r2:.3f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}",
        #         transform=ax.transAxes, va='bottom', ha='right',
        #         bbox=dict(facecolor='white', edgecolor='none'))
        ax.text(0.9, 0.1, f"$R^2$={r2:.3f}",
                transform=ax.transAxes, va='bottom', ha='right',
                bbox=dict(facecolor='white', edgecolor='none'))


        ax.legend()
        for spine in ax.spines.values():  # 四条一起改
            spine.set_color('black')
            spine.set_linewidth(border_width)

        # Panel 2: Binned mean ± std (conditional bias check)
        ax = axes[1]
        n_bins = 8
        bins = np.linspace(cond.min(), cond.max(), n_bins+1)
        bin_centers, bin_means, bin_stds = [], [], []
        for i in range(n_bins):
            mask = (cond >= bins[i]) & (cond < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append(cond[mask].mean())
                bin_means.append(pred[mask].mean())
                bin_stds.append(pred[mask].std())
        plt.plot(bin_centers, bin_means, c='#94a8b7', linewidth=linewidth, label='Segment mean')
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', capsize=4, c='#ffcc00', label='Segment std')
        # ax.plot(lims, lims, 'k--', alpha=0.7)
        ax.set_xlabel(f"Condition {out_name}s")
        ax.set_ylabel(f"Generated {out_name}s")
        ax.legend()
        for spine in ax.spines.values():  # 四条一起改
            spine.set_color('black')
            spine.set_linewidth(border_width)

        # Panel 3: Bland–Altman (difference vs mean)
        # ax = axes[2]
        # diff = pred - cond
        # mean_vals = (pred + cond) / 2
        # bias = np.mean(diff)
        # loa = 1.96 * np.std(diff)
        # ax.scatter(mean_vals, diff, c='red', alpha=0.7)
        # ax.axhline(bias, color='blue', linestyle='-',
        #            label=f'Mean bias={bias:.2f}')
        # ax.axhline(bias + loa, color='k', linestyle='--', alpha=0.7, label='±1.96σ')
        # ax.axhline(bias - loa, color='k', linestyle='--', alpha=0.7)
        # ax.set_xlabel("Mean of Conditioned & Generated")
        # ax.set_ylabel("Difference (Generated - Conditioned)")
        # ax.set_title("Bland–Altman Plot")
        # ax.legend()

        # big_ax = fig.add_subplot(111, frameon=True, zorder=-1)  # 关键：frameon=True
        # big_ax.tick_params(labelcolor='none',
        #                    top=False, bottom=False,
        #                    left=False, right=False)  # 关掉刻度、标签
        # # 可选：调粗细、颜色
        # for sp in big_ax.spines.values():
        #     sp.set_linewidth(border_width)
        #     sp.set_color('black')

        plt.savefig(os.path.join("./figs/condition", f"{out_name}_cond.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join("./figs/condition", f"{out_name}_cond.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.savefig(os.path.join("./figs/condition", f"{out_name}_cond.eps"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.show()

def condition_visual():
    data_file = "./out/var_grain_more/cond_stability/samples_0_75/test_0.npy"
    data = np.load(data_file)
    visual(data, save_dir="./figs/fig2/75.eps")

    data_file = "./out/var_grain_more/cond_stability/samples_9_125/test_0.npy"
    data = np.load(data_file)
    visual(data, save_dir="./figs/fig2/125.eps")

    data_file = "./out/var_grain_more/cond_stability/samples_10_175/test_0.npy"
    data = np.load(data_file)
    visual(data, save_dir="./figs/fig2/175.eps")

    data_file = "./out/var_grain_more/cond_stability/samples_3_225/test_0.npy"
    data = np.load(data_file)
    visual(data, save_dir="./figs/fig2/225.eps")

def plot_box_violin():
    """
    Plot boxplots and violin plots for generated values grouped by condition inputs.

    Parameters
    ----------
    cond_inputs : array-like
        Condition input values (can have duplicates, defines grouping)
    gen_values : array-like
        Generated values corresponding to each condition input
    sort : bool, optional
        Whether to sort the conditions (default: True)
    """
    # Convert to dataframe for easy grouping
    files = ['num_cond_stability_200', 'sph_cond_stability_200']
    attrs = ['nums', 'sphs']

    for f_name, attr in zip(files, attrs):
        file = os.path.join("./figs", f"{f_name}.csv")
        data = pd.read_csv(file)

        cond_inputs = data[f'cond_{attr}'].to_numpy()
        gen_values = data[attr].to_numpy()

        # Find unique condition values and group indices
        unique_conditions = np.unique(cond_inputs)

        grouped_data = [gen_values[cond_inputs == c] for c in unique_conditions]

        # --- Plot ---
        plt.rcParams.update({
            'font.family': 'Arial',
            "axes.labelsize": 20,  # x/y label font size
            "legend.fontsize": 18,  # legend
            "xtick.labelsize": 14,  # x ticks
            "ytick.labelsize": 14,  # y ticks
        })
        fig, ax = plt.subplots()
        border_width = 2


        # --- Boxplot ---
        ax.boxplot(grouped_data, positions=np.arange(len(unique_conditions)), patch_artist=True,
                   boxprops=dict(facecolor='#ab9bcb', color='#ab9bcb'),
                   whiskerprops=dict(color='#785ba7', linewidth=1),
                   capprops=dict(color='#785ba7'),
                   medianprops=dict(color='#785ba7', linewidth=1.5),  )
        ax.set_xticks(np.arange(len(unique_conditions)), unique_conditions)
        ax.set_xlabel(f"Condition {attr}")
        ax.set_ylabel(f"Generated {attr}")
        for spine in ax.spines.values():  # 四条一起改
            spine.set_color('black')
            spine.set_linewidth(border_width)

        # --- Violin Plot ---
        # ax.violinplot(grouped_data, positions=np.arange(len(unique_conditions)), showmeans=True, showextrema=True)
        # ax.set_xticks(np.arange(len(unique_conditions)), unique_conditions, rotation=45)
        # ax.set_xlabel("Condition Input")
        # ax.set_ylabel("Generated Value")

        # plt.show()
        plt.savefig(os.path.join("./figs/fig2", f"{f_name}.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join("./figs/fig2", f"{f_name}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        #plt.savefig(os.path.join("./figs/fig1", f"S2.eps"), dpi=300, bbox_inches='tight', pad_inches=0.1)

        plt.close(fig)

def fig3_visual():
    data_file = "./figs/fig3/output-3d-25-20.npy"
    data = np.load(data_file)
    visual(data, "./figs/fig3/MRF.eps")

    data_file = "./figs/fig3/SolidTexture.npy"
    data = np.load(data_file)[[2,1,0],:,:,:].transpose([1, 2, 3, 0])
    visual(data, "./figs/fig3/SolidTexture.eps")

    data_file = "./out/var_grain_more/diffusion400/samples_5/test_0.npy"
    data = np.load(data_file)
    visual(data, "./figs/fig3/PolyGen1.eps")
    visual(data, "./figs/fig3/PolyGen1.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(data[:, :, -1, :])

    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['left'].set_visible(False)  # 隐藏左边框
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框

    plt.savefig(f"./figs/fig3/PolyGen1_slice.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


    data_file = "./out/var_grain_more/diffusion400/samples_3/test_0.npy"
    data = np.load(data_file)
    visual(data, "./figs/fig3/PolyGen2.eps")
    visual(data, "./figs/fig3/PolyGen2.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(data[:, :, -1, :])

    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['left'].set_visible(False)  # 隐藏左边框
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框

    plt.savefig(f"./figs/fig3/PolyGen2_slice.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

    data_file = "./out/var_grain_more/diffusion400/samples_4/test_0.npy"
    data = np.load(data_file)
    visual(data, "./figs/fig3/PolyGen3.eps")
    visual(data, "./figs/fig3/PolyGen3.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(data[:, :, -1, :])

    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['left'].set_visible(False)  # 隐藏左边框
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框

    plt.savefig(f"./figs/fig3/PolyGen3_slice.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


# for fig4 grain isosurface
def iso_file():
    data_files = ["./out/var_grain_more/diffusion400/samples_5/test_0.npy",
                  "./figs/fig4/SolidTexture.npy",
                  "./figs/fig4/output-3d-25-20.npy"]
    names = ["model", "SolidTexture_model", "MRF_model"]
    # MRF model and SolidTexture model, is RGB integers (0-255)
    # our further process need a range in (0, 1)
    # we need shape like (n, n, n, 3) but SolidTexture model is (3, n, n, n)

    for i in range(len(data_files)):
        data = np.load(data_files[i])
        name = names[i]

        if i == 1:
            data = data[[2, 1, 0], :, :, :].transpose([1, 2, 3, 0])

        # move range to (0, 1)
        data = data / np.max(data)
        data = data[:64, :64, :64]

        visual(data, save_dir=f"./figs/fig4/{name}.pdf")

        # colors = get_colors(10)
        # grain_label = quantize_colors(data, colors)
        # opacity = (grain_label == 1)
        # visual(data, opacity=opacity)

        colors = get_colors(10, plot=False)
        # print(colors)
        # get the index model, from 0 to color_num-1
        index_model = quantize_colors(data, colors) + 1 # leave a 0 for background
        print(np.unique(index_model))

        np.save(f"./calculation/comparison/{name}.npy", (index_model - 1).astype(np.int32))
        # index_model = index_model.swapaxes(0, 2)
        visual(colors[index_model - 1], save_dir=f"./figs/fig4/{name}_quantize.pdf")

        for ori in np.unique(index_model):
            # focus on this orientation, others are background
            ori_mask = (index_model == ori)  # binary model for this orientation
            comps, n_comp = ndimage.label(ori_mask) # connectivity analysis for this orientation
            for k in range(1, n_comp+1):
                # for each grain, we still need to detect possible connected grains
                this_grain_mask = (comps == k)
                this_volume = np.sum(this_grain_mask)

                # skip small grains
                if this_volume < 50:
                    index_model[this_grain_mask] = 0 # set the small grians to 0

        index_model = index_model.swapaxes(0, 1)
        print(index_model.shape)
        print(np.unique(index_model))
        # index model from 0 to 10, 0 is background
        savemat(f"./figs/fig4/{name}.mat", {'model': index_model})

# for fig5
def model_architecture():
    # plot some images for the model architecture diagram
    # four models from the training dataset
    indices = [118, 1, 8, 122]
    for i, index in enumerate(indices):
        data_file = f"./data/var_grain/model{index}.npy"
        data = np.load(data_file)
        visual(data, save_dir=f"./figs/fig5-Method/input_{i}.pdf")

    out_file = f"./out/var_grain_more/model_architecture/test_0.npy"
    out_data = np.load(out_file)
    visual(out_data, save_dir=f"./figs/fig5-Method/out_model.pdf")

    indices = [999, 500, 0]
    for i, index in enumerate(indices):
        latent_file = f"./out/var_grain_more/model_architecture/latent_{index}.npy"
        latent_data = np.load(latent_file)
        print(latent_data.shape)
        model_max = np.max(latent_data)
        model_min = np.min(latent_data)

        latent_data = (latent_data - model_min) / (model_max - model_min)

        # latent_data = np.clip(latent_data, 0, 1)
        # latent_data = (latent_data + 1) / 2
        visual(latent_data, save_dir=f"./figs/fig5-Method/latent_{i}.pdf")

def post_workflow():
    # """visualize model"""
    # data_file = "./out/var_grain_more/diffusion400/samples_5/test_0.npy"
    # data = np.load(data_file)
    # visual(data, save_dir="./figs/fig5-Method/model1.pdf")
    #
    # colors = get_colors(10)
    # q_data = colors[quantize_colors(data, colors)]
    # visual(q_data, save_dir="./figs/fig5-Method/model2.pdf")
    #
    # """visualize a grain"""
    # grain_label, _ = get_grains(data, cut_off=200)
    #
    # grain_index = np.array(grain_label==100, dtype=np.int32)
    # grid = pv.ImageData()
    # grid.dimensions = grain_index.shape
    # grid.point_data["values"] = grain_index.flatten(order="F")
    #
    # # Extract the region where value == 1
    # thresh = grid.threshold(value=0.5)  # keeps cells > 0.5
    #
    # plotter = pv.Plotter()
    # plotter.set_background('white')
    # plotter.add_mesh(thresh, color=(1., 0.49803922, 0.05490196), show_edges=False, opacity=1.0)
    # plotter.camera_position = 'iso'
    # plotter.save_graphic("./figs/fig5-Method/grain.pdf")
    # plotter.screenshot("./figs/fig5-Method/grain.png", transparent_background=True)

    """plot a curve"""
    plt.rcParams.update({
        'font.family': 'Arial',
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 18,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2

    stress_curve = "./calculation/final/base0.csv"
    data = pd.read_csv(stress_curve)

    strain = data['e_zz']
    stress = data['stress_zz']

    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(strain, stress)

    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    for spine in ax.spines.values():  # 四条一起改
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig("./figs/fig5-Method/result.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig("./figs/fig5-Method/result.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

def curves(folder_path, out_name):
    """
    遍历目录下所有 CSV 文件，
    读取 e_zz 和 stress_zz 列并绘制曲线
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        print("No CSV files found.")
        return

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 18,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    border_width = 2


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 右下角子图
    ax_ins = fig.add_axes([0.30, 0.30, 0.54, 0.18])

    # 只画第一个文件作为示例
    for file_path in files:
        try:
            df = pd.read_csv(os.path.join(folder_path, file_path))
            ax.plot(df['e_zz'], df['stress_zz'], c='#9FBA95', lw=0.5)
            ax_ins.plot(df['e_zz'], df['stress_zz'], c='#9FBA95', lw=0.5)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 600)
    ax.set_xlabel('strain')
    ax.set_ylabel('stress')

    x_min, x_max = 0.035, 0.04
    # y_min, y_max = 537, 557 # for 125 curves
    y_min, y_max = 540, 560
    ax_ins.set_xlim(x_min, x_max)
    ax_ins.set_ylim(y_min, y_max)

    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=1.5, edgecolor='#666666', fill=False)
    ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(border_width)

    ax.annotate('', xy=((x_min+x_max)/2, 325),             # 箭头尖端
             xytext=((x_min+x_max)/2, 500),             # 箭头尾端
             arrowprops=dict(arrowstyle='->', color='black', lw=4))

    plt.savefig(f'./figs/framework/{out_name}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'./figs/framework/{out_name}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def vector_plot():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 18,
        "axes.labelsize": 20,  # x/y label font size
        "legend.fontsize": 16,  # legend
        "xtick.labelsize": 14,  # x ticks
        "ytick.labelsize": 14,  # y ticks

    })
    border_width = 2
    linewidth = 3

    vector_file = "./figs/vector_condition_200.csv"
    vector_data = pd.read_csv(vector_file)

    num_cond = vector_data['cond_nums'].to_numpy()
    num_pred = vector_data['nums'].to_numpy()

    sph_cond = vector_data['cond_sphs'].to_numpy()
    sph_pred = vector_data['sphs'].to_numpy()

    conds = [num_cond, sph_cond]
    preds = [num_pred, sph_pred]
    out_names = ['num', 'sph']

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for i in range(2):
        cond = conds[i]
        pred = preds[i]
        out_name = out_names[i]

        # ---- 1. Compute basic metrics ----
        slope, intercept, r_value, p_value, std_err = linregress(cond, pred)
        r2 = r_value**2
        mae = np.mean(np.abs(pred - cond))
        rmse = np.sqrt(np.mean((pred - cond)**2))

        # Panel 1: Parity plot with regression and identity line
        ax = axes[i]
        lims = [min(cond.min(), pred.min()), max(cond.max(), pred.max())]
        # ax.plot(lims, lims, 'k--', label='Identity line')
        xs = np.linspace(lims[0], lims[1], 100)
        ax.plot(xs, slope*xs + intercept, c='#94a8b7', linewidth=linewidth, label=f'Fitted line')
        ax.scatter(cond, pred, c='#ffcc00', s=50, label='Samples')
        ax.set_xlabel(f"Condition {out_name}s")
        ax.set_ylabel(f"Generated {out_name}s")

        # ax.text(0.95, 0.05, f"$R^2$={r2:.3f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}",
        #         transform=ax.transAxes, va='bottom', ha='right',
        #         bbox=dict(facecolor='white', edgecolor='none'))
        ax.text(0.9, 0.1, f"$R^2$={r2:.3f}",
                transform=ax.transAxes, va='bottom', ha='right',
                bbox=dict(facecolor='white', edgecolor='none'))

        ax.legend()
        for spine in ax.spines.values():  # 四条一起改
            spine.set_color('black')
            spine.set_linewidth(border_width)

        plt.savefig(os.path.join("./figs/condition", f"vector_cond.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join("./figs/condition", f"vector_cond.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)

def computation_plot():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 18,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    border_width = 2
    linewidth = 4

    data = pd.read_csv("./figs/result.csv")

    # ===============================
    #  全局 x 范围（确保两组数据一致）
    # ===============================
    stress1 = data["result125"]
    stress2 = data["result250"]
    print(stress1.mean())
    print(stress1.var())
    print(stress2.mean())
    print(stress2.var())

    print(stress1.min())
    print(stress1.max())
    print(stress2.min())
    print(stress2.max())


    # ===============================
    #  第一幅图：PDF
    # ===============================
    plt.figure()
    ax = plt.gca()

    bins_size = get_shared_bins(stress1, stress2, bins=25)
    width = 0.9 * (bins_size[1] - bins_size[0])

    ax.hist([stress1, stress2], bins=bins_size, color=['#E69191', '#92B5CA'],
            label=["d = 7.94μm", "d = 6.30μm"])

    plt.ylim(0, 8)
    plt.xlabel('Stress (MPa)')
    plt.ylabel('Count')
    plt.legend(loc='upper left')

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig('./figs/framework/computation_hist.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('./figs/framework/computation_hist.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # ===============================
    #  第二幅图：CDF （统一 x_fit！）
    # ===============================
    plt.figure()
    ax = plt.gca()

    kde1 = gaussian_kde(stress1)
    kde2 = gaussian_kde(stress2)

    xmin = min(stress1.min(), stress2.min()) - 1
    xmax = max(stress1.max(), stress2.max()) + 1
    x_fit = np.linspace(xmin, xmax, 300)

    y1_pdf = kde1(x_fit)
    y2_pdf = kde2(x_fit)

    # PDF → CDF
    dx = x_fit[1] - x_fit[0]

    y1_cdf = np.cumsum(y1_pdf) * dx
    y1_cdf /= y1_cdf[-1]

    y2_cdf = np.cumsum(y2_pdf) * dx
    y2_cdf /= y2_cdf[-1]

    plt.plot(x_fit, y1_cdf, color='#E69191', linewidth=linewidth, label='d = 7.94μm')
    plt.plot(x_fit, y2_cdf, color='#92B5CA', linewidth=linewidth, label='d = 6.30μm')

    plt.ylim(-0.01, 1.01)
    plt.xlabel('Stress (MPa)')
    plt.ylabel('CDF')
    plt.legend(loc='upper left')

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(border_width)

    plt.savefig('./figs/framework/computation_cdf.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('./figs/framework/computation_cdf.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def framework_visual():
    file = "./out/var_grain_more/125s/samples_0_125/test_0.npy"
    data = np.load(file)
    data = data.swapaxes(1, 2)
    data = data.swapaxes(0, 1)
    visual(data, save_dir="./figs/framework/125_0.eps")

    file = "./out/var_grain_more/125s/samples_10_125/test_0.npy"
    data = np.load(file)
    data = data.swapaxes(1, 2)
    data = data.swapaxes(0, 1)
    visual(data, save_dir="./figs/framework/125_1.eps")

    file = "./out/var_grain_more/125s/samples_11_125/test_0.npy"
    data = np.load(file)
    data = data.swapaxes(1, 2)
    data = data.swapaxes(0, 1)
    visual(data, save_dir="./figs/framework/125_2.eps")


# SI plot
def VAE_visual(path):
    original_files = sorted([f for f in os.listdir(path) if f.startswith('original_') and f.endswith('.npy')])
    reconstructed_files = sorted([f for f in os.listdir(path) if f.startswith('reconstructed_') and f.endswith('.npy')])

    for i in range(len(original_files)):
        original_file = os.path.join(path, original_files[i])
        reconstructed_file = os.path.join(path, reconstructed_files[i])

        # Visualize original data
        original_data = np.load(original_file)
        visual(original_data, save_dir=f"./figs/SI/original_{i}.eps")

        # Visualize reconstructed data
        reconstructed_data = np.load(reconstructed_file)

        visual(reconstructed_data, save_dir=f"./figs/SI/reconstructed_{i}.eps")

def visualize_diffusion_process():
    input_path = "./out/var_grain_more/model_architecture"
    output_path = "./figs/SI/diffusion_process"
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(input_path):
        print(f"Directory not found: {input_path}")
        return

    test_files = sorted([f for f in os.listdir(input_path) if f.startswith('test') and f.endswith('.npy')])

    if not test_files:
        print(f"No files starting with 'test' and ending with '.npy' found in {input_path}")
        return

    for filename in test_files:
        try:
            # Extract step number from filename like 'test_100.npy'
            step_str = filename.replace('test_', '').replace('.npy', '')
            step = int(step_str)

            # Check if the step is a multiple of 100 or 999
            if step % 100 == 0 or step == 999:
                # Visualize the test file
                file_path = os.path.join(input_path, filename)
                data = np.load(file_path)
                save_file_path = os.path.join(output_path, f"{step}.eps")
                print(f"Visualizing step {step} from {file_path}...")
                visual(data, save_dir=save_file_path)
                print(f"Saved to {save_file_path}")

                # Visualize the corresponding latent file
                latent_filename = f"latent_{step}.npy"
                latent_file_path = os.path.join(input_path, latent_filename)
                if os.path.exists(latent_file_path):
                    latent_data = np.load(latent_file_path)
                    model_max = np.max(latent_data)
                    model_min = np.min(latent_data)

                    latent_data = (latent_data - model_min) / (model_max - model_min)

                    latent_data = np.clip(latent_data, 0, 1)
                     # latent_data = (latent_data + 1) / 2


                    latent_save_path = os.path.join(output_path, f"{step}_latent.eps")
                    print(f"Visualizing latent step {step} from {latent_file_path}...")
                    visual(latent_data, save_dir=latent_save_path)
                    print(f"Saved to {latent_save_path}")
                else:
                    print(f"Latent file not found for step {step}: {latent_file_path}")

        except (ValueError, IndexError):
            # This will skip files that don't match the 'test_*.npy' numeric pattern
            print(f"Could not extract a valid step number from {filename}, skipping.")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

def class_cond_visual(path):
    output_dir = "./figs/SI/classcond"
    os.makedirs(output_dir, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(path):
        dir_name = os.path.basename(dirpath)
        if dir_name.startswith("classcond"):
            if "test_0.npy" in filenames:
                data_file = os.path.join(dirpath, "test_0.npy")
                data = np.load(data_file)
                base_name = dir_name.replace("classdon_", "")
                save_filename = f"{base_name}.eps"
                save_path = os.path.join(output_dir, save_filename)
                visual(data, save_dir=save_path)
                print(f"Visualized {data_file} and saved to {save_path}")

def MRF_visual(path="./figs/SI/MRF"):
    """
    Visualizes all .npy files in the specified directory.
    """
    if not os.path.isdir(path):
        print(f"Directory not found: {path}")
        return

    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]

    if not npy_files:
        print(f"No .npy files found in {path}")
        return

    for filename in npy_files:
        try:
            file_path = os.path.join(path, filename)
            data = np.load(file_path)

            # Save the visualization with the same name but .eps extension
            base_name = os.path.splitext(filename)[0]
            save_file_path = os.path.join(path, f"{base_name}.eps")

            print(f"Visualizing {file_path}...")
            visual(data, save_dir=save_file_path)
            print(f"Saved to {save_file_path}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")


def visualize_random_cond600_results():
    input_path = "./out/var_grain_more/cond600"
    output_path = "./figs/SI/more_results"
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(input_path):
        print(f"Directory not found: {input_path}")
        return

    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]

    if len(subfolders) == 0:
        print(f"No subfolders found in {input_path}")
        return

    if len(subfolders) < 16:
        print(f"Warning: Found only {len(subfolders)} subfolders, will process all of them.")
        selected_folders = subfolders
    else:
        selected_folders = random.sample(subfolders, 16)

    for folder_path in selected_folders:
        test_file_path = os.path.join(folder_path, "test_0.npy")

        if os.path.exists(test_file_path):
            try:
                data = np.load(test_file_path)

                folder_name = os.path.basename(folder_path)
                save_file_path = os.path.join(output_path, f"{folder_name}.eps")

                print(f"Visualizing {test_file_path}...")
                visual(data, save_dir=save_file_path)
                print(f"Saved to {save_file_path}")

            except Exception as e:
                print(f"An error occurred while processing {folder_path}: {e}")
        else:
            print(f"File not found: {test_file_path}")


def visualize_solid_texture(path="./figs/SI/SolidTexture"):
    """
    Visualizes all .npy files in the specified directory.
    """
    if not os.path.isdir(path):
        print(f"Directory not found: {path}")
        return

    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]

    if not npy_files:
        print(f"No .npy files found in {path}")
        return

    for filename in npy_files:
        try:
            file_path = os.path.join(path, filename)
            data = np.load(file_path)[[2,1,0],:,:,:].transpose([1, 2, 3, 0])


            # Save the visualization with the same name but .eps extension
            base_name = os.path.splitext(filename)[0]
            save_file_path = os.path.join(path, f"{base_name}_3d.eps")

            print(f"Visualizing {file_path}...")
            visual(data, save_dir=save_file_path)
            print(f"Saved to {save_file_path}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")


if __name__ == "__main__":
    # visualize_solid_texture()
    # visualize_random_cond600_results()
    # MRF_visual()
    # visualize_diffusion_process()
    # class_cond_visual("./out/voronoi_3d_KL0_lr1e-5")
    # data = np.load("./data/var_grain/model1000.npy")
    # print(data.shape)

    # visual(data)
    # framework_visual()
    # VAE_visual("./out/var_grain_more/vae_reconstruction_test")
    # plot_evaluation()
    # plot_random_generation()
    # curves("./calculation/data/ori_data/125s", "125_curves")
    # curves("./calculation/data/ori_data/250s", "250_curves")
    computation_plot()
    # model_architecture()
    # post_workflow()
