import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance
from models.VGG import VGG
from utils.post_utils import *

# --- Shared bin edges helper ---

def raw_moment(a, px, qx, rx):
    """
    Compute raw moment m_pqr for a 3D binary array.
    m_pqr = sum(x^px * y^qx * z^rx * a[x,y,z])
    """
    x, y, z = np.indices(a.shape)
    return np.sum((x ** px) * (y ** qx) * (z ** rx) * a)

"""plot the relation between aspect ratio and sphericity"""
# plt.rcParams.update({
#     'font.size' : 14,          # 基础字号
#     'axes.titlesize' : 16,
#     'axes.labelsize' : 18,
#     'xtick.labelsize' : 12,
#     'ytick.labelsize' : 12,
#     'legend.fontsize' : 12
# })
#
# info_file = "./data/var_sph/info.txt"
# data = pd.read_table(info_file, sep=',')
#
# pb_sphs = np.array(data.loc[:, 'sph_mean'])
# sphs = np.array(data.loc[:, 'sph'])
# ratios = np.array(data.loc[:, 'aspect_ratio'])
#
# print(np.mean(sphs))
# print(np.std(sphs))
#
# print(np.mean(ratios))
# print(np.std(ratios))
#
# sphs = (sphs - np.min(sphs)) / (np.max(sphs) - np.min(sphs))
# plt.hist(sphs)
# plt.show()
#
#
# fig = plt.figure(figsize=(12.8, 4.8))
# ax1 = fig.add_subplot(121)
# ax1.scatter(sphs, pb_sphs,  c='red')
# ax1.set_xlabel("sphs")
# ax1.set_ylabel("pb_sphs")
#
# ax2 = fig.add_subplot(122)
# ax2.scatter(sphs, ratios,  c='red')
# ax2.set_xlabel("sphs")
# ax2.set_ylabel("ratios")
#
# plt.savefig(f"./figs/sphs_ratios.png", dpi=300, bbox_inches='tight')


def compute_eta(grain_mask, p, q, r):
    """
    Compute normalized central moment eta_pqr for a binary 3D grain mask.
    """
    if p < 0 or q < 0 or r < 0:
        raise ValueError("p, q, r must be non-negative")

    mask = np.asarray(grain_mask)
    if mask.ndim != 3:
        raise ValueError("grain_mask must be a 3D array")

    mask = (mask > 0).astype(np.float64)

    m000 = raw_moment(mask, 0, 0, 0)
    if m000 == 0:
        raise ValueError("grain_mask is empty (no foreground voxels)")

    x_bar = raw_moment(mask, 1, 0, 0) / m000
    y_bar = raw_moment(mask, 0, 1, 0) / m000
    z_bar = raw_moment(mask, 0, 0, 1) / m000

    x, y, z = np.indices(mask.shape)
    mu_pqr = np.sum(((x - x_bar) ** p) * ((y - y_bar) ** q) * ((z - z_bar) ** r) * mask)

    gamma = 1.0 + (p + q + r) / 3.0
    eta_pqr = mu_pqr / (m000 ** gamma)
    return float(eta_pqr)


def anal_random(min_distance=6, cut_off=200):
    """
    fig 1, comparison between dataset and random generated set
    grain nubmer, grain sph, size, aspect ratio are supported
    """

    # grain information array for generated and training structures
    grain_sizes_dataset = []
    aspect_ratios_dataset = []
    sphs_dataset = []
    etas_111_dataset = []  # eta_111 for dataset grains

    grain_sizes_generated = []
    aspect_ratios_generated = []
    sphs_generated = []
    etas_111_generated = []  # eta_111 for generated grains

    # structure information array for generated structures
    # because training infos can be got directly form info_file
    grain_nums_generated = []
    mean_aspect_ratios_generated = []
    mean_sphs_generated = []

    grain_nums_dataset = []
    mean_aspect_ratios_dataset = []
    mean_sphs_dataset = []


    for i in tqdm(range(100)):
        data_file = f"./data/var_grain/model{i}.npy"
        generated_file = f"./out/var_grain_more/diffusion400/samples_{i}/test_0.npy"

        # analyze training dataset item
        grain_model, _ = get_grains(data_file, min_distance=min_distance, cut_off=cut_off)
        results = anal_grians(grain_model)
        grain_nums_dataset.append(len(results)) # record grain nums

        this_sizes = [results[gid]['volume'] for gid in results.keys()]
        grain_sizes_dataset.extend(this_sizes)  # record grain size

        this_ratios = [results[gid]['aspect_ratio'] for gid in results.keys()]
        aspect_ratios_dataset.extend(this_ratios) # record aspect ratios
        mean_aspect_ratios_dataset.append(np.mean(this_ratios)) # record mean aspect ratios

        this_sphs = [results[gid]['sphericity'] for gid in results.keys()]
        sphs_dataset.extend(this_sphs) # record sphs
        mean_sphs_dataset.append(np.mean(this_sphs)) # record mean sphs

        # compute eta_111 for all grains in dataset
        this_etas_111 = []
        for gid in results.keys():
            grain_mask = (grain_model == gid).astype(np.float64)
            eta_111 = compute_eta(grain_mask, 1, 1, 1)
            this_etas_111.append(eta_111)
        etas_111_dataset.extend(this_etas_111)  # record eta_111

        # analyze generated item
        grain_model, _ = get_grains(generated_file, min_distance=min_distance, cut_off=cut_off)
        results = anal_grians(grain_model)
        grain_nums_generated.append(len(results)) # add grain nums

        this_sizes = [results[gid]['volume'] for gid in results.keys()]
        grain_sizes_generated.extend(this_sizes) # record grain sizes

        this_ratios = [results[gid]['aspect_ratio'] for gid in results.keys()]
        aspect_ratios_generated.extend(this_ratios) # record aspect ratios
        mean_aspect_ratios_generated.append(np.mean(this_ratios)) # record mean aspect ratios

        this_sphs = [results[gid]['sphericity'] for gid in results.keys()]
        sphs_generated.extend(this_sphs) # record sphs
        mean_sphs_generated.append(np.mean(this_sphs)) # record mean sphs

        # compute eta_111 for all grains in generated
        this_etas_111 = []
        for gid in results.keys():
            grain_mask = (grain_model == gid).astype(np.float64)
            eta_111 = compute_eta(grain_mask, 1, 1, 1)
            this_etas_111.append(eta_111)
        etas_111_generated.extend(this_etas_111)  # record eta_111

    df = pd.DataFrame({'grain_sizes_dataset': grain_sizes_dataset,
                       'aspect_ratios_dataset': aspect_ratios_dataset,
                       'sphs_dataset': sphs_dataset,
                       'etas_111_dataset': etas_111_dataset})
    df.to_csv('./figs/dataset_grain_info.csv', index=False)


    df = pd.DataFrame({'grain_sizes_generated': grain_sizes_generated,
                        'aspect_ratios_generated': aspect_ratios_generated,
                        'sphs_generated': sphs_generated,
                        'etas_111_generated': etas_111_generated})
    df.to_csv('./figs/generated_grain_info.csv', index=False)

    df = pd.DataFrame({'grain_nums_generated': grain_nums_generated,
                       'aspect_ratios_generated': mean_aspect_ratios_generated,
                       'sphs_generated': mean_sphs_generated})

    df.to_csv('./figs/generated_sample_info.csv', index=False)

"""VGG score analysis"""
def anal_VGG():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VGG_model = VGG()
    VGG_model.load_state_dict(torch.load("./models/weights/vgg_conv.pth"))
    VGG_model.to(device)

    data_dir = './out/var_grain_more/diffusion400/'      # 存放 *.npy 的文件夹
    csv_path = './figs/generated_vgg_scores.csv'
    fig_path = './figs/VGGscore_hist1.png'

    # --------- 3. 主流程 ----------
    records = []
    for i in range(100):
        fname = f"samples_{i}/test_0.npy"
        path = os.path.join(data_dir, fname)
        arr = np.load(path)        # shape=(n,n,n,3)
        n = arr.shape[0]

        idx0, idx1, idx2 = np.random.randint(n, size=3) # 随机切片索引
        slice0 = arr[idx0, :, :, :]
        slice1 = arr[:, idx1, :, :]
        slice2 = arr[:, :, idx2, :] # (n,n,3)

        records.append({'file': fname, 'dir': 'x', 'slice_idx': idx0, 'score': VGG_score(VGG_model, slice0)})
        records.append({'file': fname, 'dir': 'y', 'slice_idx': idx1, 'score': VGG_score(VGG_model, slice1)})
        records.append({'file': fname, 'dir': 'z', 'slice_idx': idx2, 'score': VGG_score(VGG_model, slice2)})

    # --------- 4. 保存 CSV ----------
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f'Saved {len(df)} scores -> {csv_path}')

    # --------- 5. 绘制直方图 ----------
    plt.figure(figsize=(6,4))
    plt.hist(df['score'], bins=30, edgecolor='black')
    plt.xlabel('VGG Score')
    plt.ylabel('Count')
    plt.title('Distribution of Random-Slice Scores')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

def compare_s2_curves(dataset_dir="./data/var_grain/", generated_dir="./out/var_grain_more/diffusion400", bins=50, color_nums=10):
    """
    Compare mean ± std S2 curves for dataset vs generated data.

    dataset_samples, generated_samples : list of ndarrays
        Each element is a (n,n,n) orientation index array
    """
    colors = get_colors(color_nums)
    s2_dataset = []
    s2_generated = []

    dataset_index = np.random.randint(1000, size=100)
    for i in tqdm(range(100)):
        data_file = os.path.join(dataset_dir, f"model{i}.npy")
        data = np.load(data_file)
        color_index = quantize_colors(data, colors)
        s2_dataset.append(weighted_s2(color_index, bins=bins))

    for i in tqdm(range(50)):
        data_file = os.path.join(generated_dir, f"samples_{i}", f"test_0.npy")
        data = np.load(data_file)
        color_index = quantize_colors(data, colors)
        s2_generated.append(weighted_s2(color_index, bins=bins))

    r, _ = two_point_correlation((color_index == np.unique(color_index)[0]).astype(np.float32), bins=bins)

    s2_dataset = np.vstack(s2_dataset)
    s2_generated = np.vstack(s2_generated)

    mean_dataset = s2_dataset.mean(axis=0)
    min_dataset = s2_dataset.min(axis=0)
    max_dataset = s2_dataset.max(axis=0)

    mean_generated = s2_generated.mean(axis=0)
    min_generated = s2_generated.min(axis=0)
    max_generated = s2_generated.max(axis=0)

    df = pd.DataFrame({'r': r,
                       'mean_dataset': mean_dataset,
                       'min_dataset': min_dataset,
                       'max_dataset': max_dataset,
                       'mean_generated': mean_generated,
                       'min_generated': min_generated,
                       'max_generated': max_generated,
                       })

    df.to_csv('./figs/S2_data.csv', index=False)

def anal_RGB():
    """RGB plot"""
    # for generated models
    model_files = []
    base_dir = "./out/var_grain_more/diffusion400"
    sample_dirs = [name for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))]

    for sample_dir in sample_dirs:
        model_file = os.path.join(base_dir, sample_dir, "test_0.npy")
        if os.path.exists(model_file):
            model_files.append(model_file)

    # original model
    models = [np.load(model_file).reshape(-1, 3) * 255 for model_file in model_files]
    # post processed model
    # colors = get_colors(10)
    # models = [colors[quantize_colors(np.load(model_file), colors)].reshape(-1, 3) * 255 for model_file in model_files]

    model_arr = np.concatenate(models, axis=0)
    [R_cdf1, G_cdf1, B_cdf1] = anal_rgb(model_arr)

    # for dataset
    dataset_dir = "./data/var_grain"
    indices = random.sample(range(2000), 1000)
    model_files = [os.path.join(dataset_dir, f"model{index}.npy") for index in indices]

    models = [np.load(model_file).reshape(-1, 3) * 255 for model_file in model_files]
    model_arr = np.concatenate(models, axis=0)

    [R_cdf2, G_cdf2, B_cdf2] = anal_rgb(model_arr)

    df = pd.DataFrame({'R_cdf1': R_cdf1,
                       'G_cdf1': G_cdf1,
                       'B_cdf1': B_cdf1,
                       'R_cdf2': R_cdf2,
                       'G_cdf2': G_cdf2,
                       'B_cdf2': B_cdf2,
                       })

    df.to_csv('./figs/RGB_data.csv', index=False)


def analyze_grain_files(
        model_files,
        output_dir,
        prefix="validation",
        min_distance=6,
        cut_off=200,
        resume=True):
    """Analyze grain-level and sample-level statistics for 3D volumes.

    Grain separation is performed by :func:`utils.post_utils.get_grains`, and
    morphology descriptors are calculated by
    :func:`utils.post_utils.anal_grians`. Results are checkpointed after each
    volume because the 3D watershed and surface calculations are expensive.
    """
    model_files = [Path(path) for path in model_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grain_csv = output_dir / f"{prefix}_grain_info.csv"
    sample_csv = output_dir / f"{prefix}_sample_info.csv"

    grain_rows = []
    sample_rows = []
    completed = 0
    if resume and grain_csv.is_file() and sample_csv.is_file():
        old_grains = pd.read_csv(grain_csv)
        old_samples = pd.read_csv(sample_csv)
        if len(old_samples) == len(model_files):
            return old_grains, old_samples
        if len(old_samples) < len(model_files):
            grain_rows = old_grains.to_dict("records")
            sample_rows = old_samples.to_dict("records")
            completed = len(old_samples)

    for sample_id, path in enumerate(
            tqdm(model_files[completed:], desc=f"Grains: {prefix}"),
            start=completed):
        grain_model, _ = get_grains(
            path, min_distance=min_distance, cut_off=cut_off)
        results = anal_grians(grain_model)

        for grain_id, values in results.items():
            grain_mask = (grain_model == grain_id).astype(np.float64)
            grain_rows.append({
                "sample_id": sample_id,
                "file": os.fspath(path),
                "grain_id": int(grain_id),
                "grain_size": float(values["volume"]),
                "aspect_ratio": float(values["aspect_ratio"]),
                "sphericity": float(values["sphericity"]),
                "eta_111": compute_eta(grain_mask, 1, 1, 1),
            })

        sample_values = list(results.values())
        sample_rows.append({
            "sample_id": sample_id,
            "file": os.fspath(path),
            "grain_number": len(sample_values),
            "mean_aspect_ratio": float(np.mean([
                item["aspect_ratio"] for item in sample_values])),
            "mean_sphericity": float(np.mean([
                item["sphericity"] for item in sample_values])),
        })
        pd.DataFrame(grain_rows).to_csv(grain_csv, index=False)
        pd.DataFrame(sample_rows).to_csv(sample_csv, index=False)

    return pd.DataFrame(grain_rows), pd.DataFrame(sample_rows)


def analyze_vgg_files(
        model_files,
        output_csv,
        seed=20260811,
        weights_file="./models/weights/vgg_conv.pth"):
    """Calculate three random orthogonal-slice VGG scores per volume."""
    model_files = [Path(path) for path in model_files]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG()
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device).eval()
    rng = np.random.default_rng(seed)
    records = []

    with torch.no_grad():
        for sample_id, path in enumerate(tqdm(model_files, desc="VGG scores")):
            array = np.load(path)
            indices = [int(rng.integers(length)) for length in array.shape[:3]]
            slices = (
                array[indices[0], :, :, :],
                array[:, indices[1], :, :],
                array[:, :, indices[2], :],
            )
            for axis, index, image in zip("xyz", indices, slices):
                records.append({
                    "sample_id": sample_id,
                    "file": os.fspath(path),
                    "axis": axis,
                    "slice_index": index,
                    "score": VGG_score(model, image),
                })

    frame = pd.DataFrame(records)
    frame.to_csv(output_csv, index=False)
    return frame


def analyze_s2_files(
        validation_files,
        generated_summary_csv,
        output_csv,
        bins=50,
        color_nums=10):
    """Combine validation S2 statistics with existing generated statistics."""
    colors = get_colors(color_nums)
    curves = []
    last_indices = None
    for path in tqdm(validation_files, desc="S2: validation"):
        last_indices = quantize_colors(np.load(path), colors)
        curves.append(weighted_s2(last_indices, bins=bins))
    validation_s2 = np.vstack(curves)
    r, _ = two_point_correlation(
        (last_indices == np.unique(last_indices)[0]).astype(np.float32),
        bins=bins)
    generated = pd.read_csv(generated_summary_csv)

    frame = pd.DataFrame({
        "r": r,
        "mean_validation": validation_s2.mean(axis=0),
        "min_validation": validation_s2.min(axis=0),
        "max_validation": validation_s2.max(axis=0),
        "mean_generated": generated["mean_generated"].to_numpy(),
        "min_generated": generated["min_generated"].to_numpy(),
        "max_generated": generated["max_generated"].to_numpy(),
    })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def analyze_orientation_files(model_files, output_csv, color_nums=10):
    """Save the per-volume fractions of the discrete orientation classes."""
    return orientation_fraction_table(
        model_files,
        colors=get_colors(color_nums),
        output_csv=output_csv,
    )


def write_evaluation_summary(output_dir):
    """Write summary statistics from confirmed validation/generated CSVs."""
    output_dir = Path(output_dir)
    validation_grains = pd.read_csv(output_dir / "validation_grain_info.csv")
    generated_grains = pd.read_csv(output_dir / "generated_grain_info.csv").rename(
        columns={
            "grain_sizes_generated": "grain_size",
            "aspect_ratios_generated": "aspect_ratio",
            "sphs_generated": "sphericity",
            "etas_111_generated": "eta_111",
        })
    validation_samples = pd.read_csv(output_dir / "validation_sample_info.csv")
    generated_samples = pd.read_csv(output_dir / "generated_sample_info.csv").rename(
        columns={
            "grain_nums_generated": "grain_number",
            "aspect_ratios_generated": "mean_aspect_ratio",
            "sphs_generated": "mean_sphericity",
        })
    validation_vgg = pd.read_csv(output_dir / "validation_vgg_scores.csv")
    generated_vgg = pd.read_csv(output_dir / "generated_vgg_scores.csv")
    validation_orientation = pd.read_csv(
        output_dir / "validation_orientation_fractions.csv")
    generated_orientation = pd.read_csv(
        output_dir / "generated_orientation_fractions.csv")
    s2 = pd.read_csv(output_dir / "S2_data.csv")

    pairs = {
        "grain_size": (validation_grains["grain_size"], generated_grains["grain_size"]),
        "aspect_ratio": (validation_grains["aspect_ratio"], generated_grains["aspect_ratio"]),
        "sphericity": (validation_grains["sphericity"], generated_grains["sphericity"]),
        "eta_111": (validation_grains["eta_111"], generated_grains["eta_111"]),
        "grain_number": (validation_samples["grain_number"], generated_samples["grain_number"]),
        "vgg_score": (validation_vgg["score"], generated_vgg["score"]),
    }
    rows = []
    for name, (reference, generated) in pairs.items():
        result = ks_2samp(reference.dropna(), generated.dropna())
        rows.append({
            "metric": name,
            "validation_mean": reference.mean(),
            "validation_std": reference.std(),
            "generated_mean": generated.mean(),
            "generated_std": generated.std(),
            "ks_statistic": result.statistic,
            "ks_pvalue": result.pvalue,
        })
    orientation_error = (
        generated_orientation.mean() - validation_orientation.mean()).abs()
    rows.append({
        "metric": "orientation_fraction_mean_absolute_error",
        "generated_mean": orientation_error.mean(),
        "generated_std": orientation_error.std(),
    })
    rows.append({
        "metric": "s2_mean_curve_rmse",
        "generated_mean": np.sqrt(np.mean(
            (s2["mean_generated"] - s2["mean_validation"]) ** 2)),
    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "evaluation_summary.csv", index=False)
    return frame


def write_model_comparison_metrics(output_dir):
    """Compare PolyCrysDiff, VAE, and GAN against one validation set.

    The five quantities match Table 1. EMD is calculated after jointly
    normalizing each pair of distributions to [0, 1], consistent with
    :func:`plot.compare_distributions`.
    """
    output_dir = Path(output_dir)
    validation_grains = pd.read_csv(output_dir / "validation_grain_info.csv")
    validation_vgg = pd.read_csv(output_dir / "validation_vgg_scores.csv")

    model_files = {
        "PolyCrysDiff": (
            output_dir / "generated_grain_info.csv",
            output_dir / "generated_vgg_scores.csv",
        ),
        "VAE": (
            output_dir / "vae_grain_info.csv",
            output_dir / "vae_vgg_scores.csv",
        ),
        "GAN": (
            output_dir / "gan_grain_info.csv",
            output_dir / "gan_vgg_scores.csv",
        ),
    }
    legacy_columns = {
        "grain_sizes_generated": "grain_size",
        "aspect_ratios_generated": "aspect_ratio",
        "sphs_generated": "sphericity",
        "etas_111_generated": "eta_111",
    }
    reference = {
        "grain_size": validation_grains["grain_size"],
        "aspect_ratio": validation_grains["aspect_ratio"],
        "sphericity": validation_grains["sphericity"],
        "eta_111": validation_grains["eta_111"],
        "vgg_score": validation_vgg["score"],
    }

    rows = []
    for model_name, (grain_file, vgg_file) in model_files.items():
        grains = pd.read_csv(grain_file).rename(columns=legacy_columns)
        vgg = pd.read_csv(vgg_file)
        generated = {
            "grain_size": grains["grain_size"],
            "aspect_ratio": grains["aspect_ratio"],
            "sphericity": grains["sphericity"],
            "eta_111": grains["eta_111"],
            "vgg_score": vgg["score"],
        }
        for metric, reference_values in reference.items():
            a = np.asarray(reference_values.dropna(), dtype=np.float64)
            b = np.asarray(generated[metric].dropna(), dtype=np.float64)
            finite_a = a[np.isfinite(a)]
            finite_b = b[np.isfinite(b)]
            ks_value = ks_2samp(finite_a, finite_b).statistic
            lower = min(finite_a.min(), finite_b.min())
            upper = max(finite_a.max(), finite_b.max())
            value_range = upper - lower
            emd_value = (
                wasserstein_distance(
                    (finite_a - lower) / value_range,
                    (finite_b - lower) / value_range,
                )
                if value_range > 0 else 0.0
            )
            rows.append({
                "model": model_name,
                "metric": metric,
                "ks_statistic": ks_value,
                "emd_normalized": emd_value,
                "validation_count": len(finite_a),
                "generated_count": len(finite_b),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    table = frame.pivot(index="metric", columns="model", values=[
        "ks_statistic", "emd_normalized"])
    table = table.swaplevel(axis=1)
    model_order = ["PolyCrysDiff", "VAE", "GAN"]
    table = table.reindex(columns=pd.MultiIndex.from_product([
        model_order, ["ks_statistic", "emd_normalized"]]))
    table.columns = [
        f"{model}_{'KS' if statistic == 'ks_statistic' else 'EMD'}"
        for model, statistic in table.columns
    ]
    table.reindex([
        "grain_size", "aspect_ratio", "sphericity", "eta_111", "vgg_score"
    ]).to_csv(output_dir / "model_comparison_metrics_table.csv")
    return frame


def anal_orientation_fractions(
        training_dir="./data/var_grain",
        validation_dir="./data/var_grain_validation",
        generated_dir="./out/var_grain_more/diffusion400",
        output_dir="./figs/orientation_distribution"):
    """Save per-sample orientation fractions for all comparison datasets.

    Three ``n x 10`` tables are written for the complete training dataset, the
    independent validation dataset, and the final generated ``test_0.npy``
    samples. The validation table is the reference for the comparison plot.
    """
    def numeric_suffix(path):
        match = re.search(r"(\d+)$", path.stem)
        return int(match.group(1)) if match else path.name

    def numbered_models(folder):
        return sorted(Path(folder).glob("model*.npy"), key=numeric_suffix)

    def generated_models(folder):
        sample_dirs = [path for path in Path(folder).glob("samples_*")
                       if path.is_dir()]
        sample_dirs.sort(key=numeric_suffix)
        return [path / "test_0.npy" for path in sample_dirs
                if (path / "test_0.npy").is_file()]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = get_colors(10)
    groups = {
        "training": numbered_models(training_dir),
        "validation": numbered_models(validation_dir),
        "generated": generated_models(generated_dir),
    }

    for name, model_files in groups.items():
        if not model_files:
            raise FileNotFoundError(f"No input volumes found for {name}")
        table = orientation_fraction_table(
            model_files,
            colors=colors,
            output_csv=output_dir / f"{name}_orientation_fractions.csv",
        )
        print(f"{name}: {table.shape[0]} samples -> {table.shape}")

    return groups

def anal_condition(min_distance=6, cut_off=200):
    """
    fig2, for conditional generation, compare input value and generated value
    """
    cond_nums = []
    sample_nums = []
    sample_sphs = []
    sample_ratios = []

    base_dir = "./out/var_grain_more/cond_stability"
    out_dir = "./figs/"
    sample_dirs = [name for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))]

    for sample_dir in tqdm(sample_dirs):
        model_file = os.path.join(base_dir, sample_dir, "test_0.npy")
        if os.path.exists(model_file):
            # grain_model maybe have 0 (background), results do not have.
            grain_model, _ = get_grains(model_file, cut_off=cut_off, min_distance=min_distance)
            results = anal_grians(grain_model)
            results_df = pd.DataFrame.from_dict(results, orient='index')

            # record infos
            sample_nums.append(len(results_df))
            sample_sphs.append(results_df['sphericity'].mean())
            sample_ratios.append(results_df['aspect_ratio'].mean())

            cond_num = re.findall(r'\d+', sample_dir)[1]
            cond_nums.append(int(cond_num))

    print(sample_dirs)
    print(sample_nums)

    df = pd.DataFrame({'cond_nums': cond_nums,
                       'nums': sample_nums,
                        'aspect_ratios': sample_ratios,
                       'sphs': sample_sphs})
    df = df.sort_values(by='cond_nums', ascending=True)

    df.to_csv(os.path.join(out_dir, f"num_cond_stability_{cut_off}.csv"), index=False)   # 无行索引


    # cond_sphs = []
    # sample_nums = []
    # sample_ratios = []
    # sample_sphs = []
    #
    # base_dir = "./out/var_sph/cond_stability"
    # out_dir = './figs/'
    # sample_dirs = [name for name in os.listdir(base_dir)
    #             if os.path.isdir(os.path.join(base_dir, name))]
    #
    # for sample_dir in tqdm(sample_dirs):
    #     model_file = os.path.join(base_dir, sample_dir, "test_0.npy")
    #     if os.path.exists(model_file):
    #         grain_model, _ = get_grains(model_file, cut_off=cut_off, min_distance=min_distance)
    #         results = anal_grians(grain_model)
    #         df = pd.DataFrame.from_dict(results, orient='index')
    #
    #         sample_ratios.append(df['aspect_ratio'].mean())
    #         sample_sphs.append(df['sphericity'].mean())
    #         sample_nums.append(len(df))
    #
    #         cond_mean_sph = re.findall(r'(-?\d+(?:\.\d+)?)', sample_dir)[1]
    #         cond_sphs.append(float(cond_mean_sph))
    #
    # print(sample_dirs)
    # print(sample_sphs)
    #
    # df = pd.DataFrame({'cond_sphs': cond_sphs,
    #                    'nums': sample_nums,
    #                    'aspect_ratios': sample_ratios,
    #                    'sphs': sample_sphs})
    # df = df.sort_values(by='cond_sphs', ascending=True)
    #
    # df.to_csv(os.path.join(out_dir, f"sph_cond_stability_{cut_off}.csv"), index=False)   # 无行索引

def fig3(min_distance=8, cut_off=200):
    MRF_file = "./figs/MRF/output-3d-25-20.npy"
    MRF_data = np.load(MRF_file) / 255.0
    print(np.max(MRF_data))
    print(np.min(MRF_data))

    print(MRF_data.shape)

    grain_model, _ = get_grains(MRF_data, min_distance=min_distance, cut_off=cut_off)
    results_df = pd.DataFrame.from_dict(anal_grians(grain_model), orient='index')

    print(f"{len(results_df)} grains are detected.")
    print(f"Mean spherecity is {results_df['sphericity'].mean()}")
    print(f"Mean aspect_ratio is {results_df['aspect_ratio'].mean()}")

def anal_vector(min_distance=6, cut_off=200):
    """
    fig2, for conditional generation, compare input value and generated value
    """
    cond_nums = []
    cond_sphs = []

    sample_nums = []
    sample_sphs = []
    sample_ratios = []

    base_dir = "./out/var_grain_sph/test"
    out_dir = "./figs/"
    sample_dirs = [name for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))]

    for sample_dir in tqdm(sample_dirs):
        model_file = os.path.join(base_dir, sample_dir, "test_0.npy")
        if os.path.exists(model_file):
            # grain_model maybe have 0 (background), results do not have.
            grain_model, _ = get_grains(model_file, cut_off=cut_off, min_distance=min_distance)
            results = anal_grians(grain_model)
            results_df = pd.DataFrame.from_dict(results, orient='index')

            # record infos
            sample_nums.append(len(results_df))
            sample_sphs.append(results_df['sphericity'].mean())
            sample_ratios.append(results_df['aspect_ratio'].mean())

            parts = sample_dir.split('_')
            cond_num, cond_sph = int(float(parts[-2])), float(parts[-1])
            cond_nums.append(int(cond_num))
            cond_sphs.append(cond_sph)

    df = pd.DataFrame({'cond_nums': cond_nums,
                       'cond_sphs': cond_sphs,
                       'nums': sample_nums,
                       'sphs': sample_sphs,
                        'aspect_ratios': sample_ratios})
    df = df.sort_values(by='cond_nums', ascending=True)

    df.to_csv(os.path.join(out_dir, f"vector_condition_{cut_off}.csv"), index=False)   # 无行索引


if __name__ == "__main__":
    anal_random()
