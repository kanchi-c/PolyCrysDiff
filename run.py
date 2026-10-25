"""Project-level workflow entry points.

Low-level volume operations belong in ``utils/post_utils.py``; numerical
analyses belong in ``anal_data.py``; plotting belongs in ``plot.py``.  This
module only connects those layers into convenient, reproducible workflows.
"""

from pathlib import Path
import csv
import os
import re
import shutil
import subprocess

import numpy as np
from scipy.io import savemat

from anal_data import (
    analyze_grain_files,
    analyze_orientation_files,
    analyze_s2_files,
    analyze_vgg_files,
    write_model_comparison_metrics,
    write_evaluation_summary,
)
from plot import (
    plot_evaluation,
    plot_gan_training_losses,
    plot_training_loss_file,
    plot_vae_training_losses,
)
from utils.post_utils import get_colors, quantize_colors


ROOT = Path(__file__).resolve().parent
VALIDATION_DIR = ROOT / "data" / "var_grain_validation"
EVALUATION_DIR = ROOT / "figs" / "evaluation"
VGG_WEIGHTS = ROOT / "models" / "weights" / "vgg_conv.pth"
SAMPLE_COUNT = 100
BASELINE_SAMPLE_COUNT = 50

GENERATED_SOURCES = {
    "generated_grain_info.csv": ROOT / "figs" / "generated_grain_info.csv",
    "generated_sample_info.csv": ROOT / "figs" / "generated_sample_info.csv",
    "generated_vgg_scores.csv": ROOT / "figs" / "generated_vgg_scores.csv",
    "generated_orientation_fractions.csv": (
        ROOT / "figs" / "orientation_distribution"
        / "generated_orientation_fractions.csv"
    ),
}
GENERATED_S2_SOURCE = ROOT / "figs" / "S2_data.csv"


def validation_files():
    """Return the ordered independent validation volumes."""
    return [VALIDATION_DIR / f"model{i}.npy" for i in range(SAMPLE_COUNT)]


def verify_evaluation_inputs(files):
    """Fail early when an analysis input is absent."""
    required = list(files) + list(GENERATED_SOURCES.values())
    required.extend([GENERATED_S2_SOURCE, VGG_WEIGHTS])
    missing = [path for path in required if not path.is_file()]
    if missing:
        shown = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} required files:\n{shown}")


def copy_generated_statistics():
    """Copy existing generated-set statistics without reprocessing volumes."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, source in GENERATED_SOURCES.items():
        shutil.copy2(source, EVALUATION_DIR / output_name)


def run_validation_evaluation(seed=20260811):
    """Analyze validation data, reuse generated statistics, and redraw figures."""
    files = validation_files()
    verify_evaluation_inputs(files)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    copy_generated_statistics()

    analyze_grain_files(
        files,
        output_dir=EVALUATION_DIR,
        prefix="validation",
        min_distance=6,
        cut_off=200,
        resume=True,
    )
    analyze_vgg_files(
        files,
        output_csv=EVALUATION_DIR / "validation_vgg_scores.csv",
        seed=seed,
        weights_file=VGG_WEIGHTS,
    )
    analyze_s2_files(
        files,
        generated_summary_csv=GENERATED_S2_SOURCE,
        output_csv=EVALUATION_DIR / "S2_data.csv",
    )
    analyze_orientation_files(
        files,
        output_csv=EVALUATION_DIR / "validation_orientation_fractions.csv",
    )
    write_evaluation_summary(EVALUATION_DIR)
    plot_evaluation(EVALUATION_DIR)


def plot_validation_evaluation():
    """Redraw evaluation panels from the confirmed CSV files only."""
    plot_evaluation(EVALUATION_DIR)


def numbered_model_files(folder, count):
    """Return model0.npy through model{count-1}.npy after validation."""
    folder = Path(folder)
    files = [folder / f"model{index}.npy" for index in range(count)]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} samples; first missing file: {missing[0]}")
    return files


def render_baseline_orientation_surfaces(matlab_executable=None):
    """Render the first VAE/GAN sample as eight orientation isosurfaces."""
    comparison_dir = ROOT / "figs" / "comparison"
    renderer = ROOT / "tools" / "matlab" / "render_orientation_isosurfaces.m"
    models = {
        "VAE": ROOT / "out" / "var_grain_vae_prior" / "generated" / "model0.npy",
        "GAN": ROOT / "out" / "gan_var_grain" / "generated" / "model0.npy",
    }
    colors = get_colors(10)
    matlab_executable = matlab_executable or r"D:\Matlab\bin\matlab.exe"
    matlab_prefdir = ROOT / ".matlab_prefs"
    matlab_prefdir.mkdir(parents=True, exist_ok=True)
    matlab_env = os.environ.copy()
    matlab_env["MATLAB_PREFDIR"] = os.fspath(matlab_prefdir)

    for model_name, sample_file in models.items():
        if not sample_file.is_file():
            raise FileNotFoundError(f"Missing baseline sample: {sample_file}")
        output_dir = comparison_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        labels = quantize_colors(np.load(sample_file), colors).astype(np.uint8) + 1
        mat_file = output_dir / "model.mat"
        savemat(mat_file, {"model": labels})

        def matlab_path(path):
            return Path(path).resolve().as_posix().replace("'", "''")

        command = (
            f"addpath('{matlab_path(renderer.parent)}'); "
            f"render_orientation_isosurfaces('{matlab_path(mat_file)}', "
            f"'{matlab_path(output_dir)}')"
        )
        subprocess.run(
            [matlab_executable, "-batch", command],
            cwd=ROOT,
            env=matlab_env,
            check=True,
        )


def render_new_vae_figure7(cut_off=50, matlab_executable=None):
    """Prepare the new epoch-400 VAE example for the Figure 7 comparison.

    This follows the original Figure 7 preprocessing: RGB color quantization,
    removal of orientation-wise connected components smaller than ``cut_off``,
    conversion to a MATLAB ``model`` array, and rendering with the existing
    MATLAB isosurface script.  The full 3D overview is rendered with the
    project's unchanged PyVista ``visual`` function.
    """
    from scipy import ndimage
    import pyvista as pv

    from utils.post_utils import visual

    sample_file = (
        ROOT / "out" / "vae_var_grain" / "generated_epoch400" / "model0.npy")
    output_dir = ROOT / "figs" / "comparison" / "VAE_var_grain"
    renderer = ROOT / "tools" / "matlab" / "render_orientation_isosurfaces.m"
    if not sample_file.is_file():
        raise FileNotFoundError(f"Missing new VAE sample: {sample_file}")
    if not renderer.is_file():
        raise FileNotFoundError(f"Missing MATLAB renderer: {renderer}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(sample_file)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"Expected an RGB volume, received shape {data.shape}")
    if data.max() > 1.0:
        data = data / data.max()

    colors = get_colors(10, plot=False)
    index_model = quantize_colors(data, colors).astype(np.uint8) + 1

    # The Figure 7 overview uses the discrete ten-color representation.  The
    # vector PDF and companion PNG are both produced by the existing function.
    pv.OFF_SCREEN = True
    visual(colors[index_model - 1], save_dir=os.fspath(output_dir / "model.pdf"))

    # Match the original Figure 7 cutoff operation exactly: small connected
    # components are treated as background before orientation isosurfaces are
    # extracted.  Six-neighbour connectivity is scipy.ndimage's default in 3D.
    filtered_model = index_model.copy()
    removed_components = 0
    removed_voxels = 0
    for orientation in np.unique(index_model):
        components, component_count = ndimage.label(index_model == orientation)
        for component_index in range(1, component_count + 1):
            component_mask = components == component_index
            component_volume = int(component_mask.sum())
            if component_volume < cut_off:
                filtered_model[component_mask] = 0
                removed_components += 1
                removed_voxels += component_volume

    matlab_model = filtered_model.swapaxes(0, 1)
    mat_file = output_dir / "model.mat"
    savemat(mat_file, {"model": matlab_model})

    matlab_executable = matlab_executable or r"D:\Matlab\bin\matlab.exe"
    matlab_prefdir = ROOT / ".matlab_prefs"
    matlab_prefdir.mkdir(parents=True, exist_ok=True)
    matlab_env = os.environ.copy()
    matlab_env["MATLAB_PREFDIR"] = os.fspath(matlab_prefdir)

    def matlab_path(path):
        return Path(path).resolve().as_posix().replace("'", "''")

    command = (
        f"addpath('{matlab_path(renderer.parent)}'); "
        f"render_orientation_isosurfaces('{matlab_path(mat_file)}', "
        f"'{matlab_path(output_dir)}')"
    )
    subprocess.run(
        [matlab_executable, "-batch", command],
        cwd=ROOT,
        env=matlab_env,
        check=True,
    )
    print(
        f"Figure 7 VAE output: {output_dir}\n"
        f"Removed {removed_components} components / {removed_voxels} voxels "
        f"with cutoff < {cut_off}."
    )
    return output_dir


def evaluate_baseline_models(seed=20260811):
    """Analyze 50 VAE/GAN samples and update Table 1 comparison metrics."""
    baselines = {
        "vae": ROOT / "out" / "var_grain_vae_prior" / "generated",
        "gan": ROOT / "out" / "gan_var_grain" / "generated",
    }
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, folder in baselines.items():
        files = numbered_model_files(folder, BASELINE_SAMPLE_COUNT)
        analyze_grain_files(
            files,
            output_dir=EVALUATION_DIR,
            prefix=prefix,
            min_distance=6,
            cut_off=200,
            resume=True,
        )
        analyze_vgg_files(
            files,
            output_csv=EVALUATION_DIR / f"{prefix}_vgg_scores.csv",
            seed=seed,
            weights_file=VGG_WEIGHTS,
        )
    metrics = write_model_comparison_metrics(EVALUATION_DIR)
    print(metrics.to_string(index=False))
    return metrics


def run_baseline_comparison(seed=20260811):
    """Run all VAE/GAN figure and quantitative comparison steps."""
    render_baseline_orientation_surfaces()
    evaluate_baseline_models(seed=seed)


def plot_baseline_loss_curves():
    """Plot the saved VAE and GAN training histories for the SI."""
    vae_outputs = plot_vae_training_losses(
        recon_file=ROOT / "out" / "var_grain_vae_prior" / "recon_loss.npy",
        kl_file=ROOT / "out" / "var_grain_vae_prior" / "kl_loss.npy",
        output_dir=ROOT / "figs" / "vae-var-grain",
    )
    gan_outputs = plot_gan_training_losses(
        metrics_file=ROOT / "out" / "gan_var_grain" / "metrics.csv",
        output_dir=ROOT / "figs" / "gan-var-grain",
        smooth_window=100,
    )
    print("VAE loss curves:", *vae_outputs)
    print("GAN loss curves:", *gan_outputs)
    return {"VAE": vae_outputs, "GAN": gan_outputs}


def plot_model_convergence_losses():
    """Plot the principal autoencoder and diffusion convergence histories."""
    jobs = [
        (
            ROOT / "out" / "var_grain_more" / "recon_loss.npy",
            ROOT / "figs" / "var_grain_more",
            "reconstruction_loss_curve",
            "Reconstruction loss",
            "#D9822B",
        ),
        (
            ROOT / "out" / "var_grain_more"
            / "cond_diffusion_loss_1_400.npy",
            ROOT / "figs" / "var_grain_more",
            "conditional_diffusion_loss_curve",
            "Noise-prediction loss",
            "#4C956C",
        ),
        (
            ROOT / "out" / "var_grain_more" / "diffusion_loss_1_400.npy",
            ROOT / "figs" / "var_grain_more",
            "unconditional_diffusion_loss_curve",
            "Noise-prediction loss",
            "#315F8C",
        ),
        (
            ROOT / "out" / "voronoi_3d_KL0_lr1e-4" / "recon_loss.npy",
            ROOT / "figs" / "voronoi_3d_KL0_lr1e-4",
            "reconstruction_loss_curve",
            "Reconstruction loss",
            "#D9822B",
        ),
        (
            ROOT / "out" / "voronoi_3d_KL0_lr1e-5" / "recon_loss.npy",
            ROOT / "figs" / "voronoi_3d_KL0_lr1e-5",
            "reconstruction_loss_curve",
            "Reconstruction loss",
            "#D9822B",
        ),
    ]
    outputs = []
    for loss_file, output_dir, output_stem, ylabel, color in jobs:
        if not loss_file.is_file():
            raise FileNotFoundError(f"Missing loss history: {loss_file}")
        outputs.append(plot_training_loss_file(
            loss_file=loss_file,
            output_dir=output_dir,
            output_stem=output_stem,
            ylabel=ylabel,
            color=color,
        ))
    for png_file, pdf_file in outputs:
        print(png_file, pdf_file)
    return outputs


def extract_vae_losses_from_slurm(log_file, output_csv, max_epoch=400):
    """Extract epoch-level VAE losses from a SLURM output file.

    The training script logs an already weighted KL loss, so ``total_loss`` is
    the sum of the reported reconstruction and KL values without applying the
    KL weight a second time.
    """
    log_file = Path(log_file)
    output_csv = Path(output_csv)
    pattern = re.compile(
        r"Finished epoch:\s*(?P<epoch>\d+)\s*\|\s*"
        r"Recon Loss\s*:\s*(?P<recon>[+\-\d.eE]+)\s*\|\s*"
        r"KL Loss\s*:\s*(?P<kl>[+\-\d.eE]+)\s*\|\s*"
        r"KL Weight\s*:\s*(?P<weight>[+\-\d.eE]+)"
    )
    records = {}
    with log_file.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            if epoch <= max_epoch and epoch not in records:
                recon_loss = float(match.group("recon"))
                kl_loss = float(match.group("kl"))
                records[epoch] = {
                    "epoch": epoch,
                    "reconstruction_loss": recon_loss,
                    "weighted_kl_loss": kl_loss,
                    "total_loss": recon_loss + kl_loss,
                    "kl_weight": float(match.group("weight")),
                }

    missing = [epoch for epoch in range(1, max_epoch + 1)
               if epoch not in records]
    if missing:
        raise ValueError(
            f"SLURM log is missing {len(missing)} epochs up to {max_epoch}; "
            f"first missing epoch: {missing[0]}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "reconstruction_loss", "weighted_kl_loss",
        "total_loss", "kl_weight",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records[epoch] for epoch in range(1, max_epoch + 1))
    return [records[epoch] for epoch in range(1, max_epoch + 1)]


def prepare_new_vae_epoch400_results(sample_count=10, seed=2222):
    """Extract losses, sample the epoch-400 VAE, and render its outputs."""
    import pyvista as pv

    from utils.post_utils import crop_visual_square, visual

    run_dir = ROOT / "out" / "vae_var_grain"
    log_file = run_dir / "slurm-5390735.out"
    checkpoint = run_dir / "vae_prior_ckpt400.pth"
    config = ROOT / "config" / "var_grain_vae_config.yaml"
    sample_dir = run_dir / "generated_epoch400"
    figure_dir = ROOT / "figs" / "vae-var-grain" / "epoch400"
    csv_file = figure_dir / "vae_loss_first400.csv"

    for required in (log_file, checkpoint, config):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required VAE file: {required}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = extract_vae_losses_from_slurm(log_file, csv_file, max_epoch=400)
    recon_file = figure_dir / "_reconstruction_loss.npy"
    kl_file = figure_dir / "_weighted_kl_loss.npy"
    np.save(recon_file, np.array(
        [record["reconstruction_loss"] for record in records]))
    np.save(kl_file, np.array(
        [record["weighted_kl_loss"] for record in records]))
    try:
        loss_outputs = plot_vae_training_losses(
            recon_file=recon_file,
            kl_file=kl_file,
            output_dir=figure_dir,
        )
    finally:
        recon_file.unlink(missing_ok=True)
        kl_file.unlink(missing_ok=True)

    subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            os.fspath(ROOT / "tools" / "sample_vae3d_prior.py"),
            "--config", os.fspath(config),
            "--checkpoint", os.fspath(checkpoint),
            "--output-dir", os.fspath(sample_dir),
            "--num", str(sample_count),
            "--batch-size", "2",
            "--seed", str(seed),
        ],
        cwd=ROOT,
        check=True,
    )

    pv.OFF_SCREEN = True
    rendered = []
    for sample_index in range(sample_count):
        sample_file = sample_dir / f"model{sample_index}.npy"
        if not sample_file.is_file():
            raise FileNotFoundError(f"Missing generated VAE sample: {sample_file}")
        raw_pdf = figure_dir / f"model{sample_index}_raw.pdf"
        raw_png = raw_pdf.with_suffix(".png")
        final_png = figure_dir / f"model{sample_index}.png"
        visual(np.load(sample_file), save_dir=os.fspath(raw_pdf))
        crop_visual_square(raw_png, final_png)
        raw_pdf.unlink(missing_ok=True)
        raw_png.unlink(missing_ok=True)
        rendered.append(final_png)

    print(f"Loss CSV: {csv_file}")
    print("Loss curves:", *loss_outputs)
    print(f"Raw samples: {sample_dir}")
    print(f"Rendered samples: {figure_dir}")
    return {
        "loss_csv": csv_file,
        "loss_curves": loss_outputs,
        "sample_dir": sample_dir,
        "rendered_samples": rendered,
    }


def visualize_revision_enhance(
        input_dir=None,
        training_dir=None,
        output_dir=None,
        matrix_opacity=0.15,
        enhance_threshold=0.02,
        sample_count=10):
    """Create the revision figures for the enhanced-phase experiment."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pyvista as pv

    from utils.post_utils import (
        crop_visual_square,
        plot_loss,
        save_orthogonal_slices,
        visual_enhance,
    )

    input_dir = Path(input_dir) if input_dir else ROOT / "out" / "var_enhance"
    training_dir = (
        Path(training_dir) if training_dir else ROOT / "data" / "var_enhance")
    output_dir = (
        Path(output_dir) if output_dir else ROOT / "figs" / "revision-enhance")
    output_dir.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True

    training_path = training_dir / "model0.npy"
    if not training_path.exists():
        raise FileNotFoundError(f"Missing saved training example: {training_path}")
    training_data = np.load(training_path)
    training_mask = visual_enhance(
        training_data,
        matrix_opacity=matrix_opacity,
        enhance_threshold=enhance_threshold,
        save_dir=output_dir / "training_example_3d.pdf",
    )
    crop_visual_square(
        output_dir / "training_example_3d.png",
        output_dir / "training_example_3d_square.png",
    )
    training_indices = save_orthogonal_slices(
        training_data, output_dir, prefix="training_example")
    print(
        f"training example: enhancement fraction={training_mask.mean():.4%}, "
        f"slice indices={training_indices}")

    for sample_index in range(sample_count):
        sample_path = input_dir / f"samples_{sample_index}" / "test_0.npy"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing final sample: {sample_path}")
        data = np.load(sample_path)
        output_path = output_dir / f"sample_{sample_index:02d}.pdf"
        enhance_mask = visual_enhance(
            data,
            matrix_opacity=matrix_opacity,
            enhance_threshold=enhance_threshold,
            save_dir=output_path,
        )
        crop_visual_square(
            output_path.with_suffix(".png"),
            output_dir / f"sample_{sample_index:02d}_square.png",
        )
        print(
            f"sample {sample_index:02d}: {sample_path.name}, "
            f"enhancement fraction={enhance_mask.mean():.4%}")

        if sample_index == 0:
            visual_enhance(
                data,
                matrix_opacity=matrix_opacity,
                enhance_threshold=enhance_threshold,
                save_dir=output_dir / "generated_example0_3d.pdf",
            )
            crop_visual_square(
                output_dir / "generated_example0_3d.png",
                output_dir / "generated_example0_3d_square.png",
            )
            generated_indices = save_orthogonal_slices(
                data, output_dir, prefix="generated_example0")
            print(f"generated example 0 slice indices={generated_indices}")

    loss_path = input_dir / "diffusion_loss_1_400.npy"
    if not loss_path.exists():
        raise FileNotFoundError(f"Missing loss history: {loss_path}")
    loss = np.load(loss_path)
    plot_loss(loss)
    ax = plt.gca()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise-prediction loss")
    ax.set_title("Diffusion training loss")
    plt.tight_layout()
    plt.savefig(output_dir / "diffusion_loss_curve.png", dpi=300,
                bbox_inches="tight")
    plt.savefig(output_dir / "diffusion_loss_curve.pdf", bbox_inches="tight")
    plt.close()
    print(f"Figures saved to: {output_dir}")


def main():
    # Change this call when a different integrated workflow should be run.
    plot_model_convergence_losses()
    # render_new_vae_figure7()
    # prepare_new_vae_epoch400_results()
    # plot_baseline_loss_curves()
    # run_baseline_comparison(seed=20260811)
    # plot_validation_evaluation()
    # run_validation_evaluation(seed=20260811)
    # visualize_revision_enhance()


if __name__ == "__main__":
    main()
