import numpy as np
import pyvista as pv
pv.global_theme.transparent_background = True
import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.cm as cm
from scipy import ndimage
import os
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from numpy.fft import fftn, ifftn, fftshift
from models.VGG import GramMatrix

def label_volume_to_rgb(data, color_map="tab20"):
    """Convert a non-negative 3D integer-label volume to RGB colors."""
    ids = np.asarray(data).astype(int)
    if ids.ndim != 3:
        raise ValueError("Label volume must have shape (D, H, W).")
    if ids.min() < 0:
        raise ValueError("Integer labels must be non-negative.")
    cmap = cm.get_cmap(color_map)
    colors = cmap(np.arange(ids.max() + 1) % cmap.N)[:, :3]
    return colors[ids]


def visual(data, opacity=None, save_dir=None, color_map="tab20"):
    """Visualize an RGB volume or a 3D integer-label volume."""
    data = np.asarray(data)
    if data.ndim == 3:
        data = label_volume_to_rgb(data, color_map=color_map)
    elif data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("data must have shape (D, H, W) or (D, H, W, 3).")

    data = data.swapaxes(0, 2)
    shape = np.array(data.shape[:3])

    # generate uniform mesh
    grid = pv.ImageData(dimensions=shape+1)

    rgb = data.reshape(-1, 3)
    if opacity is None:
        grid.cell_data['rgb'] = rgb
        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.add_mesh(grid, scalars='rgb', rgb=True)
    else:
        opacity = opacity.swapaxes(0, 2).reshape(-1)
        # opacity = opacity.reshape(-1)
        rgba = np.empty((rgb.shape[0], 4), dtype=rgb.dtype)
        rgba[:, :3] = rgb
        rgba[:, 3] = opacity      # 第 4 通道 = alpha
        grid.cell_data["rgba"] = (rgba * 255).astype(np.uint8)

        # visualization
        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.add_volume(grid, scalars='rgba', preference='point')

    # use scalar to tell which one is for color data, otherwise it will use the first cell_data
    if save_dir is not None:
        plotter.save_graphic(save_dir)
        plotter.screenshot(save_dir[:-4] + ".png", transparent_background=True)
    else:
        plotter.show()
    plotter.close()


def visual_enhance(data, matrix_opacity=0.15, enhance_threshold=0.02,
                   save_dir=None, color_map="tab20"):
    """Visualize a 3D microstructure containing a black enhancement phase.

    Voxels whose RGB channels are all below ``enhance_threshold`` are treated
    as the enhancement phase and rendered as opaque. The remaining matrix is
    rendered with ``matrix_opacity`` so that the internal enhancement phase is
    visible. A 3D integer label array is also accepted and converted to RGB
    using a Matplotlib colormap before rendering.

    Parameters
    ----------
    data : numpy.ndarray
        RGB volume with shape ``(D, H, W, 3)`` or an integer label volume with
        shape ``(D, H, W)``.
    matrix_opacity : float, default=0.15
        Opacity assigned to non-enhancement voxels.
    enhance_threshold : float, default=0.02
        A voxel is classified as enhancement phase when all RGB channels are
        below this value.
    save_dir : str or path-like, optional
        Vector output path accepted by :func:`visual`, for example ``.pdf``.
        The existing :func:`visual` implementation also saves a PNG preview.
    color_map : str, default="tab20"
        Colormap used when ``data`` is an integer label volume.

    Returns
    -------
    numpy.ndarray
        Boolean enhancement-phase mask, useful for reporting or validation.
    """
    data = np.asarray(data)
    if not 0.0 <= matrix_opacity <= 1.0:
        raise ValueError("matrix_opacity must be between 0 and 1.")

    if data.ndim == 3:
        data = label_volume_to_rgb(data, color_map=color_map)
    elif data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("data must have shape (D, H, W) or (D, H, W, 3).")

    data = np.asarray(data, dtype=np.float64)
    if data.max() > 1.0:
        data = data / 255.0
    data = np.clip(data, 0.0, 1.0)

    enhance_mask = (data < enhance_threshold).all(axis=-1)
    opacity = np.where(enhance_mask, 1.0, matrix_opacity).astype(np.float64)
    if save_dir is not None:
        save_dir = os.fspath(save_dir)
    visual(data, opacity=opacity, save_dir=save_dir)
    return enhance_mask


def crop_visual_square(image_path, output_path=None, output_size=1024):
    """Crop the standard PyVista ``visual`` screenshot to a square image.

    The project's 3D screenshots are normally 4:3 images with the rendered
    volume centered horizontally and the orientation axes in the lower-left
    corner. This fixed crop keeps the full central volume, removes the axes,
    and aligns the crop with the bottom edge so the lower vertex is retained.

    Parameters
    ----------
    image_path : str or path-like
        PNG screenshot generated by :func:`visual`.
    output_path : str or path-like, optional
        Destination PNG. By default, ``_square`` is appended to the filename.
    output_size : int, default=1024
        Width and height of the standardized square output.

    Returns
    -------
    str
        Path of the saved square image.
    """
    image_path = os.fspath(image_path)
    if output_path is None:
        stem, extension = os.path.splitext(image_path)
        output_path = stem + "_square" + (extension or ".png")
    output_path = os.fspath(output_path)

    image = Image.open(image_path)
    width, height = image.size

    # Keep 94% of the screenshot height. For the standard 1024x768 output,
    # this gives a 722x722 crop: x=151..873 and y=46..768. The orientation
    # axes end to the left of the crop while the complete volume is retained.
    crop_size = min(width, int(round(height * 0.94)))
    left = max(0, int(round((width - crop_size) / 2)))
    top = max(0, height - crop_size)
    square = image.crop((left, top, left + crop_size, top + crop_size))

    if output_size is not None and square.size != (output_size, output_size):
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        square = square.resize((output_size, output_size), resampling)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    square.save(output_path)
    image.close()
    return output_path

def slice(npy_path, im_path):
    """
    读取一个形状为 (n, n, n, 3) 的 npy 文件，截取 z 轴最上面的一片，并保存为 PNG 图片。

    参数:
    npy_path (str): 输入的 npy 文件路径。
    im_path (str): 输出的 PNG 图片路径。
    """
    # 加载 npy 文件
    data = np.load(npy_path)

    # 检查数据形状是否符合要求
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("输入的 npy 文件形状必须为 (n, n, n, 3)")

    # 截取 z 轴最上面的一片（z=0）
    slice_data = data[:, :, -1, :]  # shape: (n, n, 3)

    # 将数据范围从 [0, 1] 转换到 [0, 255]，并转换为 uint8 类型
    if slice_data.max() <= 1.0:
        slice_data = (slice_data * 255).astype(np.uint8)

    # 将 NumPy 数组转换为 PIL 图像
    image = Image.fromarray(slice_data)

    # 保存为 PNG 图片
    image.save(im_path)
    print(f"图片已保存到 {im_path}")

def plot_loss(data):
    if len(data.shape) == 1:
        plt.plot(data)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss curve")
        plt.show()
    elif len(data.shape) == 2:
        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss curve")
        plt.show()

def save_slice(im, out_dir, out_file):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(im)

    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['left'].set_visible(False)  # 隐藏左边框
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框

    plt.savefig(os.path.join(out_dir, out_file), dpi=300, bbox_inches='tight', pad_inches=0)


def save_orthogonal_slices(data, out_dir, prefix, indices=None,
                           formats=("png", "pdf")):
    """Save one slice normal to each axis of an RGB volume.

    This function reuses :func:`save_slice` and defaults to the three center
    slices, ensuring that reference and generated structures can be compared
    at consistent relative positions.

    Parameters
    ----------
    data : numpy.ndarray
        RGB volume with shape ``(D, H, W, 3)``.
    out_dir : str or path-like
        Output directory.
    prefix : str
        Filename prefix, e.g. ``training_example``.
    indices : sequence of three ints, optional
        Slice indices for axes 0, 1, and 2. Center indices are used by default.
    formats : sequence of str, default=("png", "pdf")
        Output formats supported by Matplotlib.

    Returns
    -------
    dict
        Mapping from axis name to the selected slice index.
    """
    data = np.asarray(data)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("data must have shape (D, H, W, 3).")

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if indices is None:
        indices = tuple(size // 2 for size in data.shape[:3])
    if len(indices) != 3:
        raise ValueError("indices must contain one index for each spatial axis.")

    axis_slices = {
        "x": data[indices[0], :, :, :],
        "y": data[:, indices[1], :, :],
        "z": data[:, :, indices[2], :],
    }
    for axis_name, image in axis_slices.items():
        for extension in formats:
            save_slice(image, out_dir, f"{prefix}_{axis_name}.{extension}")
            plt.close()

    return dict(zip(("x", "y", "z"), indices))

def get_colors(color_num, color_map="tab20", plot=False):
    cmap = cm.get_cmap(color_map)
    # for tab 20, if you give index from 0-19, the corresponding color will be given
    # if index >= 20, the last color will be given

    # get the first color_num colors
    idx = np.arange(color_num)

    # get random color_num colors from tab20
    # idx = np.random.choice(20, color_num, replace=False)

    colors = cmap(idx)[:, :3]  # 0-1 的 RGB

    if plot:
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 2))
        for i, c in enumerate(colors):
            ax.barh(0, 1, left=i, color=c, height=2, edgecolor='k')
            # 文字：索引 + RGB
            rgb = (c * 255).astype(int)
            ax.text(i + 0.5, 0.5, f'{i}',
                    ha='center', va='center', fontsize=8, color='w')

        ax.set_xlim(0, color_num)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{color_num} colors from tab20')
        plt.tight_layout()
        plt.show()

    return colors

def quantize_colors(img, colors):
    """
    img ~ (n, n, n, 3): RGB voxels

    colors ~ (color_num, 3): color_num RGB colors

    return
    color_indices ~ (n, n, n)
    """
    img = np.asarray(img)
    colors = np.asarray(colors)

    if colors.shape != (10, 3):
        raise ValueError("colors 必须是 (10, 3) 的数组")

    n = img.shape[0]
    # 1. 拉平成 (N,3)
    flat_img = img.reshape(-1, 3)

    # 2. 计算欧氏距离平方，结果形状 (N,10)
    dist2 = np.sum((flat_img[:, None, :] - colors[None, :, :]) ** 2, axis=2)

    # 3. 取最小值所在列索引
    labels = np.argmin(dist2, axis=1)

    # 4. 还原形状
    return labels.reshape(n, n, n)


def orientation_fractions(img, colors=None):
    """Return voxel fractions for the ten encoded orientation classes.

    The RGB volume is first projected onto the same ten-color palette used by
    :func:`quantize_colors`. The result always contains ten entries, including
    zeros for orientation classes that are absent from a sample.
    """
    if colors is None:
        colors = get_colors(10)

    labels = quantize_colors(img, colors)
    counts = np.bincount(labels.reshape(-1), minlength=len(colors))
    return counts.astype(np.float64) / labels.size


def orientation_fraction_table(model_files, colors=None, output_csv=None):
    """Calculate an ``n x 10`` orientation-fraction table for RGB volumes.

    Parameters
    ----------
    model_files : iterable of path-like
        Paths to ``.npy`` RGB volumes. The input order is preserved.
    colors : numpy.ndarray, optional
        The ``(10, 3)`` RGB-to-orientation codebook.
    output_csv : path-like, optional
        Destination CSV. It contains exactly ten orientation columns.
    """
    if colors is None:
        colors = get_colors(10)

    model_files = list(model_files)
    table = np.empty((len(model_files), len(colors)), dtype=np.float64)
    for row, model_file in enumerate(model_files):
        table[row] = orientation_fractions(np.load(model_file), colors)

    if output_csv is not None:
        output_csv = os.fspath(output_csv)
        output_parent = os.path.dirname(output_csv)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        header = ",".join(f"orientation_{i + 1}" for i in range(len(colors)))
        np.savetxt(output_csv, table, delimiter=",", header=header, comments="")

    return table

def get_grains(model_input, color_num=10, cut_off=200, min_distance=5, v=False):
    # read model file
    if isinstance(model_input, np.ndarray):
        model = model_input
    elif isinstance(model_input, (str, os.PathLike)) and os.path.isfile(model_input) and str(model_input).lower().endswith('.npy'):
        model = np.load(model_input)
    else:
        raise ValueError('Input should be a ndarray or directory path of an ndarray')

    # get grain label array
    # start from 1
    grain_label = np.zeros(model.shape[:3], dtype=np.int32)

    next_grid = 1
    # get the colors to do cluster
    colors = get_colors(color_num, plot=False)
    # get the index model, from 0 to color_num-1
    index_model = quantize_colors(model, colors)
    # print(np.unique(index_model))
    # temp_model = colors[index_model]
    # visual(temp_model)
    # for color_index in np.unique(index_model):
    #     this_color = (index_model == color_index) * color_index
    #     this_color = colors[this_color]
    #     this_color[:, :32], this_color[:, 32:64] = this_color[:, 32:64].copy(), this_color[:, :32].copy()
    #     visual(this_color)
    connected_grains = []

    ori_sum = 0
    comp_grains = 0
    big_comp_grains = 0
    for ori in np.unique(index_model):
        # focus on this orientation, others are background
        ori_mask = (index_model == ori)  # binary model for this orientation
        ori_sum += np.sum(ori_mask)

        comps, n_comp = ndimage.label(ori_mask)

        assert len(np.unique(comps)) == n_comp+1
        comp_grains += (len(np.unique(comps)) - 1)

        for k in range(1, n_comp+1):
            # for each grain, we still need to detect possible connected grains
            this_grain_mask = (comps == k)
            this_volume = np.sum(this_grain_mask)

            # skip small grains
            if this_volume > cut_off:
                big_comp_grains += 1
                # detect possible connected grains
                # --- Distance transform ---
                distance = ndimage.distance_transform_edt(this_grain_mask)

                # --- Find local maxima inside the grain ---
                coords = peak_local_max(
                    distance,
                    labels=this_grain_mask,
                    footprint=np.ones((3, 3, 3)),  # 3x3x3 neighborhood
                    min_distance=min_distance
                )

                if len(coords) == 0:
                    max_idx = np.unravel_index(np.argmax(distance * this_grain_mask), distance.shape)
                    coords = np.array([max_idx])
                else:
                    coords = coords[distance[tuple(coords.T)] > 2]
                    if len(coords) == 0:
                        max_idx = np.unravel_index(np.argmax(distance * this_grain_mask), distance.shape)
                        coords = np.array([max_idx])

                markers = np.zeros_like(this_grain_mask, dtype=np.int32)
                for i, (z, y, x) in enumerate(coords, start=1):
                    markers[z, y, x] = i

                # --- Apply watershed on inverted distance map ---
                sub_grains = watershed(-distance, markers, mask=this_grain_mask)

                assert len(np.unique(sub_grains)) == len(np.unique(markers))
                assert len(np.unique(sub_grains)) == len(coords) + 1

                detected = sub_grains.max()
                connected_grains.append(detected)

                for j in range(1, detected+1):
                    sub_grain_mask = (sub_grains == j)
                    sub_grain_volume = np.sum(sub_grain_mask)

                    if sub_grain_volume > cut_off:
                        grain_label[sub_grain_mask] = next_grid
                        next_grid += 1

                        if v and detected>1:
                            # visualize grain and segmented grain
                            padded_mask = np.pad(this_grain_mask, pad_width=1, mode='constant', constant_values=0)
                            verts, faces, normals, values = measure.marching_cubes(padded_mask.astype(np.uint8))
                            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

                            padded_mask = np.pad(sub_grain_mask, pad_width=1, mode='constant', constant_values=0)
                            verts, faces, normals, values = measure.marching_cubes(padded_mask.astype(np.uint8))
                            x1, y1, z1 = verts[:, 0], verts[:, 1], verts[:, 2]

                            fig = plt.figure(figsize=(6, 6))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(x, y, z, s=2, c='tab:blue', alpha=0.6)
                            ax.scatter(x1, y1, z1, s=2, c='tab:red', alpha=0.6)
                            ax.scatter(coords[j-1, 0], coords[j-1, 1], coords[j-1, 2], c='tab:green')
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                            ax.set_zlabel("Z")
                            ax.set_title("Marching Cubes Vertices (Grain Surface)")
                            # Make aspect ratio equal
                            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                            mid_x = 0.5 * (x.max() + x.min())
                            mid_y = 0.5 * (y.max() + y.min())
                            mid_z = 0.5 * (z.max() + z.min())
                            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
                            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
                            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
                            plt.show()

    # print(comp_grains)
    # print(big_comp_grains)
    return grain_label, connected_grains

def anal_grians(grain_model, voxel_size=(1, 1, 1)):
    # voxel_size in unit nm
    # here we assume
    grain_ids = np.unique(grain_model)
    grain_ids = grain_ids[grain_ids != 0]
    results = {}
    for g in grain_ids:
        coords = np.array(np.where(grain_model == g)).T  # (M,3) in voxel indices
        M = coords.shape[0] # volume

        coords_phys = coords * np.array(voxel_size)  # apply space scaling
        centroid = coords_phys.mean(axis=0)

        C = np.cov((coords_phys - centroid).T)
        eigvals = np.linalg.eigvalsh(C)[::-1]  # λ1 ≥ λ2 ≥ λ3
        axis_lengths = np.sqrt(eigvals)  # proportional
        ar_major_minor = axis_lengths[0] / (axis_lengths[2] + 1e-12)



        # fit grain
        mask = (grain_model == g)
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

        verts, faces, normals, values = measure.marching_cubes(padded_mask.astype(np.uint8),
                                                               spacing=voxel_size)

        # x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        # # Plot using matplotlib
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, s=2, c='tab:blue', alpha=0.6)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_title("Marching Cubes Vertices (Grain Surface)")
        # # Make aspect ratio equal
        # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        # mid_x = 0.5 * (x.max() + x.min())
        # mid_y = 0.5 * (y.max() + y.min())
        # mid_z = 0.5 * (z.max() + z.min())
        # ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        # ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        # ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        # plt.show()

        # # compute surface area
        triangles = verts[faces]  # shape (n_faces, 3, 3)
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]

        A = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1).sum()

        # compute volume
        V = M * np.prod(voxel_size)

        # get sphericity
        ball_r = np.power(3 * V / (4 * np.pi), 1 / 3)
        ball_area = 4 * np.pi * ball_r ** 2
        psi = ball_area / A


        results[g] = {'volume_vox': M,
                      'volume': M * np.prod(voxel_size),
                      'centroid': centroid,
                      'axis_lengths': axis_lengths,
                      'aspect_ratio': ar_major_minor,
                      'surface_area': A,
                      'sphericity': psi
                    }

    return results

def analyze_grain_results(results):
    """
    Create multi-panel plots (subfigures) to visualize grain morphology statistics.

    Parameters
    ----------
    results : dict
        Dictionary with per-grain results. Must include:
        'volume', 'volume_vox', 'centroid', 'axis_lengths', 'aspect_ratio'
    """
    # --- Collect data ---
    grain_ids = list(results.keys())
    volumes = np.array([results[g]['volume'] for g in grain_ids])
    volumes_vox = np.array([results[g]['volume_vox'] for g in grain_ids])
    aspect_ratios = np.array([results[g]['aspect_ratio'] for g in grain_ids])
    axis_lengths = np.array([results[g]['axis_lengths'] for g in grain_ids])  # (n_grains, 3)
    centroids = np.array([results[g]['centroid'] for g in grain_ids])        # (n_grains, 3)

    # --- Print summary statistics ---
    print("="*40)
    print(f"Number of grains analyzed: {len(grain_ids)}")
    print(f"Mean grain volume: {volumes.mean():.3f} ± {volumes.std():.3f}")
    print(f"Mean aspect ratio: {aspect_ratios.mean():.3f} ± {aspect_ratios.std():.3f}")
    print(f"Mean axis lengths: major={axis_lengths[:,0].mean():.3f}, "
          f"intermediate={axis_lengths[:,1].mean():.3f}, minor={axis_lengths[:,2].mean():.3f}")
    print("="*40)

    # --- Create subplots ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Grain Morphology Statistics", fontsize=14)

    # --- Subplot 1: Grain Volume Distribution ---
    axs[0].hist(volumes, bins=30, alpha=0.7, color='tab:blue', edgecolor='black')
    axs[0].set_xlabel("Grain Volume")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Volume Distribution")
    axs[0].grid(alpha=0.3)

    # --- Subplot 2: Aspect Ratio Distribution ---
    axs[1].hist(aspect_ratios, bins=30, alpha=0.7, color='tab:orange', edgecolor='black')
    axs[1].axvline(1.0, color='red', linestyle='--', label="Equiaxed (AR=1)")
    axs[1].set_xlabel("Aspect Ratio (Major / Minor)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Aspect Ratio Distribution")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # --- Subplot 3: Axis Length Boxplot ---
    axs[2].boxplot(axis_lengths, labels=['Major', 'Intermediate', 'Minor'], patch_artist=True,
                   boxprops=dict(facecolor='lightgray'))
    axs[2].set_ylabel("Axis Length")
    axs[2].set_title("Principal Axis Lengths")
    axs[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

# analyze the RGB distribution of a model
def anal_rgb(data):
    hist_range = np.array(range(256))
    flat_im = data.reshape(-1, 3)

    R_hist, R_edges = np.histogram(flat_im[:, 0], bins=hist_range)
    G_hist, G_edges = np.histogram(flat_im[:, 1], bins=hist_range)
    B_hist, B_edges = np.histogram(flat_im[:, 2], bins=hist_range)

    R_cdf = np.cumsum(R_hist)
    G_cdf = np.cumsum(G_hist)
    B_cdf = np.cumsum(B_hist)
    R_cdf = R_cdf / R_cdf[-1]
    G_cdf = G_cdf / G_cdf[-1]
    B_cdf = B_cdf / B_cdf[-1]

    return [R_cdf, G_cdf, B_cdf]

def VGG_score(VGG_model, img, loss_layers=['r11', 'r21', 'r31', 'r41', 'r51']):
    # img is an array shapes like  (n, n, 3)
    # VGG model is nn.Module

    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    device = next(VGG_model.parameters()).device
    img_tensor = img_tensor.to(device)

    im_feature_maps = VGG_model(img_tensor, loss_layers)
    im_Gs = [GramMatrix(layer_feature) for layer_feature in im_feature_maps]

    scores = [np.linalg.norm(G.detach().cpu().numpy()) for G in im_Gs]
    return np.sum(scores)

def two_point_correlation(volume, bins=50):
    F = fftn(volume)
    autocorr = np.real(ifftn(F * np.conjugate(F)))
    autocorr = fftshift(autocorr)
    autocorr /= volume.size

    n = volume.shape[0]
    x = np.arange(-n//2, n//2)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    r_flat = r.flatten()
    s2_flat = autocorr.flatten()

    r_bins = np.linspace(0, r_flat.max(), bins+1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    s2 = np.zeros_like(r_centers)

    for i in range(bins):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(mask):
            s2[i] = np.mean(s2_flat[mask])

    return r_centers, s2

def weighted_s2(color_index, bins=50, color_num=10):
    """Compute volume-fraction-weighted S2 curve for one microstructure."""
    s2_list = []
    weights = []

    for oid in range(color_num):
        binary = (color_index == oid).astype(np.float32)
        _, s2 = two_point_correlation(binary, bins=bins)
        vol_frac = binary.mean()
        s2_list.append(s2)
        weights.append(vol_frac)

    s2_array = np.vstack(s2_list)
    weights = np.array(weights)
    # return np.average(s2_array, axis=0, weights=weights)
    return np.sum(s2_array, axis=0)
