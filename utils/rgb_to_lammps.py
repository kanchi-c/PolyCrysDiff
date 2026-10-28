"""Convert a 64^3 RGB PolyCrysDiff volume to a LAMMPS atomic data file.

The first ten tab20 colors are *labels*, not measured crystallographic
orientations. DEFAULT_EULER_DEG is an illustrative mapping only. Supply the
actual color-to-orientation angles via --orientations-json for physical use.

By default, all ten orientation classes are Al atom types. With
--visualization-elements, they are labeled H through Ne (Z=1..10) only for
OVITO display; coordinates still come from Al FCC, and that file must not be
used for physical MD. Coordinates and box lengths are in angstroms.
Grain-boundary atoms are unrelaxed: minimize a physical Al model before MD.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:  # NumPy-only fallback for environments without SciPy.
    cKDTree = None


# Same values as matplotlib's tab20 indices 0..9, used by data/Voronoi_new.py.
PALETTE = np.array(
    [
        (31, 119, 180), (174, 199, 232),
        (255, 127, 14), (255, 187, 120),
        (44, 160, 44), (152, 223, 138),
        (214, 39, 40), (255, 152, 150),
        (148, 103, 189), (197, 176, 213),
    ],
    dtype=np.float64,
) / 255.0

# Illustrative rotations (degrees): Rx, Ry, Rz, with R = Rz @ Ry @ Rx.
# Replace these with physically chosen orientations before scientific use.
DEFAULT_EULER_DEG = (
    (0, 0, 0), (20, 0, 0), (0, 20, 0), (0, 0, 20),
    (30, 15, 0), (0, 35, 25), (15, 25, 40),
    (45, 0, 20), (25, 40, 10), (40, 20, 35),
)
AL_MASS = 26.9815385
# Visualization-only mapping: type ID equals atomic number, 1 (H) through 10 (Ne).
# Positions still come from rotated Al FCC lattices, so these are not chemical species.
VISUAL_ELEMENTS = (
    ("H", 1.008), ("He", 4.002602), ("Li", 6.94), ("Be", 9.0121831),
    ("B", 10.81), ("C", 12.011), ("N", 14.007), ("O", 15.999),
    ("F", 18.998403163), ("Ne", 20.1797),
)


def _load_xsf(path):
    """Read lattice vectors and Cartesian atom positions from PRIMVEC/PRIMCOORD."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tokens = [line.strip().split() for line in lines if line.strip()]
    names = [part[0].upper() for part in tokens]
    try:
        v = names.index("PRIMVEC")
        c = names.index("PRIMCOORD")
        vectors = np.array(
            [[float(x) for x in tokens[v + k][0:3]] for k in (1, 2, 3)],
            dtype=np.float64,
        )
        count = int(tokens[c + 1][0])
        rows = tokens[c + 2:c + 2 + count]
        if len(rows) != count or any(int(row[0]) != 13 for row in rows):
            raise ValueError("The reference XSF must contain only Al atoms (Z=13).")
        basis = np.array(
            [[float(x) for x in row[1:4]] for row in rows], dtype=np.float64
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not read the Al unit cell from {path}") from exc
    volume = abs(float(np.linalg.det(vectors)))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("The XSF lattice vectors do not define a valid cell.")
    return vectors, basis, volume


def _rotation(euler_deg):
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    my = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    mz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return mz @ my @ mx


def _orientation_table(angles):
    result = np.asarray(
        DEFAULT_EULER_DEG if angles is None else angles, dtype=np.float64
    )
    if result.shape != (10, 3) or not np.all(np.isfinite(result)):
        raise ValueError("orientations must be ten [Rx, Ry, Rz] triples in degrees.")
    return result


def _labels_from_rgb(structure, generated, generated_range="auto"):
    if isinstance(structure, (str, os.PathLike)):
        rgb = np.load(structure, allow_pickle=False)
    else:
        # A torch tensor works without importing torch as a dependency.
        if hasattr(structure, "detach"):
            structure = structure.detach().cpu().numpy()
        rgb = np.asarray(structure)
    if rgb.shape == (3, 64, 64, 64):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape != (64, 64, 64, 3):
        raise ValueError(
            f"Expected (64,64,64,3) or (3,64,64,64), got {rgb.shape}."
        )
    rgb = np.asarray(rgb, dtype=np.float64)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("The RGB volume contains NaN or infinite values.")
    low, high = float(rgb.min()), float(rgb.max())
    if generated:
        if generated_range not in ("auto", "zero-one", "minus-one-one"):
            raise ValueError("generated_range must be auto, zero-one, or minus-one-one.")
        mode = generated_range
        if mode == "auto":
            mode = "minus-one-one" if low < -0.1 else "zero-one"
        if mode == "minus-one-one":
            if low < -1.1 or high > 1.1:
                raise ValueError(f"Generated values outside [-1,1]: [{low},{high}].")
            rgb = (rgb + 1.0) / 2.0
        elif low < -0.1 or high > 1.1:
            raise ValueError(f"Generated values outside [0,1]: [{low},{high}].")
        rgb = np.clip(rgb, 0.0, 1.0)
    elif low < -1e-6 or high > 1.0 + 1e-6:
        raise ValueError(f"Dataset RGB values must lie in [0,1], got [{low},{high}].")

    # Black is an extra phase in some enhanced-phase datasets, not one of
    # the ten Al orientation classes. Do not silently turn it into Al.
    if np.any(np.max(rgb, axis=-1) < 0.08):
        raise ValueError("Black/near-black voxels are not one of the ten Al colors.")

    flat = rgb.reshape(-1, 3)
    best_dist2 = np.full(len(flat), np.inf)
    labels = np.zeros(len(flat), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        dist2 = np.sum((flat - color) ** 2, axis=1)
        use = dist2 < best_dist2
        labels[use] = idx
        best_dist2[use] = dist2[use]
    if not generated and float(np.sqrt(best_dist2.max())) > 1e-4:
        raise ValueError(
            "Dataset volume contains RGB values outside the ten tab20 colors; "
            "check its palette or pass generated=True for model output."
        )
    return labels.reshape(64, 64, 64), float(np.sqrt(best_dist2.max()))


def _atoms_for_orientation(labels, label, rotation, vectors, basis, voxel_size):
    """Create one continuous rotated FCC lattice; retain atoms inside label voxels."""
    box = np.array(labels.shape, dtype=np.float64) * voxel_size
    center = box / 2.0
    corners = np.array(list(itertools.product(
        (0.0, box[0]), (0.0, box[1]), (0.0, box[2])
    )))
    # Row-vector convention: specimen = crystal @ R.T + center.
    crystal_corners = (corners - center) @ rotation
    inverse_vectors = np.linalg.inv(vectors)
    fractional = np.concatenate(
        [(crystal_corners - b) @ inverse_vectors for b in basis], axis=0
    )
    first = np.floor(fractional.min(axis=0)).astype(int) - 1
    last = np.ceil(fractional.max(axis=0)).astype(int) + 1
    j, k = np.meshgrid(
        np.arange(first[1], last[1] + 1),
        np.arange(first[2], last[2] + 1),
        indexing="ij",
    )
    jk = np.column_stack((j.ravel(), k.ravel()))
    accepted = []
    for i in range(first[0], last[0] + 1):
        lattice = (
            i * vectors[0]
            + jk[:, 0, None] * vectors[1]
            + jk[:, 1, None] * vectors[2]
        )
        for b in basis:
            xyz = (lattice + b) @ rotation.T + center
            inside = np.all((xyz >= 0.0) & (xyz < box), axis=1)
            if not inside.any():
                continue
            xyz = xyz[inside]
            index = np.floor(xyz / voxel_size).astype(np.intp)
            selected = labels[index[:, 0], index[:, 1], index[:, 2]] == label
            if selected.any():
                accepted.append(xyz[selected])
    if not accepted:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(accepted, axis=0)


def _remove_close_pairs(positions, types, distance, box):
    """Remove one atom from each pair closer than distance, including across PBC."""
    if distance <= 0 or len(positions) < 2:
        return positions, types, 0
    keep = np.ones(len(positions), dtype=bool)
    if cKDTree is not None:
        pairs = cKDTree(positions, boxsize=box).query_pairs(
            distance, output_type="ndarray"
        )
        for first, second in pairs:
            if keep[first] and keep[second]:
                keep[second] = False
    else:
        # Spatial hash: each bin is at least distance wide, so only the
        # current bin and its 26 periodic neighbors need examination.
        bins = np.maximum(1, np.floor(box / distance).astype(int))
        widths = box / bins
        buckets = {}
        distance2 = distance * distance
        for idx, point in enumerate(positions):
            key = tuple(np.floor(point / widths).astype(int))
            neighbors = {
                ((key[0] + dx) % bins[0],
                 (key[1] + dy) % bins[1],
                 (key[2] + dz) % bins[2])
                for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
            }
            overlap = False
            for neighbor in neighbors:
                for previous in buckets.get(neighbor, ()):
                    delta = np.abs(point - positions[previous])
                    delta = np.minimum(delta, box - delta)
                    if float(delta @ delta) < distance2:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                keep[idx] = False
            else:
                buckets.setdefault(key, []).append(idx)
    return positions[keep], types[keep], int((~keep).sum())


def _write_lammps(path, positions, types, box, angles, visualization_elements):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as out:
            out.write("LAMMPS data: PolyCrysDiff rotated Al FCC coordinates\n")
            if visualization_elements:
                out.write(
                    "# VISUALIZATION ONLY: types 1..10 denote Z=1..10 (H..Ne).\n"
                    "# Atom coordinates use Al FCC; masses/elements are artificial.\n"
                    "# Do not use this file for physical MD calculations.\n\n"
                )
            else:
                out.write("# Coordinates in angstroms; all atom types are Al.\n")
                out.write("# Unrelaxed grain boundaries; minimize before dynamics.\n\n")
            out.write(f"{len(positions)} atoms\n")
            out.write("10 atom types\n\n")
            for axis, length in zip("xyz", box):
                out.write(f"0.0 {length:.8f} {axis}lo {axis}hi\n")
            if visualization_elements:
                out.write("\nAtom Type Labels\n\n")
                for t, (symbol, _) in enumerate(VISUAL_ELEMENTS, start=1):
                    out.write(f"{t} {symbol}\n")
            out.write("\nMasses\n\n")
            for t in range(1, 11):
                a = angles[t - 1]
                symbol, element_mass = VISUAL_ELEMENTS[t - 1]
                mass = element_mass if visualization_elements else AL_MASS
                name = symbol if visualization_elements else "Al"
                out.write(
                    f"{t} {mass:.7f} # {name}, tab20 index {t-1}, "
                    f"Euler XYZ deg: {a[0]:g} {a[1]:g} {a[2]:g}\n"
                )
            out.write("\nAtoms # atomic\n\n")
            for start in range(0, len(positions), 50000):
                stop = min(start + 50000, len(positions))
                ids = np.arange(start + 1, stop + 1)
                data = np.column_stack((ids, types[start:stop], positions[start:stop]))
                np.savetxt(out, data, fmt="%d %d %.8f %.8f %.8f")
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def structure_to_lammps(
    structure,
    generated,
    voxel_size_angstrom,
    output_path,
    orientation_euler_deg=None,
    xsf_path=None,
    min_separation_angstrom=2.0,
    max_atoms=2_000_000,
    generated_range="auto",
    visualization_elements=False,
    dry_run=False,
):
    """Export one 64^3 RGB structure to a LAMMPS data file.

    structure: NumPy array, torch tensor, or .npy path, in DHWC or CDHW order.
    generated=False: exact tab20 dataset colors in [0,1].
    generated=True: clip/rescale model output and quantize to nearest tab20 color.
    voxel_size_angstrom: physical length of one voxel (NOT the whole cube).
    orientation_euler_deg: optional ten [Rx,Ry,Rz] angles; default is illustrative.
    min_separation_angstrom: remove boundary atom overlaps using a periodic KD-tree.
    max_atoms: safety cap, including a pre-generation density estimate.
    visualization_elements: type IDs become Z=1..10 for OVITO only, not MD.
    """
    if not isinstance(generated, bool):
        raise TypeError("generated must be a bool.")
    voxel_size = float(voxel_size_angstrom)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size_angstrom must be a positive finite number.")
    if not np.isfinite(min_separation_angstrom) or min_separation_angstrom < 0:
        raise ValueError("min_separation_angstrom must be nonnegative and finite.")
    if max_atoms < 1:
        raise ValueError("max_atoms must be positive.")
    if xsf_path is None:
        xsf_path = Path(__file__).resolve().parent.parent / "config" / "Al_fcc.xsf"
    vectors, basis, cell_volume = _load_xsf(xsf_path)
    angles = _orientation_table(orientation_euler_deg)
    labels, max_color_error = _labels_from_rgb(
        structure, generated, generated_range
    )
    box = np.array(labels.shape, dtype=np.float64) * voxel_size
    estimate = int(math.ceil(np.prod(box) * len(basis) / cell_volume))
    counts = np.bincount(labels.ravel(), minlength=10)
    info = {
        "estimated_atoms": estimate,
        "color_voxel_counts": counts.tolist(),
        "max_color_error": max_color_error,
        "box_angstrom": box.tolist(),
    }
    if dry_run:
        return info
    if estimate > max_atoms:
        raise ValueError(
            f"Estimated {estimate:,} Al atoms exceeds max_atoms={max_atoms:,}. "
            "A full 64^3 structure at micrometre-scale voxel sizes is too large "
            "for a conventional atomistic model. Use a smaller physical voxel "
            "size or an explicitly selected subvolume; raising max_atoms alone "
            "may exhaust memory and disk."
        )
    positions_by_type = []
    type_arrays = []
    for label in np.flatnonzero(counts):
        xyz = _atoms_for_orientation(
            labels, int(label), _rotation(angles[label]), vectors, basis, voxel_size
        )
        positions_by_type.append(xyz)
        type_arrays.append(np.full(len(xyz), int(label) + 1, dtype=np.int16))
    positions = np.concatenate(positions_by_type, axis=0)
    types = np.concatenate(type_arrays)
    if len(positions) > max_atoms:
        raise ValueError(f"Generated {len(positions):,} atoms exceeds max_atoms.")
    positions, types, removed = _remove_close_pairs(
        positions, types, float(min_separation_angstrom), box
    )
    _write_lammps(output_path, positions, types, box, angles, visualization_elements)
    info.update({"atoms_written": len(positions), "overlap_atoms_removed": removed,
                 "visualization_elements": visualization_elements})
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One 64^3 RGB .npy volume")
    parser.add_argument("--output", help="LAMMPS data file destination")
    parser.add_argument(
        "--voxel-size-angstrom", required=True, type=float,
        help="Physical length of one voxel in angstroms",
    )
    parser.add_argument(
        "--generated", action="store_true",
        help="Quantize generated RGB output; accepts [0,1] or [-1,1]",
    )
    parser.add_argument(
        "--generated-range", choices=("auto", "zero-one", "minus-one-one"),
        default="auto",
    )
    parser.add_argument(
        "--orientations-json",
        help="JSON file with ten [Rx,Ry,Rz] Euler-angle triples in degrees",
    )
    parser.add_argument("--xsf", help="Reference Al FCC XSF cell")
    parser.add_argument("--min-separation-angstrom", type=float, default=2.0)
    parser.add_argument("--max-atoms", type=int, default=2_000_000)
    parser.add_argument(
        "--visualization-elements", action="store_true",
        help="Map the ten colors to artificial elements H..Ne (Z=1..10); OVITO only",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report size")
    args = parser.parse_args()
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is used")
    angles = None
    if args.orientations_json:
        angles = json.loads(Path(args.orientations_json).read_text(encoding="utf-8"))
    info = structure_to_lammps(
        args.input, args.generated, args.voxel_size_angstrom, args.output,
        orientation_euler_deg=angles,
        xsf_path=args.xsf,
        min_separation_angstrom=args.min_separation_angstrom,
        max_atoms=args.max_atoms,
        generated_range=args.generated_range,
        visualization_elements=args.visualization_elements,
        dry_run=args.dry_run,
    )
    print(json.dumps(info, indent=2))
    if args.orientations_json is None:
        print(
            "WARNING: Using illustrative Euler angles. Supply --orientations-json "
            "with physically intended orientations before scientific use."
        )


if __name__ == "__main__":
    main()

