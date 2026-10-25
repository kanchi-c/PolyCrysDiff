import numpy as np
import torch


def load_volume_tensor(path, im_size, im_channels):
    """Load one NDHWC NumPy volume and normalize it to an NCDHW tensor."""
    data = np.load(path).astype(np.float32, copy=False)
    expected = (im_size, im_size, im_size, im_channels)
    if data.shape != expected:
        raise ValueError(f'Expected {expected}, got {data.shape} in {path}')
    data = np.ascontiguousarray(data.transpose(3, 0, 1, 2))
    return torch.from_numpy(data).mul(2).sub(1)
