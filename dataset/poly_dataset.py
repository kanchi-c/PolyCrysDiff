import os
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data.dataset import Dataset

from dataset.npy_utils import load_volume_tensor


class Poly_Dataset(Dataset):
    r"""
    Dataset for polycrystalline structure volumes.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """

    def __init__(self, split, root_dir, im_size, im_channels, debug=False, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param root_dir: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.root_dir = root_dir
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.debug = debug

        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.models = self.load_models(root_dir) # paths loaded on demand in __getitem__
        self.infos = self.get_infos(root_dir) # record the num seed and color num information

        # For value condition
        if 'value' in self.condition_types:
            value_config = condition_config["value_condition_config"]
            keys = value_config['keys']
            len_vector = value_config['len_vector']
            assert len_vector == len(keys)

            self.keys = keys

    def load_models(self, root_dir):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(root_dir), "images path {} does not exist".format(root_dir)

        model_files = []
        if self.debug:
            for file in os.listdir(root_dir):
                if file.endswith(".npy"):
                    model_files.append(os.path.join(root_dir, file))
                    break
        else:
            for file in os.listdir(root_dir):
                if file.endswith(".npy"):
                    model_files.append(os.path.join(root_dir, file))
        model_files.sort(key=lambda path: int(re.search(r'(?<=model)\d+(?=\.npy)', path).group()))
        self.model_ids = [int(re.search(r'(?<=model)\d+(?=\.npy)', model_file).group()) for model_file in model_files]
        self.model_files = model_files
        print(f"{len(model_files)} model paths loaded.")
        return model_files

    def get_infos(self, root_dir):
        info_file = os.path.join(root_dir, "info.txt")

        if os.path.exists(info_file):
            data = pd.read_table(info_file, sep=r"\s*,\s*", index_col='index', engine='python')
            return data
        else:
            return None

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        ## Our model are shape like (n, n, n, 3) and the range is [0, 1]
        ######## Set Conditioning Info ########
        # print(self.model_ids[index])
        # print(self.model_files[index])
        # print(self.infos.iloc[self.model_ids[index]])
        cond_inputs = {}

        model_tensor = load_volume_tensor(
            self.models[index], self.im_size, self.im_channels
        )
        if 'image' in self.condition_types:
            cond_inputs['image'] = model_tensor[:, :, :, -1]

        if 'value' in self.condition_types:
            sph_mean = 0.7684357651064376
            sph_std = 0.018839151612943705
            ratios_mean = 1.828678747187789
            ratios_std = 0.1663598292187173

            infos = self.infos.loc[self.model_ids[index]]

            selected_info = []

            for key in self.keys:
                if key == 'sph':
                    info = (infos[key] - sph_mean) / sph_std
                    selected_info.append(info)
                elif key == 'aspect_ratio':
                    info = (infos[key] - ratios_mean) / ratios_std
                    selected_info.append(info)
                else:
                    selected_info.append(infos[key])

            info_arr = np.array(selected_info).astype(np.float32)
            cond_inputs['value'] = torch.from_numpy(info_arr)
            # print(cond_inputs['value'])
            # print(self.model_files[index])

            # cond_inputs['value] should shape like (len_vector, 1)

        # Convert input to -1 to 1 range.
        if len(self.condition_types) == 0:
            return model_tensor
        else:
            return model_tensor, cond_inputs
