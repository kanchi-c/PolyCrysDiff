import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.npy_utils import load_volume_tensor


class Class_Dataset(Dataset):
    """Lazy dataset for class-labelled volumes stored in subdirectories."""

    def __init__(self, split, root_dir, im_size, im_channels, debug=False,
                 condition_config=None):
        self.root_dir = root_dir
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.debug = debug
        self.condition_types = (
            [] if condition_config is None else condition_config['condition_types']
        )
        self.models, self.classes = self.load_models(root_dir)
        self.infos = self.get_infos(root_dir)

        if 'value' in self.condition_types:
            value_config = condition_config['value_condition_config']
            self.keys = value_config['keys']
            assert value_config['len_vector'] == len(self.keys)

    def load_models(self, root_dir):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f'Data directory does not exist: {root_dir}')
        files = []
        classes = []
        labels = sorted(
            label for label in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, label))
        )
        for label in labels:
            label_dir = os.path.join(root_dir, label)
            filenames = sorted(
                filename for filename in os.listdir(label_dir)
                if filename.endswith('.npy')
            )
            if self.debug:
                filenames = filenames[:1]
            for filename in filenames:
                files.append(os.path.join(label_dir, filename))
                classes.append(label)
        if not files:
            raise FileNotFoundError(
                f'No class-subdirectory .npy volumes found in: {root_dir}'
            )
        self.label_dict = {label: index for index, label in enumerate(labels)}
        self.n_classes = len(labels)
        print(f'{len(files)} model paths loaded from {self.n_classes} classes.')
        return files, classes

    @staticmethod
    def get_infos(root_dir):
        info_file = os.path.join(root_dir, 'info.txt')
        if not os.path.exists(info_file):
            return None
        data = {}
        with open(info_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                index = int(row['index'])
                data[index] = {
                    'seeds_num': int(row['seeds_num']),
                    'color_num': int(row['color_num']),
                }
        return data

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        model_tensor = load_volume_tensor(
            self.models[index], self.im_size, self.im_channels
        )
        cond_inputs = {}
        if 'class' in self.condition_types:
            class_name = self.classes[index]
            label = torch.zeros(self.n_classes)
            label[self.label_dict[class_name]] = 1
            cond_inputs['class'] = label
        if 'image' in self.condition_types:
            cond_inputs['image'] = model_tensor[:, :, :, -1]
        if 'value' in self.condition_types:
            if self.infos is None:
                raise ValueError('Value conditioning requires info.txt.')
            selected = [[self.infos[index][key]] for key in self.keys]
            cond_inputs['value'] = torch.from_numpy(
                np.asarray(selected, dtype=np.float32)
            )
        return (model_tensor, cond_inputs) if self.condition_types else model_tensor
