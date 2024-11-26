import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.get_opt import get_opt

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def build_dict_from_txt(filename, is_style=True, is_style_text=False):
    result_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]
                if is_style and is_style_text == False:
                    value = parts[2]
                elif is_style_text:
                    value = parts[1].split("_")[0]
                else:
                    value = parts[3]

                result_dict[key] = value

    return result_dict

def random_zero_out(data, percentage=0.4, probability=0.6,noise_probability=0.8,noise_level=0.05):
    if random.random() < probability:
        # Calculate the total number of sequences to zero out
        num_sequences = data.shape[0]

        percentage = np.random.rand() * 0.5
        num_to_zero_out = int(num_sequences * percentage)

        # Randomly choose sequence indices to zero out
        indices_to_zero_out = np.random.choice(num_sequences, num_to_zero_out, replace=False)

        # Zero out the chosen sequences
        data[indices_to_zero_out, :] = 0

    # data = shuffle_segments_numpy(data,16)

    if random.random() < noise_probability:
        noise = np.random.normal(0, noise_level * np.ptp(data), data.shape)
        data += noise

    return data

class StyleMotionDataset(data.Dataset):

    def __init__(self, stage='train', nclasses=100):
        assert nclasses == 100
        data_dict = {}
        id_list = []
        data_list = []
        label_list = []
        self.max_motion_length = 60
        self.num_actions = nclasses

        txt_path = "./dataset/100STYLE_ORI"

        split_file = os.path.join(txt_path, "train_100STYLE_ORI.txt")

        self.stage = stage
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        dict_path = os.path.join(txt_path, "100STYLE_ORI_name_dict.txt")
        motion_to_label = build_dict_from_txt(dict_path)

        mean = np.load(os.path.join(txt_path, "Mean.npy"))
        std = np.load(os.path.join(txt_path, "Std.npy"))
        count = 0
        new_name_list = []

        motion_dir = os.path.join(txt_path, "new_joint_vecs")
        length_list = []

        # id_list = id_list[:500]
        print(f"Loading 100STYLE_ORI {split_file.split('/')[-1].split('.')[0]}")
        enumerator = enumerate(
            id_list
        )

        for i, name in enumerator:

            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]

            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "label": label_data,
            }

            new_name_list.append(name)
            length_list.append(len(motion))
            data_list.append(motion)
            label_list.append(int(label_data))
            count += 1

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.mean_tensor = torch.from_numpy(self.mean)
        self.std_tensor = torch.from_numpy(self.std)
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.data_list = np.stack(data_list, axis=0)
        self.label_list = label_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_tensor(self, data):
        return data * self.std_tensor.to(data.device) + self.mean_tensor.to(data.device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std

    def get_dataset(self):
        return (self.data_list- self.mean) / self.std, self.label_list, self.name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = item
        name = self.name_list[idx]
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, label = data["motion"], data["length"], data["label"]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.stage == 'train':
            motion = random_zero_out(motion)

        return {
            "inp": torch.from_numpy(motion).float().permute(1, 2, 0),
            "action": int(label),
            "lengths": len(motion),
            "action_text": name,
        }