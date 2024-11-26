import pickle
import numpy as np
import os
import torch
from torch.utils import data
import random


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

class HDM05(data.Dataset):

    def __init__(self, datapath="dataset/hdm05/hdm05", split="train",):
        """
        split="train", mode="train": return both train_data and train_label
        split="test", mode="train": return both test_data and test_label
        split="train", mode="label_only": return only train_label
        split="test", mode="label_only": return only test_label
        """
        if split=="train":
            data_path = os.path.join(datapath, "train_data.npy")
            label_path = os.path.join(datapath, "train_label.pkl")
        elif split=="test":
            data_path = os.path.join(datapath, "val_data.npy")
            label_path = os.path.join(datapath, "val_label.pkl")
        else:
            raise ValueError("Split must be either 'train' or 'test'")

        self.data = np.load(data_path)
        with open(label_path, 'rb') as f:
            self.action_names, self.labels = pickle.load(f)
        self.num_actions = 65
        train_values, train_counts = np.unique(self.labels, return_counts=True)
        print(len(train_counts))
        print(train_values)




    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data_numpy = np.array(self.data[item])
        label = self.labels[item]
        name = self.action_names[item]
        data_numpy = data_numpy[..., 0]
        data_numpy = np.transpose(data_numpy, (1, 2, 0))
        return {
            "inp": torch.from_numpy(data_numpy).float().permute(1, 2, 0),
            "action": int(label),
            "lengths": len(data_numpy),
            "action_text": name,
        }