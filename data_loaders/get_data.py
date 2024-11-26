from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate

def get_dataset_class(name):

    if name == "100style":
        from data_loaders.humanml.data.dataset import StyleMotionDataset
        return StyleMotionDataset
    elif name == "hdm05":
        from data_loaders.hdm05.hdm05 import HDM05
        return HDM05
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ['100style']:
        dataset = DATA(stage=split, nclasses=100)
    elif name in ['hdm05']:
        dataset = DATA(split=split)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate
    )

    return loader