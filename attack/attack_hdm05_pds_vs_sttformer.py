import torch
from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader


def train_attack_method(classifier_model):
    args = generate_args()
    fixseed(args.seed)

    max_frames = 60
    n_frames = max_frames
    dist_util.setup_dist(args.device)
    args.batch_size = args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model.to(dist_util.dev())
    classifier_model.to(dist_util.dev())

    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False

    data_idx = 0
    for ori_motion, model_kwargs in data:

        model.eval()
        data_idx = data_idx + 1

        device = torch.device(dist_util.dev())
        ori_motion = ori_motion.to(device)
        lengths = list(model_kwargs['y']['lengths'].numpy())
        ori_motion = ori_motion[:, :, :, :lengths[0]]

        max_iter_num = args.max_iter
        timestep_range = [20, 980]
        diffusion.our_attack_hdm05(
            motion=ori_motion, model=model, max_iter_num=max_iter_num,
            model_kwargs=model_kwargs, device=None, victim_type='sttformer', timestep_range=timestep_range,
            victim_model=victim_model,
            save_path=f'./save/ours_hdm05_sttformer'
        )

def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='train',
                              hml_mode='text_only')
    return data


if __name__ == '__main__':
    from model.sttformer.sttformer import STTFormer

    model_path = "./save/sttformer_classifier_on_hdm05/200.pt"

    model_args = {'len_parts': 6, 'num_frames': 60, 'num_joints': 25, 'num_classes': 65, 'num_heads': 3, 'kernel_size': [3, 5],
     'num_persons': 1, 'num_channels': 3, 'use_pes': True,
     'config': [[64, 64, 16], [64, 64, 16], [64, 128, 32], [128, 128, 32], [128, 256, 64], [256, 256, 64],
                [256, 256, 64], [256, 256, 64]]}
    victim_model = STTFormer(**model_args)
    victim_model.load_state_dict(torch.load(model_path))
    victim_model.eval()
    victim_model.cuda()

    train_attack_method(victim_model)