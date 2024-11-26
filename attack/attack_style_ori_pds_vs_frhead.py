import torch
from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader


def train_attack_method(victim_model):
    args = generate_args()
    fixseed(args.seed)

    max_frames = 60
    n_frames = max_frames

    dist_util.setup_dist(args.device)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    print(f"Using guidance_param: {args.guidance_param}")
    model.to(dist_util.dev())
    victim_model.to(dist_util.dev())

    model.eval()  # disable random masking
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
        diffusion.our_attack_100style(
            motion=ori_motion, model=model, max_iter_num=max_iter_num,
            model_kwargs=model_kwargs, device=None, victim_type='frhead', timestep_range=timestep_range,
            victim_model=victim_model,
            save_path=f'./save/ours_100style_frhead'
        )


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    return data


if __name__ == '__main__':
    from model.frhead.frhead import FRHEAD

    model_path = "./save/frhead_classifier_on_100style/200.pt"
    victim_model = FRHEAD(num_class=100, num_point=23, num_frame=60, num_person=1, dataset='100style_ori',
                          graph_args={'labeling_mode': 'spatial'}, in_channels=3, base_channel=64, drop_out=0,
                          adaptive=True,
                          cl_mode='ST-Multi-Level', multi_cl_weights=[0.1, 0.2, 0.5, 1], cl_version='V0',
                          pred_threshold=0, use_p_map=True, )
    victim_model.load_state_dict(torch.load(model_path))
    victim_model.eval()
    victim_model.cuda()

    train_attack_method(victim_model)