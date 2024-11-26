import torch
import torch.nn as nn
from torch import  nn
from model.style.cross_attention import  TransformerEncoder, TransformerEncoderLayer
from model.style.position_encoding import build_position_encoding
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Union
from torch.distributions.distribution import Distribution
from torch.optim import AdamW
from torch.nn import Parameter
import math
import numpy as np
    
def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class StyleClassification(nn.Module):
    def __init__(self,
                 nclasses,
                 latent_dim: list = [1, 256],  # for 100style
                 ff_size: int = 1024,  # for 100style
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 use_temporal_atten: bool = False,
                 skel_embedding_indim: int = 263,
                 **kwargs) -> None:
        
        super().__init__()
        self.style_num = nclasses
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(skel_embedding_indim, self.latent_dim)  # for 100style
        self.latent_size = latent_dim[0]
        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.abl_plus = False

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.pe_type = "mld" 

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )

        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)
    
        self.classification_layers = conv_layer(5, self.latent_dim, self.style_num)
        self.global_pool = F.max_pool1d
        self.classifier = nn.Linear(self.latent_dim, self.style_num)
    
    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            stage = "Classification",
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features.float()
        # Embed each human poses into latent vectors
        
        if skip == False:
            x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = self.global_motion_token[:, None, :].repeat(1, bs, 1)

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos(xseq)

        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)

        # if stage == "intermediate":
        #     _,intermediate = self.encoder(xseq,src_key_padding_mask=~aug_mask,is_intermediate=True)
        #
        #     style_features = []
        #     intermediate = intermediate[-2:]#[4:6]
        #
        #     for i in range(intermediate.size(0)):
        #         sub_tensor = intermediate[i]#[0]
        #
        #         mean = torch.mean(sub_tensor, dim=[0], keepdim=True)
        #         std = torch.std(sub_tensor, dim=[0], keepdim=True)
        #
        #         style_features.append((mean, std))
        #
        #     return style_features
        if stage == "intermediate":
            _, intermediate = self.encoder(xseq, src_key_padding_mask=~aug_mask, is_intermediate=True)

            style_features = []
            intermediate = intermediate[-2:]  # [4:6]
            style_features = intermediate[:, 0]
            return style_features
            
        elif stage == "Encode":
            return dist[0]
        elif stage == 'Encode_all':
            return dist
        elif stage == "Classification":
            #[2, 64, 256]
            feat = dist[0]
            output = self.classifier(feat)
            return output
        elif stage == "Both":
            feat = dist[0]
            output = self.classifier(feat)
            return output,feat
        
        elif stage == "distribution":
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

            # resampling
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            latent = dist.rsample()
            return latent, dist



    def configure_optimizers(self):
        optimizer = AdamW(params=filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        return optimizer


if __name__ == "__main__":
    model_path = "./StyleClassifier/style_encoder.pt"
    tmp_model = torch.load(model_path, map_location='cuda:0')
    classifier = StyleClassification(nclasses=100).to(torch.device('cuda:0'))
    classifier.load_state_dict(tmp_model)
    motion = np.load('./StyleClassifier/030634.npy')
    motion = torch.from_numpy(motion).unsqueeze(0).to(torch.device('cuda:0'))
    output = classifier(motion)
    _, predicted = torch.max(output, 1)
    print(predicted)