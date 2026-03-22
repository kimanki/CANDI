# adapted from: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.cfg = configs
        self.seq_len = configs.DATA.WIN_SIZE
        self.nvar = configs.DATA.N_VAR
        configs = configs.TIMESNET
        configs.seq_len = self.seq_len
        configs.enc_in = self.nvar
        configs.c_out = self.nvar
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

        self.use_normalizer = False

    def forward(self, x_enc):
        if self.use_normalizer:
            x_enc = self.normalizer(x_enc, "norm")

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)
        
        if self.use_normalizer:
            dec_out = self.normalizer(dec_out, "denorm")

        return dec_out

    @torch.no_grad()
    def get_anomaly_scores(self, x):
        is_training = copy.deepcopy(self.training)
        self.eval()

        if self.cfg.TEST.TTA.ENABLE and self.cfg.TEST.TTA.METHOD == 'CANDI':
            x_new = x + self.sana_in(x) if hasattr(self, 'sana_in') else x
            reconstructed_x_new = self.forward(x_new)
            reconstructed_x = reconstructed_x_new - self.sana_out(reconstructed_x_new) if hasattr(self, 'sana_out') else reconstructed_x_new
        else:
            reconstructed_x = self.forward(x)

        # Calculate reconstruction loss
        loss = F.mse_loss(reconstructed_x, x, reduction='none')

        score = loss.mean(dim=tuple(range(1, loss.dim())))

        # Restore the original training state of the model
        self.train(is_training)

        return score

    @torch.no_grad()
    def get_representations(self, x):
        is_training = copy.deepcopy(self.training)
        self.eval()

        if self.cfg.TEST.TTA.ENABLE and self.cfg.TEST.TTA.METHOD == 'CANDI' and hasattr(self, 'sana_in'):
            x = x + self.sana_in(x)
        
        # Forward pass through the encoder to get representations
        enc_out = self.enc_embedding(x, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        #! l2-normalize the representations
        # z = enc_out.mean(dim=1)
        z = enc_out[:, -1, :]
        # z = F.normalize(z, p=2, dim=1)
        
        # Restore the original training state of the model
        self.train(is_training)
        
        return z