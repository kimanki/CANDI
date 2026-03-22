from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from tta.adapter import Adapter
from models.optimizer import construct_optimizer
from scipy.stats import chi2


class TemporalEmbedding(nn.Module):
    def __init__(self, cfg):
        super(TemporalEmbedding, self).__init__()
        self.cfg = cfg
        self.d_model = cfg.SANA.d_model
        num_layers = 1
        kernel_size = 3
        layers = []
        in_channels = 1
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = self.d_model if i == num_layers - 1 else self.d_model
            layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation
            )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

    def forward(self, x_enc):
        # x_enc: (B, 1, L) -> (B, D, L)
        enc_out = self.tcn(x_enc)
        enc_out = enc_out.mean(dim=2) # (B, D)
        return enc_out


class SANA(nn.Module):

    def __init__(self, cfg):
        super(SANA, self).__init__()
        self.cfg = cfg
        self.cfg_model = cfg.SANA
        self.n_var = cfg.DATA.N_VAR
        self.gating = nn.Parameter(self.cfg.TEST.TTA.CANDI.GATING_INIT * torch.ones(self.n_var))

        if self.cfg.SANA.TYPE == "TCN_iTrans":
            # Variable-wise Temporal Embedding
            self.temporal_embedding = nn.ModuleList([
                TemporalEmbedding(cfg) for _ in range(self.n_var)
            ])

            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(mask_flag=False, factor=None, attention_dropout=self.cfg_model.dropout,
                                        output_attention=self.cfg_model.output_attention), self.cfg_model.d_model, self.cfg_model.n_heads),
                        self.cfg_model.d_model,
                        self.cfg_model.d_ff,
                        dropout=self.cfg_model.dropout,
                        activation=self.cfg_model.activation
                    ) for l in range(self.cfg_model.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.cfg_model.d_model)
            )

            # Projection layer
            self.projection = nn.ModuleList([
                nn.Linear(self.cfg_model.d_model, self.cfg.DATA.WIN_SIZE) for _ in range(self.n_var)
            ])
        elif self.cfg.SANA.TYPE == "Linear":
            self.projection = nn.ModuleList([
                nn.Linear(self.cfg.DATA.WIN_SIZE, self.cfg.DATA.WIN_SIZE) for _ in range(self.n_var)
            ])
        else:
            raise ValueError(f"Unsupported SANA.TYPE: {self.cfg.SANA.TYPE}")

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, N) -> (B, N, L)

        if self.cfg.SANA.TYPE == "TCN_iTrans":
            enc_out = torch.stack([embed(x[:, i:i+1, :]) for i, embed in enumerate(self.temporal_embedding)], dim=1)  # (B, N, D)
            enc_out, attns = self.encoder(enc_out)

            dec_out = torch.stack([proj(enc_out[:, i, :]) for i, proj in enumerate(self.projection)], dim=1)  # (B, N, L)
        elif self.cfg.SANA.TYPE == "Linear":
            dec_out = torch.concat([embed(x[:, i:i+1, :]) for i, embed in enumerate(self.projection)], dim=1)
        else:
            raise ValueError(f"Unsupported SANA.TYPE: {self.cfg.SANA.TYPE}")
        dec_out = dec_out.permute(0, 2, 1)  # (B, N, L) -> (B, L, N)
        dec_out = dec_out * torch.tanh(self.gating)
        return dec_out
    
    
class CANDIAdapter(Adapter):
    def __init__(self, cfg, model, thresholder, **kwargs):
        super().__init__(cfg, model, thresholder)
        if self.cfg.TEST.TTA.CANDI.USE_SANA:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.sana_in = SANA(cfg).cuda()
            for name, param in self.model.sana_in.named_parameters():
                param.requires_grad = True
            self.model.sana_out = SANA(cfg).cuda()
            for name, param in self.model.sana_out.named_parameters():
                param.requires_grad = True
            self.optimizer = construct_optimizer(self.model, self.cfg)
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        self.val_scores = kwargs['val_scores']
        self.val_inputs = kwargs['val_inputs']
        self.val_representations = self.model.get_representations(self.val_inputs)
        # Save val_representations to a file in numpy format
        val_repr_path = f"{self.cfg.RESULT_DIR}/val_representations.npy"
        np.save(val_repr_path, self.val_representations.detach().cpu().numpy())
        print(f"Validation representations saved to {val_repr_path}")

        self.val_representations_mean = torch.mean(self.val_representations, dim=0)
        self.val_representations_cov = torch.cov(self.val_representations.T)
        self.representations_cov_inv = torch.linalg.pinv(self.val_representations_cov)

        # Top-k (hard) selection
        self.topk = int(len(self.val_scores) * self.cfg.TEST.THRESHOLD.ANOMALY_RATIO / 100)
        topk_indices = np.argpartition(self.val_scores, -self.topk)[-self.topk:]
        self.topk_scores = self.val_scores[topk_indices]
        self.topk_inputs = self.val_inputs[topk_indices]
        self.topk_representations = self.val_representations[topk_indices]

        # Q1 ~ Q3 (moderate) selection
        self.q1 = np.percentile(self.val_scores, 25)
        self.q3 = np.percentile(self.val_scores, 75)
        moderate_mask = (self.val_scores > self.q1) & (self.val_scores < self.q3)
        moderate_indices = np.where(moderate_mask)[0]
        self.moderate_scores = self.val_scores[moderate_indices]
        self.moderate_inputs = self.val_inputs[moderate_indices]
        self.moderate_representations = self.val_representations[moderate_indices]

        self.samples_to_adapt_hard = []
        self.n_samples_to_adapt_hard = 0
        self.total_samples_to_adapt_hard = 0
        self.total_anomalies_in_hard = 0

        self.samples_to_adapt_moderate = []
        self.n_samples_to_adapt_moderate = 0
        self.total_samples_to_adapt_moderate = 0
        self.total_anomalies_in_moderate = 0

        self.test_labels = kwargs['test_labels']

    @torch.no_grad()
    def get_samples_to_adapt(self, x, scores):
        if not self.cfg.TEST.TTA.CANDI.USE_FPM:
            mask_moderate = scores < self.thresholder.threshold
            selected_x_moderate = x[mask_moderate]  # Handle as moderate for implementation convenience
            if len(selected_x_moderate) > 0:
                selected_indices_moderate = mask_moderate.nonzero(as_tuple=True)[0]
                offset = self.iter * len(scores)
                selected_indices_moderate_offset = (selected_indices_moderate + offset).tolist()

                selected_labels_moderate = self.test_labels[selected_indices_moderate_offset]
                self.n_samples_to_adapt_moderate += len(selected_x_moderate)
                self.samples_to_adapt_moderate.append(selected_x_moderate)
                self.total_samples_to_adapt_moderate += len(selected_x_moderate)
                self.total_anomalies_in_moderate += (selected_labels_moderate == 1).sum().item()
            return

        representations = self.model.get_representations(x)  # (B, D)
        latent_dim = representations.shape[1]

        if self.cfg.TEST.TTA.CANDI.USE_HARD:
            mahalanobis_dist_square_hard = torch.sum(
                    (representations.unsqueeze(1) - self.topk_representations.unsqueeze(0))
                    @ self.representations_cov_inv *
                    (representations.unsqueeze(1) - self.topk_representations.unsqueeze(0)),
                    dim=-1
                    )
            chi2_threshold = chi2.ppf(0.05, df=latent_dim)
            mask_hard_sim = mahalanobis_dist_square_hard < chi2_threshold
            mask_hard_sim = mask_hard_sim.any(dim=1)

            # Also consider anomaly score threshold
            mask_hard_score = scores > self.thresholder.threshold

            # Combine both masks
            mask_hard = mask_hard_sim & mask_hard_score

            selected_x_hard = x[mask_hard]
            if len(selected_x_hard) > 0:
                offset = self.iter * len(scores)
                selected_indices_hard = mask_hard.nonzero(as_tuple=True)[0]
                selected_indices_hard_offset = (selected_indices_hard + offset).tolist()
                selected_labels_hard = self.test_labels[selected_indices_hard_offset]
                self.n_samples_to_adapt_hard += len(selected_x_hard)
                self.samples_to_adapt_hard.append(selected_x_hard)
                self.total_samples_to_adapt_hard += len(selected_x_hard)
                self.total_anomalies_in_hard += (selected_labels_hard == 1).sum().item()
        else:
            selected_x_hard = torch.tensor([]).to(x.device)

        if self.cfg.TEST.TTA.CANDI.USE_MODERATE:
            # Moderate selection based on Q1, Q3 and Mahalanobis distance
            mask_moderate_score = (scores > self.q1) & (scores < self.q3)
            mask_moderate_score = scores < self.thresholder.threshold

            mahalanobis_dist_square_moderate = torch.sum(
                    (representations.unsqueeze(1) - self.moderate_representations.unsqueeze(0))
                    @ self.representations_cov_inv *
                    (representations.unsqueeze(1) - self.moderate_representations.unsqueeze(0)),
                    dim=-1
                    )
            chi2_threshold_moderate = chi2.ppf(0.05, df=latent_dim)
            mask_moderate_sim = mahalanobis_dist_square_moderate < chi2_threshold_moderate
            mask_moderate_sim = mask_moderate_sim.any(dim=1)

            mask_moderate = mask_moderate_score & mask_moderate_sim

            selected_x_moderate = x[mask_moderate]

            if len(selected_x_moderate) > 0:
                selected_indices_moderate = mask_moderate.nonzero(as_tuple=True)[0]
                offset = self.iter * len(scores)
                selected_indices_moderate_offset = (selected_indices_moderate + offset).tolist()

                selected_labels_moderate = self.test_labels[selected_indices_moderate_offset]
                self.n_samples_to_adapt_moderate += len(selected_x_moderate)
                self.samples_to_adapt_moderate.append(selected_x_moderate)
                self.total_samples_to_adapt_moderate += len(selected_x_moderate)
                self.total_anomalies_in_moderate += (selected_labels_moderate == 1).sum().item()
        else:
            selected_x_moderate = torch.tensor([]).to(x.device)

        self.iter += 1

    @torch.enable_grad()
    def adapt(self, x, scores):
        self.get_samples_to_adapt(x, scores)

        # Hard adaptation
        if self.n_samples_to_adapt_hard >= self.cfg.TEST.TTA.CANDI.MIN_SAMPLES:
            self.n_adapt += 1
            self.samples_to_adapt_hard = torch.concat(self.samples_to_adapt_hard, dim=0)
            is_eval = not deepcopy(self.model.training)
            self.model.train()
            
            for _ in range(self.cfg.TEST.TTA.STEPS):
                loss = self.calculate_loss(self.samples_to_adapt_hard)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if is_eval:
                self.model.eval()
            self.samples_to_adapt_hard = []
            self.n_samples_to_adapt_hard = 0

        # Moderate adaptation
        if self.n_samples_to_adapt_moderate >= self.cfg.TEST.TTA.CANDI.MIN_SAMPLES:
            self.n_adapt += 1
            self.samples_to_adapt_moderate = torch.concat(self.samples_to_adapt_moderate, dim=0)
            is_eval = not deepcopy(self.model.training)
            self.model.train()
            
            for _ in range(self.cfg.TEST.TTA.STEPS):
                loss = self.calculate_loss(self.samples_to_adapt_moderate)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if is_eval:
                self.model.eval()
            self.samples_to_adapt_moderate = []
            self.n_samples_to_adapt_moderate = 0
    
    def calculate_loss(self, x):
        """
        This method is a placeholder for the calculate_loss function.
        It should be overridden in subclasses to implement specific loss calculation logic.
        """
        raise NotImplementedError("The calculate_loss method should be implemented in subclasses.")
            

class MLPAdapter(CANDIAdapter):
    def __init__(self, cfg, model, thresholder, **kwargs):
        super().__init__(cfg, model, thresholder, **kwargs)

    @torch.enable_grad()
    def calculate_loss(self, x):
        if hasattr(self.model, "sana_in"):
            x_new = x + self.model.sana_in(x)
            recon = self.model(x_new)
        else:
            recon = self.model(x)
        if hasattr(self.model, "sana_out"):
            recon = recon - self.model.sana_out(recon)
        loss = F.mse_loss(recon, x)
        return loss


class TimesNetAdapter(CANDIAdapter):
    def __init__(self, cfg, model, thresholder, **kwargs):
        super().__init__(cfg, model, thresholder, **kwargs)

    def calculate_loss(self, x):
        if hasattr(self.model, "sana_in"):
            x_new = x + self.model.sana_in(x)
            recon = self.model(x_new)
        else:
            recon = self.model(x)
        if hasattr(self.model, "sana_out"):
            recon = recon - self.model.sana_out(recon)
        loss = F.mse_loss(recon, x)
        return loss
