from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2

from .base import BaseDetector
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from torch.utils.data import DataLoader
from ..utils.dataset import ReconstructDataset
from typing import Literal
        

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from scipy.stats import chi2


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        num_layers = 1
        kernel_size = 3
        layers = []
        in_channels = 1
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = d_model if i == num_layers - 1 else d_model
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

    def __init__(
            self, win_size, n_var, gating_init, 
            d_model=512, d_ff=512, n_heads=8, e_layers=1, 
            dropout=0.0, activation='gelu', output_attention=True
            ):
        super(SANA, self).__init__()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))

        # Variable-wise Temporal Embedding
        self.temporal_embedding = nn.ModuleList([
            TemporalEmbedding(d_model) for _ in range(n_var)
        ])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, factor=None, attention_dropout=dropout,
                                    output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Projection layer
        self.projection = nn.ModuleList([
            nn.Linear(d_model, win_size) for _ in range(n_var)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, N) -> (B, N, L)

        enc_out = torch.stack([embed(x[:, i:i+1, :]) for i, embed in enumerate(self.temporal_embedding)], dim=1)  # (B, N, D)
        enc_out, attns = self.encoder(enc_out)

        dec_out = torch.stack([proj(enc_out[:, i, :]) for i, proj in enumerate(self.projection)], dim=1)  # (B, N, L)
        dec_out = dec_out.permute(0, 2, 1)  # (B, N, L) -> (B, L, N)
        dec_out = dec_out * torch.tanh(self.gating)
        return dec_out
    

# models
'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''
class MLP(nn.Module):
    def __init__(self, seq_len, num_channels, latent_space_size):
        super().__init__()
        self.L, self.C = seq_len, num_channels
        self.encoder = MLPEncoder(seq_len*num_channels, latent_space_size)
        self.decoder = MLPDecoder(seq_len*num_channels, latent_space_size)

    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)
        if hasattr(self, "sana_in") and hasattr(self, "sana_out"):
            x_new = X + self.sana_in(X)
            reconstructed_x_new = self.decoder(self.encoder(x_new.reshape(B, L*C))).reshape(B, L, C)
            reconstructed_x = reconstructed_x_new - self.sana_out(reconstructed_x_new)
        else:
            reconstructed_x = self.decoder(self.encoder(X.reshape(B, L*C))).reshape(B, L, C)

        return reconstructed_x

    @torch.no_grad()
    def get_representations(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        is_training = copy.deepcopy(self.training)
        self.eval()
        if hasattr(self, "sana_in"):
            X = X + self.sana_in(X)
        z = self.encoder(X.reshape(B, L*C))
        z = F.normalize(z, p=2, dim=1)
        # Restore the original training state of the model
        self.train(is_training)
        return z

    @torch.no_grad()
    def get_anomaly_scores(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)
        is_training = copy.deepcopy(self.training)
        self.eval()

        reconstructed_x = self.forward(X)

        # Calculate reconstruction loss
        loss = F.mse_loss(reconstructed_x, X, reduction='none')
        
        score = loss.mean(dim=tuple(range(1, loss.dim())))
        
        # Restore the original training state of the model
        self.train(is_training)
        
        return score.detach().cpu().numpy()


class MLPEncoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_space_size, input_size // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return out


class MLP_Trainer:
    def __init__(
            self, model, train_loader, valid_loader=None,
            epochs=10, lr=1e-3, L2_reg=0, device='cuda'
        ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=L2_reg)

    def train(self):
        train_iterator = tqdm(
            range(1, self.epochs + 1),
            total=self.epochs,
            desc="training epochs",
            leave=True
        )
        if self.valid_loader is not None:
            early_stop = EarlyStoppingTorch(patience=5)
        for epoch in train_iterator:
            train_stats = self.train_epoch()
            if self.valid_loader is not None:
                valid_loss = self.valid()
                early_stop(valid_loss, self.model)
                if early_stop.early_stop:
                    break

    def train_epoch(self):
        self.model.train()
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary

    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.device)
        B, L, C = X.shape
        # recon
        Xhat = self.model(X)
        # optimize
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()
        out = {
            "recon_loss": loss.item(),
            "summary": loss.item(),
        }
        return out

    @torch.no_grad()
    def valid(self):
        assert self.valid_loader is not None, 'cannot valid without any data'
        self.model.eval()
        for i, batch_data in enumerate(self.valid_loader):
            X = batch_data[0].to(self.device)
            Xhat = self.model(X)
            loss = F.mse_loss(Xhat, X)
        return loss.item()

class MLP_Tester:
    def __init__(self, model, train_loader, valid_loader, test_loader, lr=1e-3, device='cuda', gating_init=0.1, tta_steps=1):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.gating_init = gating_init
        self.tta_steps = tta_steps
        self.lr = lr

    @torch.no_grad()
    def offline(self, dataloader):
        self.model.eval()
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="offline inference",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.device)
            B, L, C = X.shape
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_error = recon_error.mean(dim=(1, 2))
            recon_error = recon_error.detach().cpu().numpy()
            recon_errors.append(recon_error)
            torch.cuda.empty_cache()
        recon_errors = np.concatenate(recon_errors, axis=0) # (B,)
        anomaly_scores = recon_errors
        return anomaly_scores

    def online(self, dataloader, init_thr):
        #! CANDI test-time adaptation implementation
        # Setup validation data for representation-based adaptation
        # Use first batch as validation data for computing representations
        # Collect all validation data
        all_val_inputs = []
        for val_batch in self.valid_loader:
            val_inputs_batch = val_batch[0].to(self.device)
            all_val_inputs.append(val_inputs_batch)
        val_inputs = torch.cat(all_val_inputs, dim=0)
        
        val_scores = self.model.get_anomaly_scores(val_inputs)
        val_representations = self.model.get_representations(val_inputs)
        
        # Compute statistics for Mahalanobis distance
        val_representations_mean = torch.mean(val_representations, dim=0)
        val_representations_cov = torch.cov(val_representations.T)
        representations_cov_inv = torch.linalg.pinv(val_representations_cov)
        
        # Setup thresholds and selection criteria
        anomaly_ratio = 5.0  # 5% as default
        topk = int(len(val_scores) * anomaly_ratio / 100)
        topk_indices = np.argpartition(val_scores, -topk)[-topk:]
        topk_indices_tensor = torch.from_numpy(topk_indices).to(val_representations.device)
        topk_representations = val_representations[topk_indices_tensor]
        
        # Q1-Q3 for moderate selection
        q1 = np.percentile(val_scores, 25)
        q3 = np.percentile(val_scores, 75)
        moderate_mask = (val_scores > q1) & (val_scores < q3)
        moderate_indices = np.where(moderate_mask)[0]
        moderate_indices_tensor = torch.from_numpy(moderate_indices).to(val_representations.device)
        moderate_representations = val_representations[moderate_indices_tensor]

        # Initialize SANA modules (similar to CANDIAdapter's ccit_in/ccit_out)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.sana_in = SANA(self.model.L, self.model.C, gating_init=self.gating_init).to(self.device)
        self.model.sana_out = SANA(self.model.L, self.model.C, gating_init=self.gating_init).to(self.device)
        
        # Make SANA parameters trainable
        for param in self.model.sana_in.parameters():
            param.requires_grad = True
        for param in self.model.sana_out.parameters():
            param.requires_grad = True
            
        self.model.train()

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="online inference (CANDI)",
            leave=True
        )
        
        tau = init_thr
        TT_optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

        # CANDI adaptation variables
        samples_to_adapt_hard = []
        samples_to_adapt_moderate = []
        n_samples_to_adapt_hard = 0
        n_samples_to_adapt_moderate = 0
        min_samples = 16  # Minimum samples for adaptation
        
        As = []
        update_count = 0
        
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.device)
            B, L, C = X.shape
            
            # Get current representations and scores
            with torch.no_grad():
                representations = self.model.get_representations(X)
                latent_dim = representations.shape[1]
                
                # Compute anomaly scores for current batch
                batch_scores = self.model.get_anomaly_scores(X)  # (B,)
                
                # CANDI sample selection logic
                # Hard selection (high anomaly score + similar to known anomalies)
                mahalanobis_dist_square_hard = torch.sum(
                    (representations.unsqueeze(1) - topk_representations.unsqueeze(0))
                    @ representations_cov_inv *
                    (representations.unsqueeze(1) - topk_representations.unsqueeze(0)),
                    dim=-1
                )
                chi2_threshold = chi2.ppf(0.05, df=latent_dim)
                mask_hard_sim = mahalanobis_dist_square_hard < chi2_threshold
                mask_hard_sim = mask_hard_sim.any(dim=1)
                mask_hard_score = torch.tensor(batch_scores > tau).to(X.device)
                mask_hard = mask_hard_sim & mask_hard_score
                
                # Moderate selection (normal score + similar to moderate samples)
                mahalanobis_dist_square_moderate = torch.sum(
                    (representations.unsqueeze(1) - moderate_representations.unsqueeze(0))
                    @ representations_cov_inv *
                    (representations.unsqueeze(1) - moderate_representations.unsqueeze(0)),
                    dim=-1
                )
                mask_moderate_sim = mahalanobis_dist_square_moderate < chi2_threshold
                mask_moderate_sim = mask_moderate_sim.any(dim=1)
                mask_moderate_score = torch.tensor(batch_scores < tau).to(X.device)
                mask_moderate = mask_moderate_sim & mask_moderate_score
                
                # Collect samples for adaptation
                selected_x_hard = X[mask_hard]
                selected_x_moderate = X[mask_moderate]
                
                if len(selected_x_hard) > 0:
                    samples_to_adapt_hard.append(selected_x_hard)
                    n_samples_to_adapt_hard += len(selected_x_hard)
                    
                if len(selected_x_moderate) > 0:
                    samples_to_adapt_moderate.append(selected_x_moderate)
                    n_samples_to_adapt_moderate += len(selected_x_moderate)

            # Store anomaly scores
            As.append(batch_scores)
            
            # Perform adaptation when enough samples are collected
            adapted = False
            
            # Hard adaptation
            if n_samples_to_adapt_hard >= min_samples:
                adaptation_data = torch.cat(samples_to_adapt_hard, dim=0)
                self.model.train()
                
                for _ in range(self.tta_steps):
                    TT_optimizer.zero_grad()
                    # Apply CANDI loss with SANA modules
                    recon = self.model(adaptation_data)
                    loss = F.mse_loss(recon, adaptation_data)
                    loss.backward()
                    TT_optimizer.step()
                
                samples_to_adapt_hard = []
                n_samples_to_adapt_hard = 0
                update_count += len(adaptation_data)
                adapted = True
            
            # Moderate adaptation  
            if n_samples_to_adapt_moderate >= min_samples:
                adaptation_data = torch.cat(samples_to_adapt_moderate, dim=0)
                self.model.train()
                
                for _ in range(self.tta_steps):
                    TT_optimizer.zero_grad()
                    # Apply CANDI loss with SANA modules
                    recon = self.model(adaptation_data)
                    loss = F.mse_loss(recon, adaptation_data)
                    loss.backward()
                    TT_optimizer.step()
                
                samples_to_adapt_moderate = []
                n_samples_to_adapt_moderate = 0
                update_count += len(adaptation_data)
                adapted = True
            
            if adapted:
                self.model.eval()

        # outputs
        anoscs = np.concatenate(As, axis=0).reshape(-1)
        print(f'CANDI total update count: {update_count}')
        return anoscs

class CANDI(BaseDetector):
    def __init__(self, 
                 win_size=12,
                 stride=1,
                 num_channels=1, 
                 batch_size=64,
                 epochs=10,
                 latent_dim=128,
                 lr=1e-3,
                 ttlr=1e-3, # learning rate for online test-time adaptation
                 th=0.95, # 95 percentile == 0.95 quantile
                 valid_size=0.2,
                 infer_mode='online',
                 gating_init=0.1, tta_steps=1
                 ):
        self.model_name = 'CANDI'
        self.device = get_gpu(True)
        self.model = MLP(
            seq_len=win_size,
            num_channels=num_channels,
            latent_space_size=latent_dim,
        ).to(self.device)
        
        self.th = th
        self.lr = lr
        self.ttlr = ttlr
        self.epochs = epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.stride = stride
        self.valid_size = valid_size
        self.infer_mode = infer_mode
        self.gating_init = gating_init
        self.tta_steps = tta_steps

    def fit(self, data):
        if self.valid_size is None:
            self.train_loader = DataLoader(
                dataset=ReconstructDataset(
                    data, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=True
            )
            self.valid_loader = None
        else:
            data_train = data[:int((1-self.valid_size)*len(data))]
            data_valid = data[int((1-self.valid_size)*len(data)):]
            self.train_loader = DataLoader(
                dataset=ReconstructDataset(
                    data_train, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=True
            )
            self.valid_loader = DataLoader(
                dataset=ReconstructDataset(
                    data_valid, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=False,
            )

        self.trainer = MLP_Trainer(
            model=self.model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device
        )
        self.trainer.train()

        self.tester = MLP_Tester(
            model=self.model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            test_loader=self.train_loader,
            lr=self.ttlr,
            device=self.device,
        )
        train_anoscs = self.tester.offline(self.train_loader)
        self.tau = np.quantile(train_anoscs, self.th)
        print('tau', self.tau)

    def decision_function(self, data):
        self.test_loader = DataLoader(
            dataset=ReconstructDataset(
                data, window_size=self.win_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.tester = MLP_Tester(
            model=self.model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            test_loader=self.test_loader,
            lr=self.ttlr,
            device=self.device,
            gating_init=self.gating_init, tta_steps=self.tta_steps
        )
        if self.infer_mode == 'online':
            anoscs = self.tester.online(
                self.test_loader, self.tau)
        else:
            anoscs = self.tester.offline(self.test_loader)

        self.decision_scores_ = pad_by_edge_value(anoscs, len(data), mode='right')
        return self.decision_scores_


def pad_by_edge_value(scores, target_len, mode: Literal['both', 'left', 'right']):
    scores = np.array(scores).reshape(-1)
    assert len(scores) <= target_len, f'the length of scores is more than target one'
    print(f'origin length: {len(scores)}; target length: {target_len}')
    current_len = scores.shape[0]
    pad_total = max(target_len-current_len, 0)
    if mode == 'left':
        pad_before = pad_total
    elif mode == 'right':
        pad_before = 0
    else:
        pad_before = pad_total // 2 + 1
    pad_after = pad_total - pad_before
    padded_scores = np.pad(scores, (pad_before, pad_after), mode='edge')
    return padded_scores