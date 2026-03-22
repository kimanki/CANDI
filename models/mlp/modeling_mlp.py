# reference: https://github.com/carrtesy/M2N2/blob/master/models/MLP.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        seq_len = self.cfg.DATA.WIN_SIZE
        num_channels = self.cfg.DATA.N_VAR
        latent_space_size = self.cfg.MLP.Z_SIZE
        
        self.L, self.C = seq_len, num_channels
        self.encoder = Encoder(seq_len*num_channels, latent_space_size)
        self.decoder = Decoder(seq_len*num_channels, latent_space_size)

        self.use_normalizer = False

    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        if self.use_normalizer:
            X = self.normalizer(X, "norm")
            
        z = self.encoder(X.reshape(B, L*C))
        out = self.decoder(z).reshape(B, L, C)

        if self.use_normalizer:
            out = self.normalizer(out, "denorm")
        return out
    
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
        z = self.encoder(x.reshape(x.shape[0], -1))

        #! l2-normalize the representations
        z = F.normalize(z, p=2, dim=1)
        
        # Restore the original training state of the model
        self.train(is_training)
        
        return z


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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