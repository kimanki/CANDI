# reference: https://github.com/carrtesy/M2N2/blob/master/Exp/MLP.py


from copy import deepcopy

import torch
import torch.nn.functional as F

from tta.adapter import Adapter
from models.normalizer import Detrender


class MLPAdapter(Adapter):
    def __init__(self, cfg, model, thresholder, **kwargs):
        super().__init__(cfg, model, thresholder)
        self.model.normalizer = Detrender(num_features=cfg.DATA.N_VAR, gamma=self.cfg.TEST.TTA.M2N2.GAMMA).cuda()
        self.model.use_normalizer = True

    @torch.enable_grad()
    def adapt(self, x, scores):
        is_eval = not deepcopy(self.model.training)
        self.model.train()
        
        self.model.normalizer._update_statistics(x)

        recon = self.model(x)
        E = (recon - x) ** 2
        A = torch.mean(E, dim=-1)  # (B, T)
        ytilde = (A >= self.thresholder.threshold).float()  # (B, T)
        self.optimizer.zero_grad()
        mask = (ytilde == 0)  # (B, T)
        loss = (A * mask).mean()  # (B, T) -> (1,)
        loss.backward()
        self.optimizer.step()

        if is_eval:
            self.model.eval()


class TimesNetAdapter(Adapter):
    def __init__(self, cfg, model, thresholder, **kwargs):
        super().__init__(cfg, model, thresholder)
        self.model.normalizer = Detrender(num_features=cfg.DATA.N_VAR, gamma=self.cfg.TEST.TTA.M2N2.GAMMA).cuda()
        self.model.use_normalizer = True

    @torch.enable_grad()
    def adapt(self, x, scores):
        is_eval = not deepcopy(self.model.training)
        self.model.train()
        
        self.model.normalizer._update_statistics(x)

        recon = self.model(x)
        E = (recon - x) ** 2
        A = torch.mean(E, dim=-1)  # (B, T)
        ytilde = (A >= self.thresholder.threshold).float()  # (B, T)
        self.optimizer.zero_grad()
        mask = (ytilde == 0)  # (B, T)
        loss = (A * mask).mean()  # (B, T) -> (1,)
        loss.backward()
        self.optimizer.step()
        if is_eval:
            self.model.eval()