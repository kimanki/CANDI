import torch
import torch.nn.functional as F

from trainer import Trainer, prepare_inputs
from utils.misc import mkdir


class TimesnetTrainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    def train_step(self, inputs):
        outputs = {}
        inputs, _ = prepare_inputs(inputs)

        recon = self.model(inputs)
        loss = F.mse_loss(recon, inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        outputs["metrics"] = (loss,)
        outputs["losses"] = (loss,)

        return outputs

    @torch.no_grad()
    def eval_step(self, inputs):
        assert not self.model.training
        outputs = {}
        inputs, _ = prepare_inputs(inputs)

        recon = self.model(inputs)
        loss = F.mse_loss(recon, inputs)

        outputs["metrics"] = (loss,)
        outputs["losses"] = (loss,)

        return outputs

    def save_best_model(self):
        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(self.cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)
