import os
import time
import pickle
from typing import Optional, Tuple, Mapping, Union, List

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from tqdm import tqdm

import models.optimizer as optim
from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from utils.misc import mkdir
from utils.meters import AverageMeter, ProgressMeter


class Trainer:
    def __init__(
            self,
            cfg,
            model,
            optimizer: Optional[Union[Optimizer, List[Optimizer]]] = None
    ):
        self.cfg = cfg
        self.cfg_model = getattr(cfg, cfg.MODEL.NAME)

        self.model = model
        self.optimizer = optimizer

        self.metric_names = self.cfg_model.METRIC_NAMES
        self.loss_names = self.cfg_model.LOSS_NAMES

        self.cur_epoch = 0
        self.cur_iter = 0

        # Create the train and val (test) loaders.
        self.train_loader = get_train_dataloader(self.cfg)
        self.val_loader = get_val_dataloader(self.cfg)
        self.test_loader = get_test_dataloader(self.cfg)

        # create optimizer
        if self.optimizer is None:
            self.create_optimizer()

    def create_optimizer(self):
        self.optimizer = optim.construct_optimizer(self.model, self.cfg)

    def train(self):
        metric_best = self.cfg.TRAIN.METRIC_BEST
        for cur_epoch in tqdm(range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH)):
            self.train_epoch()

            # Evaluate the model on validation set.
            if self._is_eval_epoch(cur_epoch):
                tracking_meter = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(tracking_meter.avg, metric_best)
                # Save a checkpoint on improvement.
                if is_best:
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    print(f"[current best] Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    metric_best = tracking_meter.avg
                elif self.cfg.TRAIN.SAVE_EVERY_EVAL:
                    self.save_best_model()
                
            self.cur_epoch += 1

    def _check_improvement(self, cur_metric, metric_best):
        if self.cfg.TRAIN.IS_METRIC_LOWER_BETTER:
            return cur_metric < metric_best
        else:
            return cur_metric > metric_best

    def train_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Epoch: [{}]".format(self.cur_epoch)
        )

        # switch to train mode
        self.model.train()

        data_size = len(self.train_loader)

        start = time.time()

        for idx, inputs in enumerate(self.train_loader):
            cur_iter = idx + 1
            self.cur_iter = cur_iter
            # dictionary for logging values
            log_dict = {}

            # measure data loading time
            data_time.update(time.time() - start)

            # Update the learning rate.
            lr = optim.get_epoch_lr(self.cur_epoch + float(cur_iter) / data_size, self.cfg)
            if isinstance(self.optimizer, tuple):  # USAD
                [optim.set_lr(optimizer, lr) for optimizer in self.optimizer]
            else:
                optim.set_lr(self.optimizer, lr)

            # log to W&B
            log_dict.update({
                "lr/": lr
            })

            outputs = self.train_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)
            log_dict.update({
                f"Train/{metric_meter.name}": metric_meter.val for metric_meter in metric_meters
            })
            log_dict.update({
                f"Train/{loss_meter.name}": loss_meter.val for loss_meter in loss_meters
            })

            if self._is_display_iter(cur_iter, len(self.train_loader)):
                progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.cfg.WANDB.ENABLE:
                wandb.log(log_dict)

        log_dict = {}

        # log to W&B
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict, commit=False)

    def _get_metric_meters(self):
        return [AverageMeter(metric_name, ":.3f") for metric_name in self.metric_names]

    def _get_loss_meters(self):
        return [AverageMeter(f"Loss {loss_name}", ":.4e") for loss_name in self.loss_names]

    @staticmethod
    def _update_metric_meters(metric_meters, metrics, batch_size):
        assert len(metric_meters) == len(metrics)
        for metric_meter, metric in zip(metric_meters, metrics):
            metric_meter.update(metric.item(), batch_size)

    @staticmethod
    def _update_loss_meters(loss_meters, losses, batch_size):
        assert len(loss_meters) == len(losses)
        for loss_meter, loss in zip(loss_meters, losses):
            loss_meter.update(loss.item(), batch_size)

    def train_step(self, inputs):
        #! override for different methods
        inputs = prepare_inputs(inputs)
        outputs = self.model(inputs)
        loss = outputs["losses"][0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return outputs

    def _load_from_checkpoint(self):
        pass

    def _find_batch_size(self, inputs):
        """
        Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
        """
        if isinstance(inputs, (list, tuple)):
            for t in inputs:
                result = self._find_batch_size(t)
                if result is not None:
                    return result
        elif isinstance(inputs, Mapping):
            for key, value in inputs.items():
                result = self._find_batch_size(value)
                if result is not None:
                    return result
        elif isinstance(inputs, torch.Tensor):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None
        elif isinstance(inputs, np.ndarray):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None

    def _is_eval_epoch(self, cur_epoch):
        return (cur_epoch + 1 == self.cfg.SOLVER.MAX_EPOCH) or (cur_epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0

    @torch.no_grad()
    def eval_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Validation epoch[{}]".format(self.cur_epoch)
        )
        log_dict = {}

        # switch to eval mode
        self.model.eval()

        start = time.time()
        for idx, inputs in enumerate(self.val_loader):
            cur_iter = idx + 1
            self.cur_iter = cur_iter
            # measure data loading time
            data_time.update(time.time() - start)

            outputs = self.eval_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            if self._is_display_iter(cur_iter, len(self.val_loader)):
                progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

        log_dict.update({
            f"Val/{metric_meter.name}": metric_meter.avg for metric_meter in metric_meters
        })
        log_dict.update({
            f"Val/{loss_meter.name}": loss_meter.avg for loss_meter in loss_meters
        })

        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

        # track the best model based on the first metric
        tracking_meter = metric_meters[0]

        return tracking_meter

    @torch.no_grad()
    def eval_step(self, inputs):
        inputs = prepare_inputs(inputs)
        outputs = self.model(inputs)

        return outputs

    def _is_display_iter(self, cur_iter, loader_len):
        return cur_iter % self.cfg.TRAIN.PRINT_FREQ == 0 or cur_iter == loader_len

    def save_best_model(self):
        print('Saving best model')
        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(self.cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)

    def load_best_model(self):
        model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        if os.path.isfile(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = checkpoint['model_state']
            msg = self.model.load_state_dict(state_dict, strict=True)
            assert set(msg.missing_keys) == set()

            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        return self.model


def build_trainer(cfg, model):
    trainer_classes = {
        "TIMESNET": "models.timesnet.trainer_timesnet.TimesnetTrainer",
        "MLP": "models.mlp.trainer_mlp.MLPTrainer",
    }

    model_name = cfg.MODEL.NAME
    if model_name not in trainer_classes:
        raise ValueError(f"Unknown model name: {model_name}")

    module_path, class_name = trainer_classes[model_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    trainer_class = getattr(module, class_name)

    return trainer_class(cfg, model)


def prepare_inputs(inputs):
    # move data to the current GPU
    if isinstance(inputs, torch.Tensor):
        return inputs.cuda()
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)
