import random
from typing import Union
from pathlib import Path
import os

import numpy as np
import wandb

import torch


def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
def set_devices(devices: str):
    """
    Set cuda devices

    Args:
        devices (str): Comma separated string of device numbers. e.g., "0", "0, 1"
    """
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def set_wandb(cfg):
    wandb.init(
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        job_type=cfg.WANDB.JOB_TYPE,
        notes=cfg.WANDB.NOTES,
        dir=cfg.WANDB.DIR,
        config=cfg
    )
    # Update all CHECKPOINT_DIR and RESULT_DIR in cfg
    def update_paths(cfg, attr_name):
        for key, value in cfg.items():
            if key == attr_name:
                setattr(cfg, key, str(mkdir(os.path.join(value, 'wandb', wandb.run.id))))
            elif isinstance(value, dict):
                update_paths(getattr(cfg, key), attr_name)

    # When loading a checkpoint for evaluation, do not change the checkpoint_dir
    # if cfg.TRAIN.ENABLE:
    #     update_paths(cfg, 'CHECKPOINT_DIR')
    update_paths(cfg, 'RESULT_DIR')