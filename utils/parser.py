# adapted from: https://github.com/facebookresearch/SlowFast

"""Argument parser functions."""

import argparse
import sys
import os

from config import get_cfg_defaults
from utils.misc import mkdir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-time adaptation of unsupervised multivariate time-series anomaly detector"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Overwrite cfg.DATA.N_VAR and related variables according to dataset
    valid_datasets = {
        "SMD": 38,
        "SWaT": 51,
    }
    if cfg.DATA.NAME == 'SWaT':
        cfg.DATA.N_VAR = valid_datasets["SWaT"]
        cfg.DATA.DOWNSAMPLE_RATE = 5
    elif "SMD" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["SMD"]

    # Update model-specific configurations
    cfg.TIMESNET.enc_in = cfg.DATA.N_VAR
    cfg.TIMESNET.c_out = cfg.DATA.N_VAR
    
    cfg.TRAIN.CHECKPOINT_DIR = f'{cfg.TRAIN.CHECKPOINT_DIR}/{cfg.DATA.NAME}/{cfg.MODEL.NAME}'
    if not os.path.exists(cfg.TRAIN.CHECKPOINT_DIR):
        mkdir(cfg.TRAIN.CHECKPOINT_DIR)
        cfg.TRAIN.ENABLE = True  # Enable training if checkpoint directory does not exist

    cfg.RESULT_DIR = f'{cfg.RESULT_DIR}/{cfg.DATA.NAME}/{cfg.MODEL.NAME}/{cfg.TEST.THRESHOLD.ANOMALY_RATIO}'
    if not os.path.exists(cfg.RESULT_DIR):
        mkdir(cfg.RESULT_DIR)
    return cfg
