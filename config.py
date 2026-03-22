from yacs.config import CfgNode as CN


_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node
_C.NUM_GPUS = 4
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = "results"



_C.DATA_LOADER = CN()
# the number of data loading workers per gpu
_C.DATA_LOADER.NUM_WORKERS = 2
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True



_C.DATA = CN()
_C.DATA.BASE_DIR = "data"
_C.DATA.NAME = "SWaT"
_C.DATA.SCALE = "standard"
_C.DATA.WIN_SIZE = 10
_C.DATA.TRAIN_STEP = 1
_C.DATA.TEST_STEP = 1
_C.DATA.N_VAR = 51
_C.DATA.SAMPLING_RATES = 1.0
_C.DATA.TRAIN_RATIO = 0.8
_C.DATA.DOWNSAMPLE_RATE = 1



_C.TRAIN = CN()
_C.TRAIN.ENABLE = False
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True
_C.TRAIN.SPLIT = 'train'
# directory to save checkpoints
_C.TRAIN.CHECKPOINT_DIR = 'results'
# path to checkpoint to resume training
_C.TRAIN.RESUME = ''
# epoch period to save checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 200
# epoch period to evaluate on a validation set
_C.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.METRIC_BEST = float("inf")
_C.TRAIN.IS_METRIC_LOWER_BETTER = True
_C.TRAIN.SAVE_EVERY_EVAL = False



_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = 256
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False



_C.TEST = CN()
_C.TEST.ENABLE = True
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 256
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False
_C.TEST.THRESHOLD = CN()
_C.TEST.THRESHOLD.TYPE = 'ratio'  # ratio, best_f1
_C.TEST.THRESHOLD.ANOMALY_RATIO = 0.5  # percent
_C.TEST.ANOMALY_SCORES_DIR = ""
_C.TEST.TTA = CN()
_C.TEST.TTA.ENABLE = True
_C.TEST.TTA.METHOD = 'CANDI'  # M2N2, CANDI
_C.TEST.TTA.M2N2 = CN()
_C.TEST.TTA.M2N2.GAMMA = 0.99999
_C.TEST.TTA.CANDI = CN()
_C.TEST.TTA.CANDI.USE_FPM = True  # Whether to use FPM in CANDI adapter
_C.TEST.TTA.CANDI.USE_SANA = True  # Whether to use SANA in CANDI adapter
_C.TEST.TTA.CANDI.GATING_INIT = 0.0  # Initial value for the gating parameter in CANDI adapter
_C.TEST.TTA.CANDI.MIN_SAMPLES = 16
_C.TEST.TTA.CANDI.USE_HARD = True  # Whether to use hard samples in CANDI adapter
_C.TEST.TTA.CANDI.USE_MODERATE = True  # Whether to use moderate samples in CANDI adapter
_C.TEST.TTA.STEPS = 1  # Number of steps for TTA
_C.TEST.TTA.SOLVER = CN()
_C.TEST.TTA.SOLVER.OPTIMIZING_METHOD = 'sgd'  # sgd, Radam, sam_adam, sam_Radam, adamw
_C.TEST.TTA.SOLVER.BASE_LR = 0.0001
_C.TEST.TTA.SOLVER.WEIGHT_DECAY = 0.0001
_C.TEST.TTA.SOLVER.MOMENTUM = 0.9
_C.TEST.TTA.SOLVER.NESTEROV = True
_C.TEST.TTA.SOLVER.DAMPENING = 0.0
_C.TEST.TTA.SOLVER.GRADIENT_CLIP = True
_C.TEST.TTA.SOLVER.GRADIENT_CLIP_NORM = 0.5




_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 30
_C.SOLVER.OPTIMIZING_METHOD = 'adam'  # sgd, Radam, sam_adam, sam_Radam, adamw
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.LR_POLICY = 'cosine'
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = True
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_START_LR = 0.0001
_C.SOLVER.GRADIENT_CLIP = True
_C.SOLVER.GRADIENT_CLIP_NORM = 0.5


_C.MODEL = CN()
_C.MODEL.NAME = 'MLP'  # 'TIMESNET', 'MLP'



_C.MLP = CN()
_C.MLP.Z_SIZE = 128
_C.MLP.METRIC_NAMES = ("Recon",)
_C.MLP.LOSS_NAMES = ("Recon",)



_C.SANA = CN()
_C.SANA.TYPE = "TCN_iTrans"  # TCN_iTrans, Linear
_C.SANA.d_model = 512
_C.SANA.n_heads = 8
_C.SANA.e_layers = 1
_C.SANA.d_ff = 512
_C.SANA.dropout = 0.0
_C.SANA.activation = 'gelu'
_C.SANA.output_attention = True



_C.TIMESNET = CN()
_C.TIMESNET.e_layers = 2
_C.TIMESNET.d_model = 512
_C.TIMESNET.d_ff = 512
_C.TIMESNET.enc_in = 51
_C.TIMESNET.c_out = 51
_C.TIMESNET.top_k = 3
_C.TIMESNET.anomaly_ratio = 1
_C.TIMESNET.freq = 'h'
_C.TIMESNET.dropout = 0.1
_C.TIMESNET.num_kernels = 6
_C.TIMESNET.pred_len = 0
_C.TIMESNET.embed = 'timeF'
_C.TIMESNET.METRIC_NAMES = ("Recon",)
_C.TIMESNET.LOSS_NAMES = ("Recon",)

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = 'CANDI'
_C.WANDB.NAME = ''
_C.WANDB.JOB_TYPE = ''
_C.WANDB.NOTES = ''
_C.WANDB.DIR = ''


def get_cfg_defaults():

    return _C.clone()
