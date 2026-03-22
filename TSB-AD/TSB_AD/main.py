# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import torch
import random, argparse
import numpy as np
import wandb
from sklearn.preprocessing import MinMaxScaler
from .evaluation.metrics import get_metrics
from .utils.slidingWindows import find_length_rank
from .model_wrapper import *
from .HP_list import Optimal_Uni_algo_HP_dict

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    parser.add_argument('--filename', type=str, default='172_SWaT_id_2_Sensor_tr_23700_1st_23800.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='CANDI')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')

    #! CANDI configurations
    parser.add_argument('--ttlr', type=float, default=1e-3, help='learning rate for test time adaptation')
    parser.add_argument('--tta_steps', type=int, default=10, help='number of optimization steps for test time adaptation')
    parser.add_argument('--gating_init', type=float, default=0.5, help='initial value of gating parameter')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="CANDI",
        name='TSB-AD',
        dir="./wandb",
        config={
            "filename": args.filename,
            "data_direc": args.data_direc,
            "AD_Name": args.AD_Name,
            "seed_num": args.seed,
            "ttlr": args.ttlr,
            "tta_steps": args.tta_steps,
            "gating_init": args.gating_init,
        }
    )

    # seeding
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("CUDA Available: ", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]

    if args.AD_Name in Semisupervise_AD_Pool:
        if args.AD_Name == 'CANDI':
            Optimal_Det_HP['ttlr'] = args.ttlr
            Optimal_Det_HP['tta_steps'] = args.tta_steps
            Optimal_Det_HP['gating_init'] = args.gating_init
        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    if isinstance(output, np.ndarray):
        output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+3*np.std(output)))
        print('Evaluation Result: ', evaluation_result)
        
        # Log evaluation result to wandb
        wandb.log(evaluation_result)
    else:
        print(f'At {args.filename}: '+output)
        # Log error message to wandb
        wandb.log({"error": output})

    wandb.finish()
