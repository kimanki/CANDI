# adapted from: https://github.com/thuml/Anomaly-Transformer/data_factory/data_loader.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from datasets.util import downsample


class TSDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.dataset_name = cfg.DATA.NAME
        self.data_dir = os.path.join(cfg.DATA.BASE_DIR, cfg.DATA.NAME)
        self.win_size = cfg.DATA.WIN_SIZE
        self.step = cfg.DATA.TRAIN_STEP if split == "train" else cfg.DATA.TEST_STEP
        self.scale = cfg.DATA.SCALE
        self.split = split
        self.train_ratio = cfg.DATA.TRAIN_RATIO
        self.downsample_rate = cfg.DATA.DOWNSAMPLE_RATE

        self.train, self.val, self.test, self.train_labels, self.val_labels, self.test_labels = self._load_data()
        self.test_labels = np.squeeze(self.test_labels)
        self._normalize_data()
    
    def _save_downsampled_data(self, train, val, test, train_labels, val_labels, test_labels):
        assert self.downsample_rate > 1
        np.save(os.path.join(self.data_dir, f"downsampled_train_rate{self.downsample_rate}.npy"), train)
        np.save(os.path.join(self.data_dir, f"downsampled_val_rate{self.downsample_rate}.npy"), val)
        np.save(os.path.join(self.data_dir, f"downsampled_test_rate{self.downsample_rate}.npy"), test)
        np.save(os.path.join(self.data_dir, f"downsampled_train_labels_rate{self.downsample_rate}.npy"), train_labels)
        np.save(os.path.join(self.data_dir, f"downsampled_val_labels_rate{self.downsample_rate}.npy"), val_labels)
        np.save(os.path.join(self.data_dir, f"downsampled_test_labels_rate{self.downsample_rate}.npy"), test_labels)
    
    def _load_downsampled_data(self):
        train = np.load(os.path.join(self.data_dir, f"downsampled_train_rate{self.downsample_rate}.npy"))
        val = np.load(os.path.join(self.data_dir, f"downsampled_val_rate{self.downsample_rate}.npy"))
        test = np.load(os.path.join(self.data_dir, f"downsampled_test_rate{self.downsample_rate}.npy"))
        train_labels = np.load(os.path.join(self.data_dir, f"downsampled_train_labels_rate{self.downsample_rate}.npy"))
        val_labels = np.load(os.path.join(self.data_dir, f"downsampled_val_labels_rate{self.downsample_rate}.npy"))
        test_labels = np.load(os.path.join(self.data_dir, f"downsampled_test_labels_rate{self.downsample_rate}.npy"))
        return train, val, test, train_labels, val_labels, test_labels

    def _load_data(self):
        raise NotImplementedError

    def _split_train_val(self, train, train_labels):
        # use the latter portion of the training data as validation data.
        assert self.train_ratio < 1.0

        train_len = int(len(train) * self.train_ratio)
        val = train[train_len:].copy()
        val_labels = train_labels[train_len:].copy()
        train = train[:train_len]
        train_labels = train_labels[:train_len]
        
        return train, val, train_labels, val_labels

    def _normalize_data(self):
        if self.scale in ("standard", "instance"):
            scaler = StandardScaler()
        elif self.scale == "min-max":
            scaler = MinMaxScaler()
        elif self.scale == "none":
            return
        else:
            raise ValueError

        self.train = scaler.fit_transform(self.train)
        self.val = scaler.transform(self.val)
        self.test = scaler.transform(self.test)

    def print_data_stats(self):
        if self.split == 'train':
            print(f"Train data shape: {self.train.shape}, mean: {np.mean(self.train, axis=0)}, std: {np.std(self.train, axis=0)}")
        elif self.split == 'val':
            print(f"Validation data shape: {self.val.shape}, mean: {np.mean(self.val, axis=0)}, std: {np.std(self.val, axis=0)}")
        elif self.split == 'test':
            print(f"Test data shape: {self.test.shape}, mean: {np.mean(self.test, axis=0)}, std: {np.std(self.test, axis=0)}")

    def __len__(self):
        if self.split == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.split == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.split == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.split == "train":
            inputs = self.train
            labels = self.train_labels
        elif self.split == 'val':
            inputs = self.val
            labels = self.val_labels
        elif self.split == 'test':
            inputs = self.test
            labels = self.test_labels
        
        return np.float32(inputs[index:index + self.win_size]), int(np.any(labels[index:index + self.win_size] == 1))


class SMDSegLoader(TSDataset):
    def __init__(self, cfg, split):
        self.entity = cfg.DATA.NAME.split('_')[1]  #? e.g., SMD_1-1
        super(SMDSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'ServerMachineDataset')
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_test.pkl'), 'rb') as f:
            test = pickle.load(f)
        with open(os.path.join(self.data_dir, 'preprocessed', f'machine-{self.entity}_test_label.pkl'), 'rb') as f:
            test_labels = pickle.load(f)
        test_labels = test_labels.astype(int)
        # placeholder for train, val labels
        train_labels = np.zeros(train.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class SWaTSegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(SWaTSegLoader, self).__init__(cfg, split)
    
    def _load_data(self):
        if self.downsample_rate > 1:
            try:
                return self._load_downsampled_data()
            except:
                print(f"downsampled data not found. Load original data instead and downsample.")
        
        train_csv_path = self.data_dir + "/SWaT_Dataset_Normal_v1.csv"
        train_excel_path = self.data_dir + "/SWaT_Dataset_Normal_v1.xlsx"

        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
        elif os.path.exists(train_excel_path):
            train_df = pd.read_excel(train_excel_path)
            train_df.to_csv(train_csv_path, index=False)
        else:
            raise FileNotFoundError(f"Neither {train_csv_path} nor {train_excel_path} exists.")
        train_df = train_df.iloc[1:, 1:-1]
        train_df = train_df.astype(np.float32)

        test_csv_path = self.data_dir + "/SWaT_Dataset_Attack_v0.csv"
        test_excel_path = self.data_dir + "/SWaT_Dataset_Attack_v0.xlsx"

        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
        elif os.path.exists(test_excel_path):
            test_df = pd.read_excel(test_excel_path)
            test_df.to_csv(test_csv_path, index=False)
        else:
            raise FileNotFoundError(f"Neither {test_csv_path} nor {test_excel_path} exists.")
        test_labels = (test_df.iloc[:, -1] == 'Attack').to_numpy().astype(int)
        test_df = test_df.iloc[1:, 1:-1]
        test_df = test_df.astype(np.float32)
        
        self.var_names = list(train_df.columns)
        
        train = train_df.values
        train_labels = np.zeros(train.shape[0])
        test = test_df.values
        
        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()
        
        if self.downsample_rate > 1:
            train, val, test = downsample(train, self.downsample_rate), downsample(val, self.downsample_rate), downsample(test, self.downsample_rate)
            train_labels, val_labels, test_labels = downsample(train_labels, self.downsample_rate), downsample(val_labels, self.downsample_rate), downsample(test_labels, self.downsample_rate)
            self._save_downsampled_data(train, val, test, train_labels, val_labels, test_labels)
        
        return train, val, test, train_labels, val_labels, test_labels

def build_dataset(cfg, split):
    dataset_name = cfg.DATA.NAME

    dataset_loaders = {
        "SMD": SMDSegLoader,
        "SWaT": SWaTSegLoader,
    }

    for key in dataset_loaders:
        if key in dataset_name:
            return dataset_loaders[key](cfg, split)

    raise ValueError(f"Unknown dataset name: {dataset_name}")
