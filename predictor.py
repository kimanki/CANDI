import os
from copy import deepcopy

import numpy as np
import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from threshold import Thresholder
from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from trainer import prepare_inputs
from utils.misc import mkdir
from tta.adapter import construct_adapter


class Predictor:
    def __init__(self, cfg, model, tta=False):
        self.cfg = cfg.clone()

        self.model = deepcopy(model)
        self.tta = tta

        self.cfg.DATA.TRAIN_STEP = 1
        self.cfg.TRAIN.SHUFFLE = False
        self.cfg.TRAIN.DROP_LAST = False
        
        self.train_loader = get_train_dataloader(self.cfg)
        self.val_loader = get_val_dataloader(self.cfg)
        self.test_loader = get_test_dataloader(self.cfg)

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        log_dict = {}
        
        self.train_scores = self._get_train_scores()
        if self.tta and self.cfg.TEST.TTA.METHOD == 'CANDI':
            self.val_scores, self.val_inputs = self._get_val_scores(return_inputs=True)
        else:
            self.val_scores = self._get_val_scores()

        #! results without TTA
        self.test_scores = self._get_test_scores()
        self.test_labels = self._get_test_labels()

        assert len(self.test_scores) == len(self.test_labels)

        self.thresholder = Thresholder(self.cfg, self.test_scores, self.test_labels, self.train_scores, self.val_scores)

        pred = self.test_scores > self.thresholder.threshold

        results = self.get_results(self.test_scores, self.test_labels, pred)  # {AUROC: , AUPRC: , Precision: , Recall: , F1: }

        if self.tta:
            self.cfg.TEST.THRESHOLD.TYPE = 'ratio'
            self.thresholder = Thresholder(self.cfg, None, None, self.train_scores, self.val_scores)
            self.adapter = construct_adapter(
                self.cfg, 
                self.model, 
                self.thresholder, 
                test_labels=self.test_labels,  # for CANDIAdapter
                val_scores=self.val_scores,  # for CANDIAdapter
                val_inputs=self.val_inputs if self.cfg.TEST.TTA.METHOD == 'CANDI' else None  # for CANDIAdapter
                )

            self.test_scores_w_tta = []
            pred_w_tta = []
            for inputs in tqdm(self.test_loader, desc='test scores'):
                inputs, labels = prepare_inputs(inputs)
                scores = self.model.get_anomaly_scores(inputs)
                self.test_scores_w_tta.append(scores)
                pred_w_tta.append(scores > self.adapter.thresholder.threshold)
                
                self.adapter.adapt(inputs, scores)
                
            self.test_scores_w_tta = torch.concat(self.test_scores_w_tta, dim=0).cpu().numpy()
            pred_w_tta = torch.concat(pred_w_tta, dim=0).cpu().numpy()

            results_w_tta = self.get_results(self.test_scores_w_tta, self.test_labels, pred_w_tta)  # {AUROC: , AUPRC: , Precision: , Recall: , F1: }
            results_w_tta = {f"{k}_w_tta": v for k, v in results_w_tta.items()}
            results = {f"{k}_wo_tta": v for k, v in results.items()}
            results.update(results_w_tta)

        self.save_results(results)
        self.save_to_npy(**{
            "train_scores": self.train_scores, 
            "val_scores": self.val_scores,
            "test_scores": self.test_scores, 
            "test_labels": self.test_labels, 
            "threshold": self.thresholder.threshold
        })
        if self.tta:
            self.save_to_npy(test_scores_w_tta=self.test_scores_w_tta)

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

    @torch.no_grad()
    def _get_scores_all(self, dataloader, desc, return_inputs=False) -> np.ndarray:
        scores_all = []
        inputs_all = []
        self.model.eval()
        # for inputs in tqdm(dataloader, desc=desc):
        for inputs in dataloader:
            inputs, _ = prepare_inputs(inputs)
            scores = self.model.get_anomaly_scores(inputs)
            scores_all.append(scores)
            if return_inputs:
                inputs_all.append(inputs)
        scores_all = torch.flatten(torch.concat(scores_all, dim=0))
        scores_np = scores_all.cpu().numpy()
        if return_inputs:
            inputs_all = torch.cat(inputs_all, dim=0)
            return scores_np, inputs_all
        else:
            return scores_np

    def _get_train_scores(self, **kwargs):
        return self._get_scores_all(self.train_loader, 'train scores', **kwargs)
    
    def _get_val_scores(self, **kwargs):
        return self._get_scores_all(self.val_loader, 'val scores', **kwargs)

    def _get_test_scores(self, **kwargs):
        return self._get_scores_all(self.test_loader, 'test scores', **kwargs)

    @torch.no_grad()
    def _get_test_labels(self) -> np.ndarray:
        labels_all = []
        for idx, inputs in enumerate(self.test_loader):
            _, labels = prepare_inputs(inputs)
            labels_all.append(labels)
        labels_all = torch.flatten(torch.concat(labels_all, dim=0))

        return labels_all.cpu().numpy()

    @staticmethod
    def get_results(scores, labels, pred):
        results = {}
        
        #! AUROC
        auroc = float(roc_auc_score(labels, scores))
        results.update({"AUROC": auroc})
        
        #! AUPRC
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        auprc = float(auc(recall, precision))
        results.update({"AUPRC": auprc})
        
        #! Precision, Recall, F1
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        results.update({"Precision": precision, "Recall": recall, "F1": f1})
        
        return results

    def save_results(self, results, score_type=None):
        results_string = ", ".join([f"{metric}: {value:.04f}" for metric, value in results.items()])
        print(results_string)

        file_name = 'test' if score_type is None else f'test_{score_type}'
        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / f"{file_name}.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)
