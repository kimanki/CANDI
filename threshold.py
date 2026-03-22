from typing import Optional
from functools import cached_property

import numpy as np
import sklearn.metrics as metrics


class Thresholder:
    def __init__(
            self,
            cfg,
            test_scores: Optional[np.ndarray] = None,
            test_labels: Optional[np.ndarray] = None,
            train_scores: Optional[np.ndarray] = None,
            val_scores: Optional[np.ndarray] = None
    ):
        self.cfg = cfg
        self.test_scores = test_scores
        self.test_labels = test_labels
        self.train_scores = train_scores
        self.val_scores = val_scores

        self.threshold_type = cfg.TEST.THRESHOLD.TYPE
        assert self.threshold_type in ('ratio', 'best_f1')
        self.threshold = self._init_threshold()

    def _init_threshold(self):
        if self.threshold_type == 'ratio':
            # reference: https://github.com/thuml/Anomaly-Transformer/solver.py
            self.ratio = self.cfg.TEST.THRESHOLD.ANOMALY_RATIO
            # threshold_ratio = np.percentile(self.val_scores, 100 - self.ratio)
            threshold_ratio = np.percentile(self.val_scores, 100 - self.ratio)
            return threshold_ratio
        elif self.threshold_type == 'best_f1':
            precision, recall, thresholds = metrics.precision_recall_curve(self.test_labels, self.test_scores)
            best_f1_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-12))
            threshold_best_f1 = thresholds[best_f1_idx]
            return threshold_best_f1
        else:
            raise ValueError(f"Unknown threshold type: {self.threshold_type}")

    def set_threshold(self, threshold: float):
        """
        Set a custom threshold value.
        
        Args:
            threshold (float): The new threshold value to set.
        """
        self.threshold = threshold