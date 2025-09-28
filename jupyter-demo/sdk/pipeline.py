from __future__ import annotations

import os
from typing import Dict

import secretflow as sf

from .config import SDKConfig
from .cluster import init_secretflow
from .psi import run_psi_csv
from .data import read_vertical_csv
from .preprocess import Preprocessor
from .model import XGBTrainer
from .eval import eval_biclassification, confusion_at_threshold


class SecurePipeline:
    def __init__(self, cfg: SDKConfig):
        self.cfg = cfg
        self.devices = None
        self.spu = None
        self.vdf = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.trainer = None
        self.y_score = None

    def init_cluster(self):
        self.devices, self.spu = init_secretflow(self.cfg.cluster, self.cfg.spu)
        return self.devices, self.spu

    def run_psi(self):
        if not self.cfg.psi.enabled:
            return None
        os.makedirs(self.cfg.psi.output_dir, exist_ok=True)
        alice = self.devices["alice"]
        bob = self.devices["bob"]

        input_path = {
            alice: self.cfg.data.alice_csv,
            bob: self.cfg.data.bob_csv,
        }
        output_path = {
            alice: os.path.join(self.cfg.psi.output_dir, "alice_psi_result.csv"),
            bob: os.path.join(self.cfg.psi.output_dir, "bob_psi_result.csv"),
        }
        stats = run_psi_csv(
            spu=self.spu,
            key=self.cfg.psi.key,
            input_path=input_path,
            output_path=output_path,
            receiver=self.cfg.psi.receiver,
            protocol=self.cfg.psi.protocol,
            sort=self.cfg.psi.sort,
        )
        return stats, output_path

    def load_vertical_data(self, files_after_psi: Dict = None):
        alice = self.devices["alice"]
        bob = self.devices["bob"]
        if files_after_psi is None:
            parts = {
                alice: self.cfg.data.alice_csv,
                bob: self.cfg.data.bob_csv,
            }
        else:
            parts = files_after_psi

        self.vdf = read_vertical_csv(
            parts=parts,
            spu=self.spu,
            keys=self.cfg.data.keys,
            drop_keys=self.cfg.data.drop_keys,
        )
        return self.vdf

    def preprocess(self):
        pp = Preprocessor(
            fill_unknown_cols=self.cfg.preprocess.fill_unknown_cols,
            binary_cols=self.cfg.preprocess.binary_cols,
            ordinal_cols=self.cfg.preprocess.ordinal_cols,
            onehot_cols=self.cfg.preprocess.onehot_cols,
            label_col=self.cfg.preprocess.label_col,
            standardize=self.cfg.preprocess.standardize,
        )
        self.vdf = pp.apply(self.vdf)
        return self.vdf

    def split(self):
        from secretflow.data.split import train_test_split

        label_col = self.cfg.preprocess.label_col
        train_vdf, test_vdf = train_test_split(
            self.vdf,
            train_size=self.cfg.split.train_size,
            random_state=self.cfg.split.random_state,
        )
        self.train_x = train_vdf.drop(columns=[label_col])
        self.train_y = train_vdf[label_col]
        self.test_x = test_vdf.drop(columns=[label_col])
        self.test_y = test_vdf[label_col]
        return (self.train_x, self.train_y, self.test_x, self.test_y)

    def train_xgb(self):
        params = dict(
            num_boost_round=self.cfg.xgb.num_boost_round,
            max_depth=self.cfg.xgb.max_depth,
            sketch_eps=self.cfg.xgb.sketch_eps,
            objective=self.cfg.xgb.objective,
            reg_lambda=self.cfg.xgb.reg_lambda,
            subsample=self.cfg.xgb.subsample,
            base_score=self.cfg.xgb.base_score,
        )
        self.trainer = XGBTrainer(self.spu, params)
        self.trainer.train(self.train_x, self.train_y)
        return self.trainer.model

    def predict(self):
        to_role = self.cfg.xgb.predict_to
        to_pyu = self.devices[to_role]
        self.y_score = self.trainer.predict(self.test_x, to_pyu=to_pyu)
        return self.y_score

    def evaluate(self):
        # BiClassification metrics (AUC, KS, F1)
        metrics = eval_biclassification(self.test_y, self.y_score, bucket_size=self.cfg.eval.bucket_size)

        # Optional threshold metrics and confusion matrix
        threshold_result = None
        if self.cfg.eval.threshold is not None:
            to_role = self.cfg.eval.confusion_to
            bob_pyu = self.devices[to_role]
            threshold_result = confusion_at_threshold(
                bob_pyu=bob_pyu,
                test_y=self.test_y,
                y_score=self.y_score,
                threshold=self.cfg.eval.threshold,
            )
        return metrics, threshold_result

    def run_all(self):
        # 1) init
        self.init_cluster()
        # 2) psi
        files_after_psi = None
        stats = None
        if self.cfg.psi.enabled:
            stats, output_path = self.run_psi()
            files_after_psi = output_path
        # 3) load
        self.load_vertical_data(files_after_psi)
        # 4) preprocess
        self.preprocess()
        # 5) split
        self.split()
        # 6) train
        self.train_xgb()
        # 7) predict
        self.predict()
        # 8) evaluate
        return stats, self.evaluate()

    def cleanup(self):
        """Release remote objects BEFORE ray/secretflow shutdown."""
        try:
            # 清空持有远端引用的属性
            self.vdf = None
            self.train_x = None
            self.train_y = None
            self.test_x = None
            self.test_y = None
            self.y_score = None
            self.trainer = None
        except Exception:
            pass
        # 触发垃圾回收，让 Partition.__del__ 在 Ray 仍然存活时执行
        try:
            import gc
            gc.collect()
        except Exception:
            pass