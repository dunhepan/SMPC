from __future__ import annotations

import secretflow as sf
from secretflow.ml.boost.ss_xgb_v import Xgb


class XGBTrainer:
    def __init__(self, spu, params: dict):
        self.spu = spu
        self.params = params
        self._xgb = Xgb(spu)
        self._model = None

    def train(self, train_x, train_y):
        self._model = self._xgb.train(params=self.params, dtrain=train_x, label=train_y)
        return self._model

    def predict(self, x, to_pyu):
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        return self._model.predict(dtrain=x, to_pyu=to_pyu)

    @property
    def model(self):
        return self._model