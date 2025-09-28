from __future__ import annotations
from typing import List

import numpy as np
import secretflow as sf
from secretflow.preprocessing.encoder import VOrdinalEncoder, OneHotEncoder
from secretflow.preprocessing import StandardScaler


class Preprocessor:
    def __init__(
        self,
        fill_unknown_cols: List[str],
        binary_cols: List[str],
        ordinal_cols: List[str],
        onehot_cols: List[str],
        label_col: str = "y",
        standardize: bool = True,
    ):
        self.fill_unknown_cols = fill_unknown_cols
        self.binary_cols = binary_cols
        self.ordinal_cols = ordinal_cols
        self.onehot_cols = onehot_cols
        self.label_col = label_col
        self.standardize = standardize

        # persistent encoders/scalers if you want to reuse later
        self._ordinal_enc = VOrdinalEncoder()
        self._onehot_enc = OneHotEncoder()
        self._scaler = StandardScaler()

    def apply(self, vdf):
        # 1) Fill 'unknown' with mode for specific columns
        for col in self.fill_unknown_cols:
            vdf[col] = vdf[col].replace("unknown", np.NaN)
            vdf[col] = vdf[col].fillna(vdf[col].mode().iloc[0])

        # 2) Binary encode 'yes'/'no'
        if self.binary_cols:
            for col in self.binary_cols:
                vdf[col] = vdf[col].replace({"no": 0, "yes": 1})

        # 3) Ordinal encode ordered columns
        for col in self.ordinal_cols:
            enc_df = self._ordinal_enc.fit_transform(vdf[[col]])
            vdf[col] = enc_df[col]

        # 4) One-hot encode nominal columns
        for col in self.onehot_cols:
            transformed = self._onehot_enc.fit_transform(vdf[[col]])
            vdf[transformed.columns] = transformed
            vdf = vdf.drop(columns=[col])

        # 5) Standardize numeric features (all except label)
        if self.standardize:
            X = vdf.drop(columns=[self.label_col])
            y = vdf[self.label_col]
            X_scaled = self._scaler.fit_transform(X)
            vdf[X_scaled.columns] = X_scaled
            vdf[self.label_col] = y  # restore

        return vdf