from __future__ import annotations
from typing import Dict

import secretflow as sf
from secretflow.data.vertical import read_csv as v_read_csv


def read_vertical_csv(
    parts: Dict,   # {pyu: path}
    spu,
    keys: str = "uid",
    drop_keys: str = "uid",
):
    """
    Read CSVs from parties vertically with key alignment using SPU.
    If you've already run PSI and prepared aligned CSVs, this will still work.
    """
    vdf = v_read_csv(
        parts,
        spu=spu,
        keys=keys,
        drop_keys=drop_keys,
    )
    return vdf