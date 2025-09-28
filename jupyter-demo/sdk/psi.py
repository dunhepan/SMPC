from __future__ import annotations
import os
from typing import Dict

import secretflow as sf


def run_psi_csv(
    spu,
    key: str,
    input_path: Dict,   # {pyu: path}
    output_path: Dict,  # {pyu: path}
    receiver: str,
    protocol: str = "ECDH_PSI_2PC",
    sort: bool = True,
):
    """
    Runs PSI on CSV files for two parties using SPU.

    Returns:
      stats: list of dicts with party/original_count/intersection_count
    """
    stats = spu.psi_csv(
        key=key,
        input_path=input_path,
        output_path=output_path,
        receiver=receiver,
        protocol=protocol,
        sort=sort,
    )
    return sf.reveal(stats)