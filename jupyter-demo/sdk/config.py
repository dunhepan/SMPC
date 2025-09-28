from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class PartyConfig:
    address: str
    listen_addr: str


@dataclass
class ClusterConfig:
    parties: Dict[str, PartyConfig]
    self_party: str
    ray_address: str
    log_to_driver: bool = True
    ray_mode: bool = True


@dataclass
class SPUNodeConfig:
    party: str
    address: str


@dataclass
class SPURuntimeConfig:
    protocol: str = "SEMI2K"    # spu.spu_pb2.SEMI2K
    field: str = "FM128"        # spu.spu_pb2.FM128


@dataclass
class SPUClusterDef:
    nodes: List[SPUNodeConfig]
    runtime_config: SPURuntimeConfig
    link_desc: Dict[str, Any] = field(default_factory=lambda: {
        "connect_retry_times": 60,
        "connect_retry_interval_ms": 1000,
    })


@dataclass
class PSIConfig:
    enabled: bool = True
    key: str = "uid"
    receiver: str = "alice"
    protocol: str = "ECDH_PSI_2PC"
    sort: bool = True
    output_dir: str = "psi_results"


@dataclass
class DataConfig:
    # CSV file paths (local to each party’s machine)
    alice_csv: str
    bob_csv: str
    keys: str = "uid"
    drop_keys: str = "uid"


@dataclass
class PreprocessConfig:
    # Columns for each step; leave list empty if you don’t use that operation
    fill_unknown_cols: List[str] = field(default_factory=list)
    binary_cols: List[str] = field(default_factory=list)     # e.g., ["default", "housing", "loan", "y"]
    ordinal_cols: List[str] = field(default_factory=list)    # e.g., ["education", "month"]
    onehot_cols: List[str] = field(default_factory=list)     # e.g., ["job", "marital", "contact", "poutcome"]
    label_col: str = "y"
    # Whether to standardize numeric features (all non-label columns after encoding)
    standardize: bool = True


@dataclass
class SplitConfig:
    train_size: float = 0.8
    random_state: int = 1234


@dataclass
class XGBConfig:
    num_boost_round: int = 3
    max_depth: int = 5
    sketch_eps: float = 0.05
    objective: str = "logistic"
    reg_lambda: float = 0.5
    subsample: float = 0.4
    base_score: float = 0.11
    # to_pyu for prediction, default to "bob"
    predict_to: str = "bob"


@dataclass
class EvalConfig:
    bucket_size: int = 20
    # Optional threshold for confusion matrix preview
    threshold: Optional[float] = 0.17
    confusion_to: str = "bob"  # which party to materialize y_true / y_score for confusion matrix


@dataclass
class SDKConfig:
    role: str                    # "alice" or "bob"
    cluster: ClusterConfig
    spu: SPUClusterDef
    psi: PSIConfig
    data: DataConfig
    preprocess: PreprocessConfig
    split: SplitConfig
    xgb: XGBConfig
    eval: EvalConfig


def load_config(path: str) -> SDKConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Parties mapping -> PartyConfig
    parties_cfg = {
        name: PartyConfig(**v) for name, v in raw["cluster"]["parties"].items()
    }

    cluster = ClusterConfig(
        parties=parties_cfg,
        self_party=raw["cluster"]["self_party"],
        ray_address=raw["cluster"]["ray_address"],
        log_to_driver=raw["cluster"].get("log_to_driver", True),
        ray_mode=raw["cluster"].get("ray_mode", True),
    )

    spu_nodes = [SPUNodeConfig(**n) for n in raw["spu"]["nodes"]]
    spu_runtime = SPURuntimeConfig(**raw["spu"]["runtime_config"])
    spu = SPUClusterDef(
        nodes=spu_nodes,
        runtime_config=spu_runtime,
        link_desc=raw["spu"].get("link_desc", {
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        })
    )

    psi = PSIConfig(**raw["psi"])
    data = DataConfig(**raw["data"])
    preprocess = PreprocessConfig(**raw["preprocess"])
    split = SplitConfig(**raw["split"])
    xgb = XGBConfig(**raw["xgb"])
    eval_cfg = EvalConfig(**raw["eval"])

    cfg = SDKConfig(
        role=raw["role"],
        cluster=cluster,
        spu=spu,
        psi=psi,
        data=data,
        preprocess=preprocess,
        split=split,
        xgb=xgb,
        eval=eval_cfg,
    )
    return cfg