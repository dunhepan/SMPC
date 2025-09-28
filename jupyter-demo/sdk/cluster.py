from __future__ import annotations

import secretflow as sf
import spu as spu_lib

def _spu_protocol(protocol: str):
    # Map string to spu.spu_pb2 enum
    p = protocol.upper()
    if hasattr(spu_lib.spu_pb2, p):
        return getattr(spu_lib.spu_pb2, p)
    raise ValueError(f"Unknown SPU protocol: {protocol}")


def _spu_field(field: str):
    f = field.upper()
    if hasattr(spu_lib.spu_pb2, f):
        return getattr(spu_lib.spu_pb2, f)
    raise ValueError(f"Unknown SPU field: {field}")


def init_secretflow(
    cluster_cfg,
    spu_cfg,
):
    # Build cluster config for sf.init
    parties = {}
    for name, party in cluster_cfg.parties.items():
        parties[name] = {
            "address": party.address,
            "listen_addr": party.listen_addr,
        }

    sf.init(
        cluster_config={
            "parties": parties,
            "self_party": cluster_cfg.self_party,
        },
        address=cluster_cfg.ray_address,
        log_to_driver=cluster_cfg.log_to_driver,
        ray_mode=cluster_cfg.ray_mode,
    )

    # create devices
    devices = {name: sf.PYU(name) for name in cluster_cfg.parties.keys()}

    # SPU cluster def
    spu_cluster_def = {
        "nodes": [{"party": n.party, "address": n.address} for n in spu_cfg.nodes],
        "runtime_config": {
            "protocol": _spu_protocol(spu_cfg.runtime_config.protocol),
            "field": _spu_field(spu_cfg.runtime_config.field),
        },
    }

    spu = sf.SPU(spu_cluster_def, link_desc=spu_cfg.link_desc)
    return devices, spu


def shutdown():
    """Shutdown SecretFlow/Ray gracefully."""
    # 先尝试 SecretFlow 的 shutdown（内部会处理 Ray）
    try:
        sf.shutdown()
    except Exception:
        pass
    # 再次兜底 Ray
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass