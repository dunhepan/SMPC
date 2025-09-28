import argparse

import os
import sys

# 确保可以 import sdk 包（examples 上一级目录）
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sdk import load_config, SecurePipeline
from sdk.cluster import shutdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (bob)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.role == "bob", "This runner is for Bob; set role: bob in config."

    pipe = SecurePipeline(cfg)
    try:
        psi_stats, (metrics, threshold_result) = pipe.run_all()

        if psi_stats:
            print("PSI stats:", psi_stats)
        print("Eval metrics:", metrics)
        if threshold_result:
            print("Threshold metrics:", threshold_result)

    finally:
        # 正确的退出顺序：释放远端对象 -> shutdown -> 退出进程
        try:
            pipe.cleanup()
        except Exception:
            pass
        shutdown()
        # 避免 atexit 再触发 __del__ 引起 Ray 自动重连
        sys.exit(0)

if __name__ == "__main__":
    main()