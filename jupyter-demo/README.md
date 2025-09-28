# WeDPR SecretFlow SDK (Alice/Bob vertical XGBoost)

This SDK wraps your notebooks (alice.ipynb and bob.ipynb) into reusable modules with a config-driven workflow.

## What it does

- Cluster init (Ray + SecretFlow) and device creation (alice, bob, SPU)
- PSI for CSVs (ECDH_PSI_2PC)
- Vertical CSV loading with SecretFlow
- Preprocessing:
  - Fill 'unknown' with mode
  - Binary encoding ('yes'/'no' → 1/0)
  - Ordinal encoding (education, month, ...)
  - One-hot encoding (job, marital, contact, poutcome, ...)
  - Standardization (all features except label)
- Train/test split
- SS-XGB (vertical) training and prediction
- Evaluation (AUC/KS/F1) and thresholded confusion matrix

## How to run

Prepare two YAML configs, one per role (alice/bob). Start Ray head and make sure ports match your env.

```bash
# Alice machine
python examples/run_alice.py --config examples/config_alice.yml

# Bob machine (use bob config with self_party=bob, psi.receiver=bob, role=bob)
python examples/run_bob.py --config examples/config_bob.yml
```

## Notes

- If PSI already executed and you have aligned CSVs, you can disable PSI (`psi.enabled: false`) and point data.alice_csv / data.bob_csv to the aligned outputs.
- The SDK currently focuses on SecretFlow backend, following your notebook code. It can be extended with additional models, encoders, and custom steps.
- Align parameters with your environment (Ray/ports/addresses). Check logs for gRPC and SPU connectivity.