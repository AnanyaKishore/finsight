# FinSight — Reproducibility drill

Written by `python scripts/repro_drill.py`. The guidelines ask that every experiment be reproducible from a Git commit hash and an MLflow run ID; this drill proves it for the production forecasting model.

| | |
|---|---|
| Date | 2026-11-01 17:55 IST |
| Git commit (where the model was trained) | `e2fcca0` (requested: HEAD) |
| Original MLflow run (committed model) | `058edd597ce7440182a9667dc9a94d41` |
| Reproduced MLflow run (experiment `finsight-repro`) | `4f286f3d659948a7a3a01ec4bec21e61` |
| Fresh clone | `/tmp/finsight-repro-e2fcca0` |
| Environment | Python 3.11.17, Linux x86_64, `mlflow run --env-manager local` |

## Steps

```bash
git clone /home/ananya-kishore/finsight /tmp/finsight-repro-e2fcca0 && cd /tmp/finsight-repro-e2fcca0
git checkout e2fcca0
python -m finsight.security.secrets decrypt          # .env from secrets/.env.enc
dvc pull --with-deps evaluate_lstm                    # data + model of that commit
mlflow run . -e train_lstm --env-manager local --experiment-name finsight-repro
python -m finsight.models.lstm.evaluate               # same held-out test windows
```

## Result

**Verdict: identical.**

| Metric | Committed | Reproduced | Status |
|---|---|---|---|
| selected configuration (validation) | h32-l1-lr0.003 | h32-l1-lr0.003 | identical |
| best epoch | 3 | 3 | identical |
| validation MAPE (%) | 2.9571 | 2.9571 | identical |
| test MAPE, LSTM (%) | 3.2101 | 3.2101 | identical |
| test MAPE, naive (%) | 3.3394 | 3.3394 | identical |
| skill vs naive | 0.0387 | 0.0387 | identical |
| test windows | 150 | 150 | identical |

*identical* = equal to the precision stored in the report; *within tolerance* = differs by at most 0.01 (floating-point differences between machines or thread counts, for example a model trained inside the scheduler container and reproduced on the host); *differs* = a real difference to investigate.

Same seed (`lstm.seed` in `params.yaml`), same data (DVC hashes in `dvc.lock`), same code (the commit), same package versions (`environment.yml` → `requirements/*.txt`, also used by the Docker images).
