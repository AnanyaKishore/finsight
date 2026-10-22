"""LSTM 14-day price forecaster.

Modules:
    data      -- leak-free windows, per-window MinMax scaling, purged chronological split
    network   -- residual LSTM (forecast = last price + learned correction)
    metrics   -- RMSE / MAPE / directional accuracy vs naive and drift baselines
    pyfunc_model -- MLflow pyfunc wrapper (raw prices in, 14 prices out)
    train     -- DVC stage ``train_lstm`` (grid search + refit, MLflow tracking)
    evaluate  -- DVC stage ``evaluate_lstm`` (held-out test, plots, metrics)
"""
