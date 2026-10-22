"""MLflow "models from code" entry point for the LSTM forecaster.

Logging the model as *code* (this file) instead of a pickled Python object
means loading a model never unpickles arbitrary objects: MLflow imports this
module, which declares the model with ``set_model``. Weights and config are
plain artifacts (``weights.pt`` loaded with ``weights_only=True``, ``config.json``).
"""

from mlflow.models import set_model

from finsight.models.lstm.pyfunc_model import LSTMPricePyfunc

set_model(LSTMPricePyfunc())
