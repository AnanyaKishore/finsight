"""MLflow "models from code" entry point for the sentiment model.

As for the LSTM, the model is logged as *code* plus plain artifacts (the Hugging
Face model directory and ``config.json``), so loading it never unpickles objects.
"""

from mlflow.models import set_model

from finsight.sentiment.pyfunc_model import SentimentPyfunc

set_model(SentimentPyfunc())
