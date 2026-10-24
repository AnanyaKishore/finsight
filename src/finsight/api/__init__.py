"""FinSight REST backend (FastAPI).

Loosely coupled: the frontend only knows the base URL; model servers are
reached over HTTP (MLflow model serving); all data comes from MongoDB.
Run locally with ``python -m finsight.api`` (or the ``backend`` compose service).
"""
