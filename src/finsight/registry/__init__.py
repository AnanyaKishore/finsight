"""MLflow Model Registry: registration, champion/challenger promotion, rollback.

* ``register`` -- DVC stage: registers new model versions and runs the gate.
* ``rollback`` -- CLI: move ``@production`` back to a previous version.
* ``gate``     -- pure decision rules (unit-tested without MLflow).
* ``pointer``  -- read ``reports/promotion.json`` and resolve the production model.
"""
