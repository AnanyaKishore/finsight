"""Security helpers (Day 9): encryption of secrets and of backups at rest.

* :mod:`finsight.security.crypto`  Fernet key handling and chunked file encryption.
* :mod:`finsight.security.secrets` ``.env`` <-> ``secrets/.env.enc`` (safe to commit).
* :mod:`finsight.security.backup`  encrypted snapshots of MongoDB, reports and models.
"""
