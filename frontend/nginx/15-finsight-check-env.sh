#!/bin/sh
# Fail fast on a malformed BACKEND_URL / MLFLOW_UI_URL, before nginx starts.
# Both values are written into config.js and the Content-Security-Policy, so
# anything other than a plain http(s) URL (quotes, spaces, semicolons) is refused.
set -eu

check_url() {
    name="$1"
    value="$2"
    if ! printf '%s' "$value" | grep -Eq '^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?$'; then
        echo "finsight: $name must be a plain http(s) URL such as http://localhost:8000 (got: '$value')" >&2
        exit 1
    fi
}

check_url BACKEND_URL "${BACKEND_URL:-}"
check_url MLFLOW_UI_URL "${MLFLOW_UI_URL:-}"
echo "finsight: portal will call the API at $BACKEND_URL"
