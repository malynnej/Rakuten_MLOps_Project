#!/usr/bin/env sh
set -eu

echo "[dvc_init] dvc version:"
dvc --version || true

echo "[dvc_init] writing runtime creds to .dvc/config.local"
mkdir -p .dvc
umask 077

: "${DVC_HTTP_USER:?DVC_HTTP_USER missing}"
: "${DVC_HTTP_PASSWORD:?DVC_HTTP_PASSWORD missing}"

cat > .dvc/config.local <<EOC
[remote "origin"]
    auth = basic
    user = ${DVC_HTTP_USER}
    password = ${DVC_HTTP_PASSWORD}
EOC

echo "[dvc_init] pulling from origin..."
dvc pull -r origin --force

echo "[dvc_init] done"
touch /tmp/dvc_done

# keep container alive so service_healthy works
tail -f /dev/null
