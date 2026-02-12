#!/bin/bash

# This script generates self-signed SSL certificates for Nginx.
# It creates a directory for the certificates if it doesn't exist,
# and generates a private key and a self-signed certificate.

# IMPORTANT: for the python requests library to accept this certificate,
# the server name has to be set in two places: 
# - CN (common name)
# - subjectAltName
# Source: https://medium.com/@maciej.skorupka/hostname-mismatch-ssl-error-in-python-2901d465683

CERT_DIR="./deployments/nginx/certs"

echo "Creating certificate directory at $CERT_DIR..."
mkdir -p "$CERT_DIR"

echo "Generating self-signed SSL certificate and private key..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout deployments/nginx/certs/nginx.key \
    -out deployments/nginx/certs/nginx.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:nginx_reverse_proxy"
