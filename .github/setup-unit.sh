#!/usr/bin/env bash
set -euo pipefail

# === Configuration ===
MTLS_DIR="src/test/resources/mtls"
STOREPASS="keystorePassword"
TRUSTPASS="changeit"

mkdir -p "${MTLS_DIR}"

echo "🧹 Cleaning previous certs..."
rm -f "${MTLS_DIR}"/* || true

# === 1. Create CA ===
echo "📜 Generating CA certificate..."
openssl req -x509 -newkey rsa:4096 -days 1825 -nodes \
  -keyout "${MTLS_DIR}/ca-key.pem" \
  -out "${MTLS_DIR}/ca-cert.pem" \
  -subj "/CN=KestraTestCA"

# === 2. Create server certificate signed by CA ===
echo "🔐 Generating server certificate..."
openssl genrsa -out "${MTLS_DIR}/server-key.pem" 4096
openssl req -new -key "${MTLS_DIR}/server-key.pem" \
  -out "${MTLS_DIR}/server.csr" -subj "/CN=localhost"

openssl x509 -req -in "${MTLS_DIR}/server.csr" \
  -CA "${MTLS_DIR}/ca-cert.pem" -CAkey "${MTLS_DIR}/ca-key.pem" -CAcreateserial \
  -out "${MTLS_DIR}/server-cert.pem" -days 825 -sha256

# Combine chain (server + CA)
cat "${MTLS_DIR}/server-cert.pem" "${MTLS_DIR}/ca-cert.pem" > "${MTLS_DIR}/server-chain.pem"

# === 3. Create client certificate signed by CA ===
echo "👤 Generating client certificate..."
openssl genrsa -out "${MTLS_DIR}/client-key.pem" 4096
openssl req -new -key "${MTLS_DIR}/client-key.pem" \
  -out "${MTLS_DIR}/client.csr" -subj "/CN=KestraTestClient"

openssl x509 -req -in "${MTLS_DIR}/client.csr" \
  -CA "${MTLS_DIR}/ca-cert.pem" -CAkey "${MTLS_DIR}/ca-key.pem" -CAcreateserial \
  -out "${MTLS_DIR}/client-cert.pem" -days 825 -sha256

# Combine client cert + key for easier loading in tests
cat "${MTLS_DIR}/client-cert.pem" "${MTLS_DIR}/client-key.pem" > "${MTLS_DIR}/client-cert-key.pem"

# === 4. Create PKCS12 keystore for WireMock server (with full chain) ===
echo "🏗️ Creating server-keystore.p12 (with full chain)..."
openssl pkcs12 -export \
  -inkey "${MTLS_DIR}/server-key.pem" \
  -in "${MTLS_DIR}/server-chain.pem" \
  -out "${MTLS_DIR}/server-keystore.p12" \
  -name "wiremock-server" \
  -password pass:${STOREPASS}

# === 5. Create PKCS12 truststore for the client (containing CA) ===
echo "🧱 Creating client-truststore.p12..."
keytool -importcert -noprompt \
  -alias kestra-ca \
  -file "${MTLS_DIR}/ca-cert.pem" \
  -keystore "${MTLS_DIR}/client-truststore.p12" \
  -storetype PKCS12 \
  -storepass "${TRUSTPASS}"

# Cleanup temporary files
rm -f "${MTLS_DIR}/server.csr" "${MTLS_DIR}/client.csr" "${MTLS_DIR}/ca-cert.srl"

echo "✅ Done. Generated files:"
ls -1 "${MTLS_DIR}"
