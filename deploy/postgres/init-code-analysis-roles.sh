#!/bin/sh
set -eu

# Runs automatically only when PostgreSQL initializes a new data directory.
# It is also safe to execute manually for an existing development volume.
: "${CODE_ANALYSIS_API_DB_PASSWORD:=bizinsight-code-api-dev}"
: "${CODE_ANALYSIS_WORKER_DB_PASSWORD:=bizinsight-code-worker-dev}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=api_password="$CODE_ANALYSIS_API_DB_PASSWORD" \
  --set=worker_password="$CODE_ANALYSIS_WORKER_DB_PASSWORD" <<-'EOSQL'
SELECT 'CREATE ROLE bizinsight_code_api LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bizinsight_code_api') \gexec
SELECT 'CREATE ROLE bizinsight_code_worker LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT BYPASSRLS'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bizinsight_code_worker') \gexec

ALTER ROLE bizinsight_code_api
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS
  PASSWORD :'api_password';
ALTER ROLE bizinsight_code_worker
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT BYPASSRLS
  PASSWORD :'worker_password';
EOSQL
