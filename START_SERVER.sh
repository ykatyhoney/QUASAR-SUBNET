#!/bin/bash
# Script to start the FastAPI Validator API Server
# Run from project root
#
# Mainnet defaults: NETUID=24, SUBTENSOR_NETWORK=finney

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Validator API"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "⚠️  No .venv found — using system Python"
fi

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env file"
else
    echo "⚠️  No .env file found. Using defaults."
fi

# Set defaults if not in .env (mainnet defaults)
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./quasar_validator.db"}
export NETUID=${NETUID:-24}
export SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-"finney"}

# Check for required dependencies
echo ""
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "fastapi not found. Installing..."
    pip install -q fastapi uvicorn[standard]
fi

if ! python -c "import sqlalchemy" 2>/dev/null; then
    echo "sqlalchemy not found. Installing..."
    pip install -q sqlalchemy
fi

# Only check psycopg2 if using PostgreSQL
if [[ "$DATABASE_URL" == postgresql* ]]; then
    if ! python -c "import psycopg2" 2>/dev/null; then
        echo "psycopg2-binary not found (needed for PostgreSQL). Installing..."
        pip install -q psycopg2-binary
    fi
fi

echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  NetUID: $NETUID"
echo "  Network: $SUBTENSOR_NETWORK"
if [[ "$DATABASE_URL" == sqlite* ]]; then
    echo "  Database: SQLite (${DATABASE_URL#sqlite:///})"
else
    echo "  Database: PostgreSQL"
fi

# Show auth config
if [ -n "$VALIDATOR_HOTKEYS" ]; then
    HOTKEY_COUNT=$(echo "$VALIDATOR_HOTKEYS" | tr ',' '\n' | grep -c '[^[:space:]]')
    echo "  Authorized Validators: $HOTKEY_COUNT hotkey(s)"
else
    echo "  ⚠️  VALIDATOR_HOTKEYS not set — validator endpoints will reject all requests"
fi

if [ -n "$API_KEYS" ]; then
    echo "  API Keys: configured"
fi
echo ""

echo "Starting FastAPI server..."
echo "Server will be available at: http://$HOST:$PORT"
echo "API docs at: http://localhost:$PORT/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""

uvicorn validator_api.app:app --host "$HOST" --port "$PORT"
