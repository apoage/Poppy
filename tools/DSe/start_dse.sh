#!/bin/bash
# DSe - Easy Training Dashboard Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT=${1:-8052}

echo "=============================================="
echo "  DSe - Easy Training Dashboard"
echo "=============================================="
echo ""
echo "  Project: $PROJECT_ROOT"
echo "  Port: $PORT"
echo ""

# Check for virtual environment
if [ -d "$PROJECT_ROOT/trader_env" ]; then
    source "$PROJECT_ROOT/trader_env/bin/activate"
    echo "  Activated: trader_env"
fi

# Check dependencies
python -c "import flask" 2>/dev/null || { echo "ERROR: Flask not installed"; exit 1; }
python -c "import pandas" 2>/dev/null || { echo "ERROR: pandas not installed"; exit 1; }
python -c "import numpy" 2>/dev/null || { echo "ERROR: numpy not installed"; exit 1; }

echo ""
echo "  Starting server..."
echo "  URL: http://localhost:$PORT"
echo ""
echo "=============================================="

cd "$SCRIPT_DIR"
python server.py --port "$PORT"
