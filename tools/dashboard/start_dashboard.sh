#!/bin/bash
# Start the Training Dashboard Server
#
# Usage:
#   ./start_dashboard.sh          # Start on default port 8050
#   ./start_dashboard.sh 8080     # Start on custom port
#
# Then open http://localhost:8050 in your browser

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

PORT=${1:-8050}

echo "=================================================="
echo "  Training Dashboard Server"
echo "=================================================="
echo ""
echo "Starting server on port $PORT..."
echo "Dashboard URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/trader_env/bin/activate" ]; then
    source "$PROJECT_ROOT/trader_env/bin/activate"
fi

# Run the server
python "$SCRIPT_DIR/server.py" --port "$PORT"
