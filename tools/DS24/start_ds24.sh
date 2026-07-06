#!/bin/bash
#
# DS24 - Enhanced Training Metrics Dashboard
# Startup script
#

set -e

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default port
PORT=${1:-8051}

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║     ██████╗ ███████╗██████╗ ██╗  ██╗                     ║"
echo "║     ██╔══██╗██╔════╝╚════██╗██║  ██║                     ║"
echo "║     ██║  ██║███████╗ █████╔╝███████║                     ║"
echo "║     ██║  ██║╚════██║██╔═══╝ ╚════██║                     ║"
echo "║     ██████╔╝███████║███████╗     ██║                     ║"
echo "║     ╚═════╝ ╚══════╝╚══════╝     ╚═╝                     ║"
echo "║                                                           ║"
echo "║     Enhanced Training Metrics Dashboard                   ║"
echo "║     TRM+MuZero Trading Agent Visualization                ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 not found${NC}"
    exit 1
fi

# Check for Flask
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing Flask...${NC}"
    pip install flask flask-cors
fi

# Check for other dependencies
python3 -c "import pandas" 2>/dev/null || pip install pandas
python3 -c "import numpy" 2>/dev/null || pip install numpy

# Change to script directory
cd "$SCRIPT_DIR"

echo -e "${GREEN}Starting DS24 Dashboard on port ${PORT}...${NC}"
echo ""
echo -e "  Local:   ${CYAN}http://localhost:${PORT}${NC}"
echo -e "  Network: ${CYAN}http://$(hostname -I | awk '{print $1}'):${PORT}${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start server
python3 server.py --port "$PORT"
