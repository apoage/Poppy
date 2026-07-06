# Training Dashboard

A web-based dashboard for monitoring and analyzing MCTS trading agent training in real-time.

## Features

### Metrics Tab
- **Real-time charts**: Reward, Loss components, Action distribution, Training speed, Policy entropy
- **Value calibration scatter plot**: Visualize predicted vs actual rewards with perfect calibration line
- **Summary statistics**: Current status, MA(50) reward, latest loss, entropy, speed
- **Mastery progress bar**: Track progress toward training threshold

### Analysis Tab
- **Insights**: Automatically detected positive patterns (reward improvements, good calibration)
- **Warnings**: Automatically detected issues (entropy collapse, low sell rate, poor calibration)
- **Detailed metrics**: All computed statistics in one view
- **Phase detection**: Identify exploration, improving, and declining phases

### Checkpoints Tab
- List all available model checkpoints
- View experiment name, step number, size, and modification time

### Logs Tab
- Browse log files in the logs directory
- Real-time log viewer with configurable line count
- Auto-scroll to latest entries

### System Status
- Training status indicator (TRAINING/IDLE)
- GPU utilization, memory usage, and temperature monitoring

## Installation

The server requires Flask (should already be in your environment):

```bash
pip install flask flask-cors
```

## Quick Start

### Option 1: Start script (recommended)

```bash
# From anywhere in the project
./tools/dashboard/start_dashboard.sh

# Or with custom port
./tools/dashboard/start_dashboard.sh 8080
```

### Option 2: Direct Python

```bash
# Activate virtual environment
source trader_env/bin/activate

# Run the server
python tools/dashboard/server.py --port 8050
```

Then open **http://localhost:8050** in your browser.

## API Endpoints

The server provides a REST API for programmatic access:

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | List available training CSV files |
| `GET /api/training/<path>` | Get training data from a CSV |
| `GET /api/analysis/<path>` | Get phase analysis for a training run |
| `GET /api/checkpoints` | List all checkpoints |
| `GET /api/log-files` | List available log files |
| `GET /api/logs/<path>?lines=N` | Get last N lines of a log file |
| `GET /api/status` | Get system status (GPU, training processes) |

## Configuration

### Command line options

```bash
python tools/dashboard/server.py --help

Options:
  --port PORT   Port to run on (default: 8050)
  --host HOST   Host to bind to (default: 0.0.0.0)
  --debug       Enable debug mode
```

### Dashboard settings

- **Skip epochs**: Filter out early epochs (default: 6) for cleaner visualizations
- **Auto-refresh**: Enable 10-second automatic data refresh
- **Log scale**: Toggle logarithmic scale for loss chart
- **Stacked/Normalized**: Toggle action distribution chart modes

## Directory Structure

```
tools/dashboard/
├── server.py           # Flask backend server with REST API
├── index.html          # Dashboard HTML
├── dashboard.js        # Frontend JavaScript
├── styles.css          # Styles
├── start_dashboard.sh  # Convenience start script
└── README.md           # This file
```

## Troubleshooting

### "Error loading runs - Is server running?"
Make sure the server is running:
```bash
python tools/dashboard/server.py --port 8050
```

### Charts not updating
- Check the browser console for errors
- Verify the CSV file has new data
- Try clicking "Refresh" manually

### GPU status shows "N/A"
- `nvidia-smi` must be available in PATH
- Only NVIDIA GPUs are currently supported

### Log viewer empty
- Ensure log files exist in the `logs/` directory
- Files must have `.log` or `.txt` extension

## Legacy Mode

The old method using Python's built-in HTTP server still works:
```bash
# From project root
python -m http.server 8080
# Then open http://localhost:8080/tools/dashboard/
```

However, the new Flask server provides better API endpoints and additional features like phase analysis.
