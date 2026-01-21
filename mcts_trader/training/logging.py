import os
import csv
import logging
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List

class TrainingLogger:
    """Handles CSV, console, and DS24 logging."""
    def __init__(self, config, columns):
        self.config = config
        self.csv_path = f'logs/training_{config.name}.csv'
        self.columns = columns
        self.file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)
        self.file.flush()
        
    def log_epoch(self, metrics: Dict[str, Any]):
        """Write epoch metrics to CSV and console."""
        row = [metrics.get(col, '') for col in self.columns]
        self.writer.writerow(row)
        self.file.flush()
        
        # Console output
        print(f"Epoch {metrics['epoch']:4d} | Reward: {metrics['reward']:+7.2f} | "
              f"Loss: {metrics['total_loss']:.4f} | Val: {metrics['value_loss']:.4f}")

    def close(self):
        self.file.close()
