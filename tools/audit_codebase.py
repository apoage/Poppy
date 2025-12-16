#!/usr/bin/env python3
"""
Scan codebase and report what's ACTUALLY implemented.
"""

import os
import re
from pathlib import Path

def scan_features():
    """Scan codebase for implemented features"""

    features = {
        'data_infrastructure': {},
        'training_systems': {},
        'model_components': {},
        'curriculum': {},
        'optimizations': {}
    }

    # Data Infrastructure
    features['data_infrastructure']['fx_rates_table'] = os.path.exists('tools/build_fx_database.py')
    features['data_infrastructure']['ram_caching'] = check_method_exists('mcts_trader/data/history.py', '_preload_to_ram')
    features['data_infrastructure']['fx_matrix'] = check_attribute_exists('mcts_trader/data/history.py', 'fx_matrix')

    # Training Systems
    features['training_systems']['rainbow_table'] = check_file_contains('tools/run_scheduled_training.py', 'last_checkpoint')
    features['training_systems']['adaptive_curriculum'] = check_file_contains('tools/run_scheduled_training.py', 'current_fee')
    features['training_systems']['checkpoint_transfer'] = check_file_contains('tools/run_scheduled_training.py', '--load-checkpoint')

    # Model Components
    features['model_components']['trm_model'] = os.path.exists('mcts_trader/agent/trm.py')
    features['model_components']['muzero_wrapper'] = os.path.exists('mcts_trader/jax_agent/muzero_model.py')
    features['model_components']['mctx_search'] = os.path.exists('mcts_trader/jax_agent/mctx_search.py')

    # Curriculum (Synthetic)
    features['curriculum']['synthetic_market'] = os.path.exists('mcts_trader/data/synthetic_market.py')
    features['curriculum']['curriculum_env'] = os.path.exists('mcts_trader/env/curriculum_env.py')
    features['curriculum']['pass_criteria'] = os.path.exists('mcts_trader/curriculum/pass_criteria.py')

    # Optimizations
    features['optimizations']['mcts_worker_pool'] = os.path.exists('mcts_trader/jax_agent/mcts_worker_pool.py')
    features['optimizations']['fast_encoder'] = os.path.exists('mcts_trader/encoder/state_fast.py')

    return features

def check_method_exists(file_path, method_name):
    """Check if method exists in file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f'def {method_name}(' in content

def check_attribute_exists(file_path, attr_name):
    """Check if attribute exists in file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f'self.{attr_name}' in content

def check_file_contains(file_path, text):
    """Check if file contains text"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return text in content

def check_cli_argument(file_path, arg_name):
    """Check if CLI argument exists"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f"'--{arg_name}'" in content or f'"{arg_name}"' in content

def generate_report(features):
    """Generate markdown report"""

    report = ["# Codebase Audit Report", ""]
    report.append(f"**Generated**: {os.popen('date').read().strip()}")
    report.append("")
    report.append("## Implementation Status")
    report.append("")

    for category, items in features.items():
        report.append(f"### {category.replace('_', ' ').title()}")
        report.append("")

        for feature, implemented in items.items():
            status = "✅ IMPLEMENTED" if implemented else "❌ NOT FOUND"
            report.append(f"- **{feature.replace('_', ' ').title()}**: {status}")

        report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print("Scanning codebase...")
    features = scan_features()

    report = generate_report(features)

    # Save report
    with open('AUDIT_IMPLEMENTATION_STATUS.md', 'w') as f:
        f.write(report)

    print("Report saved to: AUDIT_IMPLEMENTATION_STATUS.md")
    print(report)
