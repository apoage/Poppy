#!/usr/bin/env python3
"""
Enhanced Log Parser - Handles all trading pair formats (with/without slash)
Usage: python log_parser.py <log_file_path>
"""

import re
import sys
from collections import defaultdict
from datetime import datetime

# Color codes
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'

def parse_log_line(line):
    """Parse log line into components."""
    pattern = r'(?:\[(.*?)\])?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(.*)'
    match = re.match(pattern, line.strip())
    
    if match:
        component = match.group(1) or "UNKNOWN"
        timestamp_str = match.group(2)
        message = match.group(3)
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        
        return {
            'component': component,
            'timestamp': timestamp,
            'timestamp_str': timestamp_str,
            'message': message,
            'raw': line.strip()
        }
    return None

# Sequence detection
def is_sequence_start(message):
    """Detect sequence start markers."""
    return any(marker in message for marker in [
        "Saving checkpoint at step:",
        "[MANAGER] - ✅ CUDA libs configured"
    ])

def is_sequence_end(message, sequence_type):
    """Detect sequence end based on type."""
    if sequence_type == "checkpoint" and "Removing checkpoint at" in message:
        return True
    if sequence_type == "startup" and "Unable to initialize backend 'tpu'" in message:
        return True
    return False

def get_sequence_type(start_message):
    """Determine sequence type."""
    if "Saving checkpoint at step:" in start_message:
        return "checkpoint"
    if "[MANAGER] - ✅ CUDA libs configured" in start_message:
        return "startup"
    return None

def normalize_message(message):
    """Normalize messages by replacing variables with placeholders."""
    
    # Capping buy orders - ULTRA ROBUST PATTERN for all formats
    if "Capping buy order for" in message and "due to insufficient cash" in message:
        # Match ANY symbol format: ATOM/USD, ETH/USDT, DOTGBP, /USDTCAD, etc.
        symbol_match = re.search(r'Capping buy order for ([A-Z0-9/]+) due', message)
        if symbol_match:
            symbol = symbol_match.group(1)
            # Clean up symbols that start/end with slash for consistency
            symbol = symbol.strip('/')
            return f"Capping buy order for {symbol} due to insufficient cash. Desired: $<AMOUNT>, Available: $<AMOUNT>"
    
    # FX rate fallback
    if "Using fallback FX rate for" in message:
        pair_match = re.search(r'Using fallback FX rate for ([A-Z/]+)', message)
        if pair_match:
            pair = pair_match.group(1)
            return f"Using fallback FX rate for {pair}: <RATE> (inaccurate for historical data)"
    
    # Checkpoint sequence
    if any(marker in message for marker in [
        "Saving checkpoint at step:",
        "Using Orbax as backend",
        "Created BasePyTreeCheckpointHandler",
        "Started saving checkpoint",
        "Scheduling D2H of",
        "Initiated.*serialize",
        "/jax/checkpoint/write/",
        "Wrote.*array_metadata",
        "Finished saving checkpoint",
        "Removing checkpoint at"
    ]):
        step_match = re.search(r'Saving checkpoint at step: (\d+)', message)
        if step_match:
            return f"Checkpoint save sequence (step {step_match.group(1)})"
        return "Checkpoint save sequence"
    
    # Startup sequence
    if any(marker in message for marker in [
        "[MANAGER] - ✅ CUDA libs configured",
        "Filtering data:",
        "Excluded.*forex pairs",
        "Selected.*random assets",
        "Config written to",
        "Loaded.*experiments",
        "Running.*experiment",
        "Loading.*specific assets",
        "Building valid timestamp index",
        "Market loaded:",
        "Unable to initialize backend 'tpu'"
    ]):
        return "Startup initialization sequence"
    
    # Existing patterns
    if "No more valid timestamps after" in message and "Ending episode" in message:
        return "No more valid timestamps after <TIMESTAMP>. Ending episode."
    
    if "Query returned 0 rows at timestamp" in message:
        return "Query returned 0 rows at timestamp <TIMESTAMP>. First 5 asset IDs: <ASSET_LIST>"
    
    return message

def extract_sequence_stats(lines, sequence_type):
    """Extract relevant stats from a sequence."""
    stats = {}
    
    if sequence_type == "checkpoint":
        for line in lines:
            match = re.search(r'Saving checkpoint at step: (\d+)', line['message'])
            if match:
                stats['step'] = match.group(1)
                break
        
        for line in lines:
            match = re.search(r'Scheduling D2H of (\d+) prioritized', line['message'])
            if match:
                stats['arrays'] = match.group(1)
                break
        
        for line in lines:
            match = re.search(r'Finished synchronous save in ([\d.]+) seconds', line['message'])
            if match:
                stats['duration'] = match.group(1)
                break
    
    elif sequence_type == "startup":
        for line in lines:
            match = re.search(r'Loading (\d+) specific assets', line['message'])
            if match:
                stats['asset_count'] = match.group(1)
                break
        
        for line in lines:
            match = re.search(r'Market loaded: (\d+) assets, ([\d,]+) valid timestamps', line['message'])
            if match:
                stats['market_assets'] = match.group(1)
                stats['market_timestamps'] = match.group(2)
                break
    
    return stats

def parse_log_file(log_file_path):
    """Parse entire log file with sequence detection."""
    entries = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    current_sequence = None
    sequence_lines = []
    sequence_type = None
    
    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        
        message = parsed['message']
        
        if current_sequence is None and is_sequence_start(message):
            current_sequence = parsed
            sequence_lines = [parsed]
            sequence_type = get_sequence_type(message)
        elif current_sequence is not None:
            sequence_lines.append(parsed)
            
            if is_sequence_end(message, sequence_type):
                normalized = normalize_message(message)
                
                entries.append({
                    'type': 'sequence',
                    'normalized': normalized,
                    'start_time': current_sequence['timestamp'],
                    'start_time_str': current_sequence['timestamp_str'],
                    'end_time': parsed['timestamp'],
                    'end_time_str': parsed['timestamp_str'],
                    'lines': sequence_lines,
                    'sequence_type': sequence_type,
                    'stats': extract_sequence_stats(sequence_lines, sequence_type)
                })
                current_sequence = None
                sequence_lines = []
                sequence_type = None
        else:
            normalized = normalize_message(message)
            entries.append({
                'type': 'single',
                'normalized': normalized,
                'data': parsed
            })
    
    # Handle unfinished sequence
    if current_sequence is not None:
        entries.append({
            'type': 'single',
            'normalized': normalize_message(current_sequence['message']),
            'data': current_sequence
        })
        for line in sequence_lines[1:]:
            entries.append({
                'type': 'single',
                'normalized': normalize_message(line['message']),
                'data': line
            })
    
    return entries

def generate_summary(log_file_path):
    """Generate comprehensive summary report."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎯 ENHANCED LOG ANALYZER: {log_file_path}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
    
    entries = parse_log_file(log_file_path)
    
    # Group by normalized message
    problem_groups = defaultdict(list)
    for entry in entries:
        key = entry['normalized']
        problem_groups[key].append(entry)
    
    # Sort by frequency
    sorted_problems = sorted(problem_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"{Colors.BOLD}{Colors.MAGENTA}📊 UNIQUE PROBLEMS & EVENTS:{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'─' * 80}{Colors.RESET}\n")
    
    for idx, (normalized_msg, occurrences) in enumerate(sorted_problems, 1):
        count = len(occurrences)
        
        # Determine severity
        if any(keyword in normalized_msg.lower() for keyword in ['error', 'no more valid', 'query returned 0 rows']):
            severity = f"{Colors.RED}⚠️  ERROR{Colors.RESET}"
        elif "Capping buy order" in normalized_msg or "fallback FX rate" in normalized_msg:
            severity = f"{Colors.YELLOW}⚡ WARN{Colors.RESET}"
        elif "checkpoint" in normalized_msg.lower():
            severity = f"{Colors.BLUE}💾 CHECKPOINT{Colors.RESET}"
        elif "startup" in normalized_msg.lower():
            severity = f"{Colors.GREEN}🚀 STARTUP{Colors.RESET}"
        else:
            severity = f"{Colors.GRAY}ℹ️  INFO{Colors.RESET}"
        
        print(f"{Colors.BOLD}{Colors.CYAN}{idx:2d}. {normalized_msg}{Colors.RESET}")
        print(f"    {severity}")
        print(f"    Occurrences: {Colors.BOLD}{count}{Colors.RESET}")
        
        # Show details
        first_entry = occurrences[0]
        if first_entry['type'] == 'single':
            if count > 1:
                print(f"    First: {first_entry['data']['timestamp_str']}")
                print(f"    Last:  {occurrences[-1]['data']['timestamp_str']}")
                time_diff = occurrences[-1]['data']['timestamp'] - first_entry['data']['timestamp']
                print(f"    Span:  {time_diff}")
            else:
                print(f"    Time:  {first_entry['data']['timestamp_str']}")
        else:  # Sequence
            print(f"    Blocks: {count} sequence(s)")
            duration = first_entry['end_time'] - first_entry['start_time']
            print(f"    Duration: {duration}")
            
            # Show stats
            if first_entry['stats']:
                stats = first_entry['stats']
                if 'step' in stats:
                    print(f"    Step: {stats['step']}")
                if 'arrays' in stats:
                    print(f"    Arrays: {stats['arrays']}")
                if 'duration' in stats:
                    print(f"    Save time: {stats['duration']}s")
                if 'asset_count' in stats:
                    print(f"    Assets: {stats['asset_count']}")
        
        print()
    
    # Analysis by category
    print(f"\n{Colors.BOLD}{Colors.YELLOW}🚨 DETAILED ANALYSIS:{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * 80}{Colors.RESET}\n")
    
    # Trading pair-specific analysis
    capping_issues = [(k, v) for k, v in problem_groups.items() if "Capping buy order" in k]
    if capping_issues:
        print(f"{Colors.YELLOW}💰 INSUFFICIENT CASH WARNINGS BY PAIR:{Colors.RESET}")
        for issue, occurrences in sorted(capping_issues, key=lambda x: len(x[1]), reverse=True):
            symbol_match = re.search(r'for ([A-Z0-9/]+) due', issue)
            symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
            count = len(occurrences)
            print(f"  • {symbol:15s} {count:3d} occurrence(s)")
        print()
    
    # FX issues
    fx_issues = [(k, v) for k, v in problem_groups.items() if "fallback FX rate" in k]
    if fx_issues:
        print(f"{Colors.YELLOW}💱 FX FALLBACK WARNINGS:{Colors.RESET}")
        for issue, occurrences in fx_issues:
            pair_match = re.search(r'for ([A-Z/]+):', issue)
            pair = pair_match.group(1) if pair_match else "UNKNOWN"
            print(f"  • {pair:15s} {len(occurrences):3d} occurrence(s)")
        print()
    
    # Query issues
    query_issues = [(k, v) for k, v in problem_groups.items() if "Query returned 0 rows" in k]
    if query_issues:
        print(f"{Colors.YELLOW}❌ QUERY FAILURES:{Colors.RESET}")
        for issue, occurrences in query_issues:
            print(f"  • {len(occurrences):3d} failures (timestamp: <TIMESTAMP>)")
        print()
    
    # Episode issues
    episode_issues = [(k, v) for k, v in problem_groups.items() if "No more valid timestamps" in k]
    if episode_issues:
        print(f"{Colors.YELLOW}🛑 EPISODE TERMINATIONS:{Colors.RESET}")
        for issue, occurrences in episode_issues:
            print(f"  • {len(occurrences):3d} termination(s)")
        print()
    
    # Statistics
    print(f"\n{Colors.BOLD}{Colors.GREEN}📈 STATISTICS:{Colors.RESET}")
    print(f"{Colors.GREEN}{'─' * 80}{Colors.RESET}\n")
    
    error_count = sum(len(v) for k, v in problem_groups.items() 
                      if any(kw in k.lower() for kw in ['error', 'no more valid', 'query returned 0 rows']))
    warn_count = sum(len(v) for k, v in problem_groups.items() 
                     if any(kw in k.lower() for kw in ['capping', 'fallback']))
    
    print(f"Total log lines: {sum(len(v) for v in problem_groups.values())}")
    print(f"Unique event types: {len(problem_groups)}")
    print(f"Error events: {error_count}")
    print(f"Warning events: {warn_count}")
    print(f"Checkpoint sequences: {len([k for k in problem_groups.keys() if 'checkpoint' in k.lower()])}")
    print(f"Startup sequences: {len([k for k in problem_groups.keys() if 'startup' in k.lower()])}")
    
    return problem_groups

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Error: Please provide log file path.{Colors.RESET}")
        sys.exit(1)
    
    generate_summary(sys.argv[1])
