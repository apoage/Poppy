#!/usr/bin/env python3
"""
Training Profiler - Identify CPU/GPU/RAM bottlenecks
Run this alongside training to find performance issues
"""

import time
import psutil
import torch
import subprocess
import sys
from datetime import datetime
from collections import defaultdict

class TrainingProfiler:
    """Profile training process in real-time"""

    def __init__(self, target_pid=None):
        self.target_pid = target_pid
        self.process = psutil.Process(target_pid) if target_pid else None
        self.samples = defaultdict(list)

    def get_gpu_stats(self):
        """Get GPU utilization and memory"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, gpu_util, mem_util, mem_used, mem_total = line.split(', ')
                    gpus.append({
                        'id': int(idx),
                        'gpu_util': float(gpu_util),
                        'mem_util': float(mem_util),
                        'mem_used_mb': float(mem_used),
                        'mem_total_mb': float(mem_total)
                    })
            return gpus
        except Exception as e:
            return []

    def get_cpu_stats(self):
        """Get CPU utilization per core"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        return {
            'per_core': cpu_percent,
            'total': sum(cpu_percent) / len(cpu_percent),
            'active_cores': sum(1 for c in cpu_percent if c > 10.0)
        }

    def get_process_tree_stats(self):
        """Get stats for main process and all children"""
        if not self.process:
            return None

        try:
            children = self.process.children(recursive=True)

            # Main process stats
            main_stats = {
                'cpu_percent': self.process.cpu_percent(),
                'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'num_threads': self.process.num_threads(),
            }

            # Children stats
            child_stats = []
            for child in children:
                try:
                    child_stats.append({
                        'pid': child.pid,
                        'cpu_percent': child.cpu_percent(),
                        'memory_mb': child.memory_info().rss / 1024 / 1024,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return {
                'main': main_stats,
                'children': child_stats,
                'num_children': len(child_stats)
            }
        except Exception as e:
            return None

    def get_ram_stats(self):
        """Get system RAM usage"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'ram_used_gb': mem.used / 1024**3,
            'ram_total_gb': mem.total / 1024**3,
            'ram_percent': mem.percent,
            'swap_used_gb': swap.used / 1024**3,
            'swap_percent': swap.percent
        }

    def sample(self):
        """Take one sample of all metrics"""
        timestamp = datetime.now()

        sample = {
            'time': timestamp,
            'cpu': self.get_cpu_stats(),
            'ram': self.get_ram_stats(),
            'gpu': self.get_gpu_stats(),
        }

        if self.process:
            sample['process'] = self.get_process_tree_stats()

        return sample

    def analyze_sample(self, sample):
        """Analyze sample and return bottleneck diagnosis"""
        issues = []

        # CPU analysis
        cpu = sample['cpu']
        if cpu['active_cores'] < 4:
            issues.append(f"⚠️  LOW CPU: Only {cpu['active_cores']}/16 cores active")

        # GPU analysis
        for gpu in sample['gpu']:
            if gpu['gpu_util'] < 50:
                issues.append(f"⚠️  GPU {gpu['id']}: {gpu['gpu_util']:.1f}% util (LOW)")
            if gpu['mem_used_mb'] < 4000:
                issues.append(f"⚠️  GPU {gpu['id']}: {gpu['mem_used_mb']:.0f}MB VRAM (should be 12-18GB)")

        # RAM analysis
        ram = sample['ram']
        if ram['swap_percent'] > 10:
            issues.append(f"⚠️  SWAP: {ram['swap_percent']:.1f}% used (RAM overflow!)")

        # Process analysis
        if 'process' in sample and sample['process']:
            proc = sample['process']
            if proc['main']['cpu_percent'] > 90:
                issues.append(f"⚠️  MAIN PROCESS: {proc['main']['cpu_percent']:.1f}% CPU (single-thread bottleneck)")

        return issues

    def print_sample(self, sample, issues):
        """Pretty print sample"""
        print("\n" + "="*80)
        print(f"[{sample['time'].strftime('%H:%M:%S')}] TRAINING PROFILER")
        print("="*80)

        # CPU
        cpu = sample['cpu']
        print(f"\n📊 CPU: {cpu['active_cores']}/16 cores active | Avg: {cpu['total']:.1f}%")
        print(f"   Per-core: {' '.join([f'{c:4.0f}' for c in cpu['per_core']])}")

        # RAM
        ram = sample['ram']
        print(f"\n💾 RAM: {ram['ram_used_gb']:.1f}/{ram['ram_total_gb']:.1f} GB ({ram['ram_percent']:.1f}%)")
        if ram['swap_used_gb'] > 0:
            print(f"   SWAP: {ram['swap_used_gb']:.2f} GB ({ram['swap_percent']:.1f}%)")

        # GPU
        print(f"\n🎮 GPU:")
        for gpu in sample['gpu']:
            print(f"   GPU {gpu['id']}: {gpu['gpu_util']:5.1f}% util | "
                  f"{gpu['mem_used_mb']:6.0f}/{gpu['mem_total_mb']:6.0f} MB "
                  f"({gpu['mem_used_mb']/gpu['mem_total_mb']*100:.1f}%)")

        # Process tree
        if 'process' in sample and sample['process']:
            proc = sample['process']
            print(f"\n🔧 Process Tree:")
            print(f"   Main: {proc['main']['cpu_percent']:6.1f}% CPU | "
                  f"{proc['main']['memory_mb']:8.0f} MB | "
                  f"{proc['main']['num_threads']} threads")
            print(f"   Children: {proc['num_children']} processes")

            # Show top 5 CPU-heavy children
            if proc['children']:
                top_children = sorted(proc['children'],
                                    key=lambda x: x['cpu_percent'],
                                    reverse=True)[:5]
                for child in top_children:
                    print(f"      PID {child['pid']}: {child['cpu_percent']:5.1f}% CPU | "
                          f"{child['memory_mb']:7.0f} MB")

        # Issues
        if issues:
            print(f"\n⚠️  BOTTLENECKS DETECTED:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"\n✓ No major bottlenecks detected")

        print("="*80)

    def run(self, interval=5, duration=None):
        """Run profiler continuously"""
        print(f"🔍 Training Profiler Started")
        if self.target_pid:
            print(f"   Tracking PID: {self.target_pid}")
        print(f"   Sample interval: {interval}s")
        if duration:
            print(f"   Duration: {duration}s")

        start_time = time.time()
        sample_count = 0

        try:
            while True:
                sample = self.sample()
                issues = self.analyze_sample(sample)
                self.print_sample(sample, issues)

                # Save sample
                self.samples['all'].append(sample)
                sample_count += 1

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Profiler stopped by user")

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print summary statistics"""
        if not self.samples['all']:
            return

        print("\n" + "="*80)
        print("📈 PROFILING SUMMARY")
        print("="*80)

        samples = self.samples['all']

        # CPU summary
        avg_cpu = sum(s['cpu']['total'] for s in samples) / len(samples)
        avg_active_cores = sum(s['cpu']['active_cores'] for s in samples) / len(samples)
        print(f"\nCPU Average: {avg_cpu:.1f}% | Active cores: {avg_active_cores:.1f}/16")

        # GPU summary
        if samples[0]['gpu']:
            for gpu_id in range(len(samples[0]['gpu'])):
                avg_gpu_util = sum(s['gpu'][gpu_id]['gpu_util'] for s in samples) / len(samples)
                avg_gpu_mem = sum(s['gpu'][gpu_id]['mem_used_mb'] for s in samples) / len(samples)
                print(f"GPU {gpu_id} Average: {avg_gpu_util:.1f}% util | {avg_gpu_mem:.0f} MB VRAM")

        # RAM summary
        avg_ram = sum(s['ram']['ram_used_gb'] for s in samples) / len(samples)
        max_ram = max(s['ram']['ram_used_gb'] for s in samples)
        print(f"RAM: Avg {avg_ram:.1f} GB | Peak {max_ram:.1f} GB")

        # Bottleneck diagnosis
        print("\n🔍 BOTTLENECK DIAGNOSIS:")

        if avg_active_cores < 4:
            print("   ❌ CPU UNDERUTILIZED - Only using {:.1f} cores out of 16".format(avg_active_cores))
            print("      → Increase collector_env_num or enable data parallelism")

        if samples[0]['gpu']:
            for gpu_id in range(len(samples[0]['gpu'])):
                avg_gpu_util = sum(s['gpu'][gpu_id]['gpu_util'] for s in samples) / len(samples)
                avg_gpu_mem = sum(s['gpu'][gpu_id]['mem_used_mb'] for s in samples) / len(samples)

                if avg_gpu_util < 50:
                    print(f"   ❌ GPU {gpu_id} UNDERUTILIZED - {avg_gpu_util:.1f}% avg")
                    print("      → Enable GPU MCTS (mcts_ctree=True) or increase batch_size")

                if avg_gpu_mem < 4000:
                    print(f"   ❌ GPU {gpu_id} VRAM UNDERUTILIZED - {avg_gpu_mem:.0f}MB / 24GB")
                    print("      → Increase batch_size to 3072+ to use available VRAM")

        if 'process' in samples[0] and samples[0]['process']:
            avg_main_cpu = sum(s['process']['main']['cpu_percent'] for s in samples
                             if s['process']) / len(samples)
            if avg_main_cpu > 90:
                print(f"   ❌ MAIN PROCESS BOTTLENECK - {avg_main_cpu:.1f}% CPU (single-thread)")
                print("      → Offload work to GPU or use multiprocessing")

        print("="*80)


def find_training_process():
    """Find the main training process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'main.py' in ' '.join(cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


if __name__ == "__main__":
    # Find training process
    pid = find_training_process()

    if not pid:
        print("⚠️  No training process found (main.py)")
        print("   Run this WHILE training is running, or specify PID manually:")
        print("   python profile_training.py <PID>")
        sys.exit(1)

    if len(sys.argv) > 1:
        pid = int(sys.argv[1])

    print(f"✓ Found training process: PID {pid}")

    # Create profiler
    profiler = TrainingProfiler(target_pid=pid)

    # Run for 5 minutes by default
    profiler.run(interval=5, duration=300)
