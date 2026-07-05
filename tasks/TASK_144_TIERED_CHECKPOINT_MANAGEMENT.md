# TASK_144: Tiered Checkpoint Management

**Status:** COMPLETE

**Objective:** Implement a tiered deletion policy to keep disk usage under 10GB without losing the ability to perform long-term Darwinian rollbacks.

## 1. Retention Policy
The system should categorize checkpoints into three tiers:

| Tier | Category | Files | Policy |
|------|----------|-------|--------|
| **1** | **Permanent** | `phase1_best.pkl`, `phase1_complete.pkl`, `merged_latest.pkl` | **Never Delete** |
| **2** | **Milestones** | `epoch100`, `epoch200`, ..., or every 10th merge | **Keep 1 per 100 epochs** |
| **3** | **Rolling** | Last 5 periodic saves | **Keep last 5, delete older** |

---

## 2. Technical Implementation Logic

### A. For `train_easy.py` (Branch Checkpoints)
Logic to parse `phase1_epoch{N}.pkl` and preserve milestones.

```python
def cleanup_tiered_checkpoints(dir_path: Path, keep_recent=5, milestone_interval=100):
    """
    Tiered retention: Keep latest N + Milestones (every 100).
    Deletes intermediate periodic saves.
    """
    import re
    # 1. Find all periodic epoch files
    files = list(dir_path.glob("phase1_epoch*.pkl"))
    if not files: return

    # 2. Map filename to epoch number using regex
    epoch_map = {}
    for f in files:
        match = re.search(r'epoch(\d+)', f.name)
        if match:
            epoch_map[int(match.group(1))] = f

    if not epoch_map: return
    
    # 3. Identify epochs to keep
    sorted_epochs = sorted(epoch_map.keys())
    latest_epochs = set(sorted_epochs[-keep_recent:])
    
    to_keep = set()
    for epoch, file_path in epoch_map.items():
        if epoch in latest_epochs or epoch % milestone_interval == 0:
            to_keep.add(file_path)
            
    # 4. Delete non-retained files
    for f in files:
        if f not in to_keep:
            f.unlink()
            print(f"[CLEANUP] Deleted intermediate checkpoint: {f.name}")
```

### B. For `e-dojo_v1.py` (Merged Checkpoints)
Logic to parse `merged_{count}_{timestamp}.pkl`.

```python
def cleanup_merged_checkpoints(merged_dir: Path, keep_recent=5, milestone_interval=10):
    """
    Tiered retention for merges: Keep last 5 + every 10th merge.
    """
    files = list(merged_dir.glob("merged_*.pkl"))
    files = [f for f in files if f.name != "merged_latest.pkl"]
    
    merge_map = {}
    for f in files:
        parts = f.name.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            merge_map[int(parts[1])] = f

    if not merge_map: return
    
    sorted_counts = sorted(merge_map.keys())
    latest_counts = set(sorted_counts[-keep_recent:])
    
    for count, file_path in merge_map.items():
        if count not in latest_counts and count % milestone_interval != 0:
            file_path.unlink()
            print(f"[DOJO] Pruned old merge checkpoint: {file_path.name}")
```

---

## 3. Integration Points

1.  **`train_easy.py`**: Call `cleanup_tiered_checkpoints(output_dir)` inside the periodic save block.
2.  **`e-dojo_v1.py`**: Call `cleanup_merged_checkpoints(self.config.merged_dir)` inside `check_and_merge()` after saving a new merge.

## 4. Expected Impact
- **Disk Usage**: Plateau at ~5GB-8GB (approx 15-20 files per branch).
- **Darwinian Safety**: Milestones remain available for rolling back to any "historical basin" at 100-epoch resolution.

---

## 5. Implementation (2026-02-10)

### Files Modified
- `train_easy.py`: Added `cleanup_tiered_checkpoints()` function (lines 628-672), called after periodic saves
- `tools/orchestration/e-dojo_v1.py`: Added `cleanup_merged_checkpoints()` function (lines 55-98), called after merges

### Retention Policy Implemented
| Type | Keep Recent | Milestones | Never Delete |
|------|-------------|------------|--------------|
| Branch (`phase1_epoch*.pkl`) | Last 5 | Every 100 epochs | `phase1_best.pkl`, `phase1_complete.pkl` |
| Merged (`merged_*.pkl`) | Last 5 | Every 10th merge | `merged_latest.pkl` |
