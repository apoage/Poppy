# Training Strategy Notes

## Massive Pre-training + Fine-tuning Approach

### Target
- **Pre-training**: 20,000+ epochs on historical data
- **Fine-tuning**: 500 epochs on recent data

### Phase 1: Pre-training (20,000 epochs)
- Full historical database
- Learn general trading patterns
- Build robust risk/position management

### Phase 2: Fine-tuning (500 epochs)
- Recent 30-90 days only
- Adapt to current market conditions

### Data Requirements
- Need more datasets if 758 timestamps is insufficient
- Run `tools/download_data.py` to expand historical coverage

### Commands
```bash
# Pre-train
python train_manager.py --auto-select volume --epochs 20000

# Fine-tune (after adding checkpoint support)
python train_jax_fast.py --checkpoint pretrained.pkl --epochs 500
```

---
2025-12-12
