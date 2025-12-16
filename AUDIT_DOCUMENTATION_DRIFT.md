# Documentation Audit Report

## Stale/Incorrect Claims

### GEMINI.md Pending Items
- [ ] **Verify Training Stability:** Run a longer training session (e.g., 50-100 epochs) to confirm the agent learns effectively with the new FX rates and does not revert to "Turtle" behavior.
- [ ] **Gym → Gymnasium migration**: Gym is unmaintained since 2022 and doesn't support NumPy 2.0; migration is partly done but Gym is still called in some places.
- [ ] **Orbax transformations API**: Warning about deprecated API.
- [ ] **tensorflow.io.gfile shim**: Install tensorflow or use alternative for GCS paths.
- [ ] **API abstraction layer**: Continue work on abstracting the API layer beyond Kraken.
