🔹 Start Phase (pretrain) -> from pretrained
🔁 Resume Phase (last.pt) -> Keeps optimizer, LR schedule, and epoch in sync
🔀 Transition to Phase 2 (best.pt from P1) -> Best weights to start new phase with fresh loss config
🔁 Resume Phase 2 (last.pt) -> Same reason — preserves training dynamics
✅ Final Evaluation (best.pt from P2) -> Use best performance model for testing or deployment