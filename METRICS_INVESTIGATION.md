
# Metrics Discrepancy Investigation Summary

## 1. Problem Description

During Phase 1 of model training, a significant discrepancy was observed between the `val_*` metrics and the `layer_1_*` metrics, despite the former being derived from the latter in this phase. For example, at Epoch 10, `val_accuracy` was reported as `0.8338` (fair), while `layer_1_accuracy` was `0.0454` (critical). According to the `HIERARCHICAL_VALIDATION_SUMMARY.md`, these two metrics should be identical in Phase 1.

## 2. Investigation Summary

The investigation involved analyzing the following files:

- `HIERARCHICAL_VALIDATION_SUMMARY.md`: To understand the expected behavior of the metrics system.
- `docs/MODEL_ARC.md`: To understand the model architecture.
- `smartcash/model/training/core/validation_metrics_computer.py`: To examine the primary metrics calculation logic.
- `smartcash/model/training/utils/research_metrics.py`: To examine the metrics standardization and research-focused reporting logic.

The root cause of the discrepancy lies in the interaction between `validation_metrics_computer.py` and `research_metrics.py`.

### 2.1. Key Findings

1.  **Correct Logic in `validation_metrics_computer.py`**: The `_update_with_classification_metrics` function in `validation_metrics_computer.py` correctly assigns the values from `layer_1_*` metrics to the top-level `accuracy`, `precision`, `recall`, and `f1` metrics during Phase 1.

2.  **Redundant and Conflicting Logic in `research_metrics.py`**: The `standardize_metric_names` function in `research_metrics.py` also attempts to standardize the metrics. For Phase 1, it directly assigns `raw_metrics['layer_1_accuracy']` to `standardized['val_accuracy']`.

3.  **Execution Order**: The `standardize_metric_names` function is called *after* the metrics are computed in `validation_metrics_computer.py`.

4.  **The Bug**: The `raw_metrics` dictionary passed to `standardize_metric_names` from `compute_final_metrics` in `validation_metrics_computer.py` does not contain the top-level `accuracy` key that was set in `_update_with_classification_metrics`. Instead, it still contains the original, very low `layer_1_accuracy` value. The `standardize_metric_names` function then incorrectly uses this low value to set `val_accuracy`, leading to the observed discrepancy.

## 3. Conclusion

The issue is not a calculation problem but a data flow and logic redundancy problem. The `research_metrics.py` module is overwriting the correctly calculated metrics from `validation_metrics_computer.py`. The fix noted in `TASK.md` was likely incomplete or did not fully address the underlying issue.

## 4. Recommended Solution

The recommended solution is to refactor the metric calculation and standardization process to remove the redundancy and ensure a single source of truth for the final metrics.

1.  **Centralize Metric Calculation**: The `validation_metrics_computer.py` module should be the sole source of truth for the final, phase-aware validation metrics.
2.  **Simplify `research_metrics.py`**: The `research_metrics.py` module should only be responsible for formatting and selecting metrics for logging and reporting, not for recalculating or re-interpreting them. The `standardize_metric_names` function should be simplified to just add the `val_` prefix and format the output, without applying any phase-specific logic.

This change will ensure that the metrics are calculated correctly in one place and that the reporting layer does not introduce inconsistencies.
