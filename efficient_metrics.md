2025-08-09 04:58:09] WARNING - smartcash.model.training.core.validation_batch_processor - Validation batch 0: Loss is zero. Targets shape: torch.Size([41, 6]), Predictions keys: not dict
[2025-08-09 04:58:09] INFO - smartcash.model.training.core.validation_batch_processor - 🔍 Phase 1 - Received prediction layers: not dict
[2025-08-09 04:58:09] INFO - smartcash.model.training.core.validation_batch_processor - 🔍 Phase 1 - Expected active layers: ['layer_1']
[2025-08-09 04:58:09] WARNING - smartcash.model.training.core.validation_batch_processor - Predictions is not a dict: <class 'list'>, converting to dict with 'layer_1' key
[2025-08-09 04:58:09] WARNING - smartcash.model.training.core.validation_map_processor - Cannot handle prediction tensor dimensions: torch.Size([16, 81, 80, 80])

2025-08-09 05:05:00] INFO - smartcash.model.training.core.validation_metrics_computer - PHASE 1: Computing validation metrics.
[2025-08-09 05:05:00] WARNING - smartcash.model.training.core.ultralytics_map_calculator - No statistics accumulated for mAP calculation (0 batches processed)                                                                                               
[2025-08-09 05:05:00] INFO - smartcash.model.training.core.validation_metrics_computer - Computed validation metrics for 1 layers
[2025-08-09 05:05:00] INFO - smartcash.model.training.core.validation_metrics_computer - 📊 Phase 1: Val Loss=0.0000, Accuracy=0.1166