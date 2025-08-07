 [INFO] 🔄 Transitioning from Phase 1 to Phase 2
ℹ️ [INFO] 🔄 Phase transition: 1 -> 2
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🔄 Phase transition: 1 -> 2
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🧹 Reset all state for Phase 2 (fresh start)
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🧹 Phase 2 state reset - starting fresh
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - ✅ Transitioned to Phase 2
ℹ️ [INFO] ✅ Phase 2 state reset completed
ℹ️ [INFO] 🔄 Scheduler, optimizer, and best checkpoint states are fresh
ℹ️ [INFO] 🔧 Re-initializing model for Phase 2 and loading Phase 1 checkpoint...
[2025-08-07 10:26:39] INFO - smartcash.model.core.weight_transfer_manager - 🔧 Re-initializing model for Phase 2...
[2025-08-07 10:26:39] INFO - smartcash.model.training.utils.progress_tracker - 🚀 Starting: build_model (4 steps)
[2025-08-07 10:26:39] INFO - model.builder - Multi-layer config detected: {'layer_1': 7, 'layer_2': 7, 'layer_3': 3} -> total classes: 17
[2025-08-07 10:26:39] INFO - model.builder - Final nc value for YOLOv5: 17
[2025-08-07 10:26:39] INFO - model.builder - Config keys: ['nc', 'depth_multiple', 'width_multiple', 'anchors', 'backbone', 'head', 'num_classes', 'img_size', 'pretrained', 'ch', 'multi_layer_nc', 'activation', 'channel_multiple']
[2025-08-07 10:26:39] INFO - model.builder - Backbone layers: 10
[2025-08-07 10:26:39] INFO - model.builder - Head layers: 15
[2025-08-07 10:26:39] INFO - model.builder - Using standard YOLOModel with multi-layer workaround
[2025-08-07 10:26:39] INFO - model.builder - Stored multi-layer config on model: 17
[2025-08-07 10:26:39] INFO - model.builder - 🔄 Created training compatibility wrapper
[2025-08-07 10:26:39] INFO - smartcash.model.training.utils.progress_tracker - ✅ build_model completed: YOLOv5 model built successfully (cspdarknet)
[2025-08-07 10:26:39] INFO - smartcash.model.training.utils.progress_tracker - ✅  completed: Model build complete
[2025-08-07 10:26:39] INFO - model.api - ✅ Model built successfully: 9,334,022 parameters
[2025-08-07 10:26:39] INFO - smartcash.model.core.weight_transfer_manager - 📂 Loading Phase 1 checkpoint: data/checkpoints/cspdarknet/best_cspdarknet_20250807.pt
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - ✅ Restored BestMetricsManager phase-specific state from checkpoint
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 📊 Restored phase best metrics: [1]
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🔄 BestMetricsManager resuming status set to True
[2025-08-07 10:26:39] INFO - model.checkpoint - ✅ Checkpoint loaded successfully from epoch 1, phase 1
[2025-08-07 10:26:39] INFO - model.checkpoint - 📊 Loaded metrics - Accuracy: 0.8192, Loss: 4.9996
[2025-08-07 10:26:39] INFO - model.api - ✅ Checkpoint loaded: data/checkpoints/cspdarknet/best_cspdarknet_20250807.pt
[2025-08-07 10:26:39] INFO - smartcash.model.core.weight_transfer_manager - ✅ Phase 1 checkpoint loaded successfully into Phase 2 model
[2025-08-07 10:26:39] INFO - smartcash.model.core.weight_transfer_manager - 🔓 All model parameters already unfrozen for Phase 2
ℹ️ [INFO] ✅ Model successfully rebuilt for Phase 2
ℹ️ [INFO] 🎯 Starting Phase 2: Fine-tuning training
[2025-08-07 10:26:39] INFO - smartcash.model.config.model_config_manager - 🔧 Configuring model for Phase 2
[2025-08-07 10:26:39] INFO - smartcash.model.config.model_config_manager - ✅ Model configured for Phase 2
[2025-08-07 10:26:39] INFO - smartcash.model.training.phases.phase_setup_manager - 🔧 Setting up Phase 2 training components
[2025-08-07 10:26:39] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🧹 Reset all state for Phase 2 (fresh start)
[2025-08-07 10:26:39] INFO - smartcash.model.training.phases.phase_setup_manager - 🧹 Reset Phase 2 metrics state for fresh start