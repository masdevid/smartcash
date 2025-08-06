ℹ️ [INFO] ✅ Phase 1 execution completed
ℹ️ [INFO] 🔄 Transitioning from Phase 1 to Phase 2
ℹ️ [INFO] 🔄 Phase transition: 1 -> 2
[2025-08-07 03:57:51] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🔄 Phase transition: 1 -> 2
[2025-08-07 03:57:51] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🧹 Reset all state for Phase 2 (fresh start)
[2025-08-07 03:57:51] INFO - smartcash.model.core.checkpoints.best_metrics_manager - 🧹 Phase 2 state reset - starting fresh
[2025-08-07 03:57:51] INFO - smartcash.model.core.checkpoints.best_metrics_manager - ✅ Transitioned to Phase 2
ℹ️ [INFO] ✅ Phase 2 state reset completed
ℹ️ [INFO] 🔄 Scheduler, optimizer, and best checkpoint states are fresh
ℹ️ [INFO] 🎯 Starting Phase 2: Fine-tuning training
[2025-08-07 03:57:51] INFO - smartcash.model.config.model_config_manager - 🔧 Configuring model for Phase 2
[2025-08-07 03:57:51] INFO - smartcash.model.config.model_config_manager - ✅ Model configured for Phase 2
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - 🔧 Setting up Phase 2 training components
📊 Training DataLoader Configuration:
   • Batch Size: 16 (source: config)
   • Dataset Size: 6333 samples
   • Batches per Epoch: 395
   • Workers: 8, Pin Memory: True
   • Prefetch Factor: 2, Drop Last: True
   • Persistent Workers: True, Non-blocking: False
📊 Validation DataLoader Configuration:
   • Batch Size: 16 (source: config)
   • Dataset Size: 343 samples
   • Batches per Epoch: 21
[2025-08-07 03:57:51] INFO - smartcash.model.training.losses.loss_coordinator - 🔧 Phase 1: Using default YOLOv5 loss
[2025-08-07 03:57:51] INFO - smartcash.model.training.loss_manager - 🔄 LossManager initialized with new modular architecture
[2025-08-07 03:57:51] INFO - smartcash.model.training.losses.loss_coordinator - 🔧 Phase 2: Using uncertainty multi-task loss (3 layers)
[2025-08-07 03:57:51] INFO - smartcash.model.training.multi_task_loss - ✅ Uncertainty-based multi-task loss initialized for 3 layers
[2025-08-07 03:57:51] INFO - smartcash.model.training.multi_task_loss -    • layer_1: 7 classes
[2025-08-07 03:57:51] INFO - smartcash.model.training.multi_task_loss -    • layer_2: 7 classes
[2025-08-07 03:57:51] INFO - smartcash.model.training.multi_task_loss -    • layer_3: 3 classes
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - 📊 Phase 2 Learning Rate Configuration:
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Head LR (Phase 1): 0.001
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Head LR (Phase 2): 0.0001
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Backbone LR: 1e-05
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - ⚙️ Phase 2 Scheduler Configuration:
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Scheduler: cosine (CosineAnnealingLR)
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Cosine eta min: 1e-06
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • T_max (epochs): 1
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Optimizer: adamw
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Weight decay: 0.01
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - • Mixed precision: Disabled
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - Fresh training: Removing existing metrics file outputs/metrics_history_cspdarknet_data_phase2.json
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - Fresh training: Removing existing latest metrics file outputs/latest_metrics_cspdarknet.json
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - Fresh training: Initialized empty metrics history
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - MetricsHistoryRecorder initialized - Backbone: cspdarknet, Data: data, Phase: 2
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - Metrics file: outputs/metrics_history_cspdarknet_data_phase2.json
[2025-08-07 03:57:51] INFO - smartcash.model.training.utils.metrics_history - Resume mode: False
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - 🎯 Using phase-specific early stopping for Phase 2
[2025-08-07 03:57:51] INFO - smartcash.model.training.components.phase_setup_manager - ✅ Phase 2 components setup completed