INFO - smartcash.model.training.training_phase_manager - Starting validation epoch 2 with 43 batches
INFO - smartcash.model.training.training_phase_manager - Processing validation batch 1/43, images: torch.Size([8, 3, 640, 640]), targets: torch.Size([53, 6])                                         
INFO - smartcash.model.training.training_phase_manager - Model predictions type: <class 'tuple'>, structure: tuple
INFO - smartcash.model.training.training_phase_manager -   Predictions list length: 2
INFO - smartcash.model.training.training_phase_manager -   First prediction type: <class 'torch.Tensor'>, shape: torch.Size([8, 25200, 12])
INFO - smartcash.model.training.training_phase_manager - First batch loss: 2.0903, loss_breakdown keys: ['val_loss', 'val_map50', 'val_map50_95', 'val_precision', 'val_recall', 'val_f1', 'val_accuracy', 'num_targets']
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: layer_preds type: <class 'tuple'>, is_list_or_tuple: True, len: 2
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: Processing tuple format with 2 scales
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: layer_preds type <class 'tuple'>, length 2
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: scale_pred type <class 'torch.Tensor'>, shape torch.Size([8, 25200, 12])
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: Using 3D format [batch=8, detections=25200, features=12]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0: Found 201600 detections above threshold 0.1
INFO - smartcash.model.training.training_phase_manager -   Max objectness: 0.5129, Max class prob: 0.7310, Max final score: 0.3684
INFO - smartcash.model.training.training_phase_manager -   Prediction tensor shape: torch.Size([8, 25200, 12]), Confidence threshold: 0.1
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 0: Adding 1000 predictions (max 1000) and 12 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3532 - 0.3682
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 1: Adding 1000 predictions (max 1000) and 1 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3454 - 0.3657
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 2: Adding 1000 predictions (max 1000) and 7 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3573 - 0.3660
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 15, 16]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 3: Adding 1000 predictions (max 1000) and 10 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3554 - 0.3679
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 4, 5]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15, 16]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 4: Adding 1000 predictions (max 1000) and 6 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3578 - 0.3669
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 5]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 15, 16]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 5: Adding 1000 predictions (max 1000) and 12 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3485 - 0.3681
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 3, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 6: Adding 1000 predictions (max 1000) and 4 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3616 - 0.3684
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 15]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 0, img 7: Adding 1000 predictions (max 1000) and 1 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3567 - 0.3667
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0]
INFO - smartcash.model.training.training_phase_manager - Layer layer_1: prediction shape torch.Size([8, 7]), target shape torch.Size([8])
🚀 Overall Training - Val Batch 1/43:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████▍                                | 74.99999999999999/100 [34:47<11:35, 27.84s/%][2025-07-28 11:06:20] INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1: layer_preds type <class 'tuple'>, length 2
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1: scale_pred type <class 'torch.Tensor'>, shape torch.Size([8, 25200, 12])
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1: Using 3D format [batch=8, detections=25200, features=12]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1: Found 201600 detections above threshold 0.1
INFO - smartcash.model.training.training_phase_manager -   Max objectness: 0.5205, Max class prob: 0.7310, Max final score: 0.3792
INFO - smartcash.model.training.training_phase_manager -   Prediction tensor shape: torch.Size([8, 25200, 12]), Confidence threshold: 0.1
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 0: Adding 1000 predictions (max 1000) and 1 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3589 - 0.3731
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 1: Adding 1000 predictions (max 1000) and 1 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3505 - 0.3667
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 2: Adding 1000 predictions (max 1000) and 6 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3564 - 0.3673
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 4, 5]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 15, 16]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 3: Adding 1000 predictions (max 1000) and 1 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3649 - 0.3716
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 4: Adding 1000 predictions (max 1000) and 13 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3464 - 0.3792
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15, 16]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 5: Adding 1000 predictions (max 1000) and 12 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3590 - 0.3696
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 6: Adding 1000 predictions (max 1000) and 12 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3417 - 0.3673
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 14, 15]
INFO - smartcash.model.training.training_phase_manager - mAP Debug - batch 1, img 7: Adding 1000 predictions (max 1000) and 9 targets to AP calculator
INFO - smartcash.model.training.training_phase_manager -   Pred scores range: 0.3611 - 0.3683
INFO - smartcash.model.training.training_phase_manager -   Pred classes: [0, 1, 2, 4, 5, 6]
INFO - smartcash.model.training.training_phase_manager -   GT classes: [0, 7, 15]
🚀 Overall Training - Val Batch 11/43:  75%|████████████████████████████████████| 74.99999999999999/100 [36:09<12:03, 28.93s/%]
INFO - smartcash.model.training.training_phase_manager - Processing validation batch 21/43, images: torch.Size([8, 3, 640, 640]), targets: torch.Size([38, 6])
🚀 Overall Training - Val Batch 21/43:  75%|████████████████████████████████████| 74.99999999999999/100 [37:31<12:30, 30.02s/%]
INFO - smartcash.model.training.training_phase_manager - Processing validation batch 31/43, images: torch.Size([8, 3, 640, 640]), targets: torch.Size([35, 6])
🚀 Overall Training - Val Batch 31/43:  75%|████████████████████████████████████| 74.99999999999999/100 [38:53<12:57, 31.12s/%]
INFO - smartcash.model.training.training_phase_manager - Processing validation batch 41/43, images: torch.Size([8, 3, 640, 640]), targets: torch.Size([53, 6])
🚀 Overall Training - Val Batch 43/43:  75%|██████████████████████████████████| 74.99999999999999/100 [40:30<13:30, 32.41s/%]
INFO - smartcash.model.training.training_phase_manager - Validation completed: 43 batches processed, running_val_loss: 85.28861725330353
INFO - smartcash.model.training.training_phase_manager - Collected predictions: ['layer_1']
INFO - smartcash.model.training.training_phase_manager - Collected targets: ['layer_1']
INFO - smartcash.model.training.training_phase_manager - mAP Calculator Data: 343000 predictions, 1769 targets
INFO - smartcash.model.training.training_phase_manager - ✅ Computed mAP metrics: mAP@0.5=0.0007, mAP@0.5:0.95=0.0002
INFO - smartcash.model.training.training_phase_manager - Computed validation metrics: {'layer_1_accuracy': 0.029154518950437316, 'layer_1_precision': 0.0008499859752314086, 'layer_1_recall': 0.029154518950437316, 'layer_1_f1': 0.0016518141048406413}
📊 METRICS [TRAINING_PHASE_2] Epoch 2:
    train_loss: 🟢 0.2563 (excellent)
    val_loss: 🔵 1.9835 (good)
    val_map50: 🔴 0.0000 (critical)
    val_map50_95: 🔴 0.0000 (critical)
    val_precision: 🔴 0.0008 (critical)
    val_recall: 🔴 0.0292 (critical)
    val_f1: 🔴 0.0017 (critical)
    val_accuracy: 🔴 0.0292 (critical)
    layer_1_accuracy: 🔴 0.2500 (critical)
    layer_1_precision: 🔴 0.1875 (critical)
    layer_1_recall: 🔴 0.2500 (critical)
    layer_1_f1: 🔴 0.2083 (critical)
    layer_2_accuracy: 🔴 0.0000 (critical)
    layer_2_precision: 🔴 0.0000 (critical)
    layer_2_recall: 🔴 0.0000 (critical)
    layer_2_f1: 🔴 0.0000 (critical)
    layer_3_accuracy: 🔴 0.0000 (critical)
    layer_3_precision: 🔴 0.0000 (critical)
    layer_3_recall: 🔴 0.0000 (critical)
    layer_3_f1: 🔴 0.0000 (critical)
    val_layer_1_accuracy: 🔴 0.0000 (critical)
    val_layer_1_precision: 🔴 0.0000 (critical)
    val_layer_1_recall: 🔴 0.0000 (critical)
    val_layer_1_f1: 🔴 0.0000 (critical)
    val_layer_2_accuracy: 🔴 0.0000 (critical)
    val_layer_2_precision: 🔴 0.0000 (critical)
    val_layer_2_recall: 🔴 0.0000 (critical)
    val_layer_2_f1: 🔴 0.0000 (critical)
    val_layer_3_accuracy: 🔴 0.0000 (critical)
    val_layer_3_precision: 🔴 0.0000 (critical)
    val_layer_3_recall: 🔴 0.0000 (critical)
    val_layer_3_f1: 🔴 0.0000 (critical)
📝 [TRAINING_CURVES] {'epoch': 2, 'train_loss': 0.2563384767176237, 'val_loss': 1.9834562151931052, 'train_accuracy': 0.25, 'val_accuracy': 0.029154518950437316, 'phase': 2}
    title: Training Progress - Phase 2
    xlabel: Epoch
    ylabel: Loss / Accuracy
📝 [LAYER_METRICS] {'epoch': 2, 'layers': {'layer_1': {'accuracy': 0.25, 'precision': 0.1875, 'recall': 0.25, 'f1': 0.20833333333333331}, 'layer_2': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, 'layer_3': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}, 'phase': 2}
    title: Layer Performance - Phase 2
    xlabel: Layer
    ylabel: Metric Value
⏳ Early stopping: val_map50 tidak improve (1/10)
📚 Epoch Progress - Epoch 2/2 completed in 589.3s - Loss: 0.2563: 100%|█████████████████████████████████████████████████████████████| 2/2 [20:08<00:00, 604.02s/epoch]
[2025-07-28 11:12:40] INFO - smartcash.model.training.utils.progress_tracker - 🚀 Starting Finalize████████████████████████████████████████████████████| 2/2 [20:08<00:00, 604.02s/epoch]
[2025-07-28 11:12:40] INFO - smartcash.model.training.utils.progress_tracker -    
🚀 Overall Training - 📊 Generating summary...:  89%|███████████████████████████████████████████████████| 88.8888888888889/100 [41:08<05:08, 27.77s/%]
INFO - smartcash.model.training.utils.progress_tracker - 🚀 Starting Summary Visualization
INFO - smartcash.model.training.utils.progress_tracker -    Generating training summary and visualizations
🚀 Overall Training - Setting up visualization manager:   0%|                                     | 0.0/100 [41:08<?, ?%/s]
INFO - smartcash.model.training.utils.progress_tracker - ❌ Summary Visualization completed in 0.0s
============================================================
✅ TRAINING COMPLETED SUCCESSFULLY
============================================================
🚀 Overall Training - Finalize:  90%|██████████  | 90/100 [41:08<04:34, 27.43s/%]
🚀 Overall Training - Setting up visualization manager:   0%|                                     | 0.0/100 [41:08<?, ?%/s]
