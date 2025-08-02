2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🔧 Coordinate conversion debug:
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Original targets shape: torch.Size([7, 6])████████████████████████████████████████████████████████▏     | 21/22 [00:11<00:00,  1.83batch/s, Loss=3.0665, Epoch=17]
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target sample (xywh): tensor([0.67031, 0.54844, 0.65938, 0.51562])
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Predictions sample (xyxy): tensor([0.79737, 0.33485, 0.80036, 0.43889])
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target widths: min=0.564, max=0.950, mean=0.765
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target heights: min=0.320, max=0.750, mean=0.578
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Oversized boxes (>80% image): 3/7
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction widths: min=0.000, max=0.178, mean=0.056
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction heights: min=0.000, max=0.187, mean=0.060
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Converted target sample (xyxy): tensor([0.34062, 0.29063, 1.00000, 0.80625])
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target coordinate ranges: x1=0.013-0.341, y1=0.125-0.295
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction coordinate ranges: x1=0.114-0.902, y1=0.045-0.770
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Sample IoU: 0.0009 (threshold: 0.05)
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Sample pred class: 6, target class: 6
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🔍 IoU Matrix Analysis:
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU matrix shape: torch.Size([653, 7])
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU range: 0.000000 - 0.085942
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU mean: 0.008037
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoUs > threshold (0.05): 58
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Best IoU location: (tensor(127), tensor(1))
[2025-08-02 11:36:21] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Best IoU value: 0.085942
                                                                                                                                                                                                                                                         [2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - ✅ Computed validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1']
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - ✅ Found layer_1_accuracy = 0.8746355685131195                                                                                                                           
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - ✅ Computed 4 validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1']
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - 📊 Phase 1: Using YOLOv5 built-in metrics for validation
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator - mAP computation completed:
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.0111
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.0261
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.0309
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.0283
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - YOLOv5 built-in metrics:
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor -   • Precision: 0.0261
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor -   • Recall: 0.0309
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor -   • F1: 0.0283
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor -   • mAP@0.5: 0.0111
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator - mAP computation completed:
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.0111
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.0261
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.0309
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.0283
[2025-08-02 11:36:21] INFO - smartcash.model.training.core.validation_executor - YOLOv5 mAP Results - mAP@0.5: 0.0111, Precision: 0.0261, Recall: 0.0309
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics - 📊 PHASE 1 - Validation Metrics Summary (YOLOv5 Only):
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics -    • val_accuracy: 0.874636
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics -    • val_precision: 0.873982
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics -    • val_recall: 0.874636
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics -    • val_f1: 0.872465
[2025-08-02 11:36:21] INFO - smartcash.model.training.utils.research_metrics -    • val_map50: 0.011071
📊 METRICS [TRAINING_PHASE_1] Epoch 17:
    train_loss: 🟢 0.1283 (excellent)
    val_loss: 🟡 3.0666 (fair)
    val_precision: 🔵 0.8740 (good)
    val_recall: 🔵 0.8746 (good)
    val_f1: 🔵 0.8725 (good)
    val_accuracy: 🔵 0.8746 (good)
    layer_1_accuracy: 🔵 0.9375 (good)
    layer_1_precision: 🟢 0.9688 (excellent)
    layer_1_recall: 🟢 0.9375 (excellent)
    layer_1_f1: 🟢 0.9375 (excellent)