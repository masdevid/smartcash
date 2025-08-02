[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🔄 BATCH UPDATE DEBUG:
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • YOLOv5 available: True███████████████████████████████████████████████████████████████████████████████      | 21/22 [00:06<00:00,  3.18batch/s, Loss=4.9957, Epoch=13]
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Predictions: torch.Size([7, 25200, 6])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Targets: torch.Size([7, 6])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🔧 Coordinate conversion debug:
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Original targets shape: torch.Size([7, 6])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target sample (xywh): tensor([0.67031, 0.54844, 0.65938, 0.51562])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Predictions sample (xyxy): tensor([0.56419, 0.40249, 0.58999, 0.41512])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target widths: min=0.564, max=0.950, mean=0.765
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target heights: min=0.320, max=0.750, mean=0.578
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Oversized boxes (>80% image): 3/7
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction widths: min=0.000, max=0.486, mean=0.053
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction heights: min=0.003, max=0.365, mean=0.048
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Converted target sample (xyxy): tensor([0.34062, 0.29063, 1.00000, 0.80625])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Target coordinate ranges: x1=0.013-0.341, y1=0.125-0.295
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Prediction coordinate ranges: x1=-0.088-0.957, y1=-0.168-0.854
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Sample IoU: 0.0010 (threshold: 0.03)
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Sample pred class: 6, target class: 6
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🔍 IoU Matrix Analysis:
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU matrix shape: torch.Size([742, 7])
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU range: 0.000000 - 0.211564
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoU mean: 0.003612
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • IoUs > threshold (0.03): 26
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Best IoU location: (tensor(361), tensor(1))
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Best IoU value: 0.211564
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 📊 STATS ACCUMULATION:
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Batch TP count: 0
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Batch predictions: 742
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Batch targets: 7
[2025-08-02 20:03:28] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total stats batches: 22
                                                                                                                                                                                                                                                            [2025-08-02 20:03:29] INFO - smartcash.model.training.core.validation_metrics_computer - ✅ Computed validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1', 'layer_2_accuracy', 'layer_2_precision', 'layer_2_recall', 'layer_2_f1', 'layer_3_accuracy', 'layer_3_precision', 'layer_3_recall', 'layer_3_f1']                                                                                                                                                                       
[2025-08-02 20:03:29] INFO - smartcash.model.training.core.validation_metrics_computer - ✅ Found layer_1_accuracy = 0.0641399416909621
[2025-08-02 20:03:29] INFO - smartcash.model.training.core.validation_metrics_computer - ✅ Computed 12 validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1', 'layer_2_accuracy', 'layer_2_precision', 'layer_2_recall', 'layer_2_f1', 'layer_3_accuracy', 'layer_3_precision', 'layer_3_recall', 'layer_3_f1']
[2025-08-02 20:03:29] INFO - smartcash.model.training.core.validation_metrics_computer - 📊 Phase 2: Using YOLOv5 built-in metrics for validation
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🎯 FINAL mAP COMPUTATION:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total accumulated batches: 22
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Stats shapes: [(21665, 1), (21665,), (21665,), (343,)]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 📈 CLASS ANALYSIS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total predictions: 21665
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total targets: 343
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total TP: 58
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Unique pred classes: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Unique target classes: [0 1 2 3 4 5 6]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Confidence range: 0.005000 - 0.314662
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 📊 AP RESULTS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP shape: (7, 1)
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP classes: [0 1 2 3 4 5 6]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP values: [          0   0.0015936   0.0020649  0.00058604           0   0.0066274  0.00012771]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Precision: [          0   0.0028944   0.0023671   0.0010262           0   0.0080168  0.00024988]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Recall: [          0    0.095238     0.18644    0.034483           0     0.62687    0.017544]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🎯 FINAL METRICS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.001571
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.002079
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.137224
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.004087
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🎯 FINAL mAP COMPUTATION:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total accumulated batches: 22
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Stats shapes: [(21665, 1), (21665,), (21665,), (343,)]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 📈 CLASS ANALYSIS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total predictions: 21665
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total targets: 343
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Total TP: 58
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Unique pred classes: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Unique target classes: [0 1 2 3 4 5 6]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Confidence range: 0.005000 - 0.314662
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 📊 AP RESULTS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP shape: (7, 1)
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP classes: [0 1 2 3 4 5 6]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • AP values: [          0   0.0015936   0.0020649  0.00058604           0   0.0066274  0.00012771]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Precision: [          0   0.0028944   0.0023671   0.0010262           0   0.0080168  0.00024988]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Recall: [          0    0.095238     0.18644    0.034483           0     0.62687    0.017544]
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator - 🎯 FINAL METRICS:
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.001571
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.002079
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.137224
[2025-08-02 20:03:29] DEBUG - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.004087
[2025-08-02 20:03:29] INFO - smartcash.model.training.utils.research_metrics - 📊 Phase 2: Val Loss=4.9958, mAP@0.5=0.0016, Accuracy=0.0641
📊 METRICS [TRAINING_PHASE_2] Epoch 13:
    train_loss: 🟢 0.5090 (excellent)
    val_loss: 🔵 4.9958 (good)
    val_precision: 🔴 0.1653 (critical)
    val_recall: 🔴 0.0641 (critical)
    val_f1: 🔴 0.0917 (critical)
    val_accuracy: 🟠 0.0641 (poor)
    layer_1_accuracy: 🟢 0.8125 (excellent)
    layer_1_precision: 🔵 0.8958 (good)
    layer_1_recall: 🔵 0.8125 (good)
    layer_1_f1: 🔵 0.8396 (good)
    layer_2_accuracy: 🟡 0.2500 (fair)
    layer_2_precision: 🔵 0.8667 (good)
    layer_2_recall: 🔴 0.2500 (critical)
    layer_2_f1: 🔴 0.1994 (critical)
    layer_3_accuracy: 🟠 0.0625 (poor)
    layer_3_precision: 🟢 1.0000 (excellent)
    layer_3_recall: 🔴 0.0625 (critical)
    layer_3_f1: 🔴 0.1176 (critical)
