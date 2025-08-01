[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor - 🔍 RAW PREDICTIONS DEBUG - Batch 0:
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • targets shape: torch.Size([61, 6])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • targets sample: tensor([[0.00000, 0.00000, 0.50000, 0.50000, 1.00000, 0.75000],
        [1.00000, 7.00000, 0.73750, 0.79688, 0.40312, 0.15000],
        [1.00000, 7.00000, 0.96406, 0.50000, 0.07031, 0.75000]], device='mps:0')
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor - 🔄 CONVERSION RESULTS:
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • map_predictions: torch.Size([16, 300, 6])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • map_targets: torch.Size([36, 6])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • prediction sample: tensor([1.13235e-01, 2.26613e-01, 3.07659e-02, 1.19420e-04, 1.53958e-02, 0.00000e+00], device='mps:0')
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • target sample: tensor([0.00000, 0.00000, 0.50000, 0.50000, 1.00000, 0.75000])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor - 🔍 mAP DEBUG - Batch 0:
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • Predictions shape: torch.Size([16, 300, 6])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • Targets shape: torch.Size([36, 6])
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • Confidence range: 0.000000 - 0.080467
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • Non-zero confidences: 688/4800
[2025-08-01 18:25:11] INFO - smartcash.model.training.core.validation_executor -   • Classes present: tensor([0., 2., 6.], device='mps:0')
2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - ✅ Computed validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1']
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - ✅ Found layer_1_accuracy = 0.26239067055393583                                                                                                                          
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - ✅ Computed 4 validation metrics: ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1']
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - 📊 Phase 1: Using YOLOv5 built-in metrics for validation
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator - 📊 mAP computation debug:
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Total stat batches: 22
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • TP shape: (37991, 1), sum: 1
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Conf shape: (37991,), range: 0.0100-0.2749
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Pred classes: [0 2 5 6]
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Target classes: [0 1 2 3 4 5 6]
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator - mAP computation completed:
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.0011
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - YOLOv5 built-in metrics:
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor -   • Precision: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor -   • Recall: 0.0011
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor -   • F1: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor -   • mAP@0.5: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator - 📊 mAP computation debug:
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Total stat batches: 22
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • TP shape: (37991, 1), sum: 1
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Conf shape: (37991,), range: 0.0100-0.2749
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Pred classes: [0 2 5 6]
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Target classes: [0 1 2 3 4 5 6]
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator - mAP computation completed:
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • mAP@0.5: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Precision: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • Recall: 0.0011
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.yolov5_map_calculator -   • F1: 0.0000
[2025-08-01 18:25:22] INFO - smartcash.model.training.core.validation_executor - YOLOv5 mAP Results - mAP@0.5: 0.0000, Precision: 0.0000, Recall: 0.0011