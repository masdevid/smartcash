"""
File: smartcash/model/utils/__init__.py
Deskripsi: Inisialisasi paket utilitas model SmartCash
"""

from smartcash.model.utils.preprocessing import (
    ModelPreprocessor, letterbox, scale_coords
)

from smartcash.model.utils.metrics import (
    box_iou, xywh2xyxy, xyxy2xywh, 
    non_max_suppression, ap_per_class, mean_average_precision,
    precision_recall_curve, compute_ap, apply_classic_nms, soft_nms
)
from smartcash.model.utils.validation import (
    ModelValidator, check_img_size, check_anchors, 
    kmeans_anchors, compute_anchor_metrics
)
from smartcash.model.utils.research import (
    clean_dataframe, format_metric_name, find_common_metrics, 
    create_benchmark_table, create_win_rate_table
)

# Fungsi dan kelas utilitas
__all__ = [
    # Preprocessing
    'ModelPreprocessor',
    'letterbox',
    'scale_coords',
    
    # Metrics
    'box_iou',
    'xywh2xyxy',
    'xyxy2xywh',
    'non_max_suppression',
    'ap_per_class',
    'mean_average_precision',
    'precision_recall_curve',
    'compute_ap',
    'apply_classic_nms',
    'soft_nms',
    
    # Validation
    'ModelValidator',
    'check_img_size',
    'check_anchors',
    'kmeans_anchors',
    'compute_anchor_metrics',
    
    # Research
    'clean_dataframe',
    'format_metric_name',
    'find_common_metrics',
    'create_benchmark_table',
    'create_win_rate_table'
]