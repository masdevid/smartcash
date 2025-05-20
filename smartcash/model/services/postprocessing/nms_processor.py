"""
File: smartcash/model/services/postprocessing/nms_processor.py
Deskripsi: Processor untuk Non-Maximum Suppression pada hasil deteksi
"""

import torch
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Any

from smartcash.common.logger import get_logger
from smartcash.model.utils.metrics.nms_metrics import non_max_suppression, apply_classic_nms


class NMSProcessor:
    """
    Processor untuk melakukan Non-Maximum Suppression (NMS) pada hasil 
    deteksi untuk menghilangkan deteksi duplikat.
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Inisialisasi NMS processor.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger()
    
    def process(
        self,
        detections: torch.Tensor,
        iou_threshold: float = 0.45,
        conf_threshold: Optional[float] = None,
        class_specific: bool = True,
        max_detections: int = 300
    ) -> torch.Tensor:
        """
        Proses deteksi dengan Non-Maximum Suppression.
        
        Args:
            detections: Tensor deteksi [N, 6] (x1, y1, x2, y2, conf, class_id) 
                       atau [N, 7] (batch_idx, x1, y1, x2, y2, conf, class_id)
            iou_threshold: Threshold IoU untuk NMS
            conf_threshold: Threshold confidence (opsional)
            class_specific: Lakukan NMS per kelas
            max_detections: Jumlah maksimum deteksi yang dipertahankan
            
        Returns:
            Tensor deteksi setelah NMS
        """
        # Validasi input
        if detections is None or detections.numel() == 0:
            return torch.zeros((0, 6), device=detections.device if detections is not None else 'cpu')
        
        # Cek apakah deteksi memiliki batch_idx (7 kolom)
        has_batch = detections.shape[1] == 7
        
        # Filter berdasarkan confidence jika diberikan
        if conf_threshold is not None:
            conf_idx = 5 if has_batch else 4
            conf_mask = detections[:, conf_idx] >= conf_threshold
            detections = detections[conf_mask]
        
        # Jika tidak ada deteksi yang tersisa
        if detections.shape[0] == 0:
            return torch.zeros((0, 6), device=detections.device)
        
        # Lakukan NMS
        if has_batch:
            # Batch processing dengan fungsi non_max_suppression
            result = non_max_suppression(
                detections.unsqueeze(0),  # Add batch dimension
                conf_thres=0.0,  # Already filtered
                iou_thres=iou_threshold,
                max_det=max_detections
            )[0]  # Take first batch result
        else:
            # Single image processing
            if class_specific:
                # Group by class
                class_idx = 5
                classes = detections[:, class_idx].cpu().unique()
                
                # Process each class separately
                keep_indices = []
                for cls in classes:
                    cls_mask = detections[:, class_idx] == cls
                    cls_dets = detections[cls_mask]
                    
                    # Apply NMS for this class
                    cls_keep = apply_classic_nms(
                        cls_dets[:, :4],
                        cls_dets[:, 4],
                        iou_threshold
                    )
                    
                    # Convert indices back to original
                    orig_indices = torch.where(cls_mask)[0]
                    cls_keep_orig = orig_indices[cls_keep]
                    
                    keep_indices.append(cls_keep_orig)
                
                # Combine all kept indices
                if keep_indices:
                    keep = torch.cat(keep_indices)
                    result = detections[keep]
                else:
                    result = torch.zeros((0, 6), device=detections.device)
            else:
                # Class-agnostic NMS
                keep = apply_classic_nms(
                    detections[:, :4],
                    detections[:, 4],
                    iou_threshold
                )
                result = detections[keep]
        
        # Limit detections
        if result.shape[0] > max_detections:
            # Sort by confidence (descending)
            conf_idx = 5 if has_batch else 4
            _, top_indices = torch.sort(result[:, conf_idx], descending=True)
            result = result[top_indices[:max_detections]]
        
        return result