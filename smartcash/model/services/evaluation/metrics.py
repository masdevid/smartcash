"""
File: smartcash/model/services/evaluation/metrics.py
Deskripsi: Implementasi perhitungan metrik evaluasi untuk model deteksi mata uang
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import time

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.model.utils.metrics import (
    box_iou, xywh2xyxy, xyxy2xywh, compute_ap, ap_per_class, 
    precision_recall_curve, MetricsCalculator as BaseMetricsCalculator
)

class MetricsComputation:
    """
    Komputasi metrik evaluasi untuk model deteksi mata uang.
    
    Fitur:
    - Perhitungan berbagai metrik per batch dan keseluruhan
    - Dukungan untuk metrik per kelas dan per layer
    - Perhitungan mAP, precision, recall, F1-score
    - Confusion matrix dan kurva PR
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Inisialisasi MetricsComputation.
        
        Args:
            config: Konfigurasi model dan evaluasi
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or get_logger("model.metrics")
        
        # Dapatkan konfigurasi layer
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config.get_layer_names())
        
        # Buat instance MetricsCalculator untuk perhitungan metrik
        self.calculator = BaseMetricsCalculator()
        
        # Metrics tambahan untuk spesifik per layer
        self.layer_metrics = {layer: BaseMetricsCalculator() for layer in self.active_layers}
        
        # Variabel tracking untuk metrik lainnya
        self.confusion_matrix = {}
        self.inference_times = []
        self.last_batch_metrics = {}
        
        self.logger.info(f"📊 MetricsComputation diinisialisasi dengan {len(self.active_layers)} layer aktif")
    
    def reset(self):
        """Reset semua metrik."""
        # Reset metrik dasar
        self.calculator.reset()
        
        # Reset metrik per layer
        for layer in self.active_layers:
            self.layer_metrics[layer].reset()
        
        # Reset metrik lainnya
        self.confusion_matrix = {}
        self.inference_times = []
        self.last_batch_metrics = {}
    
    def update(
        self,
        predictions: Union[torch.Tensor, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        inference_time: float = 0.0
    ) -> Dict[str, float]:
        """
        Update metrik dengan batch baru.
        
        * old: handlers.model.metrics.update_metrics()
        * migrated: Streamlined metrics computation
        
        Args:
            predictions: Output model [batch_size, num_pred, 6] (x,y,w,h,conf,cls) atau list per layer
            targets: Dictionary target per layer
            inference_time: Waktu inferensi dalam detik
            
        Returns:
            Dict metrik untuk batch ini
        """
        # Catat waktu inferensi
        if inference_time > 0:
            self.inference_times.append(inference_time)
        
        # Hitung metrik batch
        batch_metrics = {}
        
        # Jika predictions adalah list (untuk model multilayer)
        is_multilayer = isinstance(predictions, list)
        
        # Update metrik global terlebih dahulu
        if is_multilayer:
            # Gabungkan prediksi dari semua layer
            combined_pred = torch.cat(predictions, dim=1) if all(isinstance(p, torch.Tensor) for p in predictions) else predictions[0]
            self.calculator.update(combined_pred, targets)
        else:
            # Gunakan prediksi langsung
            self.calculator.update(predictions, targets)
        
        # Update metrik per layer
        for i, layer in enumerate(self.active_layers):
            layer_target = targets.get(layer, None)
            
            if layer_target is not None:
                layer_pred = predictions[i] if is_multilayer and i < len(predictions) else predictions
                self.layer_metrics[layer].update(layer_pred, {layer: layer_target})
                
                # Update confusion matrix untuk layer ini
                self._update_confusion_matrix(layer, layer_pred, layer_target)
        
        # Hitung metrik batch
        batch_metrics = self._calculate_batch_metrics()
        self.last_batch_metrics = batch_metrics
        
        return batch_metrics
    
    def _update_confusion_matrix(
        self,
        layer: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        iou_threshold: float = 0.5
    ) -> None:
        """
        Update confusion matrix untuk layer tertentu.
        
        Args:
            layer: Nama layer
            predictions: Prediksi model untuk layer ini
            targets: Target untuk layer ini
            iou_threshold: Threshold IoU untuk match
        """
        # Dapatkan konfigurasi layer
        layer_config = self.layer_config.get_layer_config(layer)
        class_ids = layer_config['class_ids']
        
        # Proses dan match prediksi dengan target
        if predictions.size(0) > 0 and targets.size(0) > 0:
            # Ekstrak prediksi dan target valid
            mask = predictions[:, 4] > 0.1  # filter berdasarkan confidence
            if not mask.any():
                return
                
            pred_boxes = predictions[mask, :4]  # x, y, w, h
            pred_classes = predictions[mask, 5].int()  # class id
            
            # Konversi format box jika perlu
            if pred_boxes.shape[1] == 4:  # x, y, w, h
                pred_boxes = xywh2xyxy(pred_boxes)
                
            target_boxes = xywh2xyxy(targets[:, :4]) if targets.shape[1] >= 4 else None
            target_classes = targets[:, 5].int() if targets.shape[1] >= 6 else None
            
            if target_boxes is None or target_classes is None:
                return
                
            # Hitung IoU matrix
            ious = box_iou(pred_boxes, target_boxes)
            
            # Update confusion matrix berdasarkan matches
            for pred_idx in range(len(pred_boxes)):
                pred_cls = pred_classes[pred_idx].item()
                
                # Cari match terbaik
                match_idx = torch.argmax(ious[pred_idx]).item()
                max_iou = ious[pred_idx, match_idx].item()
                
                if max_iou > iou_threshold:
                    # Ada match dengan target
                    target_cls = target_classes[match_idx].item()
                    
                    # Update confusion matrix
                    if pred_cls not in self.confusion_matrix:
                        self.confusion_matrix[pred_cls] = {}
                    if target_cls not in self.confusion_matrix[pred_cls]:
                        self.confusion_matrix[pred_cls][target_cls] = 0
                    
                    self.confusion_matrix[pred_cls][target_cls] += 1
    
    def _calculate_batch_metrics(self) -> Dict[str, float]:
        """
        Hitung metrik untuk batch terakhir.
        
        Returns:
            Dict metrik batch
        """
        # Dapatkan metrik dari calculator
        metrics = self.calculator.get_last_batch_metrics()
        
        # Tambahkan waktu inferensi rata-rata
        if self.inference_times:
            metrics['inference_time'] = np.mean(self.inference_times) * 1000  # ms
            
        return metrics
    
    def get_last_batch_metrics(self) -> Dict[str, float]:
        """Dapatkan metrik dari batch terakhir."""
        return self.last_batch_metrics
    
    def compute(self) -> Dict[str, Any]:
        """
        Hitung metrik evaluasi final.
        
        * old: handlers.model.metrics.compute_metrics()
        * migrated: Comprehensive metrics with layer and class breakdown
        
        Returns:
            Dict metrik final
        """
        # Dapatkan metrik global
        metrics = self.calculator.compute()
        
        # Tambahkan metrik per layer
        for layer in self.active_layers:
            layer_metrics = self.layer_metrics[layer].compute()
            
            # Tambahkan prefix layer ke key metrik
            for key, value in layer_metrics.items():
                metrics[f"{layer}_{key}"] = value
        
        # Tambahkan metrik per kelas
        metrics.update(self._compute_class_metrics())
        
        # Tambah confusion matrix
        metrics['confusion_matrix'] = self.confusion_matrix
        
        # Tambah kurva precision-recall
        pr_curves = self._compute_pr_curves()
        if pr_curves:
            metrics.update(pr_curves)
        
        # Waktu inferensi rata-rata
        if self.inference_times:
            metrics['inference_time'] = np.mean(self.inference_times) * 1000  # ms
            
        return metrics
    
    def _compute_class_metrics(self) -> Dict[str, float]:
        """
        Hitung metrik per kelas.
        
        Returns:
            Dict berisi metrik per kelas
        """
        class_metrics = {}
        
        # Dapatkan statistik per kelas dari base calculator
        stats = self.calculator.get_class_statistics()
        
        # Format hasil untuk setiap kelas
        for cls_id, cls_stats in stats.items():
            prefix = f"cls_{cls_id}_"
            
            for metric_name, value in cls_stats.items():
                class_metrics[f"{prefix}{metric_name}"] = value
        
        return class_metrics
    
    def _compute_pr_curves(self) -> Dict[str, np.ndarray]:
        """
        Hitung kurva precision-recall.
        
        Returns:
            Dict berisi kurva precision, recall, dan f1
        """
        # Dapatkan statistik yang dibutuhkan untuk perhitungan kurva PR
        stats = self.calculator.get_pr_curve_data()
        
        if not stats:
            return {}
            
        # Extract data
        precisions = stats.get('precision', [])
        recalls = stats.get('recall', [])
        
        # Hitung F1 curve
        f1_curve = []
        for p, r in zip(precisions, recalls):
            if p + r > 0:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
            f1_curve.append(f1)
        
        return {
            'precision_curve': np.array(precisions),
            'recall_curve': np.array(recalls),
            'f1_curve': np.array(f1_curve)
        }