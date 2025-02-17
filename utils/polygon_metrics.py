# File: utils/polygon_metrics.py
# Author: Alfrida Sabar
# Deskripsi: Kalkulator metrik khusus untuk deteksi poligon pada uang kertas

import torch
import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict, Tuple

class PolygonMetricsCalculator:
    """
    Kalkulator metrik untuk evaluasi deteksi poligon 
    dengan perhitungan akurasi geometri yang lebih presisi
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset semua metrik"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.polygon_matches = []
    
    def _convert_to_shapely_polygon(
        self, 
        polygon_points: torch.Tensor
    ) -> Polygon:
        """
        Konversi tensor polygon ke objek Shapely
        
        Args:
            polygon_points (torch.Tensor): Tensor koordinat polygon
        
        Returns:
            Polygon: Objek Shapely untuk kalkulasi geometri
        """
        if polygon_points.dim() == 1:
            polygon_points = polygon_points.view(-1, 2)
        
        return Polygon(polygon_points.cpu().numpy())
    
    def calculate_polygon_iou(
        self, 
        pred_polygon: torch.Tensor, 
        gt_polygon: torch.Tensor
    ) -> float:
        """
        Hitung Intersection over Union (IoU) untuk poligon
        
        Args:
            pred_polygon (torch.Tensor): Poligon prediksi
            gt_polygon (torch.Tensor): Poligon ground truth
        
        Returns:
            float: Nilai IoU poligon
        """
        pred_poly = self._convert_to_shapely_polygon(pred_polygon)
        gt_poly = self._convert_to_shapely_polygon(gt_polygon)
        
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        
        return intersection / union if union > 0 else 0
    
    def update(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update metrik dengan prediksi dan ground truth poligon
        
        Args:
            predictions (torch.Tensor): Prediksi poligon model
            ground_truth (torch.Tensor): Ground truth poligon
        
        Returns:
            Dict[str, float]: Metrik untuk batch ini
        """
        batch_metrics = {}
        
        for pred_poly, gt_poly in zip(predictions, ground_truth):
            # Cari pasangan poligon dengan IoU tertinggi
            best_iou = self._find_best_polygon_match(pred_poly, gt_poly)
            
            if best_iou >= self.iou_threshold:
                self.tp += 1
                self.polygon_matches.append((pred_poly, gt_poly, best_iou))
            else:
                self.fp += 1
                self.fn += 1
        
        return batch_metrics
    
    def _find_best_polygon_match(
        self, 
        pred_poly: torch.Tensor, 
        gt_poly: torch.Tensor
    ) -> float:
        """
        Temukan pasangan poligon dengan IoU tertinggi
        
        Args:
            pred_poly (torch.Tensor): Poligon prediksi
            gt_poly (torch.Tensor): Poligon ground truth
        
        Returns:
            float: IoU tertinggi
        """
        return self.calculate_polygon_iou(pred_poly, gt_poly)
    
    def compute(self) -> Dict[str, float]:
        """
        Hitung metrik akhir untuk deteksi poligon
        
        Returns:
            Dict[str, float]: Metrik deteksi poligon
        """
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 
            else 0
        )
        
        return {
            'polygon_precision': precision,
            'polygon_recall': recall,
            'polygon_f1_score': f1_score,
            'polygon_matching_rate': len(self.polygon_matches) / max(1, self.tp + self.fn)
        }