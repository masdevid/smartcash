# File: src/utils/advanced_metrics.py
# Author: Alfrida Sabar
# Deskripsi: Modul metrik lanjutan untuk evaluasi deteksi objek

import numpy as np
import torch
from scipy import stats
from utils.logging import ColoredLogger

class AdvancedObjectDetectionMetrics:
    def __init__(self, num_classes=7, iou_thresholds=None):
        """
        Inisialisasi metrik deteksi objek lanjutan
        
        Args:
            num_classes (int): Jumlah kelas yang akan dievaluasi
            iou_thresholds (list): Threshold IoU untuk evaluasi
        """
        self.logger = ColoredLogger('AdvancedMetrics')
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.75, 0.9]
        
        # Struktur untuk menyimpan statistik
        self.reset()

    def reset(self):
        """Reset semua statistik metrik"""
        self.predictions = {thresh: {cls: [] for cls in range(self.num_classes)} 
                            for thresh in self.iou_thresholds}
        self.ground_truth = {cls: [] for cls in range(self.num_classes)}

    def update(self, preds, targets, iou_fn=None):
        """
        Perbarui statistik dengan prediksi dan ground truth
        
        Args:
            preds (torch.Tensor): Prediksi model
            targets (torch.Tensor): Ground truth
            iou_fn (callable): Fungsi custom untuk menghitung IoU
        """
        iou_fn = iou_fn or self._calc_iou
        
        for pred, target in zip(preds, targets):
            for thresh in self.iou_thresholds:
                for cls in range(self.num_classes):
                    # Filter prediksi dan ground truth berdasarkan kelas
                    cls_preds = pred[pred[:, 5] == cls]
                    cls_targets = target[target[:, 4] == cls]
                    
                    # Hitung IoU dan proses prediksi
                    matched_preds = self._match_predictions(
                        cls_preds, cls_targets, thresh, iou_fn
                    )
                    self.predictions[thresh][cls].extend(matched_preds)
            
            # Simpan ground truth
            for cls in range(self.num_classes):
                cls_targets = target[target[:, 4] == cls]
                self.ground_truth[cls].extend(cls_targets)

    def _calc_iou(self, box1, box2):
        """
        Hitung Intersection over Union (IoU) standar
        
        Args:
            box1 (torch.Tensor): Kotak pertama
            box2 (torch.Tensor): Kotak kedua
        
        Returns:
            float: Nilai IoU
        """
        # Koordinat interseksi
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        # Hitung area interseksi
        intersect = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Area union
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersect
        
        return intersect / (union + 1e-6)

    def _match_predictions(self, preds, targets, iou_thresh, iou_fn):
        """
        Cocokkan prediksi dengan ground truth berdasarkan IoU
        
        Args:
            preds (torch.Tensor): Prediksi
            targets (torch.Tensor): Ground truth
            iou_thresh (float): Threshold IoU
            iou_fn (callable): Fungsi IoU
        
        Returns:
            list: Prediksi yang cocok
        """
        matched_preds = []
        
        if len(preds) == 0 or len(targets) == 0:
            return matched_preds
        
        # Hitung IoU untuk setiap kombinasi
        ious = torch.zeros((len(preds), len(targets)))
        for i, pred in enumerate(preds):
            for j, target in enumerate(targets):
                ious[i, j] = iou_fn(pred[:4], target[:4])
        
        # Temukan prediksi terbaik untuk setiap ground truth
        best_iou_per_target, _ = ious.max(dim=0)
        matched = best_iou_per_target > iou_thresh
        
        return matched_preds

    def compute_precision_recall(self):
        """
        Hitung presisi dan recall untuk setiap kelas dan threshold IoU
        
        Returns:
            dict: Presisi dan recall untuk setiap kelas dan threshold
        """
        results = {}
        
        for thresh in self.iou_thresholds:
            results[thresh] = {}
            
            for cls in range(self.num_classes):
                preds = np.array(self.predictions[thresh][cls])
                gt = len(self.ground_truth[cls])
                
                # Hitung true positives, false positives
                true_pos = np.sum(preds)
                false_pos = len(preds) - true_pos
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / gt if gt > 0 else 0
                
                results[thresh][cls] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                }
        
        return results

    def compute_statistical_significance(self, other_model_metrics):
        """
        Hitung signifikansi statistik perbedaan performa model
        
        Args:
            other_model_metrics (dict): Metrik model lain untuk perbandingan
        
        Returns:
            dict: Hasil uji statistik
        """
        statistical_tests = {}
        
        for thresh in self.iou_thresholds:
            statistical_tests[thresh] = {}
            
            for cls in range(self.num_classes):
                # Uji t untuk membandingkan kinerja
                t_stat, p_value = stats.ttest_ind(
                    self.predictions[thresh][cls],
                    other_model_metrics.predictions[thresh][cls]
                )
                
                statistical_tests[thresh][cls] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return statistical_tests

# Contoh penggunaan
def demo_advanced_metrics():
    """Demonstrasi penggunaan metrik lanjutan"""
    metrics = AdvancedObjectDetectionMetrics()
    
    # Simulasi data prediksi dan ground truth
    # TODO: Implementasi dengan data aktual
    pass