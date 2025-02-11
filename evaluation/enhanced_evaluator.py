# File: src/evaluation/enhanced_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Modul evaluasi lanjutan untuk SmartCash Detector dengan analisis komprehensif

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from utils.logging import ColoredLogger
from data.dataset import RupiahDataset
from utils.metrics import MeanAveragePrecision

class AdvancedDetectorEvaluator:
    """
    Kelas evaluator canggih untuk model deteksi mata uang
    Mendukung evaluasi multi-skenario dengan parameter kondisional
    """
    def __init__(
        self, 
        model, 
        test_dir: Optional[Path] = None, 
        conditions: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        """
        Inisialisasi evaluator dengan konfigurasi fleksibel

        Args:
            model: Model deteksi yang akan dievaluasi
            test_dir: Direktori dataset pengujian
            conditions: Kondisi spesifik untuk skenario evaluasi
            device: Perangkat komputasi (cuda/cpu)
        """
        self.logger = ColoredLogger('AdvancedEvaluator')
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Inisialisasi dataset untuk berbagai skenario
        self.test_dir = test_dir
        self.conditions = conditions or {}
        
        # Nama kelas mata uang
        self.class_names = [
            '1000', '2000', '5000', 
            '10000', '20000', '50000', '100000'
        ]
        
        # Metrik evaluasi
        self.map_metric = MeanAveragePrecision(
            num_classes=len(self.class_names)
        )

    def evaluate(
        self, 
        confidence_threshold: float = 0.25, 
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Evaluasi komprehensif model dengan kondisi yang dapat disesuaikan

        Args:
            confidence_threshold: Ambang batas kepercayaan deteksi
            iou_threshold: Ambang batas Intersection over Union

        Returns:
            Dict berisi metrik dan informasi evaluasi
        """
        # Siapkan dataset dengan kondisi
        dataset = self._prepare_dataset()
        
        # Inisialisasi variabel untuk tracking
        total_predictions = 0
        correct_predictions = 0
        class_performance = {cls: {'total': 0, 'correct': 0} for cls in self.class_names}
        
        # Proses evaluasi
        with torch.no_grad():
            for img, targets in dataset:
                img = img.unsqueeze(0).to(self.device)
                targets = targets.to(self.device)
                
                # Prediksi
                predictions = self.model(img)
                
                # Filter prediksi berdasarkan confidence
                filtered_preds = self._filter_predictions(
                    predictions, 
                    confidence_threshold
                )
                
                # Update metrik
                self.map_metric.update(filtered_preds, targets)
                
                # Analisis deteksi
                total_predictions += len(filtered_preds)
                correct_predictions += self._count_correct_predictions(
                    filtered_preds, 
                    targets, 
                    iou_threshold
                )
                
                # Analisis per kelas
                self._update_class_performance(
                    filtered_preds, 
                    targets, 
                    class_performance
                )
        
        # Hitung metrik akhir
        metrics = {
            'map': self.map_metric.compute(),
            'precision': correct_predictions / max(total_predictions, 1),
            'recall': correct_predictions / max(len(dataset), 1),
            'total_predictions': total_predictions
        }
        
        # Tambahkan informasi tambahan
        extras = {
            'class_performance': class_performance,
            'scenarios': self.conditions
        }
        
        return {
            'metrics': metrics,
            'extras': extras
        }

    def _prepare_dataset(self) -> torch.utils.data.Dataset:
        """
        Persiapkan dataset dengan kondisi spesifik

        Returns:
            Dataset yang difilter sesuai kondisi
        """
        if self.test_dir is None:
            raise ValueError("Direktori tes tidak ditentukan")
        
        # Filter dataset berdasarkan kondisi
        dataset = RupiahDataset(
            img_dir=self.test_dir,
            augment=False,  # Nonaktifkan augmentasi untuk evaluasi
            img_size=640  # Ukuran gambar standar
        )
        
        # Terapkan filter kondisional jika ada
        if self.conditions:
            filtered_dataset = self._apply_conditional_filter(dataset)
            return filtered_dataset
        
        return dataset

    def _apply_conditional_filter(self, dataset):
        """
        Terapkan filter kondisional pada dataset

        Args:
            dataset: Dataset asli

        Returns:
            Dataset yang difilter
        """
        # Implementasi filter berdasarkan kondisi
        # Contoh: filter berdasarkan pencahayaan, jarak, dll
        filtered_indices = []
        
        for idx in range(len(dataset)):
            img, target = dataset[idx]
            
            # Contoh filter sederhana
            if 'lighting' in self.conditions:
                # Simulasi filter pencahayaan (implementasi dummy)
                if self.conditions['lighting'] == 'low':
                    # Misalnya: filter gambar dengan tingkat pencahayaan rendah
                    pass
            
            # Filter berdasarkan ukuran objek
            if 'min_box_size' in self.conditions:
                min_w, min_h = self.conditions['min_box_size']
                valid_targets = [
                    t for t in target 
                    if t[2] * img.shape[1] >= min_w and t[3] * img.shape[0] >= min_h
                ]
                if valid_targets:
                    filtered_indices.append(idx)
            else:
                filtered_indices.append(idx)
        
        # Buat subset dataset
        return torch.utils.data.Subset(dataset, filtered_indices)

    def _filter_predictions(self, predictions, confidence_threshold):
        """
        Filter prediksi berdasarkan ambang kepercayaan

        Args:
            predictions: Prediksi model
            confidence_threshold: Ambang batas kepercayaan

        Returns:
            Prediksi yang difilter
        """
        filtered_preds = []
        for pred in predictions:
            # Filter prediksi dengan confidence di atas threshold
            mask = pred[..., 4] > confidence_threshold
            filtered_preds.append(pred[mask])
        return filtered_preds

    def _count_correct_predictions(self, predictions, targets, iou_threshold):
        """
        Hitung prediksi yang benar berdasarkan IoU

        Args:
            predictions: Prediksi model
            targets: Ground truth
            iou_threshold: Ambang batas IoU

        Returns:
            Jumlah prediksi yang benar
        """
        correct = 0
        for pred_batch in predictions:
            for pred in pred_batch:
                for target in targets:
                    # Hitung IoU
                    iou = self._calculate_iou(pred[:4], target[:4])
                    
                    # Periksa kelas dan IoU
                    if (iou > iou_threshold and 
                        pred[5].long() == target[4].long()):
                        correct += 1
                        break
        return correct

    def _update_class_performance(self, predictions, targets, class_performance):
        """
        Update performa untuk setiap kelas

        Args:
            predictions: Prediksi model
            targets: Ground truth
            class_performance: Dict untuk tracking performa kelas
        """
        for pred_batch in predictions:
            for pred in pred_batch:
                cls_name = self.class_names[pred[5].long()]
                class_performance[cls_name]['total'] += 1
                
                # Periksa apakah prediksi benar
                for target in targets:
                    if (self._calculate_iou(pred[:4], target[:4]) > 0.5 and 
                        pred[5].long() == target[4].long()):
                        class_performance[cls_name]['correct'] += 1
                        break

    def _calculate_iou(self, box1, box2):
        """
        Hitung Intersection over Union (IoU)

        Args:
            box1: Kotak pertama
            box2: Kotak kedua

        Returns:
            Nilai IoU
        """
        # Konversi ke format x1, y1, x2, y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Koordinat interseksi
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        # Area interseksi
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Area union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area

        # IoU
        return inter_area / (union_area + 1e-6)