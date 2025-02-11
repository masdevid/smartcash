# File: src/evaluation/evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Modul evaluasi untuk SmartCash Detector dengan berbagai skenario pengujian

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from utils.logging import ColoredLogger
from data.dataset import RupiahDataset

class DetectorEvaluator:
    def __init__(self, model, test_dir, device='cuda'):
        self.logger = ColoredLogger('DetectorEvaluator')
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Inisialisasi dataset untuk tiap skenario
        self.scenarios = {
            'normal': RupiahDataset(test_dir / 'normal'),
            'occlusion': RupiahDataset(test_dir / 'occlusion'),
            'stacked': RupiahDataset(test_dir / 'stacked'),
            'foreign': RupiahDataset(test_dir / 'foreign')
        }
        
        self.class_names = ['1000', '2000', '5000', '10000', '20000', 
                           '50000', '100000']

    def calculate_iou(self, box1, box2):
        """Menghitung Intersection over Union antara dua bounding box"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        return intersection / (box1_area + box2_area - intersection)

    def evaluate_scenario(self, scenario_name, iou_thresh=0.5, conf_thresh=0.25):
        """Evaluasi model untuk skenario tertentu"""
        self.logger.info(f'Evaluasi skenario: {scenario_name}')
        dataset = self.scenarios[scenario_name]
        
        all_preds = []
        all_targets = []
        
        for img, targets in dataset:
            img = img.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(img)
            
            # Filter prediksi berdasarkan confidence
            preds = preds[preds[:, 4] > conf_thresh]
            
            # Match prediksi dengan ground truth
            matched_preds = []
            for target in targets:
                best_iou = 0
                best_pred = None
                
                for pred in preds:
                    iou = self.calculate_iou(pred[:4], target[:4])
                    if iou > best_iou and iou > iou_thresh:
                        best_iou = iou
                        best_pred = pred
                
                if best_pred is not None:
                    matched_preds.append(best_pred)
                    all_preds.append(best_pred[5])
                    all_targets.append(target[4])
        
        return self.calculate_metrics(all_preds, all_targets)

    def calculate_metrics(self, preds, targets):
        """Menghitung metrik evaluasi"""
        preds = np.array(preds)
        targets = np.array(targets)
        
        # Hitung precision, recall, dan F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted'
        )
        
        # Buat confusion matrix
        cm = confusion_matrix(targets, preds, 
                            labels=range(len(self.class_names)))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

    def evaluate_all(self):
        """Evaluasi semua skenario dan tampilkan hasil"""
        results = {}
        
        for scenario in self.scenarios:
            self.logger.info(f'🔄 Memulai evaluasi {scenario}')
            metrics = self.evaluate_scenario(scenario)
            results[scenario] = metrics
            
            self.logger.metric('Hasil evaluasi', {
                'precision': f'{metrics["precision"]:.3f}',
                'recall': f'{metrics["recall"]:.3f}',
                'f1': f'{metrics["f1"]:.3f}'
            })
            
            # Tampilkan confusion matrix
            cm = metrics['confusion_matrix']
            self.logger.info('📊 Confusion Matrix:')
            for i, row in enumerate(cm):
                row_str = ' '.join(f'{val:4d}' for val in row)
                self.logger.info(f'{self.class_names[i]:>8}: {row_str}')
        
        return results

if __name__ == '__main__':
    from ..models.detector import SmartCashDetector
    
    model = SmartCashDetector('weights/best.pt')
    evaluator = DetectorEvaluator(model.model, Path('data/test'))
    results = evaluator.evaluate_all()