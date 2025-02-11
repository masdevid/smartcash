# File: src/metrics/test_scenarios.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi skenario pengujian lengkap untuk SmartCash Detector

from dataclasses import dataclass
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional
import albumentations as A
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from metrics.calculator import MetricsCalculator
from utils.logging import ColoredLogger

@dataclass
class ScenarioConfig:
    """Konfigurasi untuk skenario pengujian"""
    name: str
    conditions: Dict[str, any]
    metrics_thresholds: Dict[str, float]
    
class TestScenarioRunner:
    def __init__(self, model, data_loader, device='cuda'):
        self.logger = ColoredLogger('TestScenario')
        self.model = model
        self.data_loader = data_loader 
        self.device = device
        self.metrics_calc = MetricsCalculator()
        
    def evaluate_orientation(self, angles: List[int] = [0, 90, 180, 270]):
        """Evaluasi ketahanan terhadap rotasi"""
        self.logger.info("🔄 Evaluasi orientasi")
        results = {}
        
        for angle in angles:
            transformed_loader = self._rotate_dataset(angle)
            metrics = self._evaluate_batch(transformed_loader)
            results[f'angle_{angle}'] = metrics
            
        return results
    
    def evaluate_degradation(self, severities: List[float] = [0.2, 0.4, 0.6]):
        """Evaluasi ketahanan terhadap kerusakan uang"""
        self.logger.info("📉 Evaluasi degradasi")
        results = {}
        
        for severity in severities:
            degraded_loader = self._apply_degradation(severity)
            metrics = self._evaluate_batch(degraded_loader)
            results[f'degradation_{severity}'] = metrics
            
        return results
    
    def evaluate_feature_preservation(self, reference_features):
        """Evaluasi kualitas fitur pada kondisi pencahayaan rendah"""
        self.logger.info("🔍 Evaluasi preservasi fitur")
        
        with torch.no_grad():
            current_features = []
            for batch in self.data_loader:
                features = self.model.backbone(batch[0].to(self.device))
                current_features.append(features)
                
        similarity_scores = self._compute_feature_similarity(
            reference_features, current_features
        )
        return {'feature_preservation': similarity_scores.mean().item()}
    
    def _rotate_dataset(self, angle: int):
        """Rotasi dataset untuk pengujian orientasi"""
        transform = A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0),
            A.RandomBrightnessContrast(p=0.2)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        return self._transform_dataset(transform)
    
    def _apply_degradation(self, severity: float):
        """Simulasi kerusakan uang kertas"""
        transform = A.Compose([
            A.GaussNoise(var_limit=(10, 50), p=0.7),
            A.Blur(blur_limit=7, p=0.7),
            A.ImageCompression(quality_lower=40, p=0.7),
            A.CoarseDropout(max_holes=8, p=0.5)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        return self._transform_dataset(transform)
    
    def _transform_dataset(self, transform):
        """Aplikasikan transformasi ke dataset"""
        transformed_data = []
        for images, targets in self.data_loader:
            batch_transformed = []
            for img, target in zip(images, targets):
                transformed = transform(
                    image=img.numpy().transpose(1, 2, 0),
                    bboxes=target[:, :4].numpy(),
                    class_labels=target[:, 4].numpy()
                )
                batch_transformed.append((
                    torch.from_numpy(transformed['image'].transpose(2, 0, 1)),
                    torch.cat([
                        torch.tensor(transformed['bboxes']),
                        torch.tensor(transformed['class_labels']).unsqueeze(1)
                    ], dim=1)
                ))
            transformed_data.append(batch_transformed)
            
        return transformed_data
    
    def _compute_feature_similarity(self, ref_features, cur_features):
        """Hitung similarity antara reference dan current features"""
        similarities = []
        for ref, cur in zip(ref_features, cur_features):
            # Cosine similarity untuk setiap level feature
            sim = torch.nn.functional.cosine_similarity(
                ref.view(ref.size(0), -1),
                cur.view(cur.size(0), -1),
                dim=1
            )
            similarities.append(sim)
            
        return torch.cat(similarities)
    
    def _evaluate_batch(self, data_batch):
        """Evaluasi batch data dengan parallel processing"""
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._process_single, data_batch))
            
        return self.metrics_calc.aggregate_results(results)
    
    def _process_single(self, data):
        """Proses single image dengan error handling"""
        try:
            img, target = data
            with torch.no_grad():
                pred = self.model(img.unsqueeze(0).to(self.device))
            return self.metrics_calc.compute_metrics(pred, target)
        except Exception as e:
            self.logger.error(f"❌ Error processing image: {str(e)}")
            return None