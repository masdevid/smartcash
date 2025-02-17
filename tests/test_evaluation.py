# File: tests/test_evaluation.py
# Author: Alfrida Sabar
# Deskripsi: Pengujian unit untuk proses evaluasi model dalam proyek SmartCash

import pytest
import torch
import yaml
import numpy as np
from pathlib import Path

from utils.metrics import MetricsCalculator
from utils.polygon_metrics import PolygonMetricsCalculator
from handlers.evaluation_handler import EvaluationHandler
from utils.visualization import ResultVisualizer

class TestEvaluation:
    @pytest.fixture
    def config_path(self):
        """Fixture untuk path konfigurasi"""
        return Path(__file__).parent.parent / 'configs' / 'base_config.yaml'
    
    @pytest.fixture
    def sample_predictions(self):
        """Fixture prediksi model sampel"""
        # Prediksi dengan 3 bounding box
        predictions = torch.tensor([
            # [x, y, width, height, confidence, class]
            [0.5, 0.5, 0.2, 0.2, 0.9, 1],  # kelas 1, tinggi confidence
            [0.3, 0.7, 0.1, 0.1, 0.7, 2],  # kelas 2, confidence sedang
            [0.8, 0.2, 0.15, 0.15, 0.6, 1]  # kelas 1, confidence rendah
        ])
        return predictions
    
    @pytest.fixture
    def sample_targets(self):
        """Fixture target ground truth"""
        # Target dengan 2 bounding box
        targets = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 1],  # kelas 1
            [0.3, 0.7, 0.1, 0.1, 2]   # kelas 2
        ])
        return targets
    
    def test_metrics_calculator(self, sample_predictions, sample_targets):
        """Pengujian kalkulator metrik"""
        metrics_calc = MetricsCalculator()
        
        # Update dengan prediksi dan target
        metrics_calc.update(
            predictions=sample_predictions.unsqueeze(0),  # Tambah dimensi batch
            targets=sample_targets.unsqueeze(0)
        )
        
        # Hitung metrik akhir
        final_metrics = metrics_calc.compute()
        
        # Periksa keberadaan metrik kunci
        expected_metrics = [
            'precision', 'recall', 'accuracy', 
            'f1', 'mAP', 'inference_time'
        ]
        
        for metric in expected_metrics:
            assert metric in final_metrics, f"Metrik {metric} tidak ditemukan"
            assert isinstance(final_metrics[metric], float), f"Metrik {metric} harus berupa float"
        
        # Periksa rentang nilai metrik
        assert 0 <= final_metrics['precision'] <= 1, "Precision harus antara 0 dan 1"
        assert 0 <= final_metrics['recall'] <= 1, "Recall harus antara 0 dan 1"
        assert 0 <= final_metrics['accuracy'] <= 1, "Akurasi harus antara 0 dan 1"
        assert 0 <= final_metrics['f1'] <= 1, "F1-score harus antara 0 dan 1"
    
    def test_polygon_metrics_calculator(self):
        """Pengujian kalkulator metrik polygon"""
        polygon_calc = PolygonMetricsCalculator()
        
        # Polygon prediksi dan ground truth sampel
        pred_polygons = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # persegi
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]   # persegi dalam
        ])
        
        gt_polygons = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # persegi persis
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]   # persegi sedikit berbeda
        ])
        
        # Update kalkulator
        polygon_calc.update(pred_polygons, gt_polygons)
        
        # Hitung metrik
        polygon_metrics = polygon_calc.compute()
        
        # Periksa kelengkapan metrik polygon
        expected_polygon_metrics = [
            'polygon_precision', 'polygon_recall', 
            'polygon_f1_score', 'polygon_matching_rate'
        ]
        
        for metric in expected_polygon_metrics:
            assert metric in polygon_metrics, f"Metrik polygon {metric} tidak ditemukan"
            assert isinstance(polygon_metrics[metric], float), f"Metrik {metric} harus berupa float"
        
        # Periksa rentang nilai metrik
        assert 0 <= polygon_metrics['polygon_precision'] <= 1, "Precision harus antara 0 dan 1"
        assert 0 <= polygon_metrics['polygon_recall'] <= 1, "Recall harus antara 0 dan 1"
        assert 0 <= polygon_metrics['polygon_f1_score'] <= 1, "F1-score harus antara 0 dan 1"
        assert 0 <= polygon_metrics['polygon_matching_rate'] <= 1, "Matching rate harus antara 0 dan 1"
    
    def test_result_visualizer(self, tmp_path, sample_predictions, sample_targets):
        """Pengujian visualisator hasil"""
        # Persiapkan visualizer dengan direktori output sementara
        visualizer = ResultVisualizer(output_dir=str(tmp_path))
        
        # Buat gambar sampel acak
        sample_images = torch.rand(2, 3, 640, 640)  # 2 gambar, 3 channel, 640x640
        
        # Simpan visualisasi prediksi batch
        visualizer.save_batch_predictions(
            images=sample_images,
            predictions=sample_predictions.unsqueeze(0).repeat(2, 1, 1),  # Duplikat untuk 2 gambar
            targets=sample_targets.unsqueeze(0).repeat(2, 1, 1),
            scenario_name='test_scenario',
            batch_idx=0
        )
        
        # Periksa apakah file gambar dibuat
        output_dir = tmp_path / 'test_scenario' / 'batch_0'
        assert output_dir.exists(), "Direktori output visualisasi tidak dibuat"
        
        # Periksa file gambar yang dihasilkan
        image_files = list(output_dir.glob('*.png'))
        assert len(image_files) > 0, "Tidak ada gambar yang dihasilkan"
    
    def test_evaluation_comparison_visualization(self, tmp_path):
        """Pengujian visualisasi perbandingan hasil evaluasi"""
        # Siapkan visualizer
        visualizer = ResultVisualizer(output_dir=str(tmp_path))
        
        # Contoh hasil evaluasi untuk beberapa skenario
        sample_results = {
            'scenario_1': {
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1': 0.85,
                    'mAP': 0.80,
                    'inference_time': 25.5
                }
            },
            'scenario_2': {
                'metrics': {
                    'accuracy': 0.90,
                    'precision': 0.88,
                    'recall': 0.92,
                    'f1': 0.90,
                    'mAP': 0.85,
                    'inference_time': 22.3
                }
            }
        }
        
        # Buat plot perbandingan
        visualizer.create_comparison_plots(sample_results)
        
        # Periksa file-file plot yang dihasilkan
        plot_files = list(tmp_path.glob('*.png'))
        assert len(plot_files) > 0, "Tidak ada plot yang dihasilkan"
        
        # Periksa nama file plot
        plot_names = [f.name for f in plot_files]
        
        # Pastikan plot perbandingan metrik dan waktu inferensi ada
        assert any('metric_comparison' in name for name in plot_names), "Plot perbandingan metrik tidak ditemukan"
        assert any('inference_comparison' in name for name in plot_names), "Plot perbandingan waktu inferensi tidak ditemukan"
    
    def test_iou_calculation(self):
        """Pengujian perhitungan Intersection over Union (IoU)"""
        # Buat box prediksi dan ground truth
        pred_box = torch.tensor([0.5, 0.5, 0.4, 0.4])  # [x_center, y_center, width, height]
        gt_box = torch.tensor([0.6, 0.6, 0.5, 0.5])
        
        # Metode perhitungan IoU sederhana
        def calculate_iou(box1, box2):
            # Konversi ke format [x_min, y_min, x_max, y_max]
            b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
            b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
            
            b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
            b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
            
            # Hitung koordinat intersection
            inter_x1 = max(b1_x1, b2_x1)
            inter_y1 = max(b1_y1, b2_y1)
            inter_x2 = min(b1_x2, b2_x2)
            inter_y2 = min(b1_y2, b2_y2)
            
            # Hitung area intersection
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            # Hitung area box
            b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            
            # Hitung IoU
            union_area = b1_area + b2_area - inter_area
            iou = inter_area / (union_area + 1e-16)
            
            return iou
        
        # Hitung IoU
        iou = calculate_iou(pred_box, gt_box)
        
        # Periksa rentang IoU
        assert 0 <= iou <= 1, "IoU harus antara 0 dan 1"
    
    @pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
    def test_metrics_with_different_iou_thresholds(self, iou_threshold):
        """Pengujian metrik dengan threshold IoU berbeda"""
        metrics_calc = MetricsCalculator()
        
        # Prediksi dan target sampel dengan variasi IoU
        predictions = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 0.9, 1],  # overlap tinggi
            [0.7, 0.7, 0.1, 0.1, 0.7, 2],  # overlap rendah
        ])
        
        targets = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 1],  # cocok dengan prediksi pertama
            [0.6, 0.6, 0.3, 0.3, 2],  # cocok dengan prediksi kedua
        ])
        
        # Update kalkulator dengan prediksi
        metrics_calc.update(
            predictions=predictions.unsqueeze(0),
            targets=targets.unsqueeze(0)
        )
        
        # Hitung metrik
        metrics = metrics_calc.compute()
        
        # Periksa kelengkapan metrik
        assert 'precision' in metrics, "Precision tidak ditemukan"
        assert 'recall' in metrics, "Recall tidak ditemukan"
        assert 'f1' in metrics, "F1-score tidak ditemukan"