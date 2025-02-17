# File: tests/test_models.py
# Author: Alfrida Sabar
# Deskripsi: Pengujian unit untuk komponen model dalam proyek SmartCash

import pytest
import torch
import yaml
from pathlib import Path

from models.yolo5_efficient import YOLOv5Efficient
from models.backbones.efficient_adapter import EfficientNetAdapter
from models.necks.fpn_pan import FeatureProcessingNeck
from utils.logger import SmartCashLogger

class TestModelComponents:
    @pytest.fixture
    def config_path(self):
        """Fixture untuk path konfigurasi"""
        return Path(__file__).parent.parent / 'configs' / 'base_config.yaml'
    
    @pytest.fixture
    def sample_input(self):
        """Fixture input sampel untuk model"""
        return torch.randn(1, 3, 640, 640)  # Sesuai konfigurasi preprocessing
    
    def test_efficientnet_adapter(self, sample_input):
        """Pengujian EfficientNetAdapter"""
        logger = SmartCashLogger(__name__)
        backbone = EfficientNetAdapter(logger=logger)
        
        # Ekstraksi fitur
        features = backbone(sample_input)
        
        assert len(features) == 3, "Harus ada 3 feature map"
        
        # Periksa dimensi channel
        expected_channels = [56, 160, 448]
        for i, feature in enumerate(features):
            assert feature.shape[1] == expected_channels[i], f"Channel stage {i+3} tidak sesuai"
    
    def test_feature_processing_neck(self, sample_input):
        """Pengujian FeatureProcessingNeck"""
        # Siapkan backbone untuk mendapatkan fitur
        backbone = EfficientNetAdapter()
        backbone_features = backbone(sample_input)
        
        neck = FeatureProcessingNeck(
            in_channels=[56, 160, 448],
            out_channels=[128, 256, 512]
        )
        
        # Gabungkan fitur melalui neck
        neck_features = neck.fpn(backbone_features)
        neck_features = neck.pan(neck_features)
        
        assert len(neck_features) == 3, "Harus ada 3 feature map setelah pemrosesan neck"
        
        # Periksa dimensi channel output
        expected_channels = [128, 256, 512]
        for i, feature in enumerate(neck_features):
            assert feature.shape[1] == expected_channels[i], f"Channel output stage {i+3} tidak sesuai"
    
    def test_yolov5_efficient(self, config_path, sample_input):
        """Pengujian arsitektur YOLOv5 dengan EfficientNet backbone"""
        # Baca konfigurasi untuk mendapatkan jumlah kelas
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        num_classes = len(config['dataset']['classes'])
        
        model = YOLOv5Efficient(num_classes=num_classes)
        
        # Forward pass
        predictions = model(sample_input)
        
        assert len(predictions) == 3, "Harus ada 3 prediksi dari detection heads"
        
        # Periksa dimensi prediksi untuk setiap skala
        expected_channels = [128, 256, 512]
        for i, pred in enumerate(predictions):
            # Periksa bentuk prediksi
            assert pred.ndim == 5, f"Prediksi skala {i} harus 5 dimensi"
            assert pred.shape[1] == 3, f"Jumlah anchor di skala {i} harus 3"
            assert pred.shape[4] == 5 + num_classes, f"Ukuran channel prediksi di skala {i} tidak sesuai"
    
    def test_model_loss_computation(self, config_path, sample_input):
        """Pengujian komputasi loss"""
        # Baca konfigurasi
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        num_classes = len(config['dataset']['classes'])
        
        model = YOLOv5Efficient(num_classes=num_classes)
        
        # Buat target palsu untuk pengujian
        target = torch.tensor([
            [0, 0.5, 0.5, 0.2, 0.2, 1],  # format: [class_id, x_center, y_center, width, height, confidence]
            [1, 0.3, 0.7, 0.1, 0.1, 1]
        ])
        
        # Prediksi model
        predictions = model(sample_input)
        
        try:
            loss = model.compute_loss(predictions, target)
            assert loss is not None, "Komputasi loss gagal"
        except Exception as e:
            pytest.fail(f"Kesalahan dalam komputasi loss: {str(e)}")