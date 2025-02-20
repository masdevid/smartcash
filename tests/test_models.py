# File: tests/test_models.py
# Author: Alfrida Sabar
# Deskripsi: Pengujian unit untuk komponen model dalam proyek SmartCash

import pytest
import torch
import yaml
from pathlib import Path

from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.backbones.cspdarknet import CSPDarknet
from smartcash.models.backbones.efficientnet import EfficientNetBackbone
from smartcash.models.detection_head import DetectionHead
from smartcash.utils.logger import SmartCashLogger

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
        backbone = EfficientNetBackbone(logger=logger)
        
        # Ekstraksi fitur
        features = backbone(sample_input)
        
        assert len(features) == 3, "Harus ada 3 feature map"
        
        # Periksa dimensi channel
        expected_channels = [56, 160, 448]
        for i, feature in enumerate(features):
            assert feature.shape[1] == expected_channels[i], f"Channel stage {i+3} tidak sesuai"
    
    def test_cspdarknet_backbone(self, sample_input):
        """Pengujian CSPDarknet backbone"""
        backbone = CSPDarknet()
        features = backbone(sample_input)
        
        assert len(features) == 3, "Harus ada 3 feature map"
        expected_channels = [128, 256, 512]
        for i, feature in enumerate(features):
            assert feature.shape[1] == expected_channels[i], f"Channel stage {i+3} tidak sesuai"
    
    def test_detection_head_single_layer(self, sample_input):
        """Pengujian detection head untuk single layer"""
        # Setup input channels
        in_channels = [256, 512, 1024]
        
        # Single layer head
        head = DetectionHead(in_channels=in_channels)
        
        # Generate sample feature maps
        features = [
            torch.randn(1, ch, 80//s, 80//s)
            for ch, s in zip(in_channels, [1, 2, 4])
        ]
        
        # Forward pass
        predictions = head(features)
        
        # Periksa struktur output
        assert isinstance(predictions, dict), "Output harus berupa dict"
        assert "banknote" in predictions, "Harus ada layer banknote"
        assert len(predictions["banknote"]) == 3, "Harus ada 3 skala prediksi"
    
    def test_detection_head_multi_layer(self, sample_input):
        """Pengujian detection head untuk multi layer"""
        in_channels = [256, 512, 1024]
        layers = ["banknote", "nominal"]
        
        head = DetectionHead(
            in_channels=in_channels,
            layers=layers
        )
        
        features = [
            torch.randn(1, ch, 80//s, 80//s)
            for ch, s in zip(in_channels, [1, 2, 4])
        ]
        
        predictions = head(features)
        
        assert set(predictions.keys()) == set(layers), "Output harus sesuai layer yang diminta"
        for layer in layers:
            assert len(predictions[layer]) == 3, f"Layer {layer} harus memiliki 3 skala"
    
    def test_yolov5_single_layer(self, sample_input):
        """Pengujian YOLOv5 dengan single layer detection"""
        model = YOLOv5Model(
            num_classes=7,
            backbone_type="cspdarknet"
        )
        
        predictions = model(sample_input)
        
        # Single layer output
        assert isinstance(predictions, dict), "Output harus berupa dict"
        assert "banknote" in predictions, "Harus ada layer banknote"
        assert len(predictions["banknote"]) == 3, "Harus ada 3 skala prediksi"
    
    def test_yolov5_multi_layer(self, sample_input):
        """Pengujian YOLOv5 dengan multi layer detection"""
        layers = ["banknote", "nominal"]
        model = YOLOv5Model(
            backbone_type="efficientnet",
            layers=layers
        )
        
        predictions = model(sample_input)
        
        assert set(predictions.keys()) == set(layers), "Output harus sesuai layer yang diminta"
        for layer in layers:
            assert len(predictions[layer]) == 3, f"Layer {layer} harus memiliki 3 skala"
    
    def test_model_loss_computation(self, config_path, sample_input):
        """Pengujian komputasi loss"""
        # Baca konfigurasi
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = YOLOv5Model(num_classes=7)
        
        # Buat target palsu untuk pengujian
        target = torch.tensor([
            [0, 0.5, 0.5, 0.2, 0.2, 1],  # format: [class_id, x_center, y_center, width, height, confidence]
            [1, 0.3, 0.7, 0.1, 0.1, 1]
        ])
        
        # Prediksi model
        predictions = model(sample_input)
        loss = model.compute_loss(predictions, target)
        
        assert 'total_loss' in loss, "Harus ada total loss"
        assert 'box_loss' in loss, "Harus ada box loss"
        assert 'obj_loss' in loss, "Harus ada objectness loss"
        assert 'cls_loss' in loss, "Harus ada classification loss"
        assert all(v >= 0 for v in loss.values()), "Semua loss harus non-negatif"