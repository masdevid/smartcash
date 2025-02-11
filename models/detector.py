# File: src/models/detector.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi SmartCash Detector dengan YOLOv5 dan EfficientNet-B4

import os
import torch
import yaml
from torch.utils.data import DataLoader
from utils.logging import ColoredLogger
from data.dataset import RupiahDataset
from .efficientnet_backbone import SmartCashYOLODetector
from .train import Trainer

class SmartCashDetector:
    def __init__(self, backbone=None, weights_path=None, img_size=640, nc=7):
        self.logger = ColoredLogger('SmartCashDetector')
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set backbone
        self.backbone = backbone or SmartCashYOLODetector(nc=nc)
        
        # Initialize model
        if weights_path and os.path.exists(weights_path):
            self.model = torch.load(weights_path, map_location=self.device)
        else:
            self.model = SmartCashYOLODetector(nc=nc)
        
        self.model.to(self.device)
        self.logger.info(f'Model loaded to device: {self.device}')

    @property
    def backbone_info(self):
        """Dapatkan informasi detail tentang backbone"""
        if hasattr(self.backbone, 'width') and hasattr(self.backbone, 'depth'):
            return {
                'type': self.backbone.__class__.__name__,
                'width': self.backbone.width,
                'depth': self.backbone.depth
            }
        return {'type': self.backbone.__class__.__name__}

    def train(self, data_yaml, epochs=100, batch_size=16):
        self.logger.info('Memulai training...')
        
        with open(data_yaml) as f:
            data_dict = yaml.safe_load(f)
            
        # Setup data loaders
        train_dataset = RupiahDataset(data_dict['train'])
        val_dataset = RupiahDataset(data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        # Initialize trainer
        trainer = Trainer(self.model, train_loader, self.device)
        trainer.train(epochs, val_loader)
        
        self.logger.info('Training selesai!')

    def detect(self, img_path, conf_thres=0.25):
        self.logger.info('Mendeteksi nominal uang...')
        
        img = torch.from_numpy(img_path).to(self.device)
        img = img.float() / 255.0
        
        with torch.no_grad():
            pred = self.model(img)
        
        # Process predictions
        detections = []
        for *xyxy, conf, cls in pred:
            if conf > conf_thres:
                nominal = self.model.names[int(cls)]
                detections.append({
                    'bbox': xyxy,
                    'confidence': float(conf),
                    'nominal': nominal
                })
                self.logger.info(f'Terdeteksi Rp{nominal} ({conf:.2f})')
        
        if not detections:
            self.logger.warning('Tidak ada nominal yang terdeteksi')
        
        return detections

    def export_onnx(self, output_path='weights/model.onnx'):
        self.logger.info('Mengexport model ke ONNX...')
        
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        torch.onnx.export(self.model, dummy_input, output_path, 
                         opset_version=12, input_names=['input'],
                         output_names=['output'])
        
        self.logger.info('Model berhasil diexport ke ONNX')

if __name__ == '__main__':
    detector = SmartCashDetector()
    detector.train('data/rupiah.yaml')
    detector.detect('data/samples/test.jpg')