"""
File: smartcash/model/api/core.py
Deskripsi: API inti untuk operasi model SmartCash dengan progress tracker integration
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.model.core.model_builder import ModelBuilder
from smartcash.model.core.checkpoint_manager import CheckpointManager
from smartcash.model.utils.device_utils import setup_device, get_device_info
from smartcash.model.utils.progress_bridge import ModelProgressBridge

class SmartCashModelAPI:
    """🎯 API inti untuk operasi model SmartCash dengan progress tracking"""
    
    def __init__(self, config_path: Optional[str] = None, progress_callback: Optional[Callable] = None):
        """Inisialisasi API dengan konfigurasi dan progress callback"""
        self.logger = get_logger("model.api")
        self.progress_bridge = ModelProgressBridge(progress_callback)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.device = setup_device(self.config.get('device', {}))
        
        # Initialize core components
        self.model_builder = ModelBuilder(self.config, self.progress_bridge)
        self.checkpoint_manager = CheckpointManager(self.config, self.progress_bridge)
        
        # Model state
        self.model = None
        self.is_model_built = False
        
        self.logger.info(f"🚀 SmartCash Model API initialized | Device: {self.device}")
        self._log_device_info()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load konfigurasi model dengan fallback ke default"""
        default_config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'
        config_file = config_path or default_config_path
        
        try:
            if not Path(config_file).exists():
                self.logger.warning(f"⚠️ Config tidak ditemukan: {config_file}, menggunakan default")
                return self._get_default_config()
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"📁 Config loaded: {Path(config_file).name}")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration jika file tidak tersedia"""
        return {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_name': 'smartcash_yolov5',
                'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                'layer_mode': 'multi',
                'multi_layer_heads': True,
                'num_classes': {
                    'layer_1': 7,   # 7 denominations
                    'layer_2': 7,   # 7 denomination-specific features  
                    'layer_3': 3    # 3 common features
                },
                'img_size': 640,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'feature_optimization': {'enabled': True}
            },
            'data': {
                'pretrained_dir': '/data/pretrained',
                'dataset_dir': '/data/preprocessed',
                'batch_size': 16,
                'num_workers': 4
            },
            'device': {'auto_detect': True, 'preferred': 'cuda'},
            'checkpoint': {'save_dir': '/data/checkpoints', 'max_checkpoints': 5}
        }
    
    def build_model(self, **kwargs) -> Dict[str, Any]:
        """🏗️ Build model dengan konfigurasi yang diberikan"""
        try:
            self.progress_bridge.start_operation("Building Model", 4)
            
            # Update config dengan kwargs
            if kwargs:
                self._update_config(kwargs)
                self.progress_bridge.update(1, "📝 Config updated")
            
            # Build model menggunakan ModelBuilder
            self.progress_bridge.update(2, "🔧 Initializing model components...")
            self.model = self.model_builder.build(**self.config['model'])
            
            # Transfer ke device
            self.progress_bridge.update(3, f"📱 Transferring to {self.device}...")
            self.model = self.model.to(self.device)
            
            # Finalize
            self.is_model_built = True
            self.progress_bridge.complete(4, "✅ Model built successfully!")
            
            # Return model info
            return self._get_model_info()
            
        except Exception as e:
            error_msg = f"❌ Model building failed: {str(e)}"
            self.logger.error(error_msg)
            self.progress_bridge.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """📂 Load model dari checkpoint"""
        try:
            if not self.is_model_built:
                self.build_model(**kwargs)
            
            result = self.checkpoint_manager.load_checkpoint(self.model, checkpoint_path)
            self.logger.info(f"✅ Checkpoint loaded: {result['checkpoint_path']}")
            return result
            
        except Exception as e:
            error_msg = f"❌ Checkpoint loading failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def save_checkpoint(self, metrics: Optional[Dict] = None, **kwargs) -> str:
        """💾 Save model checkpoint"""
        try:
            if not self.is_model_built:
                raise RuntimeError("❌ Model belum dibangun, tidak dapat menyimpan checkpoint")
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, metrics=metrics, **kwargs
            )
            self.logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            error_msg = f"❌ Checkpoint saving failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """📋 List available checkpoints"""
        return self.checkpoint_manager.list_checkpoints()
    
    def predict(self, input_data: Union[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """🔮 Prediksi menggunakan model"""
        try:
            if not self.is_model_built:
                raise RuntimeError("❌ Model belum dibangun")
            
            self.progress_bridge.start_operation("Running Prediction", 3)
            
            # Preprocess input
            self.progress_bridge.update(1, "🔄 Preprocessing input...")
            processed_input = self._preprocess_input(input_data)
            
            # Run inference
            self.progress_bridge.update(2, "🧠 Running inference...")
            with torch.no_grad():
                predictions = self.model(processed_input)
            
            # Postprocess results
            self.progress_bridge.update(3, "📊 Processing results...")
            results = self._postprocess_predictions(predictions)
            
            self.progress_bridge.complete(3, "✅ Prediction completed!")
            return results
            
        except Exception as e:
            error_msg = f"❌ Prediction failed: {str(e)}"
            self.logger.error(error_msg)
            self.progress_bridge.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """ℹ️ Informasi detail model"""
        return self._get_model_info()
    
    def _update_config(self, updates: Dict[str, Any]) -> None:
        """Update konfigurasi dengan nested dict support"""
        def update_nested(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_nested(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.logger.debug(f"📝 Before update - backbone: {self.config.get('model', {}).get('backbone', 'N/A')}")
        self.logger.debug(f"📝 Updates received: {updates}")
        update_nested(self.config, updates)
        self.logger.debug(f"📝 After update - backbone: {self.config.get('model', {}).get('backbone', 'N/A')}")
        self.logger.debug("📝 Configuration updated")
    
    def _preprocess_input(self, input_data: Union[str, torch.Tensor]) -> torch.Tensor:
        """Preprocess input untuk inference"""
        if isinstance(input_data, str):
            # Load image from path
            from PIL import Image
            import torchvision.transforms as transforms
            
            image = Image.open(input_data).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((self.config['model']['img_size'], self.config['model']['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(image).unsqueeze(0)
        elif isinstance(input_data, torch.Tensor):
            tensor = input_data
        else:
            raise ValueError("❌ Input harus berupa path string atau torch.Tensor")
        
        return tensor.to(self.device)
    
    def _postprocess_predictions(self, predictions: Dict) -> Dict[str, Any]:
        """Postprocess predictions menjadi format yang mudah dibaca"""
        results = {
            'predictions': predictions,
            'num_detections': sum(len(pred) for pred_list in predictions.values() for pred in pred_list),
            'layers': list(predictions.keys()),
            'confidence_threshold': self.config['model']['confidence_threshold'],
            'device': str(self.device)
        }
        return results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Informasi detail model"""
        if not self.is_model_built:
            return {'status': 'not_built', 'message': 'Model belum dibangun'}
        
        # Hitung parameter
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'status': 'built',
            'model_name': self.config['model']['model_name'],
            'backbone': self.config['model']['backbone'],
            'layer_mode': self.config['model']['layer_mode'],
            'detection_layers': self.config['model']['detection_layers'],
            'num_classes': self.config['model']['num_classes'],
            'img_size': self.config['model']['img_size'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'feature_optimization': self.config['model']['feature_optimization'] if isinstance(self.config['model']['feature_optimization'], bool) else self.config['model']['feature_optimization']['enabled'],
            'memory_usage': f"{torch.cuda.memory_allocated(self.device) / 1024**2:.1f} MB" if self.device.type == 'cuda' else 'N/A'
        }
    
    def _log_device_info(self) -> None:
        """Log informasi device"""
        device_info = get_device_info()
        if device_info['cuda_available']:
            self.logger.info(f"🎮 GPU: {device_info['gpu_name']} | Memory: {device_info['gpu_memory_gb']:.1f}GB")
        else:
            self.logger.info("💻 Running on CPU")


# Factory functions untuk convenience
def create_model_api(config_path: Optional[str] = None, progress_callback: Optional[Callable] = None) -> SmartCashModelAPI:
    """Factory function untuk membuat SmartCashModelAPI"""
    return SmartCashModelAPI(config_path, progress_callback)

def quick_build_model(backbone: str = 'efficientnet_b4', progress_callback: Optional[Callable] = None) -> SmartCashModelAPI:
    """Quick build model dengan backbone tertentu"""
    api = create_model_api(progress_callback=progress_callback)
    api.build_model(model={'backbone': backbone})
    return api