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
    
    def save_initial_model(self, metrics: Optional[Dict] = None, model_name: Optional[str] = None, **kwargs) -> str:
        """💾 Save initial built model to /data/models directory"""
        try:
            if not self.is_model_built:
                raise RuntimeError("❌ Model belum dibangun, tidak dapat menyimpan model")
            
            # Create models directory
            models_dir = Path('data/models')
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate model filename
            if model_name is None:
                model_name = f"{self.config['model']['backbone']}_backbone"
            
            current_date = datetime.now()
            filename = f"{model_name}_{current_date.strftime('%Y%m%d')}.pt"
            model_path = models_dir / filename
            
            self.progress_bridge.start_operation("Saving Initial Model", 3)
            
            # Prepare model data
            self.progress_bridge.update(1, "📦 Preparing model data...")
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'model_config': getattr(self.model, 'config', {}),
                'metrics': metrics or {},
                'timestamp': current_date.isoformat(),
                'torch_version': torch.__version__,
                'model_info': self._get_model_info_dict(),
                'model_type': 'initial_build'  # Mark as initial build
            }
            
            # Save model
            self.progress_bridge.update(2, f"💾 Saving to {filename}...")
            torch.save(model_data, model_path)
            
            self.progress_bridge.complete(3, f"✅ Initial model saved: {filename}")
            
            # Log save info
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"💾 Initial model saved: {filename} ({file_size:.1f}MB)")
            
            return str(model_path)
            
        except Exception as e:
            error_msg = f"❌ Initial model saving failed: {str(e)}"
            self.logger.error(error_msg)
            self.progress_bridge.error(error_msg)
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
        
        # Define model-related parameters that should be nested under 'model'
        model_params = {
            'backbone', 'layer_mode', 'detection_layers', 'multi_layer_heads', 
            'num_classes', 'img_size', 'pretrained', 'feature_optimization', 
            'mixed_precision', 'conf_threshold', 'iou_threshold', 'device'
        }
        
        self.logger.debug(f"📝 Before update - backbone: {self.config.get('model', {}).get('backbone', 'N/A')}")
        self.logger.debug(f"📝 Updates received: {updates}")
        
        # Check if updates contain flat model parameters
        flat_model_updates = {}
        other_updates = {}
        
        for k, v in updates.items():
            if k in model_params:
                flat_model_updates[k] = v
            else:
                other_updates[k] = v
        
        # Apply flat model parameters to the model section
        if flat_model_updates:
            if 'model' not in self.config:
                self.config['model'] = {}
            update_nested(self.config['model'], flat_model_updates)
            self.logger.debug(f"📝 Applied flat model updates: {flat_model_updates}")
        
        # Apply other updates normally
        if other_updates:
            update_nested(self.config, other_updates)
            self.logger.debug(f"📝 Applied other updates: {other_updates}")
        
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
    
    def _get_model_info_dict(self) -> Dict[str, Any]:
        """Get model info as dict for saving (without status)"""
        if not self.is_model_built:
            return {}
        
        # Hitung parameter
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config['model']['model_name'],
            'backbone': self.config['model']['backbone'],
            'layer_mode': self.config['model']['layer_mode'],
            'detection_layers': self.config['model']['detection_layers'],
            'num_classes': self.config['model']['num_classes'],
            'img_size': self.config['model']['img_size'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'feature_optimization': self.config['model']['feature_optimization'] if isinstance(self.config['model']['feature_optimization'], bool) else self.config['model']['feature_optimization']['enabled']
        }
    
    def _log_device_info(self) -> None:
        """Log informasi device"""
        device_info = get_device_info()
        if device_info['cuda_available']:
            self.logger.info(f"🎮 GPU: {device_info['gpu_name']} | Memory: {device_info['gpu_memory_gb']:.1f}GB")
        else:
            self.logger.info("💻 Running on CPU")


# Training API
def run_full_training_pipeline(backbone: str = 'cspdarknet',
                              phase_1_epochs: int = 1,
                              phase_2_epochs: int = 1,
                              checkpoint_dir: str = 'data/checkpoints',
                              progress_callback: Optional[Callable] = None,
                              log_callback: Optional[Callable] = None,
                              live_chart_callback: Optional[Callable] = None,
                              metrics_callback: Optional[Callable] = None,
                              verbose: bool = True,
                              force_cpu: bool = False,
                              training_mode: str = 'two_phase',
                              single_phase_layer_mode: str = 'multi',
                              single_phase_freeze_backbone: bool = False,
                              **kwargs) -> Dict[str, Any]:
    """
    Main API for running the complete unified training pipeline with UI callbacks.
    
    Args:
        backbone: Model backbone ('cspdarknet' or 'efficientnet_b4')
        phase_1_epochs: Number of epochs for phase 1 (frozen backbone)
        phase_2_epochs: Number of epochs for phase 2 (fine-tuning)
        checkpoint_dir: Directory for checkpoint management  
        progress_callback: Callback function for progress updates
        log_callback: Callback for logging events (level, message, data)
        live_chart_callback: Callback for live chart updates (chart_type, data, config)
        metrics_callback: Callback for metrics updates (phase, epoch, metrics)
        verbose: Enable verbose logging
        force_cpu: Force CPU usage instead of auto-detecting GPU/MPS (default: False)
        training_mode: Training mode ('single_phase', 'two_phase') (default: 'two_phase')
        single_phase_layer_mode: Layer mode for single-phase training ('single', 'multi') (default: 'multi')
        single_phase_freeze_backbone: Whether to freeze backbone in single-phase training (default: False)
        **kwargs: Additional configuration overrides
        
    Returns:
        Complete training results with all phase information
        
    Note:
        single_phase_layer_mode and single_phase_freeze_backbone parameters are only used
        when training_mode='single_phase' and are ignored in two_phase mode.
    """
    from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline
    
    pipeline = UnifiedTrainingPipeline(
        progress_callback=progress_callback,
        log_callback=log_callback,
        live_chart_callback=live_chart_callback,
        metrics_callback=metrics_callback,
        verbose=verbose
    )
    return pipeline.run_full_training_pipeline(
        backbone=backbone,
        phase_1_epochs=phase_1_epochs,
        phase_2_epochs=phase_2_epochs,
        checkpoint_dir=checkpoint_dir,
        force_cpu=force_cpu,
        training_mode=training_mode,
        single_phase_layer_mode=single_phase_layer_mode,
        single_phase_freeze_backbone=single_phase_freeze_backbone,
        **kwargs
    )

# Factory functions untuk convenience
def create_model_api(config_path: Optional[str] = None, progress_callback: Optional[Callable] = None) -> SmartCashModelAPI:
    """Factory function untuk membuat SmartCashModelAPI"""
    return SmartCashModelAPI(config_path, progress_callback)

def quick_build_model(backbone: str = 'efficientnet_b4', progress_callback: Optional[Callable] = None) -> SmartCashModelAPI:
    """Quick build model dengan backbone tertentu"""
    api = create_model_api(progress_callback=progress_callback)
    api.build_model(model={'backbone': backbone})
    return api