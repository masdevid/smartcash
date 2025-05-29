"""
File: smartcash/model/services/model_loader.py
Deskripsi: Service loader untuk bridge existing model services dengan training UI
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger


class ModelServiceLoader:
    """Service loader untuk integrate existing model services dengan training UI"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger('model_loader')
        self._services_cache = {}
    
    def get_model_manager(self, config: Dict[str, Any] = None, model_type: str = 'efficient_optimized') -> Any:
        """Get model manager dari existing model structure"""
        try:
            # Import existing model manager
            from smartcash.model.manager import ModelManager
            
            cache_key = f"model_manager_{model_type}"
            if cache_key not in self._services_cache:
                manager = ModelManager(config=config, model_type=model_type, logger=self.logger)
                self._services_cache[cache_key] = manager
                self.logger.info(f"✅ Model manager loaded untuk {model_type}")
            
            return self._services_cache[cache_key]
            
        except ImportError as e:
            self.logger.error(f"❌ Model manager import error: {str(e)}")
            return self._create_fallback_model_manager(config, model_type)
    
    def get_checkpoint_manager(self, model_manager, checkpoint_dir: str = "runs/train/checkpoints") -> Any:
        """Get checkpoint manager dari existing structure"""
        try:
            from smartcash.model.manager_checkpoint import ModelCheckpointManager
            
            cache_key = f"checkpoint_manager_{checkpoint_dir}"
            if cache_key not in self._services_cache:
                manager = ModelCheckpointManager(
                    model_manager=model_manager,
                    checkpoint_dir=checkpoint_dir,
                    logger=self.logger
                )
                self._services_cache[cache_key] = manager
                self.logger.info(f"✅ Checkpoint manager loaded")
            
            return self._services_cache[cache_key]
            
        except ImportError as e:
            self.logger.error(f"❌ Checkpoint manager import error: {str(e)}")
            return self._create_fallback_checkpoint_manager(model_manager, checkpoint_dir)
    
    def get_training_service(self, config: Dict[str, Any] = None) -> Any:
        """Get training service dari existing structure"""
        try:
            # Try existing training services
            from smartcash.model.services.training.core_training_service import CoreTrainingService
            
            if 'training_service' not in self._services_cache:
                service = CoreTrainingService(config=config, logger=self.logger)
                self._services_cache['training_service'] = service
                self.logger.info(f"✅ Core training service loaded")
            
            return self._services_cache['training_service']
            
        except ImportError:
            try:
                # Fallback ke training service yang sudah ada
                from smartcash.common.services.training.training_service import TrainingService
                
                if 'fallback_training_service' not in self._services_cache:
                    service = TrainingService(config=config)
                    self._services_cache['fallback_training_service'] = service
                    self.logger.info(f"✅ Fallback training service loaded")
                
                return self._services_cache['fallback_training_service']
                
            except ImportError as e:
                self.logger.error(f"❌ Training service import error: {str(e)}")
                return self._create_fallback_training_service(config)
    
    def get_evaluation_service(self, config: Dict[str, Any] = None) -> Any:
        """Get evaluation service dari existing structure"""
        try:
            from smartcash.model.services.evaluation.core_evaluation_service import CoreEvaluationService
            
            if 'evaluation_service' not in self._services_cache:
                service = CoreEvaluationService(config=config, logger=self.logger)
                self._services_cache['evaluation_service'] = service
                self.logger.info(f"✅ Evaluation service loaded")
            
            return self._services_cache['evaluation_service']
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Evaluation service tidak tersedia: {str(e)}")
            return self._create_fallback_evaluation_service(config)
    
    def get_prediction_service(self, config: Dict[str, Any] = None) -> Any:
        """Get prediction service dari existing structure"""
        try:
            from smartcash.model.services.prediction.core_prediction_service import CorePredictionService
            
            if 'prediction_service' not in self._services_cache:
                service = CorePredictionService(config=config, logger=self.logger)
                self._services_cache['prediction_service'] = service
                self.logger.info(f"✅ Prediction service loaded")
            
            return self._services_cache['prediction_service']
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Prediction service tidak tersedia: {str(e)}")
            return self._create_fallback_prediction_service(config)
    
    def _create_fallback_model_manager(self, config: Dict[str, Any], model_type: str):
        """Create fallback model manager jika import gagal"""
        class FallbackModelManager:
            def __init__(self, config, model_type, logger):
                self.config = config or {}
                self.model_type = model_type
                self.logger = logger
                self.model = None
            
            def build_model(self):
                self.logger.warning("⚠️ Using fallback model manager - model building simulated")
                return self
            
            def save_model(self, path: str):
                self.logger.warning(f"⚠️ Fallback save model: {path}")
                return path
            
            def load_model(self, path: str):
                self.logger.warning(f"⚠️ Fallback load model: {path}")
                return self
        
        return FallbackModelManager(config, model_type, self.logger)
    
    def _create_fallback_checkpoint_manager(self, model_manager, checkpoint_dir: str):
        """Create fallback checkpoint manager"""
        class FallbackCheckpointManager:
            def __init__(self, model_manager, checkpoint_dir, logger):
                self.model_manager = model_manager
                self.checkpoint_dir = checkpoint_dir
                self.logger = logger
            
            def save_checkpoint(self, model=None, path="checkpoint.pt", **kwargs):
                self.logger.warning(f"⚠️ Fallback checkpoint save: {path}")
                return path
            
            def load_checkpoint(self, path: str, **kwargs):
                self.logger.warning(f"⚠️ Fallback checkpoint load: {path}")
                return {}
            
            def get_best_checkpoint(self):
                return None
            
            def get_latest_checkpoint(self):
                return None
        
        return FallbackCheckpointManager(model_manager, checkpoint_dir, self.logger)
    
    def _create_fallback_training_service(self, config: Dict[str, Any]):
        """Create fallback training service"""
        class FallbackTrainingService:
            def __init__(self, config, logger):
                self.config = config or {}
                self.logger = logger
                self._training_active = False
            
            def start_training(self, **kwargs):
                self.logger.warning("⚠️ Fallback training service - simulation only")
                return True
            
            def stop_training(self):
                self.logger.info("⚠️ Fallback training stop")
            
            def is_training_active(self):
                return self._training_active
            
            def get_training_status(self):
                return {'active': False, 'fallback': True}
        
        return FallbackTrainingService(config, self.logger)
    
    def _create_fallback_evaluation_service(self, config: Dict[str, Any]):
        """Create fallback evaluation service"""
        class FallbackEvaluationService:
            def __init__(self, config, logger):
                self.config = config or {}
                self.logger = logger
            
            def evaluate(self, **kwargs):
                self.logger.warning("⚠️ Fallback evaluation service")
                return {'map': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        return FallbackEvaluationService(config, self.logger)
    
    def _create_fallback_prediction_service(self, config: Dict[str, Any]):
        """Create fallback prediction service"""
        class FallbackPredictionService:
            def __init__(self, config, logger):
                self.config = config or {}
                self.logger = logger
            
            def predict(self, **kwargs):
                self.logger.warning("⚠️ Fallback prediction service")
                return {'detections': []}
        
        return FallbackPredictionService(config, self.logger)
    
    def clear_cache(self):
        """Clear services cache"""
        self._services_cache.clear()
        self.logger.info("🧹 Model services cache cleared")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status semua loaded services"""
        return {
            'loaded_services': list(self._services_cache.keys()),
            'total_services': len(self._services_cache),
            'available_loaders': [
                'model_manager', 'checkpoint_manager', 'training_service',
                'evaluation_service', 'prediction_service'
            ]
        }


# Singleton instance untuk reuse across training modules
_model_service_loader = None

def get_model_service_loader(logger=None) -> ModelServiceLoader:
    """Get singleton model service loader"""
    global _model_service_loader
    if _model_service_loader is None:
        _model_service_loader = ModelServiceLoader(logger)
    return _model_service_loader


# One-liner utilities untuk quick access
load_model_manager = lambda config=None, model_type='efficient_optimized': get_model_service_loader().get_model_manager(config, model_type)
load_checkpoint_manager = lambda model_manager, checkpoint_dir="runs/train/checkpoints": get_model_service_loader().get_checkpoint_manager(model_manager, checkpoint_dir)
load_training_service = lambda config=None: get_model_service_loader().get_training_service(config)
load_evaluation_service = lambda config=None: get_model_service_loader().get_evaluation_service(config)
load_prediction_service = lambda config=None: get_model_service_loader().get_prediction_service(config)