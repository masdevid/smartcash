"""
SmartCash Training Pipeline with YOLOv5 Integration
Multi-Layer Banknote Detection Training System
"""

import time
import json
import torch
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.api.core import create_api
from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline
from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.model.training.visualization_manager import create_visualization_manager
from smartcash.model.training.utils.progress_tracker import UnifiedProgressTracker
from smartcash.model.training.utils.summary_utils import generate_markdown_summary
from smartcash.model.training.utils.checkpoint_utils import generate_checkpoint_name, save_checkpoint_to_disk
from smartcash.model.training.utils.resume_utils import (
    handle_resume_training_pipeline, setup_training_session, validate_training_mode_and_params
)
from smartcash.model.training.utils.setup_utils import prepare_training_environment
from smartcash.model.training.training_phase_manager import TrainingPhaseManager
from smartcash.model.utils.device_utils import setup_device, model_to_device
from smartcash.model.utils.memory_optimizer import get_memory_optimizer, cleanup_training_memory, emergency_cleanup

logger = get_logger(__name__)


class TrainingPipeline(UnifiedTrainingPipeline):
    """
    SmartCash training pipeline with YOLOv5 integration
    Multi-layer banknote detection training system
    """
    
    def __init__(self, use_yolov5_integration: bool = True, **kwargs):
        """
        Initialize SmartCash training pipeline with YOLOv5 integration
        
        Args:
            use_yolov5_integration: Enable YOLOv5 integration
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        
        # Store training mode for later progress tracker setup
        self._training_mode = None
        self.use_yolov5_integration = use_yolov5_integration
        self.model_api = None
        
        logger.info(f"🚀 SmartCash training pipeline initialized")
        logger.info(f"🔧 YOLOv5 integration: {'enabled' if use_yolov5_integration else 'disabled'}")
    
    def run_full_training_pipeline(self,
                                  backbone: str = 'cspdarknet',
                                  pretrained: bool = True,
                                  phase_1_epochs: int = 1,
                                  phase_2_epochs: int = 1,
                                  checkpoint_dir: str = 'data/checkpoints',
                                  resume_from_checkpoint: bool = True,
                                  force_cpu: bool = False,
                                  training_mode: str = 'two_phase',
                                  single_phase_layer_mode: str = 'multi',
                                  single_phase_freeze_backbone: bool = False,
                                  model: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        SmartCash training pipeline with YOLOv5 architecture
        
        Args:
            backbone: Backbone type ('cspdarknet', 'efficientnet_b4')
            pretrained: Use pretrained weights
            phase_1_epochs: Epochs for phase 1 (frozen backbone)
            phase_2_epochs: Epochs for phase 2 (fine-tuning)
            checkpoint_dir: Directory for checkpoints
            resume_from_checkpoint: Enable checkpoint resuming
            force_cpu: Force CPU training
            training_mode: 'two_phase' or 'single_phase'
            single_phase_layer_mode: Layer mode for single phase
            single_phase_freeze_backbone: Freeze backbone in single phase
            model: Model configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Training result dictionary
        """
        try:
            # Generate training session ID
            self.training_session_id = str(uuid.uuid4())[:8]
            self.training_start_time = time.time()
            
            # Set up enhanced progress tracker with training mode
            if self._training_mode != training_mode:
                self._training_mode = training_mode
                # Get the progress callback from existing progress tracker
                current_callback = getattr(self.progress_tracker, 'progress_callback', None)
                self.progress_tracker = UnifiedProgressTracker(
                    progress_callback=current_callback,
                    verbose=self.verbose,
                    training_mode=training_mode
                )
            
            # Log training start
            if self.log_callback:
                self.log_callback('info', f"🚀 Starting training pipeline", {
                    'session_id': self.training_session_id,
                    'backbone': backbone,
                    'training_mode': training_mode
                })
            
            # Validate and setup training parameters
            training_config = self._setup_training_config(
                backbone=backbone,
                pretrained=pretrained,
                phase_1_epochs=phase_1_epochs,
                phase_2_epochs=phase_2_epochs,
                checkpoint_dir=checkpoint_dir,
                training_mode=training_mode,
                single_phase_layer_mode=single_phase_layer_mode,
                single_phase_freeze_backbone=single_phase_freeze_backbone,
                model=model,
                force_cpu=force_cpu,
                **kwargs
            )
            
            # Phase 1: Preparation
            self.progress_tracker.start_phase('preparation', 5)
            preparation_result = self._phase_preparation(
                backbone=training_config['backbone'],
                pretrained=training_config['pretrained'],
                phase_1_epochs=training_config['phase_1_epochs'],
                phase_2_epochs=training_config['phase_2_epochs'],
                checkpoint_dir=str(training_config['checkpoint_dir']),
                resume_from_checkpoint=training_config.get('resume_from_checkpoint', True),
                force_cpu=training_config['force_cpu'],
                training_mode=training_config['training_mode'],
                single_phase_layer_mode=training_config.get('single_phase_layer_mode', 'multi'),
                single_phase_freeze_backbone=training_config.get('single_phase_freeze_backbone', False),
                model=training_config['model']
            )
            if not preparation_result['success']:
                return preparation_result
            
            # Phase 2: Build Model
            self.progress_tracker.start_phase('build_model', 4)
            build_result = self._phase_build_model()
            if not build_result['success']:
                return build_result
            
            # Phase 3: Validate Model
            self.progress_tracker.start_phase('validate_model', 3)
            validate_result = self._phase_validate_model()
            if not validate_result['success']:
                return validate_result
            
            # Phase 4 & 5: Training phases
            training_result = self._run_training_phases(training_config)
            if not training_result['success']:
                return training_result
            
            # Final Phase: Finalize (Summary & Visualization)
            self.progress_tracker.start_phase('finalize', 3)
            summary_result = self._phase_finalize(training_result)
            
            # Final success result
            return {
                'success': True,
                'session_id': self.training_session_id,
                'architecture_type': 'yolov5',
                'training_duration': time.time() - self.training_start_time,
                'checkpoint_path': training_result.get('checkpoint_path'),
                'metrics': training_result.get('final_metrics', {}),
                'model_info': build_result.get('model_info', {}),
                'summary_path': summary_result.get('summary_path')
            }
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            
            # Emergency cleanup
            try:
                emergency_cleanup()
            except:
                pass
            
            if self.log_callback:
                self.log_callback('error', f"Training pipeline failed: {str(e)}", {
                    'session_id': self.training_session_id,
                    'error_type': type(e).__name__
                })
            
            return {
                'success': False,
                'error': str(e),
                'session_id': self.training_session_id
            }
    
    def _setup_training_config(self, **kwargs) -> Dict[str, Any]:
        """Setup training configuration"""
        
        # Base configuration
        config = {
            'backbone': kwargs.get('backbone', 'cspdarknet'),
            'pretrained': kwargs.get('pretrained', True),
            'training_mode': kwargs.get('training_mode', 'two_phase'),
            'phase_1_epochs': kwargs.get('phase_1_epochs', 1),
            'phase_2_epochs': kwargs.get('phase_2_epochs', 1),
            'checkpoint_dir': Path(kwargs.get('checkpoint_dir', 'data/checkpoints')),
            'force_cpu': kwargs.get('force_cpu', False),
            'session_id': self.training_session_id
        }
        
        # Model configuration
        model_config = kwargs.get('model', {})
        config['model'] = {
            'model_name': model_config.get('model_name', 'smartcash_yolov5_integrated'),
            'backbone': config['backbone'],
            'pretrained': config['pretrained'],
            'layer_mode': model_config.get('layer_mode', 'multi'),
            'detection_layers': model_config.get('detection_layers', ['layer_1', 'layer_2', 'layer_3']),
            'num_classes': model_config.get('num_classes', 7),
            'img_size': model_config.get('img_size', 640),
            'feature_optimization': model_config.get('feature_optimization', {'enabled': True})
        }
        
        # Training phases configuration for TrainingPhaseManager
        config['training_phases'] = {
            'phase_1': {
                'learning_rate': 0.001,
                'freeze_backbone': True,
                'layer_mode': 'single',
                'description': 'Frozen backbone training'
            },
            'phase_2': {
                'learning_rate': 0.0001,
                'freeze_backbone': False,
                'layer_mode': 'multi',
                'description': 'Fine-tuning with multi-layer'
            }
        }
        
        # Training configuration
        config['training'] = {
            'mixed_precision': False,  # Disable for CPU
            'batch_size': 8,
            'num_workers': 0 if config['force_cpu'] else 4,
            'pin_memory': False,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001,
                'monitor': 'val_loss'
            }
        }
        
        # Paths configuration
        config['paths'] = {
            'checkpoints': str(config['checkpoint_dir']),
            'visualization': 'data/visualization',
            'logs': 'data/logs'
        }
        
        # Always use YOLOv5 architecture
        
        logger.info(f"🔧 Training configuration setup: {config['backbone']} | yolov5")
        return config
    
    def _phase_build_model(self) -> Dict[str, Any]:
        """Model building phase"""
        try:
            self.progress_tracker.update_phase(1, 4, "🔧 Initializing model API...")
            
            # Create SmartCash API
            self.model_api = create_api(
                config=self.config,
                use_yolov5_integration=self.use_yolov5_integration
            )
            
            model_config = self.config.get('model', {})
            
            self.progress_tracker.update_phase(2, 4, f"🏗️ Building yolov5 model...")
            
            # Build model
            build_result = self.model_api.build_model(
                model_config=model_config
            )
            
            if not build_result['success']:
                return build_result
            
            self.model = build_result['model']
            # Model API is already set
            
            self.progress_tracker.update_phase(3, 4, "🔧 Setting up training components...")
            
            # Use device from model API build result (already transferred)
            device_str = build_result.get('device', 'cpu')
            device = torch.device(device_str)
            
            # Model is already on the correct device from build_model
            self.memory_optimizer = get_memory_optimizer(device)
            
            self.progress_tracker.update_phase(4, 4, "✅ Model building complete")
            
            if self.log_callback:
                self.log_callback('info', f"Model built successfully", {
                    'architecture_type': 'yolov5',
                    'backbone': self.config.get('backbone', 'unknown'),
                    'parameters': build_result.get('model_info', {}).get('total_parameters', 0)
                })
            
            return {
                'success': True,
                'model': self.model,
                'model_info': build_result.get('model_info', {}),
                'architecture_type': 'yolov5',
                'device': str(device)
            }
            
        except Exception as e:
            logger.error(f"❌ Build phase failed: {str(e)}")
            return {
                'success': False,
                'error': f"Model building failed: {str(e)}"
            }
    
    def _run_training_phases(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training phases with architecture-specific handling"""
        
        # Use training phase manager if available
        try:
            from smartcash.model.training.training_phase_manager import TrainingPhaseManager as DefaultTrainingPhaseManager
            
            phase_manager = DefaultTrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback
            )
            
        except ImportError:
            # Fallback to standard training phase manager
            logger.warning("Training phase manager not available, using standard manager")
            
            phase_manager = TrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback  # This might not be correct, but provides compatibility
            )
        
        # Run training phases - use the standard training phase methods
        if config['training_mode'] == 'two_phase':
            # Phase 1: Frozen backbone training
            self.progress_tracker.start_phase('training_phase_1', 6)
            phase1_result = phase_manager.run_training_phase(
                phase_num=1, 
                epochs=config['phase_1_epochs'], 
                start_epoch=0
            )
            
            if not phase1_result.get('success', False):
                return phase1_result
            
            # Phase 2: Fine-tuning training
            self.progress_tracker.start_phase('training_phase_2', 6)
            phase2_result = phase_manager.run_training_phase(
                phase_num=2, 
                epochs=config['phase_2_epochs'], 
                start_epoch=0
            )
            
            return phase2_result
        else:
            # Single phase training
            self.progress_tracker.start_phase('training_phase_single', 6)
            return phase_manager.run_training_phase(
                phase_num=1, 
                epochs=config.get('phase_1_epochs', 10), 
                start_epoch=0
            )
    
    
    def _phase_validate_model(self) -> Dict[str, Any]:
        """Run validation phase with model validation"""
        try:
            self.progress_tracker.update_phase(1, 3, "🔍 Validating model...")
            
            # Use API validation if available
            if self.model_api:
                validation_result = self.model_api.validate_model()
                
                if not validation_result['success']:
                    return {
                        'success': False,
                        'error': f"Model validation failed: {validation_result.get('error')}"
                    }
                
                self.progress_tracker.update_phase(2, 3, "📊 Analyzing model architecture...")
                
                # Log validation results
                if self.log_callback:
                    self.log_callback('info', "Model validation successful", {
                        'architecture_type': 'yolov5',
                        'validation_info': validation_result
                    })
            
            self.progress_tracker.update_phase(3, 3, "✅ Model validation complete")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                'success': False,
                'error': f"Model validation failed: {str(e)}"
            }
    
    def _phase_finalize(self, training_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run summary phase with model information"""
        try:
            self.progress_tracker.update_phase(1, 3, "📊 Generating summary...")
            
            # Get model summary
            model_summary = {}
            if self.model_api:
                model_summary = self.model_api.get_model_summary()
            
            # Generate summary with architecture information
            summary_data = {
                'session_id': self.training_session_id,
                'architecture_type': 'yolov5',
                'model_summary': model_summary,
                'training_config': self.config,
                'training_results': training_result or {},
                'total_duration': time.time() - self.training_start_time
            }
            
            # Use parent implementation for most of the summary generation
            summary_result = super()._phase_summary_visualization()
            
            # Enhance with architecture-specific information
            if summary_result.get('success'):
                summary_result['architecture_type'] = 'yolov5'
                summary_result['model_summary'] = model_summary
            
            return summary_result
            
        except Exception as e:
            logger.error(f"❌ Summary phase failed: {str(e)}")
            return {
                'success': False,
                'error': f"Summary generation failed: {str(e)}"
            }


def run_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Main entry point for training pipeline
    
    Args:
        **kwargs: Training configuration arguments
        
    Returns:
        Training result dictionary
    """
    # Extract pipeline-specific arguments
    use_yolov5_integration = kwargs.pop('use_yolov5_integration', True)
    progress_callback = kwargs.pop('progress_callback', None)
    log_callback = kwargs.pop('log_callback', None)
    metrics_callback = kwargs.pop('metrics_callback', None)
    verbose = kwargs.pop('verbose', True)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        use_yolov5_integration=use_yolov5_integration,
        progress_callback=progress_callback,
        log_callback=log_callback,
        metrics_callback=metrics_callback,
        verbose=verbose
    )
    
    return pipeline.run_full_training_pipeline(**kwargs)


# Maintain compatibility with existing API
def run_full_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Compatibility wrapper for existing training pipeline API
    Automatically uses pipeline with YOLOv5 integration
    
    Args:
        **kwargs: Training configuration arguments
        
    Returns:
        Training result dictionary
    """
    return run_training_pipeline(**kwargs)


# Export key functions
__all__ = [
    'TrainingPipeline',
    'run_training_pipeline',
    'run_full_training_pipeline'
]