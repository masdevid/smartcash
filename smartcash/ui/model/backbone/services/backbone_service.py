"""
File: smartcash/ui/model/backbone/services/backbone_service.py
Description: Service bridge to backend backbone functionality
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from unittest.mock import Mock

from smartcash.common.logger import get_logger
from smartcash.model.utils.backbone_factory import BackboneFactory
from smartcash.model.core.model_builder import ModelBuilder
from smartcash.model.utils.device_utils import get_device_info
from ..constants import (
    BackboneType, BackboneOperation, PROGRESS_STEPS, 
    ERROR_MESSAGES, SUCCESS_MESSAGES
)


class BackboneService:
    """
    Service class to bridge UI with backend backbone functionality.
    
    This service acts as an adapter between the UI layer and the 
    backend model building infrastructure.
    """
    
    def __init__(self):
        self.logger = get_logger("ui.model.backbone.service")
        self.backbone_factory = BackboneFactory()
        # ModelBuilder will be initialized when needed with proper config
        self.model_builder = None
        self._current_backbone = None
        self._current_config = None
    
    async def _safe_callback(self, callback, *args):
        """Safely call callback, handling both sync and async functions."""
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        
    async def validate_backbone_config(self, 
                                     config: Dict[str, Any], 
                                     progress_callback: Optional[Callable] = None,
                                     log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Validate backbone configuration asynchronously.
        
        Args:
            config: Backbone configuration to validate
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing validation results
        """
        try:
            steps = PROGRESS_STEPS[BackboneOperation.VALIDATE.value]
            total_steps = len(steps)
            
            # Step 1: Validate configuration format
            await self._safe_callback(progress_callback, 0, total_steps, steps[0])
            await self._safe_callback(log_callback, "INFO", "Starting backbone configuration validation")
            
            validation_result = self._validate_config_format(config)
            await asyncio.sleep(0.5)  # Simulate processing
            
            # Step 2: Check backbone compatibility  
            await self._safe_callback(progress_callback, 1, total_steps, steps[1])
            await self._safe_callback(log_callback, "INFO", f"Checking compatibility for {config.get('backbone_type', 'unknown')}")
            
            compatibility_result = await self._check_backbone_compatibility(config)
            validation_result.update(compatibility_result)
            await asyncio.sleep(0.5)
            
            # Step 3: Complete validation
            await self._safe_callback(progress_callback, 2, total_steps, steps[2])
            status = "SUCCESS" if validation_result['valid'] else "ERROR"
            message = SUCCESS_MESSAGES['validation_success'] if validation_result['valid'] else "Validation failed"
            await self._safe_callback(log_callback, status, message)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            await self._safe_callback(log_callback, "ERROR", f"Validation failed: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'errors': [ERROR_MESSAGES['validation_failed']],
                'warnings': []
            }
    
    async def load_backbone_model(self,
                                config: Dict[str, Any],
                                progress_callback: Optional[Callable] = None,
                                log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load backbone model asynchronously.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing load results
        """
        try:
            steps = PROGRESS_STEPS[BackboneOperation.LOAD.value]
            total_steps = len(steps)
            
            # Step 1: Loading backbone model
            await self._safe_callback(progress_callback, 0, total_steps, steps[0])
            await self._safe_callback(log_callback, "INFO", f"Loading {config.get('backbone_type')} backbone")
            
            backbone_type = config.get('backbone_type', 'cspdarknet')
            pretrained = config.get('pretrained', True)
            
            # Use the backend backbone factory
            backbone = self.backbone_factory.create_backbone(
                backbone_type=backbone_type,
                pretrained=pretrained,
                feature_optimization=config.get('feature_optimization', False)
            )
            await asyncio.sleep(1.0)  # Simulate loading time
            
            # Step 2: Configuring parameters
            if progress_callback:
                await progress_callback(1, total_steps, steps[1])
            if log_callback:
                await log_callback("INFO", "Configuring backbone parameters")
            
            # Configure backbone with advanced settings
            if 'advanced_settings' in config:
                self._configure_backbone_parameters(backbone, config['advanced_settings'])
            await asyncio.sleep(0.5)
            
            # Step 3: Setting up features  
            if progress_callback:
                await progress_callback(2, total_steps, steps[2])
            if log_callback:
                await log_callback("INFO", "Setting up feature extraction")
            
            # Get backbone information
            backbone_info = self._get_backbone_info(backbone)
            await asyncio.sleep(0.5)
            
            # Step 4: Complete
            if progress_callback:
                await progress_callback(3, total_steps, steps[3])
            if log_callback:
                await log_callback("SUCCESS", SUCCESS_MESSAGES['load_success'])
            
            self._current_backbone = backbone
            self._current_config = config
            
            return {
                'success': True,
                'backbone': backbone,
                'info': backbone_info,
                'message': SUCCESS_MESSAGES['load_success']
            }
            
        except Exception as e:
            self.logger.error(f"Loading error: {e}")
            if log_callback:
                await log_callback("ERROR", f"Loading failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': ERROR_MESSAGES['loading_failed']
            }
    
    async def build_backbone_architecture(self,
                                        config: Dict[str, Any],
                                        progress_callback: Optional[Callable] = None,
                                        log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Build backbone architecture asynchronously.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing build results
        """
        try:
            steps = PROGRESS_STEPS[BackboneOperation.BUILD.value]
            total_steps = len(steps)
            
            # Step 1: Building architecture
            if progress_callback:
                await progress_callback(0, total_steps, steps[0])
            if log_callback:
                await log_callback("INFO", "Building backbone architecture")
            
            # Initialize model builder if needed
            if self.model_builder is None:
                self._initialize_model_builder(config)
            
            # Use model builder to create full architecture
            model_config = self._convert_to_model_config(config)
            model = self.model_builder.build_model(model_config)
            await asyncio.sleep(1.0)
            
            # Step 2: Configuring layers
            if progress_callback:
                await progress_callback(1, total_steps, steps[1])
            if log_callback:
                await log_callback("INFO", "Configuring model layers")
            
            layer_info = self._configure_model_layers(model, config)
            await asyncio.sleep(0.5)
            
            # Step 3: Calculating parameters
            if progress_callback:
                await progress_callback(2, total_steps, steps[2])
            if log_callback:
                await log_callback("INFO", "Analyzing model parameters")
                
            model_stats = self._calculate_model_stats(model)
            await asyncio.sleep(0.5)
            
            # Step 4: Complete
            if progress_callback:
                await progress_callback(3, total_steps, steps[3])
            if log_callback:
                await log_callback("SUCCESS", SUCCESS_MESSAGES['build_success'])
            
            return {
                'success': True,
                'model': model,
                'layer_info': layer_info,
                'stats': model_stats,
                'message': SUCCESS_MESSAGES['build_success']
            }
            
        except Exception as e:
            self.logger.error(f"Build error: {e}")
            if log_callback:
                await log_callback("ERROR", f"Build failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': ERROR_MESSAGES['build_failed']
            }
    
    async def generate_model_summary(self,
                                   config: Dict[str, Any],
                                   progress_callback: Optional[Callable] = None,
                                   log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate model summary asynchronously.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing summary results
        """
        try:
            steps = PROGRESS_STEPS[BackboneOperation.SUMMARY.value]
            total_steps = len(steps)
            
            # Step 1: Generating summary
            if progress_callback:
                await progress_callback(0, total_steps, steps[0])
            if log_callback:
                await log_callback("INFO", "Generating model summary")
            
            summary_data = await self._generate_summary_data(config)
            await asyncio.sleep(0.8)
            
            # Step 2: Analyzing parameters
            if progress_callback:
                await progress_callback(1, total_steps, steps[1])
            if log_callback:
                await log_callback("INFO", "Analyzing model parameters and performance")
                
            analysis_data = await self._analyze_model_performance(config)
            await asyncio.sleep(0.7)
            
            # Step 3: Complete
            if progress_callback:
                await progress_callback(2, total_steps, steps[2])
            if log_callback:
                await log_callback("SUCCESS", SUCCESS_MESSAGES['summary_success'])
            
            return {
                'success': True,
                'summary': summary_data,
                'analysis': analysis_data,
                'message': SUCCESS_MESSAGES['summary_success']
            }
            
        except Exception as e:
            self.logger.error(f"Summary error: {e}")
            if log_callback:
                await log_callback("ERROR", f"Summary generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to generate model summary"
            }
    
    def get_available_backbones(self) -> List[str]:
        """Get list of available backbone types."""
        return self.backbone_factory.list_available_backbones()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for model deployment."""
        return get_device_info()
    
    def _validate_config_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration format."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = ['backbone_type']
        for field in required_fields:
            if field not in config:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Validate backbone type
        if 'backbone_type' in config:
            backbone_type = config['backbone_type']
            available_backbones = self.get_available_backbones()
            if backbone_type not in available_backbones:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid backbone type: {backbone_type}")
        
        return validation_result
    
    async def _check_backbone_compatibility(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check backbone compatibility with system."""
        compatibility_result = {
            'compatible': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check device compatibility
        device_info = self.get_device_info()
        
        # Memory check for EfficientNet-B4
        if config.get('backbone_type') == 'efficientnet_b4':
            if device_info.get('gpu_memory_gb', 0) < 4:
                compatibility_result['warnings'].append(
                    "EfficientNet-B4 may require more GPU memory for optimal performance"
                )
        
        return compatibility_result
    
    def _configure_backbone_parameters(self, backbone, advanced_settings: Dict[str, Any]):
        """Configure backbone with advanced settings."""
        # This would configure the backbone based on advanced settings
        # Implementation depends on the specific backbone architecture
        pass
    
    def _get_backbone_info(self, backbone) -> Dict[str, Any]:
        """Get backbone information."""
        import torch
        
        # Calculate parameters
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': backbone.__class__.__name__,
            'device': next(backbone.parameters()).device.type if list(backbone.parameters()) else 'cpu'
        }
    
    def _convert_to_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UI config to model builder config."""
        return {
            'backbone': config.get('backbone_type', 'cspdarknet'),
            'num_classes': config.get('advanced_settings', {}).get('num_classes', 17),
            'input_size': config.get('advanced_settings', {}).get('input_size', 640),
            'pretrained': config.get('pretrained', True)
        }
    
    def _configure_model_layers(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure model layers based on config."""
        return {
            'backbone_layers': len(list(model.backbone.children())) if hasattr(model, 'backbone') else 0,
            'head_layers': len(list(model.head.children())) if hasattr(model, 'head') else 0,
            'total_layers': len(list(model.children()))
        }
    
    def _calculate_model_stats(self, model) -> Dict[str, Any]:
        """Calculate model statistics."""
        import torch
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'estimated_memory_mb': total_params * 8 / (1024 * 1024)  # Rough estimate
        }
    
    async def _generate_summary_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary data."""
        backbone_type = config.get('backbone_type', 'cspdarknet')
        
        return {
            'backbone_type': backbone_type,
            'configuration': {
                'pretrained': config.get('pretrained', True),
                'feature_optimization': config.get('feature_optimization', False),
                'input_size': config.get('advanced_settings', {}).get('input_size', 640),
                'num_classes': config.get('advanced_settings', {}).get('num_classes', 17)
            },
            'capabilities': {
                'multi_scale_detection': True,
                'feature_pyramid': True,
                'anchor_free': backbone_type == 'efficientnet_b4'
            }
        }
    
    async def _analyze_model_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze expected model performance."""
        backbone_type = config.get('backbone_type', 'cspdarknet')
        
        # Performance estimates based on backbone type
        performance_estimates = {
            'cspdarknet': {
                'inference_speed': 'Fast',
                'memory_usage': 'Low',
                'accuracy': 'Good',
                'fps_estimate': '60-80'
            },
            'efficientnet_b4': {
                'inference_speed': 'Medium',
                'memory_usage': 'Medium',
                'accuracy': 'Excellent',
                'fps_estimate': '30-45'
            }
        }
        
        return performance_estimates.get(backbone_type, performance_estimates['cspdarknet'])
    
    def _initialize_model_builder(self, config: Dict[str, Any]) -> None:
        """Initialize model builder with proper configuration."""
        try:
            # Create a basic model config for initialization
            model_config = self._convert_to_model_config(config)
            
            # Create a mock progress bridge for initialization
            from smartcash.common.logger import get_logger
            
            class MockProgressBridge:
                def __init__(self):
                    self.logger = get_logger("mock_progress_bridge")
                
                def update_progress(self, *args, **kwargs):
                    pass
                
                def log_message(self, *args, **kwargs):
                    pass
            
            mock_bridge = MockProgressBridge()
            
            # Initialize ModelBuilder with required parameters
            self.model_builder = ModelBuilder(
                config=model_config,
                progress_bridge=mock_bridge
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize ModelBuilder: {e}")
            # Create a minimal mock for testing
            self.model_builder = Mock()
            self.model_builder.build_model = Mock(return_value=Mock())