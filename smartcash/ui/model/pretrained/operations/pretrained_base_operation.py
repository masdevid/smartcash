"""
Base operation class for pretrained model operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from smartcash.ui.model.pretrained.constants import EXPECTED_FILE_SIZES, PretrainedModelType
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.core.mixins.operation_mixin import OperationMixin


class PretrainedBaseOperation(OperationMixin, LoggingMixin, ABC):
    """Base class for all pretrained model operations."""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize the base operation.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        self.ui_components = ui_components
        self.config = config
        self.models_dir = config.get('models_dir', '/data/pretrained')
        
    # Logging is now handled by LoggingMixin
    
    @abstractmethod
    def execute_operation(self) -> Dict[str, Any]:
        """Execute the operation.
        
        Returns:
            Operation result dictionary
        """
        pass
    
    # ==================== SHARED HELPER METHODS ====================
    
    def check_existing_models(self, models_dir: str) -> Dict[str, Any]:
        """Check which models already exist in the directory."""
        existing_models = {}
        
        try:
            if not os.path.exists(models_dir):
                return existing_models
            
            # Check for YOLOv5s
            yolo_path = os.path.join(models_dir, 'yolov5s.pt')
            if os.path.exists(yolo_path):
                existing_models['yolov5s'] = {
                    'path': yolo_path,
                    'size': os.path.getsize(yolo_path)
                }
            
            # Check for EfficientNet-B4
            efficientnet_path = os.path.join(models_dir, 'efficientnet_b4.pth')
            if os.path.exists(efficientnet_path):
                existing_models['efficientnet_b4'] = {
                    'path': efficientnet_path,
                    'size': os.path.getsize(efficientnet_path)
                }
            
        except Exception as e:
            self.log(f"Error checking existing models: {e}", 'warning')
        
        return existing_models
    
    def validate_downloaded_models(self, models_dir: str) -> Dict[str, Any]:
        """Validate downloaded model files."""
        validation_results = {}
        
        try:
            # Validate YOLOv5s
            yolo_path = os.path.join(models_dir, 'yolov5s.pt')
            if os.path.exists(yolo_path):
                file_size = os.path.getsize(yolo_path)
                expected_size = EXPECTED_FILE_SIZES[PretrainedModelType.YOLOV5S.value]
                size_ratio = file_size / expected_size if expected_size > 0 else 0
                
                validation_results['yolov5s'] = {
                    'valid': 0.8 <= size_ratio <= 1.2,  # 20% tolerance
                    'file_path': yolo_path,
                    'size': file_size,
                    'size_mb': file_size / (1024 * 1024),
                    'expected_size': expected_size,
                    'size_ratio': size_ratio
                }
            else:
                validation_results['yolov5s'] = {
                    'valid': False,
                    'file_path': yolo_path,
                    'error': 'File not found'
                }
            
            # Validate EfficientNet-B4
            efficientnet_path = os.path.join(models_dir, 'efficientnet_b4.pth')
            if os.path.exists(efficientnet_path):
                file_size = os.path.getsize(efficientnet_path)
                expected_size = EXPECTED_FILE_SIZES[PretrainedModelType.EFFICIENTNET_B4.value]
                size_ratio = file_size / expected_size if expected_size > 0 else 0
                
                validation_results['efficientnet_b4'] = {
                    'valid': file_size > 1024,  # At least 1KB - timm models vary in size
                    'file_path': efficientnet_path,
                    'size': file_size,
                    'size_mb': file_size / (1024 * 1024),
                    'expected_size': expected_size,
                    'size_ratio': size_ratio
                }
            else:
                validation_results['efficientnet_b4'] = {
                    'valid': False,
                    'file_path': efficientnet_path,
                    'error': 'File not found'
                }
            
        except Exception as e:
            self.log(f"Error validating models: {e}", 'warning')
        
        return validation_results
    
    def scan_model_directory(self, models_dir: str) -> list:
        """Scan model directory for all model files."""
        model_files = []
        
        try:
            if not os.path.exists(models_dir):
                return model_files
            
            # Look for common model file extensions
            model_extensions = ['.pt', '.pth', '.onnx', '.pb', '.tflite']
            
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in model_extensions):
                        file_path = os.path.join(root, file)
                        model_files.append({
                            'name': file,
                            'path': file_path,
                            'size': os.path.getsize(file_path),
                            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                        })
            
        except Exception as e:
            self.log(f"Error scanning model directory: {e}", 'warning')
        
        return model_files