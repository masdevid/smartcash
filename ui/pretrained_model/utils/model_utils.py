"""
File: smartcash/ui/pretrained_model/utils/model_utils.py
Deskripsi: Fixed utilities dengan simplified progress tracker untuk avoid weak reference error
"""

from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.constants.model_constants import get_model_configs

class ModelUtils:
    """Utilities untuk model operations dengan dynamic config"""
    
    @staticmethod
    def get_all_model_names() -> List[str]:
        """Get semua model names dari config"""
        return list(get_model_configs().keys())
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get config untuk specific model"""
        return get_model_configs().get(model_name, {})
    
    @staticmethod
    def get_model_file_path(model_name: str, models_dir: str) -> Path:
        """Get file path untuk model dengan one-liner validation"""
        config = ModelUtils.get_model_config(model_name)
        return Path(models_dir) / config.get('filename', f'{model_name}.pt')
    
    @staticmethod
    def validate_model_file(file_path: Path, model_name: str) -> bool:
        """Validate model file existence dan size dengan one-liner checks"""
        config = ModelUtils.get_model_config(model_name)
        min_size = config.get('min_size_mb', 1) * 1024 * 1024
        return (file_path.exists() and file_path.stat().st_size >= min_size 
               if file_path.exists() else False)
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size dengan human readable format"""
        return (f"{size_bytes / (1024**3):.1f} GB" if size_bytes >= 1024**3 
               else f"{size_bytes / (1024**2):.1f} MB" if size_bytes >= 1024**2 
               else f"{size_bytes / 1024:.1f} KB" if size_bytes >= 1024 
               else f"{size_bytes} B")
    
    @staticmethod
    def get_models_from_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model config dari UI components dengan safe attribute access"""
        return {
            'models_dir': getattr(ui_components.get('models_dir_input'), 'value', '/content/models'),
            'drive_models_dir': getattr(ui_components.get('drive_models_dir_input'), 'value', '/content/drive/MyDrive/SmartCash/models'),
            'models': {
                'yolov5': {'url': getattr(ui_components.get('yolov5_url_input'), 'value', ''), 
                          **get_model_configs().get('yolov5', {})},
                'efficientnet_b4': {'url': getattr(ui_components.get('efficientnet_url_input'), 'value', ''), 
                                   **get_model_configs().get('efficientnet_b4', {})}
            }
        }

class ProgressTracker:
    """Simplified progress tracker untuk avoid weak reference error"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        
    def next_step(self, step_name: str, message: str) -> None:
        """Move ke step berikutnya dengan message - delegasi ke UI components"""
        update_fn = self.ui_components.get('update_progress')
        update_fn and update_fn('overall', 0, message)
        
        logger = self.ui_components.get('logger')
        logger and logger.info(f"📋 {step_name}: {message}")
    
    def update_current_step(self, progress: int, message: str) -> None:
        """Update current step progress - delegasi ke UI components"""
        update_fn = self.ui_components.get('update_progress')
        update_fn and update_fn('current', progress, message)
        
        # Optional: log significant progress updates
        progress % 25 == 0 and progress > 0 and self.ui_components.get('logger') and self.ui_components['logger'].info(f"⏳ Progress: {progress}%")