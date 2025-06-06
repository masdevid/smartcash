"""
File: smartcash/ui/pretrained_model/utils/model_utils.py
Deskripsi: Utilities untuk model operations dengan config integration dan one-liner style
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
        """Extract model config dari UI components dengan fallback"""
        return {
            'models_dir': ui_components.get('models_dir_input', {}).get('value', '/content/models'),
            'drive_models_dir': ui_components.get('drive_models_dir_input', {}).get('value', '/content/drive/MyDrive/SmartCash/models'),
            'models': {
                'yolov5': {'url': ui_components.get('yolov5_url_input', {}).get('value', ''), 
                          **get_model_configs().get('yolov5', {})},
                'efficientnet_b4': {'url': ui_components.get('efficientnet_url_input', {}).get('value', ''), 
                                   **get_model_configs().get('efficientnet_b4', {})}
            }
        }

class ProgressTracker:
    """Lightweight progress tracker wrapper untuk existing progress_tracker"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.progress_tracker = ui_components.get('progress_tracker', {}).get('tracker')
    
    def next_step(self, step_name: str, message: str) -> None:
        """Move ke step berikutnya dengan message"""
        self.progress_tracker and hasattr(self.progress_tracker, 'update_status') and self.progress_tracker.update_status(message)
    
    def update_current_step(self, progress: int, message: str) -> None:
        """Update current step progress"""
        self.progress_tracker and hasattr(self.progress_tracker, 'update_step') and self.progress_tracker.update_step(progress, message)