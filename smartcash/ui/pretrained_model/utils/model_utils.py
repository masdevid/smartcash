"""
File: smartcash/ui/pretrained_model/utils/model_utils.py
Deskripsi: Optimized utilities dengan enhanced progress tracker integration untuk backward compatibility
"""

from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.constants.model_constants import get_model_configs

class ModelUtils:
    """Utilities untuk model operations dengan dynamic config dan enhanced progress support"""
    
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

class ProgressHelper:
    """Helper untuk enhanced progress tracker dengan backward compatibility"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Detect progress tracker type dan setup delegates
        self._setup_progress_delegates()
    
    def _setup_progress_delegates(self):
        """Setup progress delegates berdasarkan tracker type yang tersedia"""
        # Check untuk enhanced progress tracker
        if 'update_overall' in self.ui_components:
            self.next_step = lambda step_name, message: self._enhanced_next_step(step_name, message)
            self.update_current_step = lambda progress, message: self._enhanced_update_current(progress, message)
        
        # Fallback untuk simple progress tracker
        elif 'update_progress' in self.ui_components:
            self.next_step = lambda step_name, message: self._simple_next_step(step_name, message)
            self.update_current_step = lambda progress, message: self._simple_update_current(progress, message)
        
        # No-op fallback jika tidak ada tracker
        else:
            self.next_step = lambda step_name, message: self._log_only(step_name, message)
            self.update_current_step = lambda progress, message: self._log_only(f"{progress}%", message)
    
    def _enhanced_next_step(self, step_name: str, message: str) -> None:
        """Enhanced progress tracker delegation"""
        self.ui_components.get('update_overall', lambda *a: None)(0, message)
        self.logger and self.logger.info(f"📋 {step_name}: {message}")
    
    def _enhanced_update_current(self, progress: int, message: str) -> None:
        """Enhanced progress tracker current update"""
        self.ui_components.get('update_current', lambda *a: None)(progress, message)
        # Log significant progress updates only
        progress % 25 == 0 and progress > 0 and self.logger and self.logger.info(f"⏳ Progress: {progress}%")
    
    def _simple_next_step(self, step_name: str, message: str) -> None:
        """Simple progress tracker delegation untuk backward compatibility"""
        self.ui_components.get('update_progress', lambda *a: None)('overall', 0, message)
        self.logger and self.logger.info(f"📋 {step_name}: {message}")
    
    def _simple_update_current(self, progress: int, message: str) -> None:
        """Simple progress tracker current update"""
        self.ui_components.get('update_progress', lambda *a: None)('current', progress, message)
        progress % 25 == 0 and progress > 0 and self.logger and self.logger.info(f"⏳ Progress: {progress}%")
    
    def _log_only(self, step_name: str, message: str) -> None:
        """Fallback ke log only jika tidak ada progress tracker"""
        self.logger and self.logger.info(f"📋 {step_name}: {message}")

# Backward compatibility alias untuk existing code
class ProgressTracker(ProgressHelper):
    """Alias untuk backward compatibility dengan existing code"""
    pass