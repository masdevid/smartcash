"""
File: smartcash/ui/pretrained_model/utils/model_utils.py
Deskripsi: Reusable utilities untuk reduce duplication dalam model operations
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
from smartcash.ui.pretrained_model.constants.model_constants import MODEL_CONFIGS, DEFAULT_MODELS_DIR, PROGRESS_STEPS

class ModelUtils:
    """Utility class untuk common model operations"""
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get model config dari constants - single source"""
        return MODEL_CONFIGS.get(model_name, {})
    
    @staticmethod
    def get_all_model_names() -> List[str]:
        """Get semua model names - single source"""
        return list(MODEL_CONFIGS.keys())
    
    @staticmethod
    def validate_model_file(file_path: Path, model_name: str) -> bool:
        """Validate model file existence dan size"""
        if not file_path.exists():
            return False
        
        config = ModelUtils.get_model_config(model_name)
        min_size = config.get('min_size_mb', 1) * 1024 * 1024
        return file_path.stat().st_size >= min_size
    
    @staticmethod
    def get_model_file_path(model_name: str, models_dir: str = None) -> Path:
        """Get full path untuk model file"""
        models_dir = models_dir or DEFAULT_MODELS_DIR
        config = ModelUtils.get_model_config(model_name)
        filename = config.get('filename', f'{model_name}.pt')
        return Path(models_dir) / filename
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size dalam MB dengan 1 decimal"""
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    @staticmethod
    def get_progress_step(step_name: str) -> Tuple[int, str]:
        """Get progress step dari constants"""
        return PROGRESS_STEPS.get(step_name, (0, 'Unknown step'))

class ProgressTracker:
    """Utility untuk step-by-step progress tracking"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.current_step = 0
        self.total_steps = 0
    
    def start_process(self, process_name: str, total_steps: int = 7):
        """Start process dengan initial progress"""
        self.total_steps = total_steps
        self.current_step = 0
        self._update_progress('INIT', f"Memulai {process_name}")
    
    def next_step(self, step_name: str, custom_message: str = None):
        """Move ke next step dengan progress update"""
        self.current_step += 1
        progress, message = ModelUtils.get_progress_step(step_name)
        
        # Override message jika ada custom message
        if custom_message:
            message = custom_message
        
        # Update progress dengan step info
        final_message = f"Step {self.current_step}/{self.total_steps}: {message}"
        self._update_progress_value(progress, final_message)
    
    def update_current_step(self, sub_progress: int, sub_message: str):
        """Update progress dalam step yang sama"""
        # Hitung progress berdasarkan current step + sub progress
        base_progress = (self.current_step - 1) / self.total_steps * 100
        step_progress = sub_progress / self.total_steps
        total_progress = min(100, base_progress + step_progress)
        
        self._update_progress_value(int(total_progress), sub_message)
    
    def complete_process(self, message: str = "Proses selesai"):
        """Complete process dengan 100% progress"""
        self._update_progress('COMPLETE', message)
        self.ui_components.get('complete_operation', lambda x: None)(message)
    
    def error_process(self, error_message: str):
        """Handle process error"""
        self.ui_components.get('error_operation', lambda x: None)(error_message)
    
    def _update_progress(self, step_name: str, message: str):
        """Update progress menggunakan step constants"""
        progress, _ = ModelUtils.get_progress_step(step_name)
        self._update_progress_value(progress, message)
    
    def _update_progress_value(self, progress: int, message: str):
        """Update progress dengan nilai dan message"""
        self.ui_components.get('update_progress', lambda *a: None)('overall', progress, message)

class StatusUpdater:
    """Utility untuk consistent status panel updates"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def update(self, message: str, status_type: str = "info"):
        """Update status panel dengan formatting konsisten"""
        from smartcash.ui.components.status_panel import update_status_panel
        if 'status_panel' in self.ui_components:
            update_status_panel(self.ui_components['status_panel'], message, status_type)
    
    def success(self, message: str):
        """Update dengan success status"""
        self.update(message, "success")
    
    def error(self, message: str):
        """Update dengan error status"""
        self.update(message, "error")
    
    def info(self, message: str):
        """Update dengan info status"""
        self.update(message, "info")
    
    def warning(self, message: str):
        """Update dengan warning status"""
        self.update(message, "warning")

class UILogger:
    """Utility untuk UI logger operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def reset(self):
        """Reset UI logger dan clear outputs"""
        for key in ['log_output', 'status']:
            widget = self.ui_components.get(key)
            if widget and hasattr(widget, 'clear_output'):
                widget.clear_output(wait=True)
        self.ui_components.get('reset_all', lambda: None)()
    
    def log(self, message: str, level: str = "info"):
        """Log message dengan level"""
        logger = self.ui_components.get('logger')
        if logger:
            getattr(logger, level, logger.info)(message)