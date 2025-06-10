"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Completed progress utilities dengan full Progress Bridge callback integration dan streamlined backend sync
"""

from typing import Dict, Any, Callable, Optional
from smartcash.common.logger import get_logger

class ProgressBridgeManager:
    """🌉 Enhanced manager untuk Progress Bridge callback system"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = get_logger('progress_utils')
        self._callback_registered = False
        
    def create_backend_callback(self) -> Callable[[str, int, int, str], None]:
        """📊 Create backend-compatible progress callback dengan dual tracker integration"""
        def backend_progress_callback(level: str, current: int, total: int, message: str):
            try:
                # Map backend levels ke UI dual tracker
                if level in ['overall', 'step']:
                    self._update_overall_progress(current, total, message)
                elif level in ['current', 'batch', 'file']:
                    self._update_current_progress(current, total, message)
                else:
                    # Default mapping
                    self._update_overall_progress(current, total, message)
                
                # Log milestone progress (prevent flooding)
                if self._is_milestone_progress(current, total):
                    self.logger.info(f"🔄 {message} ({current}/{total})")
                    
            except Exception as e:
                self.logger.debug(f"Progress callback error: {str(e)}")
        
        self._callback_registered = True
        return backend_progress_callback
    
    def _update_overall_progress(self, current: int, total: int, message: str):
        """📊 Update overall progress tracker"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_overall'):
            progress_pct = int((current / max(total, 1)) * 100)
            progress_tracker.update_overall(progress_pct, message)
    
    def _update_current_progress(self, current: int, total: int, message: str):
        """⚡ Update current operation progress tracker"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_current'):
            progress_pct = int((current / max(total, 1)) * 100)
            progress_tracker.update_current(progress_pct, f"Processing: {current}/{total}")
    
    def _is_milestone_progress(self, current: int, total: int) -> bool:
        """🎯 Check if progress adalah milestone untuk prevent log flooding"""
        if total <= 10:
            return True  # Log semua untuk small tasks
        
        milestones = [0, 10, 25, 50, 75, 90, 100]
        progress_pct = (current / total) * 100
        return any(abs(progress_pct - milestone) < 1 for milestone in milestones) or current == total
    
    def setup_for_operation(self, operation_name: str = "Processing"):
        """🚀 Setup progress tracking untuk backend operation"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
        
        self.logger.info(f"🚀 Setup progress tracking untuk {operation_name}")
    
    def complete_operation(self, message: str = "Operation completed"):
        """✅ Complete progress tracking dengan success status"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete(message)
            else:
                progress_tracker.update_overall(100, f"✅ {message}")
        
        self.logger.success(f"✅ {message}")
    
    def error_operation(self, error_msg: str = "Operation failed"):
        """❌ Set error state pada progress tracking"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            else:
                progress_tracker.update_overall(0, f"❌ {error_msg}")
        
        self.logger.error(f"❌ {error_msg}")
    
    def reset_tracking(self):
        """🔄 Reset progress tracking state"""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        
        self.logger.info("🔄 Progress tracking direset")

def create_dual_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """🏭 Factory untuk create dual progress callback dengan backend compatibility"""
    manager = ProgressBridgeManager(ui_components)
    callback = manager.create_backend_callback()
    
    # Store manager reference untuk future operations
    ui_components['progress_manager'] = manager
    return callback

def setup_dual_progress_tracker(ui_components: Dict[str, Any], operation_name: str = "Dataset Preprocessing"):
    """🚀 Setup dual progress tracker dengan enhanced backend integration"""
    manager = ui_components.get('progress_manager')
    if manager:
        manager.setup_for_operation(operation_name)
    else:
        # Fallback setup
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")

def complete_progress_tracker(ui_components: Dict[str, Any], message: str):
    """✅ Complete dual progress tracker dengan enhanced feedback"""
    manager = ui_components.get('progress_manager')
    if manager:
        manager.complete_operation(message)
    else:
        # Fallback completion
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete(message)
            else:
                progress_tracker.update_overall(100, f"✅ {message}")

def error_progress_tracker(ui_components: Dict[str, Any], error_msg: str):
    """❌ Set error state pada dual progress tracker"""
    manager = ui_components.get('progress_manager')
    if manager:
        manager.error_operation(error_msg)
    else:
        # Fallback error handling
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            else:
                progress_tracker.update_overall(0, f"❌ {error_msg}")

def reset_progress_tracker(ui_components: Dict[str, Any]):
    """🔄 Reset dual progress tracker state"""
    manager = ui_components.get('progress_manager')
    if manager:
        manager.reset_tracking()
    else:
        # Fallback reset
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()

def register_progress_callback_to_backend(ui_components: Dict[str, Any], backend_service) -> bool:
    """🔗 Register progress callback ke backend service dengan validation"""
    try:
        callback = ui_components.get('progress_callback')
        if not callback:
            # Create callback jika belum ada
            callback = create_dual_progress_callback(ui_components)
            ui_components['progress_callback'] = callback
        
        # Register ke backend service
        if hasattr(backend_service, 'register_progress_callback'):
            backend_service.register_progress_callback(callback)
            get_logger('progress_utils').info("🔗 Progress callback registered to backend")
            return True
        elif hasattr(backend_service, 'progress_callback'):
            backend_service.progress_callback = callback
            get_logger('progress_utils').info("🔗 Progress callback set on backend")
            return True
        else:
            get_logger('progress_utils').warning("⚠️ Backend service tidak support progress callback")
            return False
            
    except Exception as e:
        get_logger('progress_utils').error(f"❌ Error registering progress callback: {str(e)}")
        return False

def create_progress_reporter(ui_components: Dict[str, Any], operation_type: str = "processing") -> Callable:
    """📈 Create progress reporter function dengan operation-specific mapping"""
    manager = ui_components.get('progress_manager') or ProgressBridgeManager(ui_components)
    
    def report_progress(current: int, total: int, message: str = None, level: str = "current"):
        """Report progress dengan smart level mapping"""
        try:
            # Operation-specific message formatting
            if not message:
                message = f"{operation_type.capitalize()}: {current}/{total}"
            
            # Map ke appropriate level
            if level == "overall" or operation_type in ["validation", "setup"]:
                manager._update_overall_progress(current, total, message)
            else:
                manager._update_current_progress(current, total, message)
                
        except Exception as e:
            get_logger('progress_utils').debug(f"Progress report error: {str(e)}")
    
    return report_progress

def sync_progress_with_config_operations(ui_components: Dict[str, Any], config_handler) -> bool:
    """🔄 Sync progress tracking dengan config handler operations"""
    try:
        # Set progress callback ke config handler
        if hasattr(config_handler, 'set_progress_callback'):
            callback = ui_components.get('progress_callback')
            if not callback:
                callback = create_dual_progress_callback(ui_components)
                ui_components['progress_callback'] = callback
            
            config_handler.set_progress_callback(callback)
            get_logger('progress_utils').info("🔄 Config handler progress sync enabled")
            return True
            
    except Exception as e:
        get_logger('progress_utils').warning(f"⚠️ Config progress sync warning: {str(e)}")
        return False

# Backward compatibility functions (streamlined)
def update_progress_simple(ui_components: Dict[str, Any], current: int, total: int, message: str = None):
    """🔄 Simple progress update untuk backward compatibility"""
    manager = ui_components.get('progress_manager')
    if manager:
        manager._update_current_progress(current, total, message or f"Processing {current}/{total}")

def is_milestone_step(step: str, progress: int) -> bool:
    """🎯 Backward compatibility untuk milestone checking"""
    return progress in [0, 25, 50, 75, 100] or 'error' in step.lower()

# One-liner utilities
create_progress_callback = lambda ui_components: create_dual_progress_callback(ui_components)
get_progress_manager = lambda ui_components: ui_components.get('progress_manager') or ProgressBridgeManager(ui_components)
is_progress_available = lambda ui_components: 'progress_tracker' in ui_components
setup_progress = lambda ui_components, name="Processing": setup_dual_progress_tracker(ui_components, name)
complete_progress = lambda ui_components, msg="Completed": complete_progress_tracker(ui_components, msg)
error_progress = lambda ui_components, msg="Error": error_progress_tracker(ui_components, msg)