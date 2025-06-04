"""
File: smartcash/ui/dataset/downloader/handlers/progress_handler.py
Deskripsi: Progress handler dengan callback management dan step tracking
"""

from typing import Dict, Any, Callable, Optional
import time
from smartcash.common.logger import get_logger

class ProgressCallbackManager:
    """Manager untuk progress callbacks dengan step tracking."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = get_logger('downloader.progress')
        self.current_step = 0
        self.total_steps = 5
        self.step_names = ['validate', 'connect', 'download', 'extract', 'organize']
        self.start_time = None
        self.step_start_time = None
    
    def get_progress_callback(self) -> Callable:
        """Get progress callback function untuk download service."""
        return self._progress_callback
    
    def _progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Progress callback implementation dengan step tracking."""
        try:
            # Map step ke step index
            step_index = self._get_step_index(step)
            
            # Update current step jika berubah
            if step_index != self.current_step:
                self._complete_current_step()
                self.current_step = step_index
                self._start_new_step(step, message)
            
            # Calculate overall progress
            overall_progress = self._calculate_overall_progress(current, total)
            
            # Update UI components
            self._update_progress_display(step, current, total, message, overall_progress)
            
            # Log progress (tidak terlalu verbose)
            self._log_progress(step, current, total, message)
            
        except Exception as e:
            self.logger.error(f"❌ Progress callback error: {str(e)}")
    
    def start_download_process(self) -> None:
        """Start download process tracking."""
        self.start_time = time.time()
        self.current_step = 0
        self.step_start_time = self.start_time
        
        # Show progress container
        self._show_progress_container()
        
        # Log start
        self.logger.info("🚀 Download process started")
    
    def complete_download_process(self, message: str = "Download completed") -> None:
        """Complete download process dengan success state."""
        # Complete final step
        self._complete_current_step()
        
        # Update progress to 100%
        self._update_overall_progress(100, f"✅ {message}")
        
        # Calculate total duration
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.success(f"🎉 {message} dalam {duration:.1f} detik")
        
        # Complete progress tracking
        self._complete_progress_tracking()
    
    def error_download_process(self, error_message: str) -> None:
        """Error state untuk download process."""
        # Update progress dengan error state
        self._update_progress_error(f"❌ {error_message}")
        
        # Log error
        self.logger.error(f"💥 Download process error: {error_message}")
    
    def _get_step_index(self, step: str) -> int:
        """Get step index dari step name."""
        step_mapping = {
            'validate': 0,
            'connect': 1,
            'download': 2,
            'extract': 3,
            'organize': 4,
            'verify': 4  # Same as organize
        }
        return step_mapping.get(step.lower(), self.current_step)
    
    def _start_new_step(self, step: str, message: str) -> None:
        """Start new step tracking."""
        self.step_start_time = time.time()
        
        # Update step display
        self._update_step_display(step, 0, 100, f"Starting {step}: {message}")
        
        # Log step start (minimal)
        if self.current_step < len(self.step_names):
            step_name = self.step_names[self.current_step]
            self.logger.info(f"🔄 Step {self.current_step + 1}/{self.total_steps}: {step_name}")
    
    def _complete_current_step(self) -> None:
        """Complete current step dengan timing."""
        if self.step_start_time:
            step_duration = time.time() - self.step_start_time
            
            # Update step display to complete
            if self.current_step < len(self.step_names):
                step_name = self.step_names[self.current_step]
                self._update_step_display(step_name, 100, 100, f"✅ {step_name} completed")
                
                # Log completion (minimal)
                self.logger.debug(f"✅ Step {step_name} completed in {step_duration:.1f}s")
    
    def _calculate_overall_progress(self, current: int, total: int) -> int:
        """Calculate overall progress berdasarkan current step dan progress."""
        if self.total_steps == 0:
            return 0
        
        # Base progress dari completed steps
        completed_steps_progress = (self.current_step / self.total_steps) * 100
        
        # Current step progress
        if total > 0:
            current_step_progress = (current / total) * (100 / self.total_steps)
        else:
            current_step_progress = 0
        
        overall = int(completed_steps_progress + current_step_progress)
        return min(100, max(0, overall))
    
    def _update_progress_display(self, step: str, current: int, total: int, 
                               message: str, overall_progress: int) -> None:
        """Update progress display pada UI components."""
        # Update overall progress
        self._update_overall_progress(overall_progress, message)
        
        # Update step progress
        step_progress = int((current / max(total, 1)) * 100)
        self._update_step_display(step, step_progress, 100, message)
        
        # Update current operation progress (detail)
        if self.ui_components.get('detailed_progress_checkbox', {}).get('value', False):
            self._update_current_progress(current, total, message)
    
    def _update_overall_progress(self, progress: int, message: str) -> None:
        """Update overall progress bar."""
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('overall', progress, f"📊 {message}")
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].update('overall', progress, message)
    
    def _update_step_display(self, step: str, progress: int, total: int, message: str) -> None:
        """Update step progress display."""
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('step', progress, f"🔄 {message}")
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].update('step', progress, message)
    
    def _update_current_progress(self, current: int, total: int, message: str) -> None:
        """Update current operation progress (detailed)."""
        if 'update_progress' in self.ui_components:
            progress = int((current / max(total, 1)) * 100)
            self.ui_components['update_progress']('current', progress, f"⚡ {message}")
    
    def _update_progress_error(self, error_message: str) -> None:
        """Update progress dengan error state."""
        if 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](error_message)
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].error(error_message)
    
    def _complete_progress_tracking(self) -> None:
        """Complete progress tracking dengan success state."""
        if 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation']("Download completed!")
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].complete("Download completed!")
    
    def _show_progress_container(self) -> None:
        """Show progress container untuk download operation."""
        if 'show_for_operation' in self.ui_components:
            self.ui_components['show_for_operation']('download')
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].show('download')
    
    def _log_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Log progress dengan selective logging untuk avoid spam."""
        # Hanya log setiap 25% progress atau di milestone tertentu
        if total > 0:
            progress_pct = int((current / total) * 100)
            
            # Log pada milestones: 25%, 50%, 75%, 100%
            if progress_pct in [25, 50, 75, 100] and hasattr(self, '_last_logged_progress'):
                if getattr(self, '_last_logged_progress', 0) != progress_pct:
                    self.logger.info(f"📈 {step}: {progress_pct}% - {message}")
                    self._last_logged_progress = progress_pct
            elif not hasattr(self, '_last_logged_progress'):
                # First log
                self.logger.info(f"📈 {step}: {progress_pct}% - {message}")
                self._last_logged_progress = progress_pct

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking handlers."""
    logger = get_logger('downloader.progress_setup')
    
    try:
        # Ensure progress tracking components ada
        if 'tracker' not in ui_components:
            from smartcash.ui.components.progress_tracking import create_progress_tracking_container
            progress_components = create_progress_tracking_container()
            ui_components.update(progress_components)
        
        # Setup progress callback manager
        progress_manager = ProgressCallbackManager(ui_components)
        ui_components['progress_manager'] = progress_manager
        
        logger.debug("✅ Progress handlers configured")
        return {'progress_handlers': True, 'progress_manager': progress_manager}
        
    except Exception as e:
        logger.error(f"❌ Progress handlers setup error: {str(e)}")
        return {'progress_handlers': False, 'error': str(e)}

def create_simple_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create simple progress callback tanpa step tracking."""
    
    def simple_callback(step: str, current: int, total: int, message: str) -> None:
        """Simple callback yang hanya update progress bar."""
        try:
            progress = int((current / max(total, 1)) * 100)
            
            if 'update_progress' in ui_components:
                ui_components['update_progress']('overall', progress, f"{step}: {message}")
            
            # Log progress setiap 20%
            if progress % 20 == 0:
                logger = ui_components.get('logger')
                logger and logger.info(f"📈 {step}: {progress}% - {message}")
                
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Simple progress callback error: {str(e)}")
    
    return simple_callback