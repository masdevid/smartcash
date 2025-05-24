"""
File: smartcash/ui/dataset/download/services/progress_bridge.py
Deskripsi: Fixed progress bridge dengan step progress bar yang benar dan log suppression
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.components.observer import notify, EventTopics
import logging
import sys

class ProgressBridge:
    """Enhanced bridge dengan dual progress tracking yang benar."""
    
    def __init__(self, observer_manager=None, namespace: str = "download"):
        self.observer_manager = observer_manager
        self.namespace = namespace
        
        # Overall progress tracking
        self.total_steps = 0
        self.current_step_index = 0
        self.completed_steps = 0
        
        # Current step progress
        self.current_step_name = ""
        self.current_step_progress = 0
        
        # UI components reference
        self._ui_components_ref = None
        
        # Step definitions
        self.steps = []
        self.step_weights = {}
        
        # Setup log suppression
        self._setup_log_suppression()
        
    def _setup_log_suppression(self):
        """Suppress backend service logs dari console."""
        # Suppress common backend loggers
        backend_loggers = [
            'requests', 'urllib3', 'http.client', 'requests.packages.urllib3',
            'smartcash.dataset.services', 'tensorflow', 'torch'
        ]
        
        for logger_name in backend_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    def set_ui_components_reference(self, ui_components: Dict[str, Any]) -> None:
        """Set reference ke UI components untuk direct updates."""
        self._ui_components_ref = ui_components
    
    def define_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Define steps yang akan dijalankan dengan weights."""
        self.steps = steps
        self.total_steps = len(steps)
        self.step_weights = {step['name']: step['weight'] for step in steps}
        
        if self._ui_components_ref:
            self._log_info(f"📋 Proses akan berjalan dalam {self.total_steps} tahap")
    
    def notify_start(self, message: str = "Memulai proses") -> None:
        """Start notification dengan step initialization."""
        self.current_step_index = 0
        self.completed_steps = 0
        self.current_step_progress = 0
        
        # Reset UI
        self._direct_ui_update(
            overall_progress=0,
            step_progress=0,
            overall_message=message,
            step_message="Siap memulai"
        )
        
        self._send_observer_event('DOWNLOAD_START', {
            'progress': 0, 'message': message, 'namespace': self.namespace,
            'total_steps': self.total_steps, 'current_step': 0
        })
    
    def notify_step_start(self, step_name: str, step_description: str = "") -> None:
        """Start step baru dengan reset step progress."""
        self.current_step_name = step_name
        self.current_step_progress = 0
        
        # Calculate overall progress dari completed steps
        overall_progress = self._calculate_overall_progress_from_completed()
        
        # Update UI - step progress reset ke 0
        message = step_description or f"Memulai {step_name}"
        self._direct_ui_update(
            overall_progress=overall_progress,
            step_progress=0,  # Reset step progress
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {step_name}",
            step_message=message
        )
        
        if self._ui_components_ref:
            self._log_info(f"🚀 {step_name}: {message}")
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': step_name, 'step_progress': 0,
            'step_description': step_description, 'current_step': self.current_step_index + 1,
            'total_steps': self.total_steps, 'message': message, 'namespace': self.namespace
        })
    
    def notify_step_progress(self, step_progress: int, step_message: str = "") -> None:
        """Update progress untuk step saat ini."""
        # Clamp step progress
        step_progress = max(0, min(100, step_progress))
        self.current_step_progress = step_progress
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress_with_current_step(step_progress)
        
        # Update UI dengan step progress yang benar
        self._direct_ui_update(
            overall_progress=overall_progress,
            step_progress=step_progress,  # Use actual step progress
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {self.current_step_name}",
            step_message=step_message or f"{self.current_step_name}: {step_progress}%"
        )
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': self.current_step_name,
            'step_progress': step_progress, 'current_step': self.current_step_index + 1,
            'total_steps': self.total_steps, 'message': step_message, 'namespace': self.namespace
        })
    
    def notify_step_complete(self, step_message: str = "") -> None:
        """Complete step saat ini dan lanjut ke step berikutnya."""
        self.current_step_progress = 100
        self.completed_steps += 1
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress_from_completed()
        
        # Update UI dengan step progress 100%
        complete_message = step_message or f"{self.current_step_name} selesai"
        self._direct_ui_update(
            overall_progress=overall_progress,
            step_progress=100,  # Complete step progress
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {self.current_step_name}",
            step_message=complete_message
        )
        
        if self._ui_components_ref:
            self._log_success(f"✅ {complete_message}")
        
        # Move to next step
        self.current_step_index += 1
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': self.current_step_name,
            'step_progress': 100, 'current_step': self.current_step_index,
            'total_steps': self.total_steps, 'message': complete_message, 'namespace': self.namespace
        })
    
    def notify_complete(self, message: str = "Proses selesai", duration: float = 0) -> None:
        """Complete notification dengan final UI update."""
        self._direct_ui_update(
            overall_progress=100, step_progress=100,
            overall_message=message, step_message="Semua tahap selesai"
        )
        
        self._send_observer_event('DOWNLOAD_COMPLETE', {
            'message': message, 'duration': duration, 'namespace': self.namespace,
            'final_progress': 100, 'total_steps': self.total_steps
        })
    
    def notify_error(self, message: str = "Terjadi error", error_details: Dict = None) -> None:
        """Error notification dengan UI reset."""
        self._direct_ui_update(
            overall_progress=0, step_progress=0,
            overall_message=f"❌ {message}", step_message=f"Error pada {self.current_step_name}"
        )
        
        self._send_observer_event('DOWNLOAD_ERROR', {
            'message': message, 'namespace': self.namespace,
            'error_details': error_details or {}
        })
    
    def _calculate_overall_progress_from_completed(self) -> int:
        """Calculate overall progress berdasarkan completed steps saja."""
        if self.total_steps == 0:
            return 0
        
        if self.total_steps == 1:
            return self.current_step_progress
        
        return int((self.completed_steps / self.total_steps) * 100)
    
    def _calculate_overall_progress_with_current_step(self, step_progress: int) -> int:
        """Calculate overall progress termasuk current step progress."""
        if self.total_steps == 0:
            return 0
        
        if self.total_steps == 1:
            return step_progress
        
        completed_contribution = (self.completed_steps / self.total_steps) * 100
        current_step_contribution = (step_progress / 100) * (100 / self.total_steps)
        
        overall = int(completed_contribution + current_step_contribution)
        return min(100, max(0, overall))
    
    def _direct_ui_update(self, overall_progress: int, step_progress: int, 
                         overall_message: str, step_message: str) -> None:
        """Direct UI update untuk immediate feedback."""
        if not self._ui_components_ref:
            return
            
        try:
            ui = self._ui_components_ref
            
            # Update main progress bar (overall progress)
            if 'progress_bar' in ui and hasattr(ui['progress_bar'], 'value'):
                ui['progress_bar'].value = overall_progress
                ui['progress_bar'].description = f"Overall: {overall_progress}%"
                if hasattr(ui['progress_bar'], 'layout'):
                    ui['progress_bar'].layout.visibility = 'visible'
            
            # Update current progress (step progress) - FIX: Use actual step_progress
            if 'current_progress' in ui and hasattr(ui['current_progress'], 'value'):
                ui['current_progress'].value = step_progress  # Use actual step progress
                ui['current_progress'].description = f"Step: {step_progress}%"
                if hasattr(ui['current_progress'], 'layout'):
                    ui['current_progress'].layout.visibility = 'visible'
            
            # Update labels
            if 'overall_label' in ui and hasattr(ui['overall_label'], 'value'):
                ui['overall_label'].value = overall_message
                if hasattr(ui['overall_label'], 'layout'):
                    ui['overall_label'].layout.visibility = 'visible'
            
            if 'step_label' in ui and hasattr(ui['step_label'], 'value'):
                ui['step_label'].value = step_message
                if hasattr(ui['step_label'], 'layout'):
                    ui['step_label'].layout.visibility = 'visible'
            
            # Ensure progress container is visible
            if 'progress_container' in ui and hasattr(ui['progress_container'], 'layout'):
                ui['progress_container'].layout.display = 'block'
                ui['progress_container'].layout.visibility = 'visible'
                
        except Exception:
            pass
    
    def _send_observer_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send observer event dengan multiple fallback methods."""
        import time
        data['timestamp'] = time.time()
        data['event_type'] = event_type
        
        try:
            if self.observer_manager and hasattr(self.observer_manager, 'notify'):
                self.observer_manager.notify(event_type, self, **data)
                return
        except Exception:
            pass
        
        try:
            from smartcash.components.observer import EventDispatcher
            EventDispatcher.notify(event_type, self, **data)
            return
        except Exception:
            pass
        
        try:
            notify(event_type, self, **data)
        except Exception:
            pass
    
    def _log_info(self, message: str) -> None:
        """Log info message via UI logger."""
        if self._ui_components_ref and 'logger' in self._ui_components_ref:
            try:
                self._ui_components_ref['logger'].info(message)
            except Exception:
                pass
    
    def _log_success(self, message: str) -> None:
        """Log success message via UI logger."""
        if self._ui_components_ref and 'logger' in self._ui_components_ref:
            try:
                self._ui_components_ref['logger'].success(message)
            except Exception:
                pass
    
    def reset(self) -> None:
        """Reset progress tracking state."""
        self.current_step_index = 0
        self.completed_steps = 0
        self.current_step_progress = 0
        self.current_step_name = ""
        
        if self._ui_components_ref:
            self._direct_ui_update(0, 0, "Siap memulai", "")
            
            ui = self._ui_components_ref
            if 'progress_container' in ui and hasattr(ui['progress_container'], 'layout'):
                ui['progress_container'].layout.display = 'none'