"""
File: smartcash/ui/dataset/download/services/progress_bridge.py
Deskripsi: Enhanced progress bridge dengan tqdm persistent tracking dan detail per step
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.components.observer import notify, EventTopics
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.manager_observer import get_observer_manager
import logging
import sys
import time

class ProgressBridge(BaseObserver):
    """Enhanced bridge dengan tqdm persistent tracking dan detail progress yang mengimplementasikan BaseObserver."""
    
    def __init__(self, observer_manager=None, namespace: str = "download"):
        # Inisialisasi BaseObserver
        super().__init__(name=f"ProgressBridge_{namespace}", priority=0)
        
        # Dapatkan observer_manager jika tidak diberikan
        self.observer_manager = observer_manager or get_observer_manager()
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
        
        # Progress persistence
        self._last_update_time = 0
        self._update_throttle = 0.1  # Update setiap 100ms minimum
        
        # Setup log suppression
        self._setup_log_suppression()
        
    def _setup_log_suppression(self):
        """Suppress backend service logs dari console."""
        backend_loggers = [
            'requests', 'urllib3', 'http.client', 'requests.packages.urllib3',
            'smartcash.dataset.services', 'tensorflow', 'torch'
        ]
        
        for logger_name in backend_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    def set_ui_components_reference(self, ui_components: Dict[str, Any]) -> None:
        """Set reference ke UI components untuk persistent updates."""
        self._ui_components_ref = ui_components
    
    def define_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Define steps yang akan dijalankan dengan weights."""
        self.steps = steps
        self.total_steps = len(steps)
        self.step_weights = {step['name']: step['weight'] for step in steps}
        
        if self._ui_components_ref:
            self._log_info(f"📋 Proses akan berjalan dalam {self.total_steps} tahap")
    
    def notify_start(self, message: str = "Memulai proses") -> None:
        """Start notification dengan step initialization yang persisten."""
        self.current_step_index = 0
        self.completed_steps = 0
        self.current_step_progress = 0
        
        # Setup progress untuk operation dengan semua bars aktif
        self._setup_persistent_progress()
        self._persistent_ui_update(
            overall_progress=0, step_progress=0, current_progress=0,
            overall_message=message, step_message="Siap memulai", current_message=""
        )
        
        self._send_observer_event('DOWNLOAD_START', {
            'progress': 0, 'message': message, 'namespace': self.namespace,
            'total_steps': self.total_steps, 'current_step': 0
        })
    
    def notify_step_start(self, step_name: str, step_description: str = "") -> None:
        """Start step baru dengan persistent tracking."""
        self.current_step_name = step_name
        self.current_step_progress = 0
        
        overall_progress = self._calculate_overall_progress_from_completed()
        
        # Persistent UI update
        message = step_description or f"Memulai {step_name}"
        self._persistent_ui_update(
            overall_progress=overall_progress, step_progress=0, current_progress=0,
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {step_name}",
            step_message=f"🔄 {message}", current_message=""
        )
        
        if self._ui_components_ref:
            self._log_info(f"🚀 {step_name}: {message}")
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': step_name, 'step_progress': 0,
            'step_description': step_description, 'current_step': self.current_step_index + 1,
            'total_steps': self.total_steps, 'message': message, 'namespace': self.namespace
        })
    
    def notify_step_progress(self, step_progress: int, step_message: str = "", 
                           current_progress: int = None, current_message: str = "") -> None:
        """Update progress untuk step saat ini dengan throttling."""
        current_time = time.time()
        if current_time - self._last_update_time < self._update_throttle:
            return  # Skip update untuk menghindari spam
        
        self._last_update_time = current_time
        
        step_progress = max(0, min(100, step_progress))
        self.current_step_progress = step_progress
        
        overall_progress = self._calculate_overall_progress_with_current_step(step_progress)
        
        # Persistent UI update dengan detail
        self._persistent_ui_update(
            overall_progress=overall_progress,
            step_progress=step_progress,
            current_progress=current_progress or 0,
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {self.current_step_name}",
            step_message=step_message or f"🔄 {self.current_step_name}: {step_progress}%",
            current_message=current_message
        )
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': self.current_step_name,
            'step_progress': step_progress, 'current_step': self.current_step_index + 1,
            'total_steps': self.total_steps, 'message': step_message, 'namespace': self.namespace
        })
    
    def notify_step_complete(self, step_message: str = "") -> None:
        """Complete step saat ini dengan persistent update."""
        self.current_step_progress = 100
        self.completed_steps += 1
        
        overall_progress = self._calculate_overall_progress_from_completed()
        
        complete_message = step_message or f"{self.current_step_name} selesai"
        self._persistent_ui_update(
            overall_progress=overall_progress, step_progress=100, current_progress=100,
            overall_message=f"Tahap {self.current_step_index + 1}/{self.total_steps}: {self.current_step_name}",
            step_message=f"✅ {complete_message}", current_message=""
        )
        
        if self._ui_components_ref:
            self._log_success(f"✅ {complete_message}")
        
        self.current_step_index += 1
        
        self._send_observer_event('DOWNLOAD_PROGRESS', {
            'progress': overall_progress, 'step_name': self.current_step_name,
            'step_progress': 100, 'current_step': self.current_step_index,
            'total_steps': self.total_steps, 'message': complete_message, 'namespace': self.namespace
        })
    
    def notify_organize_split(self, split_name: str, split_progress: int, split_message: str) -> None:
        """Special notification untuk organize per split (current progress)."""
        if self._ui_components_ref and 'update_progress' in self._ui_components_ref:
            self._ui_components_ref['update_progress']('current', split_progress, f"📂 {split_name}: {split_message}")
    
    def notify_complete(self, message: str = "Proses selesai", duration: float = 0) -> None:
        """Complete notification dengan final persistent update."""
        self._persistent_ui_update(
            overall_progress=100, step_progress=100, current_progress=100,
            overall_message=f"✅ {message}", step_message="✅ Semua tahap selesai", current_message=""
        )
        
        self._send_observer_event('DOWNLOAD_COMPLETE', {
            'message': message, 'duration': duration, 'namespace': self.namespace,
            'final_progress': 100, 'total_steps': self.total_steps
        })
    
    def notify_error(self, message: str = "Terjadi error", error_details: Dict = None) -> None:
        """Error notification dengan UI reset."""
        self._persistent_ui_update(
            overall_progress=0, step_progress=0, current_progress=0,
            overall_message=f"❌ {message}", step_message=f"❌ Error pada {self.current_step_name}",
            current_message=""
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
    
    def _setup_persistent_progress(self) -> None:
        """Setup progress bars untuk operation dengan semua bars aktif."""
        try:
            if not self._ui_components_ref:
                return
            
            # Show progress untuk download dengan semua bars
            if 'show_for_operation' in self._ui_components_ref:
                self._ui_components_ref['show_for_operation']('download')  # Aktifkan semua bars
            
        except Exception:
            pass
    
    def _persistent_ui_update(self, overall_progress: int, step_progress: int, current_progress: int,
                            overall_message: str, step_message: str, current_message: str) -> None:
        """Persistent UI update untuk immediate feedback."""
        if not self._ui_components_ref:
            return
            
        try:
            ui = self._ui_components_ref
            
            # Update overall progress
            if 'update_progress' in ui:
                ui['update_progress']('overall', overall_progress, overall_message)
                ui['update_progress']('step', step_progress, step_message)
                if current_message:
                    ui['update_progress']('current', current_progress, current_message)
                
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
        except Exception as e:
            if self._ui_components_ref and 'logger' in self._ui_components_ref:
                self._ui_components_ref['logger'].debug(f"Observer notify error: {str(e)}")
        
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
        self._last_update_time = 0
        
        if self._ui_components_ref:
            self._persistent_ui_update(0, 0, 0, "Siap memulai", "", "")
            
            ui = self._ui_components_ref
            try:
                if 'hide_container' in ui:
                    ui['hide_container']()
            except Exception:
                pass
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """Implementasi metode update dari BaseObserver."""
        # Metode ini diperlukan untuk implementasi BaseObserver
        # Kita tidak perlu implementasi khusus karena ProgressBridge adalah pengirim event, bukan penerima
        pass
    
    def should_process_event(self, event_type: str) -> bool:
        """Implementasi metode should_process_event dari BaseObserver."""
        # Selalu return True karena kita tidak memfilter event
        return True