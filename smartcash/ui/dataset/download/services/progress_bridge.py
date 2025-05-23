"""
File: smartcash/ui/dataset/download/services/progress_bridge.py
Deskripsi: Enhanced progress bridge dengan step-by-step progress tracking yang akurat dan reliable
"""

from typing import Dict, Any, Optional, Callable
from smartcash.components.observer import notify, EventTopics

class ProgressBridge:
    """Enhanced bridge untuk step-by-step progress tracking yang akurat dengan robust error handling."""
    
    def __init__(self, observer_manager=None, namespace: str = "download"):
        self.observer_manager = observer_manager
        self.namespace = namespace
        self.current_progress = 0
        self.current_step = ""
        self.total_steps = 5
        self.step_weights = {
            'validate': 5,    # 5%
            'metadata': 15,   # 15% 
            'download': 60,   # 60%
            'extract': 15,    # 15%
            'verify': 5       # 5%
        }
        self._ui_components_ref = None
        
    def set_ui_components_reference(self, ui_components: Dict[str, Any]) -> None:
        """Set reference ke UI components untuk direct updates."""
        self._ui_components_ref = ui_components
        
    def notify_start(self, message: str = "Memulai proses", total_steps: int = 5) -> None:
        """Start notification dengan step initialization."""
        self.total_steps = total_steps
        self.current_progress = 0
        
        self._direct_ui_update(0, message, "Inisialisasi")
        self._send_observer_event('DOWNLOAD_START', {
            'progress': 0, 'message': message, 'namespace': self.namespace,
            'total_steps': total_steps, 'current_step': 0
        })
    
    def notify_step_progress(self, step_name: str, step_progress: int, message: str = "") -> None:
        """Progress untuk step tertentu dengan weight calculation."""
        try:
            # Clamp step progress
            step_progress = max(0, min(100, step_progress))
            
            # Calculate overall progress berdasarkan step weights
            step_weight = self.step_weights.get(step_name, 20)  # Default 20% jika tidak ada
            overall_progress = self._calculate_weighted_progress(step_name, step_progress)
            
            # Update internal state
            self.current_progress = overall_progress
            self.current_step = step_name
            
            # Direct UI update untuk immediate feedback
            self._direct_ui_update(overall_progress, message, step_name)
            
            # Observer notification
            self._send_observer_event('DOWNLOAD_PROGRESS', {
                'progress': overall_progress,
                'step_progress': step_progress,
                'step_name': step_name,
                'message': message,
                'namespace': self.namespace
            })
            
        except Exception as e:
            # Fallback ke basic progress jika calculation gagal
            self._direct_ui_update(step_progress, f"Error: {str(e)}", step_name)
    
    def _calculate_weighted_progress(self, current_step: str, step_progress: int) -> int:
        """Calculate overall progress berdasarkan step weights yang lebih akurat."""
        total_progress = 0
        step_order = ['validate', 'metadata', 'download', 'extract', 'verify']
        
        try:
            current_index = step_order.index(current_step) if current_step in step_order else 0
            
            # Add completed steps
            for i, step in enumerate(step_order):
                if i < current_index:
                    total_progress += self.step_weights.get(step, 20)
                elif i == current_index:
                    # Current step contribution
                    step_weight = self.step_weights.get(step, 20)
                    total_progress += int((step_progress / 100) * step_weight)
                    break
            
            return min(100, max(0, total_progress))
            
        except Exception:
            # Fallback calculation
            return min(100, step_progress)
    
    def notify_complete(self, message: str = "Proses selesai", duration: float = 0) -> None:
        """Complete notification dengan final UI update."""
        self.current_progress = 100
        
        # Final UI update
        self._direct_ui_update(100, message, "Selesai")
        
        # Observer notification
        self._send_observer_event('DOWNLOAD_COMPLETE', {
            'message': message, 'duration': duration, 'namespace': self.namespace,
            'final_progress': 100
        })
    
    def notify_error(self, message: str = "Terjadi error", error_details: Dict = None) -> None:
        """Error notification dengan UI reset."""
        # Reset progress ke 0 untuk indicate error
        self._direct_ui_update(0, f"❌ {message}", "Error")
        
        # Observer notification
        self._send_observer_event('DOWNLOAD_ERROR', {
            'message': message, 'namespace': self.namespace,
            'error_details': error_details or {}
        })
    
    def _direct_ui_update(self, progress: int, message: str, step_name: str = "") -> None:
        """Direct UI update sebagai primary method untuk immediate feedback."""
        if not self._ui_components_ref:
            return
            
        try:
            ui = self._ui_components_ref
            
            # Update main progress bar
            if 'progress_bar' in ui and hasattr(ui['progress_bar'], 'value'):
                ui['progress_bar'].value = progress
                ui['progress_bar'].description = f"Progress: {progress}%"
                if hasattr(ui['progress_bar'], 'layout'):
                    ui['progress_bar'].layout.visibility = 'visible'
            
            # Update current progress (step progress)
            if 'current_progress' in ui and hasattr(ui['current_progress'], 'value'):
                # Show step progress untuk current step
                if step_name and step_name != "Selesai":
                    ui['current_progress'].value = progress
                    ui['current_progress'].description = f"{step_name}: {progress}%"
                else:
                    ui['current_progress'].value = 100
                    ui['current_progress'].description = "Selesai: 100%"
                    
                if hasattr(ui['current_progress'], 'layout'):
                    ui['current_progress'].layout.visibility = 'visible'
            
            # Update labels
            if 'overall_label' in ui and hasattr(ui['overall_label'], 'value'):
                ui['overall_label'].value = message
                if hasattr(ui['overall_label'], 'layout'):
                    ui['overall_label'].layout.visibility = 'visible'
            
            if 'step_label' in ui and hasattr(ui['step_label'], 'value'):
                step_message = f"{step_name}: {message}" if step_name else message
                ui['step_label'].value = step_message
                if hasattr(ui['step_label'], 'layout'):
                    ui['step_label'].layout.visibility = 'visible'
            
            # Ensure progress container is visible
            if 'progress_container' in ui and hasattr(ui['progress_container'], 'layout'):
                ui['progress_container'].layout.display = 'block'
                ui['progress_container'].layout.visibility = 'visible'
                
        except Exception:
            # Ignore UI update errors
            pass
    
    def _send_observer_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send observer event dengan multiple fallback methods."""
        import time
        data['timestamp'] = time.time()
        data['event_type'] = event_type
        
        # Try observer manager first
        if self.observer_manager and hasattr(self.observer_manager, 'notify'):
            try:
                self.observer_manager.notify(event_type, self, **data)
                return
            except Exception:
                pass
        
        # Try direct EventDispatcher
        try:
            from smartcash.components.observer import EventDispatcher
            EventDispatcher.notify(event_type, self, **data)
            return
        except Exception:
            pass
        
        # Try global notify
        try:
            notify(event_type, self, **data)
        except Exception:
            pass
        
        # If all fail, silent continue - UI update sudah dilakukan
    
    def reset(self) -> None:
        """Reset progress tracking state."""
        self.current_progress = 0
        self.current_step = ""
        
        # Reset UI jika ada reference
        if self._ui_components_ref:
            self._direct_ui_update(0, "Siap memulai", "")
            
            # Hide progress container
            ui = self._ui_components_ref
            if 'progress_container' in ui and hasattr(ui['progress_container'], 'layout'):
                ui['progress_container'].layout.display = 'none'