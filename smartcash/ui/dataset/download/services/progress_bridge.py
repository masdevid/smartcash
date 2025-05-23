"""
File: smartcash/ui/dataset/download/services/progress_bridge.py
Deskripsi: Enhanced progress bridge untuk komunikasi yang lebih reliable antara service dan UI
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import notify, EventTopics

class ProgressBridge:
    """Enhanced bridge untuk komunikasi progress dari service ke UI dengan robust error handling."""
    
    def __init__(self, observer_manager=None, namespace: str = "download"):
        self.observer_manager = observer_manager
        self.namespace = namespace
        self.current_progress = 0
        self.current_step = ""
        self.total_steps = 1
        self.step_progress = {}  # Track progress per step
        
    def notify_start(self, message: str = "Memulai proses", total_steps: int = 5) -> None:
        """Enhanced start notification dengan step tracking initialization."""
        self.total_steps = total_steps
        self.current_progress = 0
        self.step_progress = {}
        
        event_data = {
            'progress': 0,
            'message': message,
            'namespace': self.namespace,
            'total_steps': total_steps,
            'current_step': 0
        }
        
        self._send_event('DOWNLOAD_START', event_data)
        
        # Also send initial progress
        self.notify_progress(0, 100, message, current_step=0, total_steps=total_steps)
    
    def notify_progress(self, progress: int, total: int = 100, message: str = "", 
                       step: Optional[str] = None, current_step: int = 1, total_steps: int = 5) -> None:
        """Enhanced progress notification dengan step awareness dan validation."""
        
        # Validate dan clamp progress values
        progress = max(0, min(total, progress))
        percentage = int((progress / total) * 100) if total > 0 else 0
        
        # Update internal state
        self.current_progress = percentage
        if step:
            self.current_step = step
            self.step_progress[step] = percentage
        
        # Main progress event
        event_data = {
            'progress': percentage,
            'message': message,
            'namespace': self.namespace,
            'raw_progress': progress,
            'raw_total': total
        }
        
        self._send_event('DOWNLOAD_PROGRESS', event_data)
        
        # Step-specific progress jika ada step info
        if step:
            step_data = {
                'step_progress': percentage,
                'step_message': f"{step}: {message}",
                'step_name': step,
                'current_step': current_step,
                'total_steps': total_steps,
                'namespace': self.namespace
            }
            self._send_event('DOWNLOAD_STEP_PROGRESS', step_data)
    
    def notify_step_complete(self, step_name: str, message: str = None) -> None:
        """Notify completion of a specific step."""
        final_message = message or f"{step_name} selesai"
        self.step_progress[step_name] = 100
        
        step_data = {
            'step_progress': 100,
            'step_message': final_message,
            'step_name': step_name,
            'namespace': self.namespace,
            'step_complete': True
        }
        
        self._send_event('DOWNLOAD_STEP_PROGRESS', step_data)
    
    def notify_complete(self, message: str = "Proses selesai", duration: float = 0) -> None:
        """Enhanced completion notification dengan summary information."""
        self.current_progress = 100
        
        # Calculate average step completion
        completed_steps = len([s for s, p in self.step_progress.items() if p >= 100])
        
        event_data = {
            'message': message,
            'duration': duration,
            'namespace': self.namespace,
            'final_progress': 100,
            'completed_steps': completed_steps,
            'total_steps': self.total_steps,
            'step_summary': self.step_progress.copy()
        }
        
        self._send_event('DOWNLOAD_COMPLETE', event_data)
    
    def notify_error(self, message: str = "Terjadi error", error_details: Dict = None) -> None:
        """Enhanced error notification dengan detailed error information."""
        event_data = {
            'message': message,
            'namespace': self.namespace,
            'current_progress': self.current_progress,
            'current_step': self.current_step,
            'error_details': error_details or {}
        }
        
        self._send_event('DOWNLOAD_ERROR', event_data)
    
    def notify_substep_progress(self, substep_name: str, progress: int, 
                              parent_step: str, message: str = "") -> None:
        """Notify progress untuk substep dalam step utama (misal: download chunks)."""
        substep_data = {
            'substep_name': substep_name,
            'substep_progress': progress,
            'parent_step': parent_step,
            'message': message,
            'namespace': self.namespace
        }
        
        self._send_event('DOWNLOAD_SUBSTEP_PROGRESS', substep_data)
    
    def _send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Enhanced event sending dengan multiple fallback methods dan error handling."""
        
        # Add timestamp untuk debugging
        import time
        data['timestamp'] = time.time()
        data['event_type'] = event_type
        
        success = False
        
        # Method 1: Observer manager jika tersedia
        if self.observer_manager and hasattr(self.observer_manager, 'notify'):
            try:
                self.observer_manager.notify(event_type, self, **data)
                success = True
            except Exception as e:
                # Log error tapi lanjutkan ke fallback
                pass
        
        # Method 2: Direct EventDispatcher notification
        if not success:
            try:
                from smartcash.components.observer import EventDispatcher
                EventDispatcher.notify(event_type, self, **data)
                success = True
            except Exception as e:
                pass
        
        # Method 3: Global notify function
        if not success:
            try:
                notify(event_type, self, **data)
                success = True
            except Exception as e:
                pass
        
        # Method 4: Direct UI update sebagai ultimate fallback
        if not success:
            self._direct_ui_update(event_type, data)
    
    def _direct_ui_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Direct UI update sebagai ultimate fallback ketika observer system gagal."""
        try:
            # Ini adalah fallback terakhir - update UI components secara langsung
            # Hanya untuk kasus observer system benar-benar gagal
            
            if hasattr(self, '_ui_components_ref'):
                ui_components = self._ui_components_ref
                
                if event_type.endswith('_PROGRESS'):
                    progress = data.get('progress', 0)
                    message = data.get('message', '')
                    
                    # Update progress bar langsung
                    if 'progress_bar' in ui_components:
                        ui_components['progress_bar'].value = progress
                        ui_components['progress_bar'].description = f"Progress: {progress}%"
                    
                    # Update message label
                    if 'overall_label' in ui_components:
                        ui_components['overall_label'].value = message
                        
        except Exception:
            # Fallback gagal total - tidak ada yang bisa dilakukan
            pass
    
    def set_ui_components_reference(self, ui_components: Dict[str, Any]) -> None:
        """Set reference ke UI components untuk direct fallback updates."""
        self._ui_components_ref = ui_components
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status untuk monitoring dan debugging."""
        return {
            'namespace': self.namespace,
            'current_progress': self.current_progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'step_progress': self.step_progress.copy(),
            'has_observer_manager': self.observer_manager is not None,
            'completed_steps': len([s for s, p in self.step_progress.items() if p >= 100])
        }
    
    def reset(self) -> None:
        """Reset progress tracking state."""
        self.current_progress = 0
        self.current_step = ""
        self.total_steps = 1
        self.step_progress = {}