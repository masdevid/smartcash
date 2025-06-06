"""
File: smartcash/ui/components/progress_tracker.py
Deskripsi: SmartProgressTracker dengan konfigurasi dinamis untuk single/dual/triple tracking
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import get_layout

class SmartProgressTracker:
    """Unified progress tracker dengan konfigurasi dinamis untuk single/dual/triple tracking"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SmartProgressTracker dengan konfigurasi fleksibel
        
        Config options:
        - mode: 'single'/'dual'/'triple' (default: 'single')
        - height: str (default: auto-calculated)
        - show_timer: bool (default: True)
        - steps: List[str] untuk single mode
        - phases: List[str] untuk dual/triple mode
        """
        config = config or {}
        self.mode = config.get('mode', 'single')
        self.steps = config.get('steps', ['setup', 'process', 'complete'])
        self.phases = config.get('phases', ['init', 'execute', 'finalize'])
        self.show_timer = config.get('show_timer', True)
        
        # Auto-calculate height berdasarkan mode
        base_height = {'single': 120, 'dual': 160, 'triple': 240}
        self.height = config.get('height', f"{base_height[self.mode]}px")
        
        # Initialize tracking state
        self.current_step, self.current_phase, self.overall_progress = 0, 0, 0
        self.start_time, self.is_complete, self.callbacks = None, False, []
        
        # Create progress bars berdasarkan mode
        self._create_progress_bars()
        self._create_container()
        self.hide()
    
    def _create_progress_bars(self) -> None:
        """Create progress bars berdasarkan mode dengan one-liner initialization"""
        tracker_height = f"{int(self.height.replace('px', '')) // {'single': 1, 'dual': 2, 'triple': 3}[self.mode]}px"
        
        # Overall progress (semua mode)
        self.overall_bar = widgets.IntProgress(value=0, min=0, max=100, description='Overall:', 
                                             layout=get_layout('container', width='100%', margin='2px 0'))
        
        # Phase progress (dual/triple mode)
        if self.mode in ['dual', 'triple']:
            self.phase_bar = widgets.IntProgress(value=0, min=0, max=100, description=f'{self.phases[0].title()}:', 
                                               layout=get_layout('container', width='100%', margin='2px 0'))
            self.phase_progress = {phase: 0 for phase in self.phases}
        
        # Step progress (triple mode atau single dengan steps)
        if self.mode == 'triple' or (self.mode == 'single' and len(self.steps) > 1):
            self.step_bar = widgets.IntProgress(value=0, min=0, max=100, description=f'{self.steps[0].title()}:', 
                                              layout=get_layout('container', width='100%', margin='2px 0'))
            self.step_progress = {step: 0 for step in self.steps}
        
        # Status dan timer
        self.status_label = widgets.HTML(f"<div style='padding:4px 8px;color:{COLORS.get('text','#333')};font-size:13px;'>⏳ Siap memulai</div>",
                                       layout=get_layout('container', margin='2px 0'))
        
        if self.show_timer:
            self.timer_label = widgets.HTML(f"<div style='padding:2px 8px;color:{COLORS.get('muted','#666')};font-size:11px;'>🕐 00:00</div>",
                                          layout=get_layout('container', margin='0'))
    
    def _create_container(self) -> None:
        """Create container berdasarkan mode dengan conditional widget inclusion"""
        children = [self.overall_bar]
        
        # Add widgets berdasarkan mode
        if hasattr(self, 'phase_bar'): children.append(self.phase_bar)
        if hasattr(self, 'step_bar'): children.append(self.step_bar)
        
        children.append(self.status_label)
        if hasattr(self, 'timer_label'): children.append(self.timer_label)
        
        self.container = widgets.VBox(children, layout=get_layout('container', height=self.height, 
                                                                 border='1px solid #ddd', border_radius='4px', 
                                                                 padding='8px', margin='5px 0'))
    
    def start(self, message: str = "Memulai proses") -> None:
        """Start tracking dengan mode-aware initialization"""
        self.start_time, self.current_step, self.current_phase, self.is_complete = datetime.now(), 0, 0, False
        self.show(), self._reset_all_progress(), self.update_status(message)
        self._trigger_callbacks('start', {'message': message, 'mode': self.mode})
    
    def next_step(self, message: str = None) -> None:
        """Next step dengan mode-aware behavior"""
        if self.mode == 'single':
            self._next_single_step(message)
        elif self.mode == 'dual':
            self._next_dual_step(message)
        elif self.mode == 'triple':
            self._next_triple_step(message)
    
    def next_phase(self, message: str = None) -> None:
        """Next phase untuk dual/triple mode"""
        if self.mode in ['dual', 'triple'] and self.current_phase < len(self.phases) - 1:
            self.phase_progress[self.phases[self.current_phase]] = 100
            self.current_phase += 1
            self.phase_bar.description = f'{self.phases[self.current_phase].title()}:'
            self.phase_bar.value = 0
            
            # Reset step tracking untuk triple mode
            if self.mode == 'triple' and hasattr(self, 'step_bar'):
                self.current_step = 0
                self.step_bar.description = f'{self.steps[0].title()}:'
                self.step_bar.value = 0
            
            message = message or f"Memulai {self.phases[self.current_phase]}"
            self.update_status(message)
            self._update_progress_hierarchy()
            self._trigger_callbacks('next_phase', {'phase': self.current_phase, 'message': message})
    
    def update_progress(self, level: str, value: int, message: str = None) -> None:
        """Update progress di level tertentu dengan auto-propagation"""
        value = max(0, min(100, value))
        
        if level == 'overall':
            self.overall_progress = value
            self.overall_bar.value = value
        elif level == 'phase' and hasattr(self, 'phase_bar'):
            self.phase_progress[self.phases[self.current_phase]] = value
            self.phase_bar.value = value
            self._update_progress_hierarchy()
        elif level == 'step' and hasattr(self, 'step_bar'):
            self.step_progress[self.steps[self.current_step]] = value
            self.step_bar.value = value
            self._update_progress_hierarchy()
        
        message and self.update_status(message)
        self._trigger_callbacks('update_progress', {'level': level, 'value': value, 'message': message})
    
    def update_status(self, message: str, status_type: str = 'info') -> None:
        """Update status dengan styling dan timer"""
        color_map = {'info': COLORS.get('primary', '#007bff'), 'success': COLORS.get('success', '#28a745'), 
                    'warning': COLORS.get('warning', '#ffc107'), 'error': COLORS.get('danger', '#dc3545')}
        emoji_map = {'info': '⏳', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        
        color, emoji = color_map.get(status_type, color_map['info']), emoji_map.get(status_type, emoji_map['info'])
        self.status_label.value = f"<div style='padding:4px 8px;color:{color};font-size:13px;'>{emoji} {message}</div>"
        self._update_timer()
        self._trigger_callbacks('update_status', {'message': message, 'status_type': status_type})
    
    def complete(self, message: str = "Proses selesai") -> None:
        """Complete dengan mode-aware finalization"""
        self.is_complete = True
        self.overall_bar.value = 100
        hasattr(self, 'phase_bar') and setattr(self.phase_bar, 'value', 100)
        hasattr(self, 'step_bar') and setattr(self.step_bar, 'value', 100)
        self.update_status(message, 'success')
        self._trigger_callbacks('complete', {'message': message})
    
    def error(self, message: str = "Terjadi kesalahan") -> None:
        """Set error state"""
        self.update_status(message, 'error')
        self._trigger_callbacks('error', {'message': message})
    
    def reset(self) -> None:
        """Reset dengan mode-aware cleanup"""
        self.current_step, self.current_phase, self.overall_progress, self.is_complete = 0, 0, 0, False
        self.start_time = None
        self._reset_all_progress(), self.update_status("Siap memulai"), self.hide()
        self._trigger_callbacks('reset', {})
    
    def _next_single_step(self, message: str = None) -> None:
        """Next step untuk single mode"""
        if hasattr(self, 'step_bar') and self.current_step < len(self.steps) - 1:
            self.step_progress[self.steps[self.current_step]] = 100
            self.current_step += 1
            self.step_bar.description = f'{self.steps[self.current_step].title()}:'
            self.step_bar.value = 0
            message = message or f"Memulai {self.steps[self.current_step]}"
            self.update_status(message)
            self._update_overall_from_steps()
    
    def _next_dual_step(self, message: str = None) -> None:
        """Next step untuk dual mode (step dalam phase)"""
        if hasattr(self, 'step_bar') and self.current_step < len(self.steps) - 1:
            self.current_step += 1
            message = message or f"Step {self.current_step + 1} dalam {self.phases[self.current_phase]}"
            self.update_status(message)
    
    def _next_triple_step(self, message: str = None) -> None:
        """Next step untuk triple mode"""
        if hasattr(self, 'step_bar') and self.current_step < len(self.steps) - 1:
            self.step_progress[self.steps[self.current_step]] = 100
            self.current_step += 1
            self.step_bar.description = f'{self.steps[self.current_step].title()}:'
            self.step_bar.value = 0
            message = message or f"Step {self.steps[self.current_step]} dalam {self.phases[self.current_phase]}"
            self.update_status(message)
            self._update_progress_hierarchy()
    
    def _update_progress_hierarchy(self) -> None:
        """Update progress hierarchy berdasarkan mode"""
        if self.mode == 'triple':
            # Step → Phase → Overall
            if hasattr(self, 'step_progress'):
                phase_progress = sum(self.step_progress.values()) / len(self.step_progress)
                self.phase_progress[self.phases[self.current_phase]] = phase_progress
                self.phase_bar.value = phase_progress
            
            if hasattr(self, 'phase_progress'):
                overall_progress = sum(self.phase_progress.values()) / len(self.phase_progress)
                self.overall_progress = overall_progress
                self.overall_bar.value = overall_progress
        
        elif self.mode == 'dual':
            # Phase → Overall
            if hasattr(self, 'phase_progress'):
                overall_progress = sum(self.phase_progress.values()) / len(self.phase_progress)
                self.overall_progress = overall_progress
                self.overall_bar.value = overall_progress
    
    def _update_overall_from_steps(self) -> None:
        """Update overall dari step completion untuk single mode"""
        if hasattr(self, 'step_progress'):
            completed_steps = sum(min(100, progress) for progress in self.step_progress.values())
            self.overall_progress = int(completed_steps / len(self.step_progress))
            self.overall_bar.value = self.overall_progress
    
    def _reset_all_progress(self) -> None:
        """Reset semua progress bars ke zero"""
        self.overall_bar.value = 0
        hasattr(self, 'phase_bar') and setattr(self.phase_bar, 'value', 0)
        hasattr(self, 'step_bar') and setattr(self.step_bar, 'value', 0)
        
        # Reset progress dictionaries
        if hasattr(self, 'phase_progress'): self.phase_progress = {phase: 0 for phase in self.phases}
        if hasattr(self, 'step_progress'): self.step_progress = {step: 0 for step in self.steps}
    
    def _update_timer(self) -> None:
        """Update timer jika enabled"""
        if hasattr(self, 'timer_label') and self.start_time:
            elapsed = datetime.now() - self.start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            self.timer_label.value = f"<div style='padding:2px 8px;color:{COLORS.get('muted','#666')};font-size:11px;'>🕐 {minutes:02d}:{seconds:02d}</div>"
    
    def show(self) -> None:
        """Show tracker"""
        self.container.layout.display, self.container.layout.visibility = 'block', 'visible'
    
    def hide(self) -> None:
        """Hide tracker"""
        self.container.layout.display, self.container.layout.visibility = 'none', 'hidden'
    
    def add_callback(self, callback: Callable) -> None:
        """Add event callback"""
        callback not in self.callbacks and self.callbacks.append(callback)
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger callbacks dengan error handling"""
        [self._safe_call_callback(cb, event, data) for cb in self.callbacks]
    
    def _safe_call_callback(self, callback: Callable, event: str, data: Dict[str, Any]) -> None:
        """Safe callback execution"""
        try: callback(event, data)
        except Exception: pass
    
    # Backward compatibility methods
    def show_for_operation(self, operation: str) -> None: self.show(), self.start(f"Memulai {operation}")
    def complete_operation(self, message: str) -> None: self.complete(message)
    def error_operation(self, message: str) -> None: self.error(message)
    def reset_all(self) -> None: self.reset()
    def update_step(self, progress: int, message: str = None) -> None: self.update_progress('step', progress, message)
    def update_overall(self, progress: int, message: str = None) -> None: self.update_progress('overall', progress, message)


# Factory functions dengan config-based creation
def create_progress_tracker(mode: str = 'single', **kwargs) -> SmartProgressTracker:
    """
    Factory untuk SmartProgressTracker dengan mode konfigurasi
    
    Args:
        mode: 'single'/'dual'/'triple'
        **kwargs: steps, phases, height, show_timer, dll
    """
    config = {'mode': mode, **kwargs}
    return SmartProgressTracker(config)

def create_single_progress_tracker(steps: List[str] = None, **kwargs) -> SmartProgressTracker:
    """Create single mode tracker"""
    return create_progress_tracker('single', steps=steps, **kwargs)

def create_dual_progress_tracker(phases: List[str] = None, **kwargs) -> SmartProgressTracker:
    """Create dual mode tracker"""
    return create_progress_tracker('dual', phases=phases, **kwargs)

def create_triple_progress_tracker(phases: List[str] = None, **kwargs) -> SmartProgressTracker:
    """Create triple mode tracker"""
    return create_progress_tracker('triple', phases=phases, **kwargs)
