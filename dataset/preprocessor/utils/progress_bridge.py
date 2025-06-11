"""
File: smartcash/dataset/preprocessor/utils/progress_bridge.py
Deskripsi: Enhanced progress bridge dengan kompatibilitas penuh untuk ui/components/progress_tracker
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import time

from smartcash.common.logger import get_logger

class ProgressLevel(Enum):
    """🎯 Progress levels yang kompatibel dengan progress_tracker"""
    OVERALL = "overall"
    STEP = "step" 
    CURRENT = "current"
    PRIMARY = "primary"  # Alias untuk single level

@dataclass
class ProgressUpdate:
    """📊 Progress update data structure"""
    level: str
    current: int
    total: int
    message: str
    percentage: float
    timestamp: float
    
    @property
    def is_complete(self) -> bool:
        return self.current >= self.total

class ProgressBridge:
    """🌉 Enhanced progress bridge dengan full compatibility untuk progress_tracker"""
    
    def __init__(self, throttle_ms: int = 100):
        self.logger = get_logger(__name__)
        self.callbacks: List[Callable] = []
        self.last_updates: Dict[str, float] = {}
        self.throttle_interval = throttle_ms / 1000.0  # Convert ke seconds
        self.total_operations = 0
        self.completed_operations = 0
        
        # Progress state tracking
        self.progress_state = {
            'overall': {'current': 0, 'total': 100, 'message': ''},
            'step': {'current': 0, 'total': 100, 'message': ''},
            'current': {'current': 0, 'total': 100, 'message': ''},
            'primary': {'current': 0, 'total': 100, 'message': ''}
        }
    
    def register_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """📝 Register progress callback dengan signature yang sesuai"""
        if callback and callback not in self.callbacks:
            self.callbacks.append(callback)
            self.logger.debug(f"✅ Progress callback registered: {len(self.callbacks)} total")
    
    def unregister_callback(self, callback: Callable) -> None:
        """🗑️ Unregister progress callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.debug(f"🗑️ Progress callback unregistered: {len(self.callbacks)} remaining")
    
    def update(self, level: str, current: int, total: int, message: str = "", 
               force_update: bool = False) -> None:
        """🔄 Enhanced progress update dengan throttling dan state tracking"""
        try:
            # Normalize level untuk compatibility
            level = self._normalize_level(level)
            
            # Validate input
            current = max(0, min(current, total))
            total = max(1, total)
            percentage = (current / total) * 100
            
            # Throttling check
            now = time.time()
            last_update = self.last_updates.get(level, 0)
            
            if not force_update and (now - last_update) < self.throttle_interval:
                return
            
            # Update state
            self.progress_state[level] = {
                'current': current,
                'total': total, 
                'message': message,
                'percentage': percentage,
                'timestamp': now
            }
            
            self.last_updates[level] = now
            
            # Create progress update object
            update = ProgressUpdate(level, current, total, message, percentage, now)
            
            # Call all registered callbacks
            self._notify_callbacks(update)
            
            # Log milestone progress
            if current % max(1, total // 10) == 0 or current == total or force_update:
                emoji = "✅" if current == total else "🔄"
                self.logger.debug(f"{emoji} {level}: {current}/{total} ({percentage:.1f}%) - {message}")
        
        except Exception as e:
            self.logger.error(f"❌ Progress update error: {str(e)}")
    
    def update_overall(self, current: int, total: int = 100, message: str = "") -> None:
        """🎯 Update overall progress (compatibility method)"""
        self.update('overall', current, total, message)
    
    def update_step(self, current: int, total: int = 100, message: str = "") -> None:
        """🎯 Update step progress (compatibility method)"""
        self.update('step', current, total, message)
    
    def update_current(self, current: int, total: int = 100, message: str = "") -> None:
        """🎯 Update current operation progress (compatibility method)"""
        self.update('current', current, total, message)
    
    def update_primary(self, current: int, total: int = 100, message: str = "") -> None:
        """🎯 Update primary progress untuk single level tracker"""
        self.update('primary', current, total, message)
    
    def increment(self, level: str, message: str = "", step: int = 1) -> None:
        """➕ Increment progress dengan step"""
        state = self.progress_state.get(level, {'current': 0, 'total': 100})
        new_current = min(state['current'] + step, state['total'])
        self.update(level, new_current, state['total'], message)
    
    def complete_level(self, level: str, message: str = "Completed") -> None:
        """✅ Mark level sebagai complete"""
        state = self.progress_state.get(level, {'total': 100})
        self.update(level, state['total'], state['total'], message, force_update=True)
    
    def reset_level(self, level: str, total: int = 100, message: str = "Reset") -> None:
        """🔄 Reset progress level"""
        self.update(level, 0, total, message, force_update=True)
    
    def reset_all(self) -> None:
        """🔄 Reset semua progress levels"""
        for level in self.progress_state.keys():
            self.reset_level(level, message="Reset all progress")
    
    def get_progress_state(self, level: Optional[str] = None) -> Dict[str, Any]:
        """📊 Get current progress state"""
        if level:
            return self.progress_state.get(level, {}).copy()
        return {k: v.copy() for k, v in self.progress_state.items()}
    
    def is_complete(self, level: str) -> bool:
        """✅ Check if level is complete"""
        state = self.progress_state.get(level, {})
        return state.get('current', 0) >= state.get('total', 100)
    
    def get_overall_percentage(self) -> float:
        """📊 Get overall completion percentage"""
        overall_state = self.progress_state.get('overall', {})
        current = overall_state.get('current', 0)
        total = overall_state.get('total', 100)
        return (current / total) * 100 if total > 0 else 0
    
    def _normalize_level(self, level: str) -> str:
        """🔧 Normalize level name untuk compatibility"""
        level_mapping = {
            'main': 'primary',
            'total': 'overall',
            'sub': 'current',
            'substep': 'current'
        }
        return level_mapping.get(level.lower(), level.lower())
    
    def _notify_callbacks(self, update: ProgressUpdate) -> None:
        """📢 Notify all registered callbacks"""
        for callback in self.callbacks.copy():  # Copy untuk avoid modification during iteration
            try:
                callback(update.level, update.current, update.total, update.message)
            except Exception as e:
                self.logger.warning(f"⚠️ Progress callback error: {str(e)}")
                # Remove callback yang error untuk prevent future errors
                if callback in self.callbacks:
                    self.callbacks.remove(callback)
    
    # === ADVANCED FEATURES ===
    
    def create_sub_bridge(self, parent_level: str, weight: float = 1.0) -> 'SubProgressBridge':
        """🌉 Create sub-progress bridge untuk nested operations"""
        return SubProgressBridge(self, parent_level, weight)
    
    def batch_update(self, updates: List[Tuple[str, int, int, str]]) -> None:
        """📦 Batch update multiple progress levels"""
        for level, current, total, message in updates:
            self.update(level, current, total, message)
    
    def create_phase_manager(self, phases: List[Tuple[str, int]]) -> 'PhaseProgressManager':
        """🎭 Create phase manager untuk multi-phase operations"""
        return PhaseProgressManager(self, phases)

class SubProgressBridge:
    """🌉 Sub-progress bridge untuk nested operations"""
    
    def __init__(self, parent_bridge: ProgressBridge, parent_level: str, weight: float = 1.0):
        self.parent_bridge = parent_bridge
        self.parent_level = parent_level
        self.weight = weight
        self.parent_state = parent_bridge.get_progress_state(parent_level)
        self.base_progress = self.parent_state.get('current', 0)
    
    def update(self, level: str, current: int, total: int, message: str = "") -> None:
        """🔄 Update sub-progress dan propagate ke parent"""
        # Update local level
        self.parent_bridge.update(level, current, total, message)
        
        # Calculate parent progress contribution
        if total > 0:
            sub_percentage = (current / total) * self.weight
            parent_total = self.parent_state.get('total', 100)
            parent_increment = int(sub_percentage * parent_total / 100)
            parent_current = self.base_progress + parent_increment
            
            # Update parent level
            self.parent_bridge.update(
                self.parent_level, 
                parent_current, 
                parent_total, 
                f"Sub-operation: {message}"
            )

class PhaseProgressManager:
    """🎭 Phase manager untuk multi-phase operations"""
    
    def __init__(self, bridge: ProgressBridge, phases: List[Tuple[str, int]]):
        self.bridge = bridge
        self.phases = phases  # [(phase_name, weight), ...]
        self.total_weight = sum(weight for _, weight in phases)
        self.current_phase_index = 0
        self.phase_start_progress = 0
    
    def start_phase(self, phase_name: str) -> None:
        """🎬 Start new phase"""
        # Find phase index
        for i, (name, _) in enumerate(self.phases):
            if name == phase_name:
                self.current_phase_index = i
                break
        
        # Calculate start progress
        self.phase_start_progress = sum(
            weight for _, weight in self.phases[:self.current_phase_index]
        ) * 100 // self.total_weight
        
        self.bridge.update('overall', self.phase_start_progress, 100, f"Starting {phase_name}")
    
    def update_phase_progress(self, current: int, total: int, message: str = "") -> None:
        """🔄 Update current phase progress"""
        if self.current_phase_index >= len(self.phases):
            return
        
        phase_name, phase_weight = self.phases[self.current_phase_index]
        phase_percentage = (current / total) if total > 0 else 0
        phase_contribution = (phase_weight * phase_percentage * 100) // self.total_weight
        
        overall_progress = self.phase_start_progress + int(phase_contribution)
        self.bridge.update('overall', overall_progress, 100, f"{phase_name}: {message}")
        self.bridge.update('current', current, total, message)

# === FACTORY FUNCTIONS ===

def create_progress_bridge(throttle_ms: int = 100) -> ProgressBridge:
    """🏭 Factory untuk create ProgressBridge"""
    return ProgressBridge(throttle_ms)

def create_compatible_bridge(ui_components: Dict[str, Any]) -> ProgressBridge:
    """🏭 Create bridge yang kompatibel dengan UI components"""
    bridge = ProgressBridge()
    
    # Register UI progress tracker jika ada
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        def ui_callback(level: str, current: int, total: int, message: str):
            """Callback yang kompatibel dengan progress_tracker"""
            try:
                if hasattr(progress_tracker, f'update_{level}'):
                    getattr(progress_tracker, f'update_{level}')(current, message)
                elif hasattr(progress_tracker, 'update'):
                    progress_tracker.update(level, current, message)
            except Exception as e:
                get_logger(__name__).debug(f"⚠️ UI callback error: {str(e)}")
        
        bridge.register_callback(ui_callback)
    
    return bridge

# === CONVENIENCE FUNCTIONS ===

def update_progress_safe(bridge: Optional[ProgressBridge], level: str, current: int, 
                        total: int, message: str = "") -> None:
    """🔄 One-liner safe progress update"""
    if bridge:
        bridge.update(level, current, total, message)

def create_preprocessing_bridge(ui_components: Dict[str, Any] = None) -> ProgressBridge:
    """🏭 Create bridge khusus untuk preprocessing operations"""
    if ui_components:
        return create_compatible_bridge(ui_components)
    return create_progress_bridge()

def create_validation_bridge(ui_components: Dict[str, Any] = None) -> ProgressBridge:
    """🏭 Create bridge khusus untuk validation operations"""
    bridge = create_preprocessing_bridge(ui_components)
    
    # Setup phases untuk validation
    validation_phases = [
        ("File Scanning", 20),
        ("Image Validation", 40), 
        ("Label Validation", 30),
        ("Consistency Check", 10)
    ]
    
    return bridge

def create_augmentation_bridge(ui_components: Dict[str, Any] = None) -> ProgressBridge:
    """🏭 Create bridge khusus untuk augmentation operations"""
    bridge = create_preprocessing_bridge(ui_components)
    
    # Setup phases untuk augmentation
    augmentation_phases = [
        ("Preparation", 10),
        ("Image Processing", 60),
        ("Label Processing", 20),
        ("Finalization", 10)
    ]
    
    return bridge