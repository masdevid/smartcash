"""
File: smartcash/ui/dataset/preprocessing/utils/progress_tracker.py
Deskripsi: Progress tracker dengan 2-level tracking (overall + step) untuk preprocessing
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
import time
from concurrent.futures import ThreadPoolExecutor, Future

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import COLORS, ICONS

logger = get_logger(__name__)

class ProcessingSteps(Enum):
    """Langkah-langkah preprocessing."""
    PREPARATION = auto()
    TRAIN_SPLIT = auto() 
    VAL_SPLIT = auto()
    TEST_SPLIT = auto()
    FINALIZATION = auto()

class ProgressTracker:
    """Progress tracker dengan 2 level: overall dan step."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi progress tracker.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', logger)
        
        # Progress state
        self.overall_progress = 0
        self.overall_total = 100
        self.overall_message = ""
        
        self.step_progress = 0
        self.step_total = 100
        self.step_message = ""
        self.current_step = 0
        self.total_steps = 5
        
        # Step definitions
        self.step_definitions = {
            ProcessingSteps.PREPARATION: {
                'title': 'Persiapan',
                'description': 'Mempersiapkan data dan validasi',
                'weight': 10  # 10% dari total
            },
            ProcessingSteps.TRAIN_SPLIT: {
                'title': 'Training Split',
                'description': 'Memproses data training',
                'weight': 40  # 40% dari total
            },
            ProcessingSteps.VAL_SPLIT: {
                'title': 'Validation Split', 
                'description': 'Memproses data validasi',
                'weight': 25  # 25% dari total
            },
            ProcessingSteps.TEST_SPLIT: {
                'title': 'Test Split',
                'description': 'Memproses data testing',
                'weight': 20  # 20% dari total
            },
            ProcessingSteps.FINALIZATION: {
                'title': 'Finalisasi',
                'description': 'Menyelesaikan dan cleanup',
                'weight': 5   # 5% dari total
            }
        }
        
        # UI components
        self._setup_progress_widgets()
        
    def _setup_progress_widgets(self) -> None:
        """Setup widget progress di UI."""
        # Overall progress
        if 'progress_bar' not in self.ui_components:
            import ipywidgets as widgets
            self.ui_components['progress_bar'] = widgets.FloatProgress(
                value=0, min=0, max=100,
                description='',
                bar_style='info',
                layout=widgets.Layout(width='100%', visibility='hidden')
            )
        
        # Overall label
        if 'overall_label' not in self.ui_components:
            import ipywidgets as widgets
            self.ui_components['overall_label'] = widgets.HTML(
                value="",
                layout=widgets.Layout(margin='5px 0', visibility='hidden')
            )
        
        # Step progress
        if 'current_progress' not in self.ui_components:
            import ipywidgets as widgets
            self.ui_components['current_progress'] = widgets.FloatProgress(
                value=0, min=0, max=100,
                description='',
                bar_style='info',
                layout=widgets.Layout(width='100%', visibility='hidden')
            )
        
        # Step label
        if 'step_label' not in self.ui_components:
            import ipywidgets as widgets
            self.ui_components['step_label'] = widgets.HTML(
                value="",
                layout=widgets.Layout(margin='0 0 5px 0', visibility='hidden')
            )
    
    def start_processing(self, steps: Optional[List[ProcessingSteps]] = None) -> None:
        """
        Mulai progress tracking.
        
        Args:
            steps: List step yang akan dijalankan (default: semua step)
        """
        if steps is None:
            steps = list(ProcessingSteps)
        
        self.total_steps = len(steps)
        self.current_step = 0
        
        # Reset progress
        self.overall_progress = 0
        self.step_progress = 0
        
        # Tampilkan progress widgets
        self._show_progress_widgets()
        
        # Update initial display
        self.update_overall_progress(0, "Memulai preprocessing dataset...")
        
        self.logger.info(f"🚀 Progress tracking dimulai dengan {self.total_steps} step")
    
    def start_step(self, step: ProcessingSteps, total_items: int = 100) -> None:
        """
        Mulai step baru.
        
        Args:
            step: Step yang dimulai
            total_items: Total item untuk step ini
        """
        self.current_step += 1
        self.step_progress = 0
        self.step_total = total_items
        
        step_info = self.step_definitions.get(step, {'title': str(step), 'description': ''})
        
        # Update step progress
        self.step_message = f"Step {self.current_step}/{self.total_steps}: {step_info['title']}"
        self._update_step_display()
        
        # Update overall progress berdasarkan step
        previous_weight = sum(
            self.step_definitions[s]['weight'] 
            for i, s in enumerate(ProcessingSteps) 
            if i < self.current_step - 1
        )
        self.overall_progress = previous_weight
        self.overall_message = f"{step_info['title']} - {step_info['description']}"
        
        self._update_overall_display()
        
        self.logger.info(f"📋 Mulai {self.step_message}: {step_info['description']}")
    
    def update_step_progress(self, current: int, message: str = "") -> None:
        """
        Update progress step saat ini.
        
        Args:
            current: Progress saat ini
            message: Pesan progress
        """
        self.step_progress = min(current, self.step_total)
        
        # Update step display
        step_percentage = (self.step_progress / self.step_total * 100) if self.step_total > 0 else 0
        
        if message:
            self.step_message = f"Step {self.current_step}/{self.total_steps}: {message}"
        
        self._update_step_display()
        
        # Update overall progress berdasarkan step progress
        current_step_enum = list(ProcessingSteps)[self.current_step - 1] if self.current_step > 0 else None
        if current_step_enum:
            step_weight = self.step_definitions[current_step_enum]['weight']
            step_contribution = (step_percentage / 100) * step_weight
            
            previous_weight = sum(
                self.step_definitions[s]['weight'] 
                for i, s in enumerate(ProcessingSteps) 
                if i < self.current_step - 1
            )
            
            self.overall_progress = previous_weight + step_contribution
            self._update_overall_display()
    
    def complete_step(self, step: ProcessingSteps, message: str = "") -> None:
        """
        Selesaikan step saat ini.
        
        Args:
            step: Step yang diselesaikan
            message: Pesan completion
        """
        step_info = self.step_definitions.get(step, {'title': str(step)})
        
        # Complete step progress
        self.step_progress = self.step_total
        completion_message = message or f"{step_info['title']} selesai"
        
        self.step_message = f"Step {self.current_step}/{self.total_steps}: {completion_message}"
        self._update_step_display()
        
        # Update overall progress
        completed_weight = sum(
            self.step_definitions[s]['weight'] 
            for i, s in enumerate(ProcessingSteps) 
            if i < self.current_step
        )
        self.overall_progress = completed_weight
        self._update_overall_display()
        
        self.logger.info(f"✅ {completion_message}")
    
    def update_overall_progress(self, progress: float, message: str = "") -> None:
        """
        Update overall progress secara manual.
        
        Args:
            progress: Progress overall (0-100)
            message: Pesan overall
        """
        self.overall_progress = min(progress, 100)
        if message:
            self.overall_message = message
        
        self._update_overall_display()
    
    def complete_processing(self, message: str = "Preprocessing selesai") -> None:
        """
        Selesaikan seluruh proses.
        
        Args:
            message: Pesan completion
        """
        # Complete all progress
        self.overall_progress = 100
        self.step_progress = self.step_total
        self.overall_message = message
        self.step_message = f"✅ Selesai - {message}"
        
        # Update displays
        self._update_overall_display()
        self._update_step_display()
        
        self.logger.info(f"🎉 {message}")
        
        # Auto-hide setelah delay
        def hide_after_delay():
            import time
            time.sleep(3)
            self._hide_progress_widgets()
        
        # Gunakan ThreadPoolExecutor untuk delay tanpa blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(hide_after_delay)
    
    def handle_error(self, step: ProcessingSteps, error_message: str) -> None:
        """
        Handle error pada step tertentu.
        
        Args:
            step: Step yang error
            error_message: Pesan error
        """
        step_info = self.step_definitions.get(step, {'title': str(step)})
        
        # Update displays dengan error state
        self.overall_message = f"❌ Error pada {step_info['title']}: {error_message}"
        self.step_message = f"Step {self.current_step}/{self.total_steps}: ❌ Error"
        
        # Change bar style to danger
        if 'progress_bar' in self.ui_components:
            self.ui_components['progress_bar'].bar_style = 'danger'
        if 'current_progress' in self.ui_components:
            self.ui_components['current_progress'].bar_style = 'danger'
        
        self._update_overall_display()
        self._update_step_display()
        
        self.logger.error(f"❌ Error pada {step_info['title']}: {error_message}")
    
    def reset(self) -> None:
        """Reset progress tracker."""
        self.overall_progress = 0
        self.step_progress = 0
        self.current_step = 0
        self.overall_message = ""
        self.step_message = ""
        
        # Reset bar styles
        if 'progress_bar' in self.ui_components:
            self.ui_components['progress_bar'].bar_style = 'info'
        if 'current_progress' in self.ui_components:
            self.ui_components['current_progress'].bar_style = 'info'
        
        self._hide_progress_widgets()
        
        self.logger.debug("🔄 Progress tracker direset")
    
    def _update_overall_display(self) -> None:
        """Update tampilan overall progress."""
        if 'progress_bar' in self.ui_components:
            progress_bar = self.ui_components['progress_bar']
            progress_bar.value = self.overall_progress
            
            # Update bar style berdasarkan progress
            if self.overall_progress < 30:
                progress_bar.bar_style = 'info'
            elif self.overall_progress < 70:
                progress_bar.bar_style = 'warning'  
            else:
                progress_bar.bar_style = 'success'
        
        if 'overall_label' in self.ui_components and self.overall_message:
            self.ui_components['overall_label'].value = f"""
            <div style="font-weight: bold; color: {COLORS['primary']};">
                {ICONS['progress']} {self.overall_message} ({self.overall_progress:.1f}%)
            </div>
            """
    
    def _update_step_display(self) -> None:
        """Update tampilan step progress."""
        if 'current_progress' in self.ui_components:
            step_percentage = (self.step_progress / self.step_total * 100) if self.step_total > 0 else 0
            current_progress = self.ui_components['current_progress']
            current_progress.value = step_percentage
            
            # Update description
            current_progress.description = f"Step {self.current_step}/{self.total_steps}"
        
        if 'step_label' in self.ui_components and self.step_message:
            self.ui_components['step_label'].value = f"""
            <div style="color: {COLORS['secondary']}; font-size: 0.9em;">
                {self.step_message}
            </div>
            """
    
    def _show_progress_widgets(self) -> None:
        """Tampilkan progress widgets."""
        widgets_to_show = ['progress_bar', 'overall_label', 'current_progress', 'step_label']
        
        for widget_name in widgets_to_show:
            if widget_name in self.ui_components:
                widget = self.ui_components[widget_name]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
        
        # Tampilkan progress container jika ada
        if 'progress_container' in self.ui_components:
            container = self.ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'block'
    
    def _hide_progress_widgets(self) -> None:
        """Sembunyikan progress widgets."""
        widgets_to_hide = ['progress_bar', 'overall_label', 'current_progress', 'step_label']
        
        for widget_name in widgets_to_hide:
            if widget_name in self.ui_components:
                widget = self.ui_components[widget_name]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'hidden'

def create_progress_tracker(ui_components: Dict[str, Any]) -> ProgressTracker:
    """
    Factory function untuk membuat progress tracker.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ProgressTracker: Instance progress tracker
    """
    tracker = ProgressTracker(ui_components)
    ui_components['progress_tracker'] = tracker
    return tracker

def create_progress_callback(progress_tracker: ProgressTracker) -> Callable:
    """
    Buat callback function untuk integrasi dengan backend service.
    
    Args:
        progress_tracker: Instance progress tracker
        
    Returns:
        Callable: Callback function
    """
    def progress_callback(**kwargs) -> bool:
        """
        Callback untuk progress updates dari backend.
        
        Args:
            **kwargs: Parameter progress
            
        Returns:
            bool: False untuk stop, True untuk lanjut
        """
        try:
            # Extract parameters
            step_name = kwargs.get('step', '')
            current = kwargs.get('current', 0)
            total = kwargs.get('total', 100)
            message = kwargs.get('message', '')
            split = kwargs.get('split', '')
            
            # Map step name ke ProcessingSteps
            step_mapping = {
                'preparation': ProcessingSteps.PREPARATION,
                'train': ProcessingSteps.TRAIN_SPLIT,
                'validation': ProcessingSteps.VAL_SPLIT,
                'test': ProcessingSteps.TEST_SPLIT,
                'finalization': ProcessingSteps.FINALIZATION
            }
            
            # Tentukan step dari parameter
            current_step = None
            for key, enum_step in step_mapping.items():
                if key in step_name.lower() or key in split.lower():
                    current_step = enum_step
                    break
            
            # Update progress berdasarkan step
            if current_step:
                # Start step jika belum
                if progress_tracker.current_step == 0 or current == 0:
                    progress_tracker.start_step(current_step, total)
                
                # Update step progress
                progress_tracker.update_step_progress(current, message)
                
                # Complete step jika selesai
                if current >= total:
                    progress_tracker.complete_step(current_step, message)
            else:
                # Update overall jika tidak ada step spesifik
                progress_percentage = (current / total * 100) if total > 0 else 0
                progress_tracker.update_overall_progress(progress_percentage, message)
            
            # Check stop request
            stop_requested = progress_tracker.ui_components.get('stop_requested', False)
            return not stop_requested
            
        except Exception as e:
            logger.error(f"❌ Error progress callback: {str(e)}")
            return True  # Continue on error
    
    return progress_callback