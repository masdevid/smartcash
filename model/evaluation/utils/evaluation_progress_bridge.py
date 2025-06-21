"""
File: smartcash/model/evaluation/utils/evaluation_progress_bridge.py
Deskripsi: Progress tracking bridge untuk evaluation pipeline dengan UI integration
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class EvaluationProgressBridge:
    """Progress bridge untuk evaluation pipeline dengan multi-level tracking"""
    
    def __init__(self, ui_components: Dict[str, Any] = None, callback: Optional[Callable] = None):
        self.ui_components = ui_components or {}
        self.progress_callback = callback
        self.logger = get_logger('evaluation_progress')
        
        # Progress state
        self.current_operation = None
        self.total_scenarios = 0
        self.current_scenario = 0
        self.total_checkpoints = 0
        self.current_checkpoint = 0
        self.current_metrics_progress = 0
        
    def start_evaluation(self, scenarios: list, checkpoints: list, operation: str = "Evaluation") -> None:
        """🚀 Initialize evaluation progress tracking"""
        self.current_operation = operation
        self.total_scenarios = len(scenarios)
        self.total_checkpoints = len(checkpoints)
        self.current_scenario = 0
        self.current_checkpoint = 0
        self.current_metrics_progress = 0
        
        self._update_progress(0, f"🚀 Memulai {operation}")
        self.logger.info(f"🚀 Started {operation}: {self.total_scenarios} scenarios, {self.total_checkpoints} checkpoints")
    
    def update_scenario(self, scenario_idx: int, scenario_name: str, message: str = "") -> None:
        """🎯 Update scenario progress"""
        self.current_scenario = scenario_idx
        overall_progress = self._calculate_overall_progress()
        
        status_message = f"🎯 Scenario {scenario_idx + 1}/{self.total_scenarios}: {scenario_name}"
        if message:
            status_message += f" - {message}"
        
        self._update_progress(overall_progress, status_message)
        self.logger.info(f"🎯 Scenario progress: {scenario_name} ({scenario_idx + 1}/{self.total_scenarios})")
    
    def update_checkpoint(self, checkpoint_idx: int, checkpoint_name: str, message: str = "") -> None:
        """🏗️ Update checkpoint progress"""
        self.current_checkpoint = checkpoint_idx
        overall_progress = self._calculate_overall_progress()
        
        status_message = f"🏗️ Checkpoint {checkpoint_idx + 1}/{self.total_checkpoints}: {checkpoint_name}"
        if message:
            status_message += f" - {message}"
        
        self._update_progress(overall_progress, status_message)
        self.logger.info(f"🏗️ Checkpoint progress: {checkpoint_name} ({checkpoint_idx + 1}/{self.total_checkpoints})")
    
    def update_metrics(self, metrics_progress: int, message: str = "Calculating metrics") -> None:
        """📊 Update metrics calculation progress"""
        self.current_metrics_progress = min(metrics_progress, 100)
        overall_progress = self._calculate_overall_progress()
        
        status_message = f"📊 {message} ({metrics_progress}%)"
        
        self._update_progress(overall_progress, status_message)
        self.logger.debug(f"📊 Metrics progress: {metrics_progress}%")
    
    def complete_evaluation(self, message: str = "Evaluation completed successfully!") -> None:
        """✅ Mark evaluation complete"""
        self._update_progress(100, f"✅ {message}")
        self.logger.info(f"✅ {self.current_operation} completed")
        
        # Reset state
        self.current_operation = None
        self.current_scenario = 0
        self.current_checkpoint = 0
        self.current_metrics_progress = 0
    
    def evaluation_error(self, error_message: str) -> None:
        """❌ Handle evaluation errors"""
        self._update_progress(None, f"❌ Error: {error_message}")
        self.logger.error(f"❌ {self.current_operation} error: {error_message}")
    
    def update_substep(self, substep_message: str, substep_progress: int = None) -> None:
        """🔄 Update current substep progress"""
        if substep_progress is not None:
            self.current_metrics_progress = substep_progress
        
        overall_progress = self._calculate_overall_progress()
        self._update_progress(overall_progress, f"🔄 {substep_message}")
        self.logger.debug(f"🔄 Substep: {substep_message}")
    
    def _calculate_overall_progress(self) -> int:
        """📊 Calculate overall progress percentage"""
        if self.total_scenarios == 0 or self.total_checkpoints == 0:
            return 0
        
        # Progress calculation:
        # Each scenario-checkpoint combination contributes equally
        total_combinations = self.total_scenarios * self.total_checkpoints
        completed_combinations = (self.current_scenario * self.total_checkpoints + 
                                self.current_checkpoint)
        
        # Add partial progress dari current metrics calculation
        current_combination_progress = self.current_metrics_progress / 100.0
        
        overall_progress = ((completed_combinations + current_combination_progress) / 
                          total_combinations * 100)
        
        return min(int(overall_progress), 100)
    
    def _update_progress(self, progress: Optional[int], message: str) -> None:
        """📡 Update progress via callback dan UI components"""
        progress_data = {
            'overall_progress': progress,
            'scenario_progress': (self.current_scenario / self.total_scenarios * 100) if self.total_scenarios > 0 else 0,
            'checkpoint_progress': (self.current_checkpoint / self.total_checkpoints * 100) if self.total_checkpoints > 0 else 0,
            'metrics_progress': self.current_metrics_progress,
            'message': message,
            'operation': self.current_operation,
            'current_scenario': self.current_scenario,
            'total_scenarios': self.total_scenarios,
            'current_checkpoint': self.current_checkpoint,
            'total_checkpoints': self.total_checkpoints
        }
        
        # Call progress callback jika tersedia
        if self.progress_callback:
            try:
                self.progress_callback(progress_data)
            except Exception as e:
                self.logger.debug(f"⚠️ Progress callback error: {str(e)}")
        
        # Update UI components jika tersedia
        self._update_ui_components(progress_data)
    
    def _update_ui_components(self, progress_data: Dict[str, Any]) -> None:
        """🎨 Update UI components dengan progress data"""
        # Update progress tracker jika tersedia
        tracker = self.ui_components.get('progress_tracker')
        if tracker and hasattr(tracker, 'update'):
            try:
                if progress_data['overall_progress'] is not None:
                    tracker.update('overall', progress_data['overall_progress'], progress_data['message'])
                
                if progress_data['scenario_progress'] > 0:
                    tracker.update('current', progress_data['scenario_progress'], 
                                 f"Scenario {progress_data['current_scenario'] + 1}/{progress_data['total_scenarios']}")
            except Exception as e:
                self.logger.debug(f"⚠️ Progress tracker update error: {str(e)}")
        
        # Update status widget jika tersedia
        status_widget = self.ui_components.get('status')
        if status_widget and hasattr(status_widget, 'clear_output'):
            try:
                with status_widget:
                    from IPython.display import display, HTML
                    display(HTML(f"<div style='color: #007bff;'>{progress_data['message']}</div>"))
            except Exception as e:
                self.logger.debug(f"⚠️ Status widget update error: {str(e)}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """📋 Get current progress summary"""
        return {
            'operation': self.current_operation,
            'overall_progress': self._calculate_overall_progress(),
            'scenario_progress': {
                'current': self.current_scenario,
                'total': self.total_scenarios,
                'percentage': (self.current_scenario / self.total_scenarios * 100) if self.total_scenarios > 0 else 0
            },
            'checkpoint_progress': {
                'current': self.current_checkpoint,
                'total': self.total_checkpoints,
                'percentage': (self.current_checkpoint / self.total_checkpoints * 100) if self.total_checkpoints > 0 else 0
            },
            'metrics_progress': self.current_metrics_progress
        }


# Factory functions
def create_evaluation_progress_bridge(ui_components: Dict[str, Any] = None, 
                                    callback: Optional[Callable] = None) -> EvaluationProgressBridge:
    """🏭 Factory untuk EvaluationProgressBridge"""
    return EvaluationProgressBridge(ui_components, callback)

def create_progress_callback(bridge: EvaluationProgressBridge) -> Callable:
    """📡 Create progress callback function untuk backend services"""
    def progress_callback(level: str, current: int, total: int, message: str = ""):
        if level == 'scenario':
            bridge.update_scenario(current - 1, f"Scenario {current}", message)
        elif level == 'checkpoint':
            bridge.update_checkpoint(current - 1, f"Checkpoint {current}", message)
        elif level == 'metrics':
            progress_percent = int((current / total) * 100) if total > 0 else 0
            bridge.update_metrics(progress_percent, message)
        else:
            bridge.update_substep(message, int((current / total) * 100) if total > 0 else None)
    
    return progress_callback