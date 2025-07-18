"""
Operation Container Demo Script

This script demonstrates all functionality of the operation container with progress tracker:
- Single, Dual, and Triple progress levels
- Modern compact styling
- Multiple operations with different patterns
- Error handling and completion states
- Button-triggered operations for interactive testing
"""

import time
import random
import threading
from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel


class OperationContainerDemo:
    """Interactive demo for testing all operation container functionality."""
    
    def __init__(self):
        """Initialize the demo with UI components."""
        self.containers = {}
        self.is_running = False
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Set up the comprehensive demo UI."""
        # Main title
        title = widgets.HTML(
            '<h1 style="color: #2c3e50; margin: 0 0 20px 0;">Operation Container Demo</h1>'
            '<p style="color: #7f8c8d; margin: 0 0 30px 0;">Test all progress tracker functionality with modern compact styling</p>',
            layout=widgets.Layout(width='100%')
        )
        
        # Create different operation containers
        self.create_containers()
        
        # Create control panels
        control_panels = self.create_control_panels()
        
        # Assemble main UI
        self.main_ui = widgets.VBox([
            title,
            *control_panels,
            widgets.HTML('<hr style="margin: 30px 0;">'),
            self.create_operation_displays()
        ], layout=widgets.Layout(
            width='100%',
            padding='20px',
            border='1px solid #e0e0e0',
            border_radius='8px'
        ))
        
        # Display the UI
        display(self.main_ui)
    
    def create_containers(self) -> None:
        """Create operation containers for different test scenarios."""
        # Single-level progress container
        self.containers['single'] = create_operation_container(
            show_progress=True,
            show_logs=True,
            show_dialog=True,
            progress_levels='single',
            log_module_name="Single Progress Demo",
            log_height="150px"
        )
        
        # Dual-level progress container
        self.containers['dual'] = create_operation_container(
            show_progress=True,
            show_logs=True,
            show_dialog=True,
            progress_levels='dual',
            log_module_name="Dual Progress Demo",
            log_height="150px"
        )
        
        # Triple-level progress container
        self.containers['triple'] = create_operation_container(
            show_progress=True,
            show_logs=True,
            show_dialog=True,
            progress_levels='triple',
            log_module_name="Triple Progress Demo",
            log_height="150px"
        )
    
    def create_control_panels(self) -> List[widgets.Widget]:
        """Create control panels for different operations."""
        panels = []
        
        # Single Progress Controls
        single_panel = self.create_single_controls()
        panels.append(single_panel)
        
        # Dual Progress Controls
        dual_panel = self.create_dual_controls()
        panels.append(dual_panel)
        
        # Triple Progress Controls
        triple_panel = self.create_triple_controls()
        panels.append(triple_panel)
        
        # Utility Controls
        utility_panel = self.create_utility_controls()
        panels.append(utility_panel)
        
        return panels
    
    def create_single_controls(self) -> widgets.Widget:
        """Create controls for single-level progress testing."""
        title = widgets.HTML('<h3 style="color: #3498db;">Single Level Progress</h3>')
        
        steps_input = widgets.IntText(value=5, description='Steps:', min=1, max=20, layout=widgets.Layout(width='150px'))
        
        btn_linear = widgets.Button(description='Linear Progress', button_style='info', layout=widgets.Layout(width='150px'))
        btn_linear.on_click(lambda b: self.run_linear_progress('single', steps_input.value))
        
        btn_batch = widgets.Button(description='Batch Process', button_style='primary', layout=widgets.Layout(width='150px'))
        btn_batch.on_click(lambda b: self.run_batch_process('single', steps_input.value))
        
        btn_error = widgets.Button(description='Test Error', button_style='danger', layout=widgets.Layout(width='150px'))
        btn_error.on_click(lambda b: self.test_error_handling('single'))
        
        controls = widgets.HBox([steps_input, btn_linear, btn_batch, btn_error], 
                               layout=widgets.Layout(margin='10px 0', align_items='center'))
        
        return widgets.VBox([title, controls])
    
    def create_dual_controls(self) -> widgets.Widget:
        """Create controls for dual-level progress testing."""
        title = widgets.HTML('<h3 style="color: #e67e22;">Dual Level Progress</h3>')
        
        stages_input = widgets.IntText(value=3, description='Stages:', min=1, max=10, layout=widgets.Layout(width='150px'))
        tasks_input = widgets.IntText(value=4, description='Tasks/Stage:', min=1, max=10, layout=widgets.Layout(width='150px'))
        
        btn_nested = widgets.Button(description='Nested Process', button_style='warning', layout=widgets.Layout(width='150px'))
        btn_nested.on_click(lambda b: self.run_nested_progress('dual', stages_input.value, tasks_input.value))
        
        btn_parallel = widgets.Button(description='Parallel Tasks', button_style='info', layout=widgets.Layout(width='150px'))
        btn_parallel.on_click(lambda b: self.run_parallel_tasks('dual', tasks_input.value))
        
        controls = widgets.HBox([stages_input, tasks_input, btn_nested, btn_parallel], 
                               layout=widgets.Layout(margin='10px 0', align_items='center'))
        
        return widgets.VBox([title, controls])
    
    def create_triple_controls(self) -> widgets.Widget:
        """Create controls for triple-level progress testing."""
        title = widgets.HTML('<h3 style="color: #27ae60;">Triple Level Progress</h3>')
        
        projects_input = widgets.IntText(value=2, description='Projects:', min=1, max=5, layout=widgets.Layout(width='150px'))
        phases_input = widgets.IntText(value=3, description='Phases:', min=1, max=8, layout=widgets.Layout(width='150px'))
        
        btn_complex = widgets.Button(description='Complex Process', button_style='success', layout=widgets.Layout(width='150px'))
        btn_complex.on_click(lambda b: self.run_complex_process('triple', projects_input.value, phases_input.value))
        
        btn_stress = widgets.Button(description='Stress Test', button_style='danger', layout=widgets.Layout(width='150px'))
        btn_stress.on_click(lambda b: self.run_stress_test('triple'))
        
        controls = widgets.HBox([projects_input, phases_input, btn_complex, btn_stress], 
                               layout=widgets.Layout(margin='10px 0', align_items='center'))
        
        return widgets.VBox([title, controls])
    
    def create_utility_controls(self) -> widgets.Widget:
        """Create utility controls for testing."""
        title = widgets.HTML('<h3 style="color: #9b59b6;">Utility Controls</h3>')
        
        btn_reset_all = widgets.Button(description='Reset All', button_style='', layout=widgets.Layout(width='150px'))
        btn_reset_all.on_click(lambda b: self.reset_all_containers())
        
        btn_test_styling = widgets.Button(description='Test Styling', button_style='info', layout=widgets.Layout(width='150px'))
        btn_test_styling.on_click(lambda b: self.test_styling_variations())
        
        btn_stop_all = widgets.Button(description='Stop All', button_style='danger', layout=widgets.Layout(width='150px'))
        btn_stop_all.on_click(lambda b: self.stop_all_operations())
        
        controls = widgets.HBox([btn_reset_all, btn_test_styling, btn_stop_all], 
                               layout=widgets.Layout(margin='10px 0', align_items='center'))
        
        return widgets.VBox([title, controls])
    
    def create_operation_displays(self) -> widgets.Widget:
        """Create the operation container displays."""
        single_container = widgets.VBox([
            widgets.HTML('<h4>Single Level Container</h4>'),
            self.containers['single']['container']
        ])
        
        dual_container = widgets.VBox([
            widgets.HTML('<h4>Dual Level Container</h4>'),
            self.containers['dual']['container']
        ])
        
        triple_container = widgets.VBox([
            widgets.HTML('<h4>Triple Level Container</h4>'),
            self.containers['triple']['container']
        ])
        
        return widgets.VBox([single_container, dual_container, triple_container])
    
    # Operation Methods
    
    def run_linear_progress(self, container_type: str, steps: int) -> None:
        """Run a simple linear progress operation."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log'](f"Starting linear progress with {steps} steps", 'info')
                
                for step in range(1, steps + 1):
                    progress = int((step / steps) * 100)
                    container['update_progress'](progress, f"Processing step {step}/{steps}", 'primary')
                    container['log'](f"Completed step {step}", 'success')
                    time.sleep(0.3)
                
                container['show_dialog']("Success", f"Linear process completed with {steps} steps!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_batch_process(self, container_type: str, batches: int) -> None:
        """Run a batch processing operation."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log'](f"Starting batch process with {batches} batches", 'info')
                
                for batch in range(1, batches + 1):
                    # Simulate variable batch sizes
                    batch_size = random.randint(10, 100)
                    container['log'](f"Processing batch {batch} with {batch_size} items", 'info')
                    
                    # Process batch with incremental updates
                    for item in range(1, batch_size + 1):
                        progress = int(((batch - 1) * 100 + (item / batch_size) * 100) / batches)
                        container['update_progress'](progress, f"Batch {batch}: Item {item}/{batch_size}", 'primary')
                        
                        if item % 10 == 0:  # Log every 10 items
                            time.sleep(0.1)
                    
                    container['log'](f"Batch {batch} completed", 'success')
                
                container['show_dialog']("Success", f"All {batches} batches processed!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_nested_progress(self, container_type: str, stages: int, tasks_per_stage: int) -> None:
        """Run a nested progress operation with dual levels."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log'](f"Starting nested process: {stages} stages, {tasks_per_stage} tasks each", 'info')
                
                for stage in range(1, stages + 1):
                    stage_progress = int(((stage - 1) / stages) * 100)
                    container['update_progress'](stage_progress, f"Stage {stage}/{stages}", 'primary')
                    container['log'](f"Starting stage {stage}", 'info')
                    
                    for task in range(1, tasks_per_stage + 1):
                        task_progress = int((task / tasks_per_stage) * 100)
                        container['update_progress'](task_progress, f"Task {task}/{tasks_per_stage}", 'secondary')
                        time.sleep(0.2)
                    
                    container['update_progress'](100, f"Stage {stage} complete", 'secondary')
                    container['log'](f"Stage {stage} completed", 'success')
                
                container['update_progress'](100, "All stages complete", 'primary')
                container['show_dialog']("Success", f"Nested process completed!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_parallel_tasks(self, container_type: str, num_tasks: int) -> None:
        """Simulate parallel task processing."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log'](f"Starting {num_tasks} parallel tasks", 'info')
                completed_tasks = 0
                
                # Simulate tasks completing at random intervals
                while completed_tasks < num_tasks:
                    time.sleep(random.uniform(0.1, 0.5))
                    completed_tasks += 1
                    
                    overall_progress = int((completed_tasks / num_tasks) * 100)
                    container['update_progress'](overall_progress, f"Tasks: {completed_tasks}/{num_tasks}", 'primary')
                    
                    # Simulate current task progress
                    if completed_tasks < num_tasks:
                        current_progress = random.randint(0, 90)
                        container['update_progress'](current_progress, f"Processing task {completed_tasks + 1}", 'secondary')
                    else:
                        container['update_progress'](100, "All tasks complete", 'secondary')
                    
                    container['log'](f"Task {completed_tasks} completed", 'success')
                
                container['show_dialog']("Success", f"All {num_tasks} parallel tasks completed!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_complex_process(self, container_type: str, projects: int, phases: int) -> None:
        """Run a complex three-level process."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log'](f"Starting complex process: {projects} projects, {phases} phases each", 'info')
                
                for project in range(1, projects + 1):
                    project_progress = int(((project - 1) / projects) * 100)
                    container['update_progress'](project_progress, f"Project {project}/{projects}", 'primary')
                    container['log'](f"Starting project {project}", 'info')
                    
                    for phase in range(1, phases + 1):
                        phase_progress = int((phase / phases) * 100)
                        container['update_progress'](phase_progress, f"Phase {phase}/{phases}", 'secondary')
                        
                        # Simulate subtasks within each phase
                        subtasks = random.randint(3, 8)
                        for subtask in range(1, subtasks + 1):
                            subtask_progress = int((subtask / subtasks) * 100)
                            container['update_progress'](subtask_progress, f"Subtask {subtask}/{subtasks}", 'tertiary')
                            time.sleep(0.1)
                        
                        container['log'](f"Project {project}, Phase {phase} completed", 'success')
                    
                    container['update_progress'](100, f"Project {project} complete", 'secondary')
                    container['update_progress'](100, "All subtasks complete", 'tertiary')
                
                container['update_progress'](100, "All projects complete", 'primary')
                container['show_dialog']("Success", f"Complex process completed: {projects} projects!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_stress_test(self, container_type: str) -> None:
        """Run a stress test with rapid updates."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log']("Starting stress test with rapid updates", 'info')
                
                total_iterations = 100
                for i in range(total_iterations):
                    overall_progress = int((i / total_iterations) * 100)
                    container['update_progress'](overall_progress, f"Iteration {i+1}/{total_iterations}", 'primary')
                    
                    # Rapid secondary updates
                    for j in range(10):
                        secondary_progress = int((j / 10) * 100)
                        container['update_progress'](secondary_progress, f"Sub-iteration {j+1}/10", 'secondary')
                        
                        # Rapid tertiary updates
                        for k in range(5):
                            tertiary_progress = int((k / 5) * 100)
                            container['update_progress'](tertiary_progress, f"Micro-task {k+1}/5", 'tertiary')
                            time.sleep(0.01)  # Very fast updates
                    
                    if i % 20 == 0:
                        container['log'](f"Stress test: {i+1} iterations completed", 'info')
                
                container['show_dialog']("Success", "Stress test completed successfully!", "success")
                
            except Exception as e:
                container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def test_error_handling(self, container_type: str) -> None:
        """Test error handling and recovery."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            container = self.containers[container_type]
            
            try:
                container['log']("Testing error handling", 'info')
                
                # Simulate progress before error
                for i in range(1, 6):
                    progress = i * 20
                    container['update_progress'](progress, f"Step {i}/5", 'primary')
                    time.sleep(0.3)
                
                # Simulate error
                container['log']("Simulating error condition", 'warning')
                raise Exception("Simulated error for testing")
                
            except Exception as e:
                container['log'](f"Error occurred: {str(e)}", 'error')
                container['show_dialog']("Error", f"Error handling test: {str(e)}", "error")
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def test_styling_variations(self) -> None:
        """Test different styling variations."""
        for container_type, container in self.containers.items():
            container['log'](f"Testing {container_type} styling variations", 'info')
            
            # Test different status types
            status_types = ['info', 'success', 'warning', 'error']
            for i, status_type in enumerate(status_types):
                progress = (i + 1) * 25
                container['update_progress'](progress, f"Testing {status_type} styling", 'primary')
                time.sleep(0.5)
    
    def reset_all_containers(self) -> None:
        """Reset all containers to initial state."""
        for container_type, container in self.containers.items():
            try:
                if 'progress_tracker' in container and container['progress_tracker']:
                    container['progress_tracker'].reset()
                if 'log_accordion' in container and container['log_accordion']:
                    container['log_accordion'].clear()
                container['log'](f"{container_type.title()} container reset", 'info')
            except Exception as e:
                print(f"Error resetting {container_type}: {e}")
    
    def stop_all_operations(self) -> None:
        """Stop all running operations."""
        self.is_running = False
        for container in self.containers.values():
            container['log']("All operations stopped", 'warning')


# Create and display the demo
if __name__ == "__main__" or 'get_ipython' in globals():
    print("=== Operation Container Comprehensive Demo ===")
    demo = OperationContainerDemo()