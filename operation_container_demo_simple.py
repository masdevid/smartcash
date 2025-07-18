"""
Simplified Operation Container Demo

This script demonstrates a single operation container that can change state 
based on button tests, including dialog functionality.
"""

import time
import random
import threading
from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.components.operation_container import create_operation_container


class SimplifiedOperationDemo:
    """Simplified demo with one operation container that changes state based on buttons."""
    
    def __init__(self):
        """Initialize the demo with UI components."""
        self.container = None
        self.current_level = 'single'
        self.is_running = False
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Set up the simplified demo UI."""
        # Create title
        title = widgets.HTML(
            '<h1 style="color: #2c3e50; margin: 0 0 20px 0;">Operation Container Demo</h1>'
            '<p style="color: #7f8c8d; margin: 0 0 20px 0;">Single container that changes state based on selected operation type</p>',
            layout=widgets.Layout(width='100%')
        )
        
        # Create operation container (start with triple level)
        self.create_container('triple')
        
        # Create control panels
        controls = self.create_controls()
        
        # Create progress level selector
        level_selector = self.create_level_selector()
        
        # Assemble main UI
        self.main_ui = widgets.VBox([
            title,
            level_selector,
            widgets.HTML('<hr style="margin: 20px 0;">'),
            controls,
            widgets.HTML('<hr style="margin: 20px 0;">'),
            self.container['container']
        ], layout=widgets.Layout(
            width='100%',
            padding='20px',
            border='1px solid #e0e0e0',
            border_radius='8px'
        ))
        
        # Display the UI
        display(self.main_ui)
    
    def create_container(self, level: str) -> None:
        """Create or recreate the operation container with specified level."""
        self.current_level = level
        self.container = create_operation_container(
            show_progress=True,
            show_logs=True,
            show_dialog=True,
            progress_levels=level,
            log_module_name=f"{level.title()} Level Demo",
            log_height="200px"
        )
    
    def create_level_selector(self) -> widgets.Widget:
        """Create level selector controls."""
        title = widgets.HTML('<h3 style="color: #8e44ad;">Progress Level Configuration</h3>')
        
        btn_single = widgets.Button(description='Single Level', button_style='info', layout=widgets.Layout(width='150px'))
        btn_dual = widgets.Button(description='Dual Level', button_style='warning', layout=widgets.Layout(width='150px'))
        btn_triple = widgets.Button(description='Triple Level', button_style='success', layout=widgets.Layout(width='150px'))
        
        def change_to_single(b):
            self.change_level('single')
            self.update_button_styles(btn_single, [btn_dual, btn_triple])
        
        def change_to_dual(b):
            self.change_level('dual')
            self.update_button_styles(btn_dual, [btn_single, btn_triple])
        
        def change_to_triple(b):
            self.change_level('triple')
            self.update_button_styles(btn_triple, [btn_single, btn_dual])
        
        btn_single.on_click(change_to_single)
        btn_dual.on_click(change_to_dual)
        btn_triple.on_click(change_to_triple)
        
        # Start with triple selected
        self.update_button_styles(btn_triple, [btn_single, btn_dual])
        
        controls = widgets.HBox([btn_single, btn_dual, btn_triple], 
                               layout=widgets.Layout(margin='10px 0', justify_content='center'))
        
        return widgets.VBox([title, controls])
    
    def update_button_styles(self, active_button, inactive_buttons):
        """Update button styles to show active state."""
        active_button.button_style = 'primary'
        for btn in inactive_buttons:
            if btn.description == 'Single Level':
                btn.button_style = 'info'
            elif btn.description == 'Dual Level':
                btn.button_style = 'warning'
            else:
                btn.button_style = 'success'
    
    def change_level(self, new_level: str) -> None:
        """Change the operation container level."""
        if new_level != self.current_level and not self.is_running:
            self.create_container(new_level)
            # Update the container in the main UI
            children = list(self.main_ui.children)
            children[-1] = self.container['container']
            self.main_ui.children = children
            
            self.container['log'](f"Switched to {new_level} level progress tracking", 'info')
    
    def create_controls(self) -> widgets.Widget:
        """Create operation controls."""
        title = widgets.HTML('<h3 style="color: #e74c3c;">Operation Controls</h3>')
        
        # Input controls
        self.steps_input = widgets.IntText(value=3, description='Steps:', min=1, max=10, layout=widgets.Layout(width='150px'))
        self.delay_input = widgets.FloatText(value=0.3, description='Delay (s):', min=0.1, max=2.0, layout=widgets.Layout(width='150px'))
        
        # Operation buttons
        btn_linear = widgets.Button(description='Linear Progress', button_style='info', layout=widgets.Layout(width='150px'))
        btn_nested = widgets.Button(description='Nested Progress', button_style='warning', layout=widgets.Layout(width='150px'))
        btn_complex = widgets.Button(description='Complex Process', button_style='success', layout=widgets.Layout(width='150px'))
        btn_error = widgets.Button(description='Test Error', button_style='danger', layout=widgets.Layout(width='150px'))
        
        # Utility buttons
        btn_dialog_success = widgets.Button(description='Success Dialog', button_style='success', layout=widgets.Layout(width='150px'))
        btn_dialog_error = widgets.Button(description='Error Dialog', button_style='danger', layout=widgets.Layout(width='150px'))
        btn_reset = widgets.Button(description='Reset', button_style='', layout=widgets.Layout(width='150px'))
        btn_clear_logs = widgets.Button(description='Clear Logs', button_style='', layout=widgets.Layout(width='150px'))
        
        # Wire up button handlers
        btn_linear.on_click(lambda b: self.run_linear_progress())
        btn_nested.on_click(lambda b: self.run_nested_progress())
        btn_complex.on_click(lambda b: self.run_complex_progress())
        btn_error.on_click(lambda b: self.test_error_handling())
        btn_dialog_success.on_click(lambda b: self.test_success_dialog())
        btn_dialog_error.on_click(lambda b: self.test_error_dialog())
        btn_reset.on_click(lambda b: self.reset_container())
        btn_clear_logs.on_click(lambda b: self.clear_logs())
        
        # Layout controls
        input_row = widgets.HBox([self.steps_input, self.delay_input], 
                                layout=widgets.Layout(margin='10px 0', justify_content='center'))
        
        operation_row = widgets.HBox([btn_linear, btn_nested, btn_complex, btn_error], 
                                   layout=widgets.Layout(margin='10px 0', justify_content='center'))
        
        utility_row = widgets.HBox([btn_dialog_success, btn_dialog_error, btn_reset, btn_clear_logs], 
                                  layout=widgets.Layout(margin='10px 0', justify_content='center'))
        
        return widgets.VBox([title, input_row, operation_row, utility_row])
    
    # Operation Methods
    
    def run_linear_progress(self) -> None:
        """Run linear progress operation."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            steps = self.steps_input.value
            delay = self.delay_input.value
            
            try:
                self.container['log'](f"Starting linear progress: {steps} steps", 'info')
                
                for step in range(1, steps + 1):
                    progress = int((step / steps) * 100)
                    self.container['update_progress'](progress, f"Step {step}/{steps}", 'primary')
                    
                    if self.current_level in ['dual', 'triple']:
                        # Add secondary progress for multi-level
                        sub_progress = int((step % 2) * 100)  # Alternating pattern
                        self.container['update_progress'](sub_progress, f"Sub-step {step}", 'secondary')
                    
                    if self.current_level == 'triple':
                        # Add tertiary progress for triple level
                        detail_progress = random.randint(20, 100)
                        self.container['update_progress'](detail_progress, f"Detail {step}", 'tertiary')
                    
                    self.container['log'](f"Completed step {step}", 'success')
                    time.sleep(delay)
                
                # Complete all levels
                self.container['update_progress'](100, "Linear process complete!", 'primary')
                if self.current_level in ['dual', 'triple']:
                    self.container['update_progress'](100, "All sub-steps complete", 'secondary')
                if self.current_level == 'triple':
                    self.container['update_progress'](100, "All details complete", 'tertiary')
                
                self.container['show_dialog']("Success", f"Linear process completed with {steps} steps!", "success")
                
            except Exception as e:
                self.container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_nested_progress(self) -> None:
        """Run nested progress operation."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            stages = self.steps_input.value
            delay = self.delay_input.value
            
            try:
                self.container['log'](f"Starting nested process: {stages} stages", 'info')
                
                for stage in range(1, stages + 1):
                    # Update primary progress
                    primary_progress = int(((stage - 1) / stages) * 100)
                    self.container['update_progress'](primary_progress, f"Stage {stage}/{stages}", 'primary')
                    self.container['log'](f"Starting stage {stage}", 'info')
                    
                    # Process tasks within each stage
                    tasks = random.randint(2, 5)
                    for task in range(1, tasks + 1):
                        if self.current_level in ['dual', 'triple']:
                            task_progress = int((task / tasks) * 100)
                            self.container['update_progress'](task_progress, f"Task {task}/{tasks}", 'secondary')
                        
                        # Process subtasks for triple level
                        if self.current_level == 'triple':
                            subtasks = random.randint(2, 4)
                            for subtask in range(1, subtasks + 1):
                                subtask_progress = int((subtask / subtasks) * 100)
                                self.container['update_progress'](subtask_progress, f"Subtask {subtask}/{subtasks}", 'tertiary')
                                time.sleep(delay * 0.3)
                        
                        time.sleep(delay * 0.5)
                    
                    # Complete stage
                    if self.current_level in ['dual', 'triple']:
                        self.container['update_progress'](100, f"Stage {stage} complete", 'secondary')
                    if self.current_level == 'triple':
                        self.container['update_progress'](100, "All subtasks complete", 'tertiary')
                    
                    self.container['log'](f"Stage {stage} completed", 'success')
                
                # Complete all
                self.container['update_progress'](100, "Nested process complete!", 'primary')
                self.container['show_dialog']("Success", f"Nested process completed with {stages} stages!", "success")
                
            except Exception as e:
                self.container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def run_complex_progress(self) -> None:
        """Run complex progress operation."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            projects = self.steps_input.value
            delay = self.delay_input.value
            
            try:
                self.container['log'](f"Starting complex process: {projects} projects", 'info')
                
                for project in range(1, projects + 1):
                    # Update primary
                    primary_progress = int(((project - 1) / projects) * 100)
                    self.container['update_progress'](primary_progress, f"Project {project}/{projects}", 'primary')
                    self.container['log'](f"Starting project {project}", 'info')
                    
                    # Phases within project
                    phases = random.randint(2, 4)
                    for phase in range(1, phases + 1):
                        if self.current_level in ['dual', 'triple']:
                            phase_progress = int(((phase - 1) / phases) * 100)
                            self.container['update_progress'](phase_progress, f"Phase {phase}/{phases}", 'secondary')
                        
                        # Tasks within phase
                        if self.current_level == 'triple':
                            tasks = random.randint(3, 6)
                            for task in range(1, tasks + 1):
                                task_progress = int((task / tasks) * 100)
                                self.container['update_progress'](task_progress, f"Task {task}/{tasks}", 'tertiary')
                                time.sleep(delay * 0.2)
                        
                        time.sleep(delay * 0.4)
                    
                    # Complete project
                    if self.current_level in ['dual', 'triple']:
                        self.container['update_progress'](100, f"Project {project} complete", 'secondary')
                    if self.current_level == 'triple':
                        self.container['update_progress'](100, "All tasks complete", 'tertiary')
                    
                    self.container['log'](f"Project {project} completed", 'success')
                
                # Complete all
                self.container['update_progress'](100, "Complex process complete!", 'primary')
                self.container['show_dialog']("Success", f"Complex process completed with {projects} projects!", "success")
                
            except Exception as e:
                self.container['log'](f"Error: {str(e)}", 'error')
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def test_error_handling(self) -> None:
        """Test error handling."""
        if self.is_running:
            return
        
        def operation():
            self.is_running = True
            
            try:
                self.container['log']("Testing error handling", 'info')
                
                # Simulate some progress
                for i in range(1, 4):
                    progress = i * 25
                    self.container['update_progress'](progress, f"Step {i}/3", 'primary')
                    if self.current_level in ['dual', 'triple']:
                        self.container['update_progress'](progress + 10, f"Sub-step {i}", 'secondary')
                    if self.current_level == 'triple':
                        self.container['update_progress'](progress + 20, f"Detail {i}", 'tertiary')
                    time.sleep(0.5)
                
                # Simulate error
                self.container['log']("Simulating error condition", 'warning')
                raise Exception("Simulated error for testing purposes")
                
            except Exception as e:
                self.container['log'](f"Error occurred: {str(e)}", 'error')
                self.container['show_dialog']("Error", f"Operation failed: {str(e)}", "error")
            finally:
                self.is_running = False
        
        threading.Thread(target=operation, daemon=True).start()
    
    def test_success_dialog(self) -> None:
        """Test success dialog."""
        self.container['show_dialog'](
            "Success Test", 
            "This is a test of the success dialog functionality. Everything is working correctly!", 
            "success"
        )
        self.container['log']("Success dialog displayed", 'success')
    
    def test_error_dialog(self) -> None:
        """Test error dialog."""
        self.container['show_dialog'](
            "Error Test", 
            "This is a test of the error dialog functionality. This simulates an error condition.", 
            "error"
        )
        self.container['log']("Error dialog displayed", 'error')
    
    def reset_container(self) -> None:
        """Reset the operation container."""
        try:
            if 'progress_tracker' in self.container and self.container['progress_tracker']:
                self.container['progress_tracker'].reset()
            self.container['log']("Container reset to initial state", 'info')
        except Exception as e:
            print(f"Error resetting container: {e}")
    
    def clear_logs(self) -> None:
        """Clear the logs."""
        try:
            if 'log_accordion' in self.container and self.container['log_accordion']:
                self.container['log_accordion'].clear()
        except Exception as e:
            print(f"Error clearing logs: {e}")


# Create and display the demo
def main():
    """Main function to create and display the demo."""
    print("=== Simplified Operation Container Demo ===")
    demo = SimplifiedOperationDemo()
    return demo

if __name__ == "__main__":
    main()
else:
    # Check if we're in a Jupyter notebook environment
    try:
        get_ipython()
        # We're in Jupyter, create the demo
        demo = main()
    except NameError:
        # Not in Jupyter, just import
        pass