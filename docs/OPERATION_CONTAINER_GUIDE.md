# 📋 Operation Container Setup & Usage Guide

## Overview

The Operation Container provides a unified interface for managing operation-related UI components including progress tracking, logging, and dialogs. This guide covers setup, usage, and troubleshooting.

## 🚀 Quick Start

### Basic Setup

```python
from smartcash.ui.components.operation_container import create_operation_container

# Create operation container with all features
operation_container = create_operation_container(
    show_progress=True,
    show_logs=True,
    show_dialog=True,
    title="My Operation"
)

# Access components
container_widget = operation_container['container']
progress_tracker = operation_container['progress_tracker']
log_accordion = operation_container['log_accordion']
```

### Integration with OperationHandler

```python
from smartcash.ui.core.handlers.operation_handler import OperationHandler

class MyOperationManager(OperationHandler):
    def __init__(self, config, operation_container):
        super().__init__(
            module_name='my_module',
            parent_module='parent',
            operation_container=operation_container
        )
        
    def execute_operation(self):
        # Clear previous logs
        self.clear_logs()
        
        # Update progress
        self.update_progress(25, "Starting operation...")
        
        # Log messages
        self.log("Operation started", 'info')
        
        # Do work...
        
        # Complete
        self.update_progress(100, "Operation completed")
        self.log("✅ Operation completed successfully", 'success')
```

## 📊 Progress Tracking

### Basic Progress Updates

```python
# Update progress (0-100)
operation_container['update_progress'](50, "Processing data...")

# Mark as complete
operation_container['update_progress'](100, "Completed!")

# Reset progress
operation_container['progress_tracker'].reset()
```

### Multi-level Progress

```python
# Create with multiple progress levels
operation_container = create_operation_container(
    show_progress=True,
    progress_levels='triple'  # 'single', 'dual', 'triple'
)

# Update different levels
operation_container['update_progress'](30, "Overall progress", 'primary')
operation_container['update_progress'](60, "Current step", 'secondary')
operation_container['update_progress'](80, "Sub-task", 'tertiary')
```

### Progress Visibility Control

```python
progress_tracker = operation_container['progress_tracker']

# Show/hide progress tracker
progress_tracker.show()
progress_tracker.hide()

# Show/hide specific levels
progress_tracker.set_progress_visibility('secondary', False)
```

## 📝 Logging

### Basic Logging

```python
# Log messages with different levels
log_fn = operation_container['log_message']

log_fn("Operation started", 'info')
log_fn("Warning: High memory usage", 'warning')
log_fn("Error occurred", 'error')
log_fn("Operation completed", 'success')
```

### Log Filtering

```python
# Filter logs by namespace
operation_container = create_operation_container(
    show_logs=True,
    log_namespace_filter='smartcash.model'  # Only show model logs
)

# Access log accordion for advanced operations
log_accordion = operation_container['log_accordion']
log_accordion.clear_logs()
log_accordion.set_filter('my_module')
```

### Integration with Python Logging

```python
import logging
from smartcash.ui.components.log_accordion import setup_ui_logging

# Setup logging to redirect to UI
logger = setup_ui_logging(
    log_accordion=operation_container['log_accordion'],
    module_name='my_module'
)

# Use standard logging
logger.info("This will appear in the UI")
logger.warning("This is a warning")
```

## 💬 Dialog Management

### Confirmation Dialogs

```python
# Show confirmation dialog
def on_confirm():
    print("User confirmed!")

def on_cancel():
    print("User cancelled!")

operation_container['show_dialog'](
    title="Confirm Operation",
    message="Are you sure you want to continue?",
    dialog_type="confirmation",
    on_confirm=on_confirm,
    on_cancel=on_cancel
)
```

### Info Dialogs

```python
# Show info dialog
operation_container['show_info_dialog'](
    title="Operation Complete",
    message="The operation completed successfully!",
    auto_close=3000  # Auto-close after 3 seconds
)
```

### Custom Dialogs

```python
import ipywidgets as widgets

# Create custom dialog content
custom_content = widgets.VBox([
    widgets.HTML("<h3>Custom Dialog</h3>"),
    widgets.Text(placeholder="Enter value"),
    widgets.Button(description="Submit")
])

operation_container['show_dialog'](
    title="Custom Dialog",
    content=custom_content,
    dialog_type="custom"
)
```

## 🔧 Advanced Configuration

### Custom Progress Configuration

```python
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel

# Custom progress configuration
progress_config = ProgressConfig(
    level=ProgressLevel.DUAL,
    operation="Custom Operation",
    auto_hide=True,
    auto_hide_delay=2
)

operation_container = create_operation_container(
    show_progress=True,
    progress_config=progress_config
)
```

### Custom Log Configuration

```python
operation_container = create_operation_container(
    show_logs=True,
    log_module_name="Custom Module",
    log_height="400px",
    log_namespace_filter="smartcash.custom"
)
```

## 🛠️ Integration Patterns

### With UIModule

```python
class MyUIModule(UIModule):
    def __init__(self):
        super().__init__(module_name='my_module')
        
        # Create operation container
        self.operation_container = create_operation_container(
            show_progress=True,
            show_logs=True,
            log_module_name=self.module_name
        )
        
        # Add to UI components
        self.ui_components['operation_container'] = self.operation_container
    
    def get_main_widget(self):
        # Include operation container in main UI
        return widgets.VBox([
            self.create_form(),
            self.create_buttons(),
            self.operation_container['container']
        ])
```

### With Backend Services

```python
class ServiceWithProgress:
    def __init__(self, operation_container):
        self.operation_container = operation_container
        
    def long_running_operation(self):
        steps = ['Init', 'Process', 'Validate', 'Complete']
        
        for i, step in enumerate(steps):
            progress = (i + 1) * 25
            self.operation_container['update_progress'](
                progress, 
                f"Step {i+1}/{len(steps)}: {step}"
            )
            self.operation_container['log_message'](
                f"Executing {step}...", 
                'info'
            )
            
            # Do actual work
            time.sleep(1)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Progress Tracker Not Visible

**Problem**: Progress tracker shows tqdm instead of UI widgets

**Solution**:
```python
# Ensure progress tracker is properly initialized
operation_container = create_operation_container(show_progress=True)
progress_tracker = operation_container['progress_tracker']

# Manually initialize if needed
if not progress_tracker._initialized:
    progress_tracker.initialize()

# Show the tracker
progress_tracker.show()
```

#### 2. Logs Not Appearing

**Problem**: Log messages not showing in accordion

**Solution**:
```python
# Check log accordion initialization
log_accordion = operation_container['log_accordion']
if not log_accordion._initialized:
    log_accordion.initialize()

# Check namespace filtering
log_accordion.set_filter(None)  # Remove filters

# Test direct logging
log_accordion.add_log("Test message", 'info')
```

#### 3. Operation Container Empty

**Problem**: Container has no visible children

**Solution**:
```python
# Check container children
container = operation_container['container']
print(f"Children: {len(container.children)}")

# Recreate with explicit options
operation_container = create_operation_container(
    show_progress=True,
    show_logs=True,
    show_dialog=True
)
```

#### 4. Button Handlers Not Working

**Problem**: Operation buttons don't respond

**Solution**:
```python
# In OperationHandler, ensure proper async handling
class MyOperationManager(OperationHandler):
    def execute_operation(self):
        try:
            # Clear logs at start
            self.clear_logs()
            
            # Disable buttons during operation
            button_states = self.disable_all_buttons("⏳ Processing...")
            
            # Your operation code here
            
            return {'success': True}
        finally:
            # Re-enable buttons
            self.enable_all_buttons(button_states)
```

### Debug Mode

```python
# Enable debug logging for operation container
import logging
logging.getLogger('smartcash.ui.components.operation_container').setLevel(logging.DEBUG)

# Test each component individually
operation_container = create_operation_container(show_progress=True, show_logs=True)

# Test progress tracker
progress_tracker = operation_container['progress_tracker']
print(f"Progress tracker initialized: {progress_tracker._initialized}")
print(f"Container exists: {progress_tracker.container is not None}")

# Test log accordion
log_accordion = operation_container['log_accordion']
print(f"Log accordion initialized: {log_accordion._initialized}")
log_accordion.add_log("Test log", 'info')

# Test update functions
operation_container['update_progress'](50, "Test progress")
operation_container['log_message']("Test message", 'info')
```

## 📚 Best Practices

### 1. Initialization Order

```python
# ✅ Correct order
operation_container = create_operation_container(...)
operation_manager = MyOperationManager(config, operation_container)
operation_manager.initialize()

# ❌ Wrong order
operation_manager = MyOperationManager(config, None)
operation_container = create_operation_container(...)
```

### 2. Error Handling

```python
class RobustOperationManager(OperationHandler):
    def execute_operation(self):
        button_states = None
        try:
            self.clear_logs()
            button_states = self.disable_all_buttons("⏳ Processing...")
            
            # Operation code here
            result = self.do_work()
            
            if result['success']:
                self.update_progress(100, "Completed!")
                self.log("✅ Success", 'success')
            else:
                self.log(f"❌ Failed: {result['error']}", 'error')
            
            return result
            
        except Exception as e:
            self.log(f"❌ Exception: {str(e)}", 'error')
            return {'success': False, 'error': str(e)}
        finally:
            if button_states:
                self.enable_all_buttons(button_states)
```

### 3. Resource Cleanup

```python
class CleanupOperationManager(OperationHandler):
    def cleanup(self):
        # Clear progress and logs
        if hasattr(self, '_operation_container'):
            if 'progress_tracker' in self._operation_container:
                self._operation_container['progress_tracker'].reset()
            if 'log_accordion' in self._operation_container:
                self._operation_container['log_accordion'].clear_logs()
        
        super().cleanup()
```

## 🔗 API Reference

### create_operation_container()

```python
def create_operation_container(
    show_progress: bool = True,
    show_dialog: bool = True, 
    show_logs: bool = True,
    log_module_name: str = "Operation",
    log_namespace_filter: Optional[str] = None,
    progress_levels: Literal['single', 'dual', 'triple'] = 'single',
    progress_config: Optional[ProgressConfig] = None,
    **kwargs
) -> Dict[str, Any]
```

**Returns**:
- `container`: Main container widget
- `progress_tracker`: ProgressTracker instance
- `log_accordion`: LogAccordion instance
- `show_dialog`: Function to show dialogs
- `show_info_dialog`: Function to show info dialogs
- `clear_dialog`: Function to clear dialogs
- `update_progress`: Function to update progress
- `log_message`: Function to log messages

### OperationHandler Methods

```python
# Logging
self.log(message: str, level: str = 'info', namespace: str = None)
self.clear_logs()

# Progress
self.update_progress(progress: int, message: str = "")
self.reset_progress()

# Button Management
self.disable_all_buttons(message: str = "") -> Dict[str, bool]
self.enable_all_buttons(button_states: Dict[str, bool])

# Dialogs  
self.show_dialog(title: str, message: str, dialog_type: str = "info")
self.show_info_dialog(title: str, message: str)
self.clear_dialog()
```

## 📖 Examples

See the following modules for complete examples:
- `smartcash.ui.model.pretrained.operations.pretrained_operation_manager`
- `smartcash.ui.model.backbone.operations.backbone_operation_manager`
- `smartcash.ui.model.training.operations.training_operation_manager`