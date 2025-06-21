# 📊 SmartCash Progress Tracker API Documentation

## 🏗️ Arsitektur Modular

```
smartcash/ui/components/progress_tracker/
├── __init__.py                    # Main exports
├── progress_tracker.py            # Core tracker class
├── progress_config.py             # Configuration & enums
├── callback_manager.py            # Event handling
├── tqdm_manager.py                # Progress bar management
├── ui_components.py               # UI widgets manager
└── factory.py                     # Factory functions
```

## 🚀 Core API

### ProgressTracker Class

```python
from smartcash.ui.components.progress_tracker import ProgressTracker, ProgressConfig, ProgressLevel

# Initialization
config = ProgressConfig(level=ProgressLevel.DUAL, operation="Dataset Processing")
tracker = ProgressTracker(config)

# Display methods
tracker.show(operation="Training", steps=["Load", "Train", "Validate"])
tracker.hide()
tracker.reset()

# Progress updates
tracker.update(level_name="overall", progress=75, message="Processing...")
tracker.update_primary(progress=50, message="Loading data")
tracker.update_overall(progress=80, message="Training model")
tracker.update_current(progress=25, message="Current batch")

# State management
tracker.complete(message="Training completed successfully!")
tracker.error(message="Training failed due to memory error")
```

### Factory Functions

```python
from smartcash.ui.components.progress_tracker.factory import (
    create_single_progress_tracker,
    create_dual_progress_tracker,
    create_triple_progress_tracker,
    create_three_progress_tracker
)

# Single level - primary progress only
single_tracker = create_single_progress_tracker("Data Loading", auto_hide=True)

# Dual level - overall + current
dual_tracker = create_dual_progress_tracker("Model Training", auto_hide=False)

# Triple level - overall + step + current
triple_tracker = create_triple_progress_tracker(
    operation="Full Pipeline",
    steps=["Preprocessing", "Training", "Evaluation"],
    auto_hide=True
)

# Backward compatibility wrapper
legacy_components = create_three_progress_tracker(auto_hide=True)
container = legacy_components['container']
tracker = legacy_components['tracker']
```

## ⚙️ Configuration API

### ProgressConfig

```python
from smartcash.ui.components.progress_tracker.progress_config import (
    ProgressConfig, ProgressLevel, ProgressBarConfig
)

config = ProgressConfig(
    level=ProgressLevel.TRIPLE,           # SINGLE, DUAL, TRIPLE
    operation="Model Training",           # Operation title
    steps=["Load", "Train", "Validate"],  # Step names
    step_weights={"Load": 20, "Train": 60, "Validate": 20},  # Step weights
    auto_advance=True,                    # Auto step progression
    auto_hide=False,                      # Auto hide after completion
    auto_hide_delay=3600.0,              # Auto hide delay (1 hour)
    animation_speed=0.1,                  # Animation speed
    show_step_info=False                  # Show step information
)

# Get level configurations
bar_configs = config.get_level_configs()
for bar_config in bar_configs:
    print(f"{bar_config.name}: {bar_config.description}")

# Get container height based on level
height = config.get_container_height()  # "120px", "160px", "200px"
```

### ProgressLevel Enum

```python
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

ProgressLevel.SINGLE   # 1 bar: primary only
ProgressLevel.DUAL     # 2 bars: overall + current
ProgressLevel.TRIPLE   # 3 bars: overall + step + current
```

## 🎯 Callback System

### Event Registration

```python
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager

tracker = ProgressTracker()

# Register callbacks
progress_id = tracker.on_progress_update(lambda level, progress, msg: print(f"{level}: {progress}%"))
complete_id = tracker.on_complete(lambda: print("Operation completed!"))
error_id = tracker.on_error(lambda msg: print(f"Error: {msg}"))
reset_id = tracker.on_reset(lambda: print("Tracker reset"))

# Remove specific callback
tracker.remove_callback(progress_id)

# Clear all callbacks
tracker.callback_manager.clear_all()
```

### Callback Signatures

```python
# Progress update callback
def on_progress_update(level_name: str, progress: int, message: str) -> None:
    """Called when progress is updated"""
    
# Step completion callback  
def on_step_complete(step_name: str, step_index: int) -> None:
    """Called when a step is completed"""
    
# Operation completion callback
def on_complete() -> None:
    """Called when entire operation is completed"""
    
# Error callback
def on_error(error_message: str) -> None:
    """Called when an error occurs"""
    
# Reset callback
def on_reset() -> None:
    """Called when tracker is reset"""
```

## 🎨 UI Components API

### Widget Access

```python
tracker = ProgressTracker()

# Main container
container = tracker.container
display(container)

# Individual widgets
status_widget = tracker.status_widget
step_info_widget = tracker.step_info_widget  # Always None (disabled)

# Progress bars dictionary
progress_bars = tracker.progress_bars
main_bar = progress_bars.get('main')  # For SINGLE level
overall_bar = progress_bars.get('overall')  # For DUAL/TRIPLE
current_bar = progress_bars.get('current')   # For DUAL/TRIPLE
```

### Manual Widget Management

```python
from smartcash.ui.components.progress_tracker.ui_components import UIComponentsManager

ui_manager = UIComponentsManager(config)

# Show/hide operations
ui_manager.show()
ui_manager.hide()

# Status updates
ui_manager.update_status("Processing data...", style='info')
ui_manager.update_status("Operation completed!", style='success')
ui_manager.update_status("Warning: Low memory", style='warning')
ui_manager.update_status("Error occurred", style='error')

# Header updates
ui_manager.update_header("New Operation Name")
```

## 📈 Progress Management API

### Tqdm Integration

```python
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager

tqdm_manager = TqdmManager(ui_manager)

# Initialize progress bars
bar_configs = config.get_level_configs()
tqdm_manager.initialize_bars(bar_configs)

# Update specific bar
tqdm_manager.update_bar("overall", 75, "Training epoch 3/4")
tqdm_manager.update_bar("current", 50, "Batch 128/256")

# Set completion state
tqdm_manager.set_all_complete("Training completed successfully!")
tqdm_manager.set_all_error("Training failed")

# Cleanup
tqdm_manager.close_all_bars()
tqdm_manager.reset()
```

### Progress Values

```python
# Get current progress values
overall_progress = tqdm_manager.get_progress_value("overall")
current_message = tqdm_manager.get_progress_message("current")

# Direct bar access
tqdm_bars = tqdm_manager.tqdm_bars
overall_bar = tqdm_bars.get("overall")
if overall_bar:
    current_value = overall_bar.n
    total_value = overall_bar.total
```

## 🔄 Integration Patterns

### With UI Components Dictionary

```python
def setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Integrate progress tracker into UI components"""
    tracker = create_dual_progress_tracker("Operation", auto_hide=True)
    
    # Add to ui_components
    ui_components.update({
        'progress_tracker': tracker,
        'progress_container': tracker.container,
        'progress_status': tracker.status_widget
    })
    
    # Register callbacks
    tracker.on_complete(lambda: print("✅ Operation completed"))
    tracker.on_error(lambda msg: print(f"❌ Error: {msg}"))

def update_operation_progress(ui_components: Dict[str, Any], progress: int, message: str):
    """Update progress through ui_components"""
    tracker = ui_components.get('progress_tracker')
    if tracker:
        tracker.update_overall(progress, message)
```

### With Backend Services

```python
class DatasetProcessor:
    def __init__(self, ui_components: Dict[str, Any] = None):
        self.ui_components = ui_components or {}
        self.tracker = ui_components.get('progress_tracker')
    
    def process_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.tracker:
            self.tracker.show("Dataset Processing", ["Load", "Process", "Save"])
        
        try:
            # Step 1: Load data
            self._update_progress("overall", 0, "Loading dataset...")
            data = self._load_data(config)
            self._update_progress("overall", 33, "Data loaded")
            
            # Step 2: Process data  
            self._update_progress("overall", 33, "Processing data...")
            processed = self._process_data(data)
            self._update_progress("overall", 66, "Data processed")
            
            # Step 3: Save results
            self._update_progress("overall", 66, "Saving results...")
            self._save_results(processed)
            self._update_progress("overall", 100, "Results saved")
            
            if self.tracker:
                self.tracker.complete("Dataset processing completed!")
            
            return {"success": True, "message": "Processing completed"}
            
        except Exception as e:
            if self.tracker:
                self.tracker.error(f"Processing failed: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def _update_progress(self, level: str, progress: int, message: str):
        if self.tracker:
            self.tracker.update(level, progress, message)
```

### Custom Progress Callback

```python
def create_progress_callback(tracker: ProgressTracker) -> Callable:
    """Create progress callback for backend services"""
    def progress_callback(level: str, current: int, total: int, message: str = ""):
        progress_percent = int((current / total) * 100) if total > 0 else 0
        tracker.update(level, progress_percent, message)
    return progress_callback

# Usage with backend service
tracker = create_dual_progress_tracker("Data Processing")
callback = create_progress_callback(tracker)

# Pass to backend service
result = preprocess_dataset(config, progress_callback=callback)
```

## 🎛️ Auto-Hide & Timing

### Auto-Hide Configuration

```python
# Auto-hide after 1 hour (default)
tracker = create_single_progress_tracker("Operation", auto_hide=True)

# Custom auto-hide delay
config = ProgressConfig(auto_hide=True, auto_hide_delay=1800.0)  # 30 minutes
tracker = ProgressTracker(config)

# Disable auto-hide
tracker = create_dual_progress_tracker("Operation", auto_hide=False)

# Cancel auto-hide on error (automatic)
tracker.error("Something went wrong")  # Cancels auto-hide timer
```

### Manual Timer Control

```python
# Access UI manager for timer control
ui_manager = tracker.ui_manager

# Start auto-hide timer manually
ui_manager._start_auto_hide_timer()

# Cancel auto-hide timer
ui_manager._cancel_auto_hide_timer()

# Check if visible
is_visible = ui_manager.is_visible
```

## 🔧 Utility Functions

### Message Cleaning

```python
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager

# Clean messages from duplicate emojis and progress indicators
cleaned = TqdmManager._clean_message("📊 Processing... [50%] (5/10)")
# Result: "Processing..."

# Truncate messages for UI
truncated = TqdmManager._truncate_message("Very long message here", 20)
# Result: "Very long message..."
```

### Progress State Management

```python
# Check completion state
is_complete = tracker.is_complete
is_error = tracker.is_error

# Get current step index
current_step = tracker.current_step_index

# Get active levels
active_levels = tracker.active_levels
```

## 📝 Error Handling

### Safe Operations

```python
# All progress updates are safe - won't crash if components missing
tracker.update("nonexistent_level", 50, "Message")  # Silently ignored

# Safe callback registration
callback_id = tracker.on_progress_update(None)  # Handles None callback

# Safe widget access
container = tracker.container  # Always returns valid widget
status = tracker.status_widget  # Always returns valid widget
```

### Error Recovery

```python
# Reset after error
tracker.error("Operation failed")
# ... fix the issue ...
tracker.reset()  # Clears error state and resets all progress

# Continue after partial failure
tracker.update_overall(50, "Partial completion")
# ... handle the issue ...
tracker.update_overall(100, "Completed with warnings")
tracker.complete("Operation completed with some issues")
```

## 🎯 Best Practices

### Initialization

```python
# ✅ Good: Use factory functions
tracker = create_dual_progress_tracker("Operation")

# ✅ Good: Configure before showing
config = ProgressConfig(level=ProgressLevel.TRIPLE, auto_hide=True)
tracker = ProgressTracker(config)
tracker.show("Operation", ["Step1", "Step2", "Step3"])
```

### Progress Updates

```python
# ✅ Good: Clear, descriptive messages
tracker.update_overall(25, "Loading dataset (1/4)")
tracker.update_current(50, "Processing batch 128/256")

# ✅ Good: Meaningful completion messages
tracker.complete("Model training completed successfully!")

# ❌ Avoid: Generic messages
tracker.update_overall(50, "Processing...")
```

### Integration

```python
# ✅ Good: Check tracker availability
if tracker:
    tracker.update_overall(progress, message)

# ✅ Good: Use callbacks for decoupling
tracker.on_complete(lambda: save_results())
tracker.on_error(lambda msg: log_error(msg))

# ✅ Good: Clean up resources
try:
    # ... processing ...
    tracker.complete("Success!")
except Exception as e:
    tracker.error(f"Failed: {str(e)}")
finally:
    # Cleanup handled automatically
    pass
```