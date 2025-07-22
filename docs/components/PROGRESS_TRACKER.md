# Progress Tracker Components

Components for tracking and displaying the progress of long-running operations.

## Components

### ProgressTracker (`progress_tracker.py`)
Main component for tracking and displaying progress of operations.

**Props:**
- `total` (int): Total number of steps
- `current` (int): Current step number
- `message` (str, optional): Current status message
- `show_percentage` (bool): Whether to show percentage complete
- `show_steps` (bool): Whether to show step count (e.g., "2/10")
- `variant` (str): Visual variant ('linear' or 'circular')

**Example:**
```python
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

# In your operation:
progress = ProgressTracker(total=100)
for i in range(100):
    # Update progress
    progress.update(
        current=i+1,
        message=f"Processing item {i+1}..."
    )
    # Your operation here
```

### TqdmManager (`tqdm_manager.py`)
Integrates with tqdm for progress tracking in the UI.

**Usage:**
```python
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager

with TqdmManager(total=100, desc="Processing") as pbar:
    for i in range(100):
        # Your operation here
        pbar.update(1)
```

### CallbackManager (`callback_manager.py`)
Manages progress update callbacks.

**Features:**
- Register multiple callbacks
- Batch updates for performance
- Thread-safe operations

## Best Practices

- Update progress at meaningful intervals (not too frequently)
- Provide clear, actionable status messages
- Handle cancellation gracefully
- Show estimated time remaining when possible
- Use appropriate progress indicators for operation length

## Integration

1. **Basic Usage:**
   ```python
   from smartcash.ui.components.progress_tracker.factory import create_progress_tracker
   
   tracker = create_progress_tracker(
       total=100,
       description="Processing data..."
   )
   
   try:
       for i in range(100):
           # Update progress
           tracker.update(
               current=i+1,
               message=f"Processing item {i+1}"
           )
   finally:
       tracker.close()
   ```

2. **With Context Manager:**
   ```python
   from contextlib import contextmanager
   
   @contextmanager
   def track_progress(total, description):
       tracker = create_progress_tracker(total=total, description=description)
       try:
           yield tracker
       finally:
           tracker.close()
   ```

## Error Handling

- Progress components handle errors gracefully
- Errors are logged and displayed to the user
- Failed operations can be retried when appropriate

## Performance

- Minimal overhead during progress updates
- Batched updates for high-frequency operations
- Memory-efficient for large numbers of items
