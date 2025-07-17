"""
Debug Progress Tracker

This script provides detailed debugging information for the ProgressTracker component.
"""

import time
from IPython.display import display, HTML
import ipywidgets as widgets
import inspect

# Import progress tracker components
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel

def debug_progress_tracker():
    """Debug the progress tracker with detailed information."""
    print("=== Debugging Progress Tracker ===\n")
    
    # Create a simple progress config
    config = ProgressConfig(
        level=ProgressLevel.SINGLE,
        operation="Debug Test",
        steps=["progress"],
        step_weights={"progress": 1}
    )
    
    print("1. Creating ProgressTracker instance...")
    progress = ProgressTracker("debug_test", config=config)
    
    # Print available methods
    print("\n2. Available methods in ProgressTracker:")
    methods = [m for m in dir(progress) if not m.startswith('_') or m.startswith('_create')]
    print("\n".join([f"  - {m}" for m in methods]))
    
    print("\n3. Initializing UI components...")
    if not hasattr(progress, '_ui_components'):
        print("  - _ui_components does not exist before initialization")
    
    progress.initialize()
    
    if not hasattr(progress, '_ui_components'):
        print("  - ERROR: _ui_components not created after initialization")
        return
        
    print(f"  - _ui_components keys: {list(progress._ui_components.keys())}")
    
    # Check TQDM manager
    if not hasattr(progress, 'tqdm_manager'):
        print("  - ERROR: tqdm_manager not found")
        return
        
    print(f"  - tqdm_manager: {progress.tqdm_manager}")
    
    # Display the container
    if 'container' not in progress._ui_components:
        print("  - ERROR: container not found in _ui_components")
        return
        
    print("\n4. Displaying container...")
    display(progress._ui_components['container'])
    
    # Test setting progress
    print("\n5. Testing progress updates...")
    try:
        for i in range(1, 6):  # 5 steps
            progress_val = i * 20
            print(f"  - Setting progress to {progress_val}%")
            progress.set_progress(
                progress=progress_val,
                message=f"Progress: {progress_val}%",
                level="progress"
            )
            time.sleep(0.5)
            
            # Check TQDM bars
            if hasattr(progress.tqdm_manager, 'tqdm_bars'):
                print(f"    - TQDM bars: {progress.tqdm_manager.tqdm_bars}")
            else:
                print("    - No tqdm_bars found in tqdm_manager")
                
    except Exception as e:
        print(f"  - ERROR during progress update: {str(e)}")
    
    print("\n6. Test complete!")

# Run the debug function
if __name__ == "__main__":
    debug_progress_tracker()
