"""Test script to isolate the progress container issue"""
import ipywidgets as widgets
from ipywidgets import VBox, FloatProgress, HTML
from smartcash.ui.setup.env_config.utils.dual_progress_tracker import DualProgressTracker

def test_progress_container():
    """Test the progress container initialization"""
    print("1. Creating empty components dictionary...")
    components = {}
    
    print("2. Creating progress tracker...")
    tracker = DualProgressTracker(components)
    
    print("3. Checking components:", components.keys())
    print("4. Container type:", type(components.get('progress_container')))
    print("5. Container children:", 
          len(components.get('progress_container', VBox()).children) if hasattr(components.get('progress_container'), 'children') else 'No children')
    
    return components

if __name__ == "__main__":
    test_progress_container()
