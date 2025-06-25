"""Comprehensive test for progress container functionality"""
import time
import ipywidgets as widgets
from ipywidgets import VBox, FloatProgress, HTML
from smartcash.ui.setup.env_config.utils.dual_progress_tracker import DualProgressTracker, SetupStage

def test_progress_container():
    """Test the progress container initialization and updates"""
    print("🚀 Starting progress container test...")
    
    # 1. Test basic initialization
    print("\n1. Testing initialization...")
    components = {}
    tracker = DualProgressTracker(components)
    
    # 2. Verify components
    print("\n2. Verifying components:")
    print(f"- Components: {list(components.keys())}")
    container = components.get('progress_container')
    print(f"- Container type: {type(container).__name__}")
    print(f"- Container children: {len(container.children) if hasattr(container, 'children') else 'N/A'}")
    
    # 3. Test stage updates
    print("\n3. Testing stage updates...")
    for stage in SetupStage:
        print(f"\n   Updating to stage: {stage.name}")
        tracker.update_stage(stage, f"Testing {stage.name}...")
        print(f"   - Progress: {tracker.overall_progress}%")
        print(f"   - Stage progress: {tracker.stage_progress}%")
        time.sleep(0.5)
    
    # 4. Test progress updates
    print("\n4. Testing progress updates...")
    for i in range(0, 101, 20):
        tracker.update_progress(i, f"Progress: {i}%")
        print(f"   - Progress: {i}%")
        time.sleep(0.1)
    
    # 5. Test within-stage progress updates
    print("\n5. Testing within-stage progress updates...")
    total_items = 5
    for i in range(1, total_items + 1):
        tracker.update_within_stage(i, total_items, f"Processing item {i}/{total_items}")
        print(f"   - Item {i}/{total_items}: {tracker.stage_progress:.1f}%")
        time.sleep(0.1)
    
    # 6. Test edge cases for update_within_stage
    print("\n6. Testing edge cases...")
    # Test with zero items
    tracker.update_within_stage(0, 0, "Should not update")
    print("   - Zero items test passed")
    
    # Test with single item
    tracker.update_within_stage(1, 1, "Single item complete")
    print(f"   - Single item test: {tracker.stage_progress:.1f}%")
    
    # 7. Test completion
    print("\n7. Testing completion...")
    tracker.complete("Test completed successfully!")
    print(f"   - Final progress: {tracker.overall_progress}%")
    
    print("\n✅ All tests completed!")
    return components

if __name__ == "__main__":
    test_progress_container()
