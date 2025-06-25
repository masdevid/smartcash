"""Test script for DualProgressTracker"""
import time
from smartcash.ui.setup.env_config.utils.dual_progress_tracker import DualProgressTracker, SetupStage

def test_progress_tracker():
    # Initialize tracker
    tracker = DualProgressTracker()
    
    # Test stage updates
    print("Testing stage updates...")
    for stage in SetupStage:
        if stage == SetupStage.COMPLETE:
            break
            
        tracker.update_stage(stage, f"Starting {stage.name.replace('_', ' ').title()}")
        
        # Simulate progress within stage
        for i in range(1, 6):
            time.sleep(0.3)
            tracker.update_within_stage(i, 5, f"Processing item {i}/5")
        
        tracker.complete_stage(f"Completed {stage.name.replace('_', ' ').title()}")
    
    # Test completion
    tracker.complete("All stages completed successfully!")
    print("Progress tracker test completed successfully!")

if __name__ == "__main__":
    test_progress_tracker()
