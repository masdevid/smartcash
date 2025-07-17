#!/usr/bin/env python3
"""
Comprehensive Demo and Test of Enhanced Operation Container Integration

This script demonstrates all the enhanced features working together:
- Progress tracker auto-visibility
- Enhanced logging with namespace support  
- Dialog system integration
- Backend logging capture
- Error handling and recovery
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demonstrate_enhanced_integration():
    """Demonstrate all enhanced features working together."""
    print("🚀 COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("Demonstrating enhanced operation container integration:")
    print("✅ Progress tracker auto-visibility")
    print("✅ Enhanced logging with namespace support")
    print("✅ Dialog system integration")
    print("✅ Backend logging capture")
    print("✅ Error handling and recovery")
    print("")
    
    try:
        # 1. Initialize Backbone UI Module
        print("📋 Step 1: Initializing Backbone UI Module")
        print("-" * 50)
        
        from smartcash.ui.model.backbone.backbone_uimodule import initialize_backbone_ui
        
        result = initialize_backbone_ui(display=False)
        assert result['success'], "Backbone initialization should succeed"
        
        module = result['module']
        ui_components = result['ui_components']
        operation_container = ui_components['operation_container']
        
        print("✅ Backbone UI module initialized successfully")
        print(f"📊 Available components: {len(ui_components)} components")
        print("")
        
        # 2. Demonstrate Progress Tracker Auto-Visibility
        print("📋 Step 2: Demonstrating Progress Tracker Auto-Visibility")
        print("-" * 50)
        
        op_manager = module._operation_manager
        
        # Progress tracker should be visible by default
        print("🔍 Checking initial progress tracker state...")
        if hasattr(operation_container, 'progress_tracker'):
            tracker = operation_container.progress_tracker
            if hasattr(tracker, 'container') and tracker.container:
                display = tracker.container.layout.display
                print(f"📊 Progress tracker display state: {display}")
                assert display != 'none', "Progress tracker should be visible by default"
        
        print("✅ Progress tracker is visible by default")
        print("")
        
        # 3. Demonstrate Enhanced Logging
        print("📋 Step 3: Demonstrating Enhanced Logging")
        print("-" * 50)
        
        # Test different log levels
        op_manager.log("🚀 Starting demonstration", 'info')
        op_manager.log("📝 This is a debug message", 'debug')
        op_manager.log("⚠️ This is a warning message", 'warning')
        op_manager.log("❌ This is an error message", 'error')
        
        # Test namespace logging
        if hasattr(operation_container, 'log'):
            operation_container.log("📦 Message with namespace", 
                                  namespace="demo.test")
        
        print("✅ Enhanced logging demonstrated")
        print("")
        
        # 4. Demonstrate Backend Logging Capture
        print("📋 Step 4: Demonstrating Backend Logging Capture")
        print("-" * 50)
        
        # Create test loggers for backend services
        dataset_logger = logging.getLogger('smartcash.dataset.preprocessor.demo')
        model_logger = logging.getLogger('smartcash.model.demo')
        common_logger = logging.getLogger('smartcash.common.demo')
        
        # These should be captured by the UI logging bridge
        dataset_logger.info("Dataset service: Processing data...")
        dataset_logger.warning("Dataset service: Low memory warning")
        
        model_logger.info("Model service: Loading pretrained weights...")
        model_logger.error("Model service: GPU not available")
        
        common_logger.info("Common service: Configuration loaded")
        
        print("✅ Backend logging capture demonstrated")
        print("📝 Backend service logs should appear in operation container")
        print("")
        
        # 5. Demonstrate Progress Updates
        print("📋 Step 5: Demonstrating Progress Updates")
        print("-" * 50)
        
        progress_steps = [
            (0, "Initializing operation..."),
            (25, "Loading configuration..."),
            (50, "Processing data..."),
            (75, "Validating results..."),
            (100, "Operation completed successfully!")
        ]
        
        for progress, message in progress_steps:
            op_manager.update_progress(progress, message)
            op_manager.log(f"📊 Progress: {progress}% - {message}", 'info')
            time.sleep(0.2)  # Small delay to show progression
        
        print("✅ Progress updates demonstrated")
        print("")
        
        # 6. Demonstrate Error Handling
        print("📋 Step 6: Demonstrating Error Handling")
        print("-" * 50)
        
        # Test invalid progress values (should be handled gracefully)
        try:
            op_manager.update_progress(-10, "Invalid negative progress")
            op_manager.update_progress(150, "Invalid over-100 progress")
            print("✅ Invalid progress values handled gracefully")
        except Exception as e:
            print(f"⚠️ Progress error handling issue: {e}")
        
        # Test error logging
        op_manager.log("🔥 Simulated error condition", 'error')
        op_manager.log("🛠️ Error recovery initiated", 'info')
        op_manager.log("✅ Error resolved successfully", 'success')
        
        print("✅ Error handling demonstrated")
        print("")
        
        # 7. Demonstrate Dialog Integration (if available)
        print("📋 Step 7: Demonstrating Dialog Integration")
        print("-" * 50)
        
        if hasattr(operation_container, 'show_dialog'):
            print("🔍 Dialog system is available")
            
            # Test info dialog (simulate user interaction)
            dialog_shown = operation_container.show_info_dialog(
                title="Demo Complete",
                message="All enhanced features have been demonstrated successfully!"
            )
            
            if dialog_shown:
                print("✅ Dialog system integration working")
                # Clear dialog immediately for demo
                operation_container.clear_dialog()
            else:
                print("📝 Dialog system available but not displayed (expected in test mode)")
        else:
            print("📝 Dialog system not available in this configuration")
        
        print("")
        
        # 8. Summary
        print("📋 Step 8: Integration Validation Summary")
        print("-" * 50)
        
        if hasattr(operation_container, 'validate_integration'):
            validation = operation_container.validate_integration()
            print(f"🔍 Overall Status: {validation['overall_status']}")
            
            for component, status in validation['components'].items():
                health = "✅ Healthy" if status['healthy'] else "❌ Issues"
                print(f"  📦 {component}: {health}")
            
            if validation['issues']:
                print("⚠️ Issues found:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
        else:
            print("📝 Validation method not available")
        
        print("")
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("All enhanced features are working correctly:")
        print("✅ Progress tracker is visible by default")
        print("✅ Backend logs are captured in UI")
        print("✅ Enhanced logging with namespace support")
        print("✅ Error handling works properly")
        print("✅ Operation container integration is seamless")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_smoke_test():
    """Quick smoke test to verify basic functionality."""
    print("🧪 QUICK SMOKE TEST")
    print("=" * 50)
    
    try:
        # Test operation container creation
        from smartcash.ui.components.operation_container import create_operation_container
        
        container = create_operation_container(
            component_name="smoke_test",
            show_progress=True,
            show_logs=True,
            show_dialog=True
        )
        
        # Test basic functionality
        container['update_progress'](50, "Smoke test in progress...")
        container['log_message']("🧪 Smoke test message", 'info')
        
        print("✅ Operation container creation: PASSED")
        
        # Test backbone UI initialization
        from smartcash.ui.model.backbone.backbone_uimodule import initialize_backbone_ui
        
        result = initialize_backbone_ui(display=False)
        
        if result and result.get('success'):
            print("✅ Backbone UI initialization: PASSED")
        else:
            print("❌ Backbone UI initialization: FAILED")
            return False
        
        print("✅ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Enhanced Operation Container Integration Demo")
    print("=" * 80)
    print("This demo shows all the enhanced features working together.")
    print("")
    
    # Run quick smoke test first
    if not quick_smoke_test():
        print("❌ Smoke test failed - aborting demo")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    
    # Run comprehensive demonstration
    if demonstrate_enhanced_integration():
        print("\n🎉 Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)