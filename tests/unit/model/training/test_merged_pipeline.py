#!/usr/bin/env python3
"""
Test script for the merged training pipeline.

This script validates that the merged training pipeline works correctly
and all components are properly integrated.
"""

import sys
import traceback
from unittest.mock import Mock, patch

def test_merged_pipeline_imports():
    """Test that all merged pipeline components can be imported."""
    print("🧪 Testing merged pipeline imports...")
    
    try:
        # Test main pipeline import
        from smartcash.model.training.training_pipeline import TrainingPipeline
        print("  ✅ TrainingPipeline import successful")
        
        # Test pipeline components import
        from smartcash.model.training.pipeline import (
            ConfigurationBuilder, PipelineExecutor, SessionManager
        )
        print("  ✅ Pipeline components import successful")
        
        # Test function imports
        from smartcash.model.training.training_pipeline import (
            run_training_pipeline, run_full_training_pipeline
        )
        print("  ✅ Function imports successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_instantiation():
    """Test that the merged pipeline can be instantiated."""
    print("🧪 Testing pipeline instantiation...")
    
    try:
        from smartcash.model.training.training_pipeline import TrainingPipeline
        
        # Test basic instantiation
        pipeline = TrainingPipeline()
        print("  ✅ Basic instantiation successful")
        
        # Test with callbacks
        def mock_progress(phase, current, total, message, **kwargs):
            pass
        
        def mock_log(level, message, data):
            pass
        
        def mock_metrics(phase, epoch, metrics):
            pass
        
        pipeline = TrainingPipeline(
            progress_callback=mock_progress,
            log_callback=mock_log,
            metrics_callback=mock_metrics
        )
        print("  ✅ Instantiation with callbacks successful")
        
        # Verify basic components are initialized
        assert hasattr(pipeline, 'progress_tracker')
        assert hasattr(pipeline, 'use_yolov5_integration')
        assert hasattr(pipeline, 'log_callback')
        print("  ✅ Components properly initialized")
        
        # Note: session_manager and other components are initialized during pipeline execution
        
        return True
        
    except Exception as e:
        print(f"  ❌ Instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_builder():
    """Test the configuration builder component."""
    print("🧪 Testing configuration builder...")
    
    try:
        from smartcash.model.training.pipeline import ConfigurationBuilder
        
        # Test instantiation
        config_builder = ConfigurationBuilder("test_session")
        print("  ✅ ConfigurationBuilder instantiation successful")
        
        # Test configuration building
        config = config_builder.build_training_config(
            backbone='cspdarknet',
            pretrained=True,
            phase_1_epochs=1,
            phase_2_epochs=1
        )
        
        # Verify configuration structure
        required_keys = ['backbone', 'model', 'training_phases', 'training', 'paths']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
        print("  ✅ Configuration building successful")
        
        # Test validation
        is_valid = config_builder.validate_configuration(config)
        assert is_valid, "Configuration validation failed"
        print("  ✅ Configuration validation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration builder test failed: {e}")
        traceback.print_exc()
        return False

def test_session_manager():
    """Test the session manager component."""
    print("🧪 Testing session manager...")
    
    try:
        from smartcash.model.training.pipeline import SessionManager
        
        # Test instantiation
        session_manager = SessionManager()
        print("  ✅ SessionManager instantiation successful")
        
        # Test session creation
        session_id, resume_info = session_manager.create_session(
            backbone='cspdarknet',
            training_mode='two_phase'
        )
        
        assert session_id is not None, "Session ID should not be None"
        assert len(session_id) == 8, "Session ID should be 8 characters"
        print(f"  ✅ Session creation successful: {session_id}")
        
        # Test parameter validation
        is_valid = session_manager.validate_session_params('two_phase')
        assert is_valid, "Parameter validation should succeed"
        print("  ✅ Parameter validation successful")
        
        # Test session info
        session_info = session_manager.get_session_info()
        assert 'session_id' in session_info
        print("  ✅ Session info retrieval successful")
        
        # Test cleanup
        session_manager.cleanup_session()
        print("  ✅ Session cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session manager test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests for the merged pipeline."""
    print("🚀 Starting merged training pipeline tests...\\n")
    
    tests = [
        test_merged_pipeline_imports,
        test_pipeline_instantiation,
        test_configuration_builder,
        test_session_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed\\n")
            else:
                failed += 1
                print("❌ Test failed\\n")
        except Exception as e:
            failed += 1
            print(f"❌ Test failed with exception: {e}\\n")
    
    print("📊 Test Summary:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All tests passed! Merged pipeline is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)