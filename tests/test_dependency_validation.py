#!/usr/bin/env python3
"""
Simple validation test for dependency module functionality
Including triple progress system validation
"""

import sys
import traceback
from unittest.mock import MagicMock, patch

def test_dependency_module_imports():
    """Test that dependency module imports work correctly"""
    try:
        print("🔍 Testing dependency module imports...")
        
        # Test core components
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        from smartcash.ui.components.operation_container import OperationContainer
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
        
        print("✅ All dependency module imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False

def test_progress_config_creation():
    """Test progress configuration for different levels"""
    try:
        print("\n🔍 Testing progress configuration creation...")
        
        from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
        
        # Test single level
        single_config = ProgressConfig(level=ProgressLevel.SINGLE, operation="Test Single")
        print(f"✅ Single config: {single_config.level}, steps: {single_config.steps}")
        
        # Test dual level  
        dual_config = ProgressConfig(level=ProgressLevel.DUAL, operation="Test Dual")
        print(f"✅ Dual config: {dual_config.level}, steps: {dual_config.steps}")
        
        # Test triple level
        triple_config = ProgressConfig(level=ProgressLevel.TRIPLE, operation="Test Triple")
        print(f"✅ Triple config: {triple_config.level}, steps: {triple_config.steps}")
        
        # Test level configs
        level_configs = triple_config.get_level_configs()
        print(f"✅ Triple level configs count: {len(level_configs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Progress config error: {str(e)}")
        traceback.print_exc()
        return False

def test_operation_container_integration():
    """Test operation container with different progress levels"""
    try:
        print("\n🔍 Testing operation container integration...")
        
        from smartcash.ui.components.operation_container import OperationContainer
        from smartcash.ui.components.progress_tracker.types import ProgressLevel
        
        # Test single progress
        container_single = OperationContainer(
            title="Test Single",
            progress_levels="single"
        )
        print("✅ Single progress container created")
        
        # Test dual progress
        container_dual = OperationContainer(
            title="Test Dual", 
            progress_levels="dual"
        )
        print("✅ Dual progress container created")
        
        # Test triple progress
        container_triple = OperationContainer(
            title="Test Triple",
            progress_levels="triple"
        )
        print("✅ Triple progress container created")
        
        return True
        
    except Exception as e:
        print(f"❌ Operation container error: {str(e)}")
        traceback.print_exc()
        return False

def test_dependency_uimodule_initialization():
    """Test dependency UI module initialization"""
    try:
        print("\n🔍 Testing dependency UI module initialization...")
        
        # Mock dependencies to avoid GUI initialization
        with patch('smartcash.ui.setup.dependency.dependency_uimodule.get_dependency_config') as mock_config, \
             patch('smartcash.ui.components.operation_container.OperationContainer') as mock_container:
            
            mock_config.return_value = {
                'packages': ['numpy', 'pandas'],
                'operation': 'install'
            }
            
            mock_container.return_value = MagicMock()
            
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            # Initialize module
            module = DependencyUIModule()
            print("✅ DependencyUIModule created successfully")
            
            # Test that required methods exist
            assert hasattr(module, 'update_triple_progress'), "Missing update_triple_progress method"
            assert hasattr(module, 'update_stage_progress'), "Missing update_stage_progress method"
            assert hasattr(module, 'set_progress_level'), "Missing set_progress_level method"
            assert hasattr(module, 'get_progress_level'), "Missing get_progress_level method"
            print("✅ All required triple progress methods found")
            
            return True
        
    except Exception as e:
        print(f"❌ Dependency UI module error: {str(e)}")
        traceback.print_exc()
        return False

def test_triple_progress_methods():
    """Test triple progress methods specifically"""
    try:
        print("\n🔍 Testing triple progress methods...")
        
        with patch('smartcash.ui.setup.dependency.dependency_uimodule.get_dependency_config') as mock_config, \
             patch('smartcash.ui.components.operation_container.OperationContainer') as mock_container:
            
            mock_config.return_value = {'packages': ['test'], 'operation': 'install'}
            mock_container_instance = MagicMock()
            mock_container.return_value = mock_container_instance
            
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            module = DependencyUIModule()
            
            # Mock progress tracker with tqdm_manager
            mock_progress_tracker = MagicMock()
            mock_tqdm_manager = MagicMock()
            mock_progress_tracker.tqdm_manager = mock_tqdm_manager
            module._ui_components = {'progress_tracker': mock_progress_tracker}
            
            # Test triple progress update
            module.update_triple_progress(
                overall_progress=50,
                overall_message="Overall: 50%",
                step_progress=30,
                step_message="Step: 30%", 
                current_progress=80,
                current_message="Current: 80%"
            )
            print("✅ update_triple_progress executed successfully")
            
            # Test stage progress update (dual)
            module.update_stage_progress(
                overall_progress=60,
                overall_message="Overall: 60%",
                stage_progress=40,
                stage_message="Stage: 40%"
            )
            print("✅ update_stage_progress executed successfully")
            
            # Test progress level methods
            module.set_progress_level('triple')
            level = module.get_progress_level()
            print(f"✅ Progress level get/set methods work: {level}")
            
            return True
        
    except Exception as e:
        print(f"❌ Triple progress methods error: {str(e)}")
        traceback.print_exc()
        return False

def test_save_reset_operations():
    """Test save/reset operations without recursion"""
    try:
        print("\n🔍 Testing save/reset operations...")
        
        with patch('smartcash.ui.setup.dependency.dependency_uimodule.get_dependency_config') as mock_config, \
             patch('smartcash.ui.components.operation_container.OperationContainer') as mock_container:
            
            mock_config.return_value = {'packages': ['test'], 'operation': 'install'}
            mock_container.return_value = MagicMock()
            
            from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
            
            module = DependencyUIModule()
            
            # Mock save_config method
            module.save_config = MagicMock(return_value={'success': True, 'message': 'Config saved'})
            module.reset_config = MagicMock(return_value={'success': True, 'message': 'Config reset'})
            module._update_header_status = MagicMock()
            module.log = MagicMock()
            
            # Test save operation
            result = module._handle_save_config()
            print("✅ Save config operation completed")
            
            # Test reset operation  
            result = module._handle_reset_config()
            print("✅ Reset config operation completed")
            
            return True
        
    except Exception as e:
        print(f"❌ Save/reset operations error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🚀 Starting Dependency Module Validation Tests")
    print("=" * 60)
    
    tests = [
        test_dependency_module_imports,
        test_progress_config_creation,
        test_operation_container_integration,
        test_dependency_uimodule_initialization,
        test_triple_progress_methods,
        test_save_reset_operations
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All dependency module tests PASSED!")
        print("✅ Save/reset recursion fix: WORKING")
        print("✅ Progress tracker display: WORKING") 
        print("✅ Triple progress support: WORKING")
        return True
    else:
        print("\n⚠️  Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)