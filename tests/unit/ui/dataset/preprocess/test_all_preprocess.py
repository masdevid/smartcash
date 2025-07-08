"""
File: tests/unit/ui/dataset/preprocess/test_all_preprocess.py
Description: Comprehensive test runner for all preprocessing module tests
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestPreprocessModuleIntegration:
    """Integration tests for the complete preprocessing module"""
    
    def test_module_structure_compliance(self):
        """Test that module structure complies with documentation"""
        # Test main module structure
        try:
            from smartcash.ui.dataset.preprocess import (
                initialize_preprocessing_ui, PreprocessInitializer
            )
            assert initialize_preprocessing_ui is not None
            assert PreprocessInitializer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import main module components: {e}")
        
        # Test components structure
        try:
            from smartcash.ui.dataset.preprocess.components import (
                create_preprocessing_main_ui, create_preprocessing_input_options
            )
            assert create_preprocessing_main_ui is not None
            assert create_preprocessing_input_options is not None
        except ImportError as e:
            pytest.fail(f"Failed to import component modules: {e}")
        
        # Test handlers structure
        try:
            from smartcash.ui.dataset.preprocess.handlers import (
                PreprocessUIHandler
            )
            assert PreprocessUIHandler is not None
        except ImportError as e:
            pytest.fail(f"Failed to import handler modules: {e}")
        
        # Test configs structure
        try:
            from smartcash.ui.dataset.preprocess.configs import (
                PreprocessConfigHandler, get_default_preprocessing_config
            )
            assert PreprocessConfigHandler is not None
            assert get_default_preprocessing_config is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config modules: {e}")
        
        # Test operations structure
        try:
            from smartcash.ui.dataset.preprocess.operations import (
                PreprocessOperation, CheckOperation, CleanupOperation
            )
            assert PreprocessOperation is not None
            assert CheckOperation is not None
            assert CleanupOperation is not None
        except ImportError as e:
            pytest.fail(f"Failed to import operation modules: {e}")
        
        # Test services structure
        try:
            from smartcash.ui.dataset.preprocess.services import (
                PreprocessUIService
            )
            assert PreprocessUIService is not None
        except ImportError as e:
            pytest.fail(f"Failed to import service modules: {e}")
    
    def test_constants_availability(self):
        """Test that all required constants are available"""
        try:
            from smartcash.ui.dataset.preprocess.constants import (
                PreprocessingOperation, YOLO_PRESETS, BANKNOTE_CLASSES,
                BUTTON_CONFIG, UI_CONFIG, SUCCESS_MESSAGES, ERROR_MESSAGES
            )
            
            # Test enum values
            assert PreprocessingOperation.PREPROCESS is not None
            assert PreprocessingOperation.CHECK is not None
            assert PreprocessingOperation.CLEANUP is not None
            
            # Test configuration dictionaries
            assert 'yolov5s' in YOLO_PRESETS
            assert 'yolov5l' in YOLO_PRESETS
            assert 'yolov5x' in YOLO_PRESETS
            
            assert len(BANKNOTE_CLASSES) > 0
            assert all('display' in cls_info for cls_info in BANKNOTE_CLASSES.values())
            
            assert 'preprocess' in BUTTON_CONFIG
            assert 'check' in BUTTON_CONFIG
            assert 'cleanup' in BUTTON_CONFIG
            
            assert 'title' in UI_CONFIG
            assert 'module_name' in UI_CONFIG
            
        except ImportError as e:
            pytest.fail(f"Failed to import constants: {e}")
    
    def test_modern_ui_container_structure(self):
        """Test that module uses modern UI container structure"""
        from smartcash.ui.dataset.preprocess.preprocess_initializer import PreprocessInitializer
        from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import get_default_preprocessing_config
        
        initializer = PreprocessInitializer()
        config = get_default_preprocessing_config()
        
        ui_components = initializer.create_ui_components(config)
        
        # Check for modern container structure
        modern_containers = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container',  # Modern: unified progress, dialogs, logs
            'footer_container'
        ]
        
        for container in modern_containers:
            assert container in ui_components, f"Missing modern container: {container}"
        
        # Check that operation_container includes all necessary components
        operation_container = ui_components.get('operation_container')
        assert operation_container is not None
        
        # Check modern components are available
        modern_components = [
            'progress_tracker',
            'log_accordion', 
            'confirmation_dialog',
            'summary_container'
        ]
        
        for component in modern_components:
            assert component in ui_components, f"Missing modern component: {component}"
    
    def test_operation_handler_inheritance(self):
        """Test that operations inherit from OperationHandler"""
        from smartcash.ui.dataset.preprocess.operations import (
            PreprocessOperation, CheckOperation, CleanupOperation
        )
        from smartcash.ui.core.handlers.operation_handler import OperationHandler
        
        # Create mock UI components and config
        mock_ui_components = {'test': Mock()}
        mock_config = {'test': 'config'}
        
        # Test that operations can be created and inherit from OperationHandler
        preprocess_op = PreprocessOperation(
            ui_components=mock_ui_components,
            config=mock_config,
            progress_callback=Mock(),
            log_callback=Mock()
        )
        
        check_op = CheckOperation(
            ui_components=mock_ui_components,
            config=mock_config,
            progress_callback=Mock(),
            log_callback=Mock()
        )
        
        cleanup_op = CleanupOperation(
            ui_components=mock_ui_components,
            config=mock_config,
            progress_callback=Mock(),
            log_callback=Mock()
        )
        
        # Test inheritance
        assert isinstance(preprocess_op, OperationHandler)
        assert isinstance(check_op, OperationHandler)
        assert isinstance(cleanup_op, OperationHandler)
        
        # Test that shared methods are available
        shared_methods = ['execute', 'update_progress', 'log_message']
        for op in [preprocess_op, check_op, cleanup_op]:
            for method in shared_methods:
                assert hasattr(op, method), f"Operation missing shared method: {method}"
                assert callable(getattr(op, method))
    
    def test_backend_integration_no_simulation(self):
        """Test that operations use real backend integration without simulation"""
        from smartcash.ui.dataset.preprocess.operations.preprocess_operation import PreprocessOperation
        from smartcash.ui.dataset.preprocess.operations.check_operation import CheckOperation
        from smartcash.ui.dataset.preprocess.operations.cleanup_operation import CleanupOperation
        
        # Read operation source code to ensure no simulation
        import inspect
        
        # Check PreprocessOperation
        preprocess_source = inspect.getsource(PreprocessOperation._processing_phase)
        assert 'simulate' not in preprocess_source.lower()
        assert 'mock' not in preprocess_source.lower()
        assert 'smartcash.dataset.preprocessor' in preprocess_source
        
        # Check CheckOperation  
        check_source = inspect.getsource(CheckOperation._processing_phase)
        assert 'simulate' not in check_source.lower()
        assert 'mock' not in check_source.lower()
        assert 'smartcash.dataset.preprocessor' in check_source
        
        # Check CleanupOperation
        cleanup_source = inspect.getsource(CleanupOperation._processing_phase)
        assert 'simulate' not in cleanup_source.lower()
        assert 'mock' not in cleanup_source.lower()
        assert 'smartcash.dataset.preprocessor' in cleanup_source
    
    def test_service_bridge_integration(self):
        """Test service bridge between UI and backend"""
        from smartcash.ui.dataset.preprocess.services import PreprocessUIService
        
        mock_ui_components = {'test': Mock()}
        service = PreprocessUIService(mock_ui_components)
        
        # Test service interface
        service_methods = [
            'check_existing_data',
            'execute_preprocess_operation', 
            'execute_check_operation',
            'execute_cleanup_operation',
            'get_last_operation_results',
            'is_backend_available',
            'get_service_status'
        ]
        
        for method in service_methods:
            assert hasattr(service, method), f"Service missing method: {method}"
            assert callable(getattr(service, method))
        
        # Test that service manages confirmation workflows
        assert hasattr(service, '_load_backend_modules')
        assert hasattr(service, 'operation_results')
        assert hasattr(service, 'current_operation')
    
    def test_component_summary_and_confirmation(self):
        """Test operation summary and confirmation dialog components"""
        from smartcash.ui.dataset.preprocess.components.operation_summary import (
            create_operation_summary, create_operation_results_summary
        )
        
        # Test summary creation
        config = {'preprocessing': {'normalization': {'preset': 'yolov5s'}}}
        summary = create_operation_summary(config)
        
        assert summary is not None
        assert hasattr(summary, 'update_summary')
        assert callable(summary.update_summary)
        
        # Test results summary
        results = {
            'operation': 'preprocess',
            'success': True,
            'stats': {'total_files': 100, 'processed_files': 100}
        }
        results_summary = create_operation_results_summary(results)
        
        assert results_summary is not None
        assert len(results_summary.children) > 0


class TestPreprocessModuleCompliance:
    """Test compliance with project documentation and standards"""
    
    def test_file_structure_compliance(self):
        """Test that file structure follows documented patterns"""
        import os
        
        base_path = '/Users/masdevid/Projects/smartcash/smartcash/ui/dataset/preprocess'
        
        # Check main structure
        required_dirs = [
            'components',
            'configs', 
            'handlers',
            'operations',
            'services'
        ]
        
        for dir_name in required_dirs:
            dir_path = os.path.join(base_path, dir_name)
            assert os.path.exists(dir_path), f"Missing required directory: {dir_name}"
        
        # Check main files
        required_files = [
            'preprocess_initializer.py',
            'constants.py',
            '__init__.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(base_path, file_name)
            assert os.path.exists(file_path), f"Missing required file: {file_name}"
    
    def test_naming_conventions(self):
        """Test that naming conventions are followed"""
        from smartcash.ui.dataset.preprocess.constants import UI_CONFIG, BUTTON_CONFIG
        
        # Test module naming
        assert UI_CONFIG['module_name'] == 'preprocess'
        assert UI_CONFIG['parent_module'] == 'dataset'
        
        # Test consistent naming in buttons
        button_names = list(BUTTON_CONFIG.keys())
        assert 'preprocess' in button_names
        assert 'check' in button_names
        assert 'cleanup' in button_names
    
    def test_initialization_pattern_compliance(self):
        """Test that initialization follows ModuleInitializer pattern"""
        from smartcash.ui.dataset.preprocess.preprocess_initializer import PreprocessInitializer
        from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
        
        initializer = PreprocessInitializer()
        
        # Test inheritance
        assert isinstance(initializer, ModuleInitializer)
        
        # Test required methods are implemented
        required_methods = [
            'create_ui_components',
            'create_config_handler',
            'create_module_handler',
            'setup_handlers',
            'get_critical_components'
        ]
        
        for method in required_methods:
            assert hasattr(initializer, method), f"Missing required method: {method}"
            assert callable(getattr(initializer, method))


def run_all_preprocessing_tests():
    """Run all preprocessing module tests"""
    test_files = [
        'test_preprocess_comprehensive.py',
        'test_preprocess_service.py', 
        'test_operation_summary.py',
        'test_service_integration.py',
        'test_all_preprocess.py'
    ]
    
    test_dir = os.path.dirname(__file__)
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"\n{'='*60}")
            print(f"Running tests from: {test_file}")
            print(f"{'='*60}")
            
            # Run pytest on the specific file
            result = pytest.main([test_path, '-v', '--tb=short'])
            
            if result != 0:
                print(f"❌ Tests failed in {test_file}")
                return False
            else:
                print(f"✅ All tests passed in {test_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    print(f"\n{'='*60}")
    print("🎉 All preprocessing module tests completed!")
    print(f"{'='*60}")
    return True


if __name__ == '__main__':
    success = run_all_preprocessing_tests()
    sys.exit(0 if success else 1)