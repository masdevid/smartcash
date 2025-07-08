"""
File: tests/unit/ui/dataset/test_preprocess_comprehensive.py
Description: Comprehensive tests for preprocessing module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import ipywidgets as widgets

# Test imports
from smartcash.ui.dataset.preprocess.preprocess_initializer import (
    PreprocessInitializer, initialize_preprocessing_ui, _legacy_initialize_preprocessing_ui
)
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler
from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import get_default_preprocessing_config
from smartcash.ui.dataset.preprocess.handlers.preprocess_ui_handler import PreprocessUIHandler
from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocess.components.input_options import create_preprocessing_input_options
from smartcash.ui.dataset.preprocess.constants import (
    PreprocessingOperation, YOLO_PRESETS, BUTTON_CONFIG, UI_CONFIG
)


class TestPreprocessInitializer:
    """Test PreprocessInitializer class"""
    
    def test_initializer_creation(self):
        """Test initializer can be created successfully"""
        initializer = PreprocessInitializer()
        
        assert initializer.module_name == 'preprocess'
        assert initializer.parent_module == 'dataset'
        assert hasattr(initializer, 'module_metadata')
    
    def test_create_ui_components(self):
        """Test UI components creation"""
        initializer = PreprocessInitializer()
        config = get_default_preprocessing_config()
        
        ui_components = initializer.create_ui_components(config)
        
        # Check essential components
        assert 'ui' in ui_components
        assert 'main_container' in ui_components
        assert 'header_container' in ui_components
        assert 'form_container' in ui_components
        assert 'action_container' in ui_components
        assert 'operation_container' in ui_components
        assert 'footer_container' in ui_components
        
        # Check buttons
        assert 'preprocess_btn' in ui_components
        assert 'check_btn' in ui_components
        assert 'cleanup_btn' in ui_components
        
        # Check progress and logging
        assert 'progress_tracker' in ui_components
        assert 'log_accordion' in ui_components
        
        # Check form components
        assert 'resolution_dropdown' in ui_components
        assert 'normalization_dropdown' in ui_components
        assert 'target_splits_select' in ui_components
        
        # Check metadata
        assert ui_components['module_name'] == 'preprocess'
        assert ui_components['parent_module'] == 'dataset'
        assert ui_components['ui_initialized'] is True
    
    def test_create_config_handler(self):
        """Test config handler creation"""
        initializer = PreprocessInitializer()
        
        config_handler = initializer.create_config_handler()
        
        assert isinstance(config_handler, PreprocessConfigHandler)
        assert config_handler.module_name == 'preprocess'
        assert config_handler.parent_module == 'dataset'
    
    def test_create_module_handler(self):
        """Test module handler creation"""
        initializer = PreprocessInitializer()
        
        # Create mock UI components
        ui_components = {
            'preprocess_btn': Mock(),
            'check_btn': Mock(),
            'cleanup_btn': Mock(),
            'progress_tracker': Mock(),
            'log_accordion': Mock()
        }
        
        # Create config handler first
        initializer.config_handler = Mock()
        
        module_handler = initializer.create_module_handler(ui_components)
        
        assert isinstance(module_handler, PreprocessUIHandler)
        assert module_handler.module_name == 'preprocess'
        assert module_handler.parent_module == 'dataset'
    
    def test_setup_handlers(self):
        """Test handlers setup"""
        initializer = PreprocessInitializer()
        
        # Mock UI components
        ui_components = {
            'preprocess_btn': Mock(),
            'check_btn': Mock(),
            'cleanup_btn': Mock(),
            'progress_tracker': Mock(),
            'log_accordion': Mock()
        }
        
        # Create config handler
        initializer.config_handler = Mock()
        
        updated_components = initializer.setup_handlers(ui_components)
        
        assert 'config_handler' in updated_components
        assert 'module_handler' in updated_components
        assert isinstance(updated_components['module_handler'], PreprocessUIHandler)
    
    def test_get_critical_components(self):
        """Test critical components list"""
        initializer = PreprocessInitializer()
        
        critical = initializer.get_critical_components()
        
        expected_components = [
            'ui', 'main_container', 'preprocess_btn', 'check_btn', 'cleanup_btn',
            'operation_container', 'progress_tracker', 'log_accordion', 
            'header_container', 'form_container', 'action_container', 'footer_container'
        ]
        
        for component in expected_components:
            assert component in critical
    
    def test_pre_initialize_checks(self):
        """Test pre-initialization checks"""
        initializer = PreprocessInitializer()
        
        # Should not raise with IPython available
        with patch('builtins.__import__'):
            initializer.pre_initialize_checks()
        
        # Should raise without IPython
        with patch('builtins.__import__', side_effect=ImportError()):
            with pytest.raises(RuntimeError, match="IPython environment"):
                initializer.pre_initialize_checks()


class TestPreprocessConfigHandler:
    """Test PreprocessConfigHandler class"""
    
    def test_config_handler_creation(self):
        """Test config handler creation"""
        handler = PreprocessConfigHandler()
        
        assert handler.module_name == 'preprocess'
        assert handler.parent_module == 'dataset'
        assert hasattr(handler, 'default_config')
    
    def test_get_default_config(self):
        """Test default configuration"""
        handler = PreprocessConfigHandler()
        
        config = handler.get_default_config()
        
        assert 'preprocessing' in config
        assert 'data' in config
        assert 'performance' in config
        
        # Check preprocessing section
        preprocessing = config['preprocessing']
        assert 'target_splits' in preprocessing
        assert 'normalization' in preprocessing
        assert 'validation' in preprocessing
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        handler = PreprocessConfigHandler()
        config = get_default_preprocessing_config()
        
        is_valid, errors = handler.validate_config(config)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_invalid_splits(self):
        """Test configuration validation with invalid splits"""
        handler = PreprocessConfigHandler()
        config = get_default_preprocessing_config()
        config['preprocessing']['target_splits'] = ['invalid_split']
        
        is_valid, errors = handler.validate_config(config)
        
        assert is_valid is False
        assert any('Invalid split' in error for error in errors)
    
    def test_validate_config_invalid_preset(self):
        """Test configuration validation with invalid preset"""
        handler = PreprocessConfigHandler()
        config = get_default_preprocessing_config()
        config['preprocessing']['normalization']['preset'] = 'invalid_preset'
        
        is_valid, errors = handler.validate_config(config)
        
        assert is_valid is False
        assert any('Invalid preset' in error for error in errors)
    
    def test_validate_config_invalid_target_size(self):
        """Test configuration validation with invalid target size"""
        handler = PreprocessConfigHandler()
        config = get_default_preprocessing_config()
        config['preprocessing']['normalization']['target_size'] = [10000, 10000]  # Too large
        
        is_valid, errors = handler.validate_config(config)
        
        assert is_valid is False
        assert any('target_size values must be integers between 32 and 2048' in error for error in errors)
    
    def test_extract_config_from_ui(self):
        """Test extracting configuration from UI components"""
        handler = PreprocessConfigHandler()
        
        # Mock UI components
        ui_components = {
            'resolution_dropdown': Mock(value='yolov5l'),
            'normalization_dropdown': Mock(value='zscore'),
            'preserve_aspect_checkbox': Mock(value=False),
            'target_splits_select': Mock(value=['train', 'test']),
            'batch_size_input': Mock(value=64),
            'validation_checkbox': Mock(value=True),
            'move_invalid_checkbox': Mock(value=True),
            'invalid_dir_input': Mock(value='data/bad'),
            'cleanup_target_dropdown': Mock(value='both'),
            'backup_checkbox': Mock(value=False)
        }
        
        config = handler.extract_config_from_ui(ui_components)
        
        preprocessing = config['preprocessing']
        assert preprocessing['normalization']['preset'] == 'yolov5l'
        assert preprocessing['normalization']['method'] == 'zscore'
        assert preprocessing['normalization']['preserve_aspect_ratio'] is False
        assert preprocessing['target_splits'] == ['train', 'test']
        assert preprocessing['batch_size'] == 64
        assert preprocessing['validation']['enabled'] is True
        assert preprocessing['move_invalid'] is True
        assert preprocessing['invalid_dir'] == 'data/bad'
        assert preprocessing['cleanup_target'] == 'both'
        assert preprocessing['backup_enabled'] is False
    
    def test_update_ui_from_config(self):
        """Test updating UI components from configuration"""
        handler = PreprocessConfigHandler()
        
        # Create config
        config = get_default_preprocessing_config()
        config['preprocessing']['normalization']['preset'] = 'yolov5x'
        config['preprocessing']['batch_size'] = 128
        
        # Mock UI components
        ui_components = {
            'resolution_dropdown': Mock(),
            'batch_size_input': Mock()
        }
        
        handler.update_ui_from_config(ui_components, config)
        
        assert ui_components['resolution_dropdown'].value == 'yolov5x'
        assert ui_components['batch_size_input'].value == '128'
    
    def test_get_yolo_preset_config(self):
        """Test YOLO preset configuration retrieval"""
        handler = PreprocessConfigHandler()
        
        preset_config = handler.get_yolo_preset_config('yolov5l')
        
        assert preset_config['target_size'] == [832, 832]
        assert preset_config['preserve_aspect_ratio'] is True
        
        # Test invalid preset returns default
        default_config = handler.get_yolo_preset_config('invalid')
        assert default_config['target_size'] == [640, 640]
    
    def test_get_effective_normalization_config(self):
        """Test effective normalization configuration"""
        handler = PreprocessConfigHandler()
        
        # Update config to use yolov5l preset
        config = handler.get_config()
        config['preprocessing']['normalization']['preset'] = 'yolov5l'
        handler.update_config(config)
        
        effective = handler.get_effective_normalization_config()
        
        assert effective['target_size'] == [832, 832]  # From yolov5l preset
        assert effective['preset'] == 'yolov5l'
    
    def test_get_processing_splits(self):
        """Test processing splits retrieval"""
        handler = PreprocessConfigHandler()
        
        splits = handler.get_processing_splits()
        
        assert isinstance(splits, list)
        assert 'train' in splits
        assert 'valid' in splits
    
    def test_get_data_directories(self):
        """Test data directories configuration"""
        handler = PreprocessConfigHandler()
        
        dirs = handler.get_data_directories()
        
        assert 'source_dir' in dirs
        assert 'preprocessed_dir' in dirs
        assert dirs['source_dir'] == 'data'
        assert dirs['preprocessed_dir'] == 'data/preprocessed'
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults"""
        handler = PreprocessConfigHandler()
        
        # Modify config
        config = handler.get_config()
        config['preprocessing']['batch_size'] = 999
        handler.update_config(config)
        
        # Reset to defaults
        handler.reset_to_defaults()
        
        new_config = handler.get_config()
        assert new_config['preprocessing']['batch_size'] == 32  # Default value


class TestPreprocessUIHandler:
    """Test PreprocessUIHandler class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_ui_components = {
            'preprocess_btn': Mock(),
            'check_btn': Mock(),
            'cleanup_btn': Mock(),
            'operation_container': Mock(),
            'progress_tracker': Mock(),
            'log_accordion': Mock(),
            'update_status': Mock(),
            'resolution_dropdown': Mock(value='yolov5s'),
            'batch_size_input': Mock(value=32)
        }
        
        self.mock_config_handler = Mock(spec=PreprocessConfigHandler)
        self.mock_config_handler.extract_config_from_ui.return_value = get_default_preprocessing_config()
        self.mock_config_handler.validate_config.return_value = (True, [])
        
        self.handler = PreprocessUIHandler(
            ui_components=self.mock_ui_components,
            config_handler=self.mock_config_handler
        )
    
    def test_handler_creation(self):
        """Test handler creation"""
        assert self.handler.module_name == 'preprocess'
        assert self.handler.parent_module == 'dataset'
        assert self.handler.is_processing is False
        assert self.handler.current_operation is None
    
    def test_handle_preprocess_click_success(self):
        """Test successful preprocess button click"""
        with patch('smartcash.dataset.preprocessor.preprocess_dataset') as mock_preprocess:
            mock_preprocess.return_value = {'success': True, 'message': 'Completed'}
            
            self.handler.handle_preprocess_click()
            
            # Verify config extraction and validation
            self.mock_config_handler.extract_config_from_ui.assert_called_once()
            self.mock_config_handler.validate_config.assert_called_once()
            self.mock_config_handler.update_config.assert_called_once()
            
            # Verify backend call
            mock_preprocess.assert_called_once()
    
    def test_handle_preprocess_click_validation_error(self):
        """Test preprocess click with validation error"""
        self.mock_config_handler.validate_config.return_value = (False, ['Invalid config'])
        
        self.handler.handle_preprocess_click()
        
        # Should not proceed to backend
        assert not self.handler.is_processing
    
    def test_handle_preprocess_click_during_processing(self):
        """Test preprocess click while already processing"""
        self.handler.is_processing = True
        
        self.handler.handle_preprocess_click()
        
        # Should not extract config again
        self.mock_config_handler.extract_config_from_ui.assert_not_called()
    
    def test_handle_check_click(self):
        """Test check button click"""
        with patch('smartcash.dataset.preprocessor.get_preprocessing_status') as mock_check:
            mock_check.return_value = {
                'service_ready': True,
                'file_statistics': {'train': {'raw_images': 100, 'preprocessed_files': 0}}
            }
            
            self.handler.handle_check_click()
            
            mock_check.assert_called_once()
    
    def test_handle_cleanup_click(self):
        """Test cleanup button click"""
        with patch('smartcash.dataset.preprocessor.api.cleanup_api.cleanup_preprocessing_files') as mock_cleanup:
            mock_cleanup.return_value = {'success': True, 'files_removed': 50}
            
            self.handler.handle_cleanup_click()
            
            mock_cleanup.assert_called_once()
    
    def test_execute_operation(self):
        """Test operation execution"""
        with patch.object(self.handler, '_run_backend_operation') as mock_backend:
            config = get_default_preprocessing_config()
            
            self.handler._execute_operation(PreprocessingOperation.PREPROCESS, config)
            
            mock_backend.assert_called_once_with(PreprocessingOperation.PREPROCESS, config)
    
    def test_create_progress_callback(self):
        """Test progress callback creation"""
        callback = self.handler._create_progress_callback()
        
        # Test callback execution
        callback('overall', 50, 100, 'Processing...')
        
        # Should not raise exceptions
        assert callable(callback)
    
    def test_set_buttons_enabled(self):
        """Test button state management"""
        self.handler._set_buttons_enabled(False)
        
        # Check that buttons are disabled
        for btn_name in ['preprocess_btn', 'check_btn', 'cleanup_btn']:
            if hasattr(self.mock_ui_components[btn_name], 'disabled'):
                assert self.mock_ui_components[btn_name].disabled is True
    
    def test_setup_config_handlers(self):
        """Test configuration change handlers setup"""
        # Mock components with observe method
        for component_name in ['resolution_dropdown', 'batch_size_input']:
            self.mock_ui_components[component_name].observe = Mock()
        
        self.handler.setup_config_handlers(self.mock_ui_components)
        
        # Check that observe was called for each component
        for component_name in ['resolution_dropdown', 'batch_size_input']:
            self.mock_ui_components[component_name].observe.assert_called()
    
    def test_handle_config_change(self):
        """Test configuration change handling"""
        change = {'new': 'yolov5l'}
        
        self.handler._handle_config_change(change)
        
        self.mock_config_handler.extract_config_from_ui.assert_called_once()
        self.mock_config_handler.update_config.assert_called_once()
    
    def test_cleanup(self):
        """Test handler cleanup"""
        self.handler.is_processing = True
        self.handler.current_operation = PreprocessingOperation.PREPROCESS
        
        self.handler.cleanup()
        
        assert self.handler.is_processing is False
        assert self.handler.current_operation is None


class TestPreprocessComponents:
    """Test preprocessing UI components"""
    
    def test_create_preprocessing_main_ui(self):
        """Test main UI creation"""
        config = get_default_preprocessing_config()
        
        ui_components = create_preprocessing_main_ui(config)
        
        # Check essential components
        assert 'ui' in ui_components
        assert 'main_container' in ui_components
        assert 'header_container' in ui_components
        assert 'form_container' in ui_components
        assert 'action_container' in ui_components
        assert 'operation_container' in ui_components
        assert 'footer_container' in ui_components
        
        # Check buttons
        assert 'preprocess_btn' in ui_components
        assert 'check_btn' in ui_components
        assert 'cleanup_btn' in ui_components
        
        # Check helper methods
        assert callable(ui_components['update_status'])
        assert callable(ui_components['update_title'])
        assert callable(ui_components['update_section'])
        
        # Check metadata
        assert ui_components['module_name'] == 'preprocess'
        assert ui_components['ui_initialized'] is True
    
    def test_create_preprocessing_input_options(self):
        """Test input options creation"""
        config = get_default_preprocessing_config()
        
        input_options = create_preprocessing_input_options(config)
        
        # Check that it's a VBox widget
        assert isinstance(input_options, widgets.VBox)
        
        # Check attached form components
        assert hasattr(input_options, 'resolution_dropdown')
        assert hasattr(input_options, 'normalization_dropdown')
        assert hasattr(input_options, 'preserve_aspect_checkbox')
        assert hasattr(input_options, 'target_splits_select')
        assert hasattr(input_options, 'batch_size_input')
        assert hasattr(input_options, 'validation_checkbox')
        assert hasattr(input_options, 'cleanup_target_dropdown')
        assert hasattr(input_options, 'backup_checkbox')
        
        # Check initial values
        assert input_options.resolution_dropdown.value == 'yolov5s'
        assert input_options.normalization_dropdown.value == 'minmax'
        assert input_options.batch_size_input.value == 32
    
    def test_input_options_with_custom_config(self):
        """Test input options with custom configuration"""
        config = {
            'preprocessing': {
                'normalization': {
                    'preset': 'yolov5l',
                    'method': 'zscore',
                    'preserve_aspect_ratio': False
                },
                'target_splits': ['train', 'test'],
                'batch_size': 64,
                'validation': {'enabled': True},
                'move_invalid': True,
                'cleanup_target': 'both',
                'backup_enabled': False
            }
        }
        
        input_options = create_preprocessing_input_options(config)
        
        # Check custom values are applied
        assert input_options.resolution_dropdown.value == 'yolov5l'
        assert input_options.normalization_dropdown.value == 'zscore'
        assert input_options.preserve_aspect_checkbox.value is False
        assert input_options.batch_size_input.value == 64
        assert input_options.validation_checkbox.value is True
        assert input_options.move_invalid_checkbox.value is True
        assert input_options.cleanup_target_dropdown.value == 'both'
        assert input_options.backup_checkbox.value is False


class TestPreprocessConstants:
    """Test preprocessing constants and enums"""
    
    def test_yolo_presets(self):
        """Test YOLO presets configuration"""
        assert 'yolov5s' in YOLO_PRESETS
        assert 'yolov5l' in YOLO_PRESETS
        assert 'yolov5x' in YOLO_PRESETS
        
        # Check yolov5s preset
        yolov5s = YOLO_PRESETS['yolov5s']
        assert yolov5s['target_size'] == [640, 640]
        assert yolov5s['preserve_aspect_ratio'] is True
        
        # Check yolov5l preset
        yolov5l = YOLO_PRESETS['yolov5l']
        assert yolov5l['target_size'] == [832, 832]
    
    def test_button_config(self):
        """Test button configuration"""
        assert 'preprocess' in BUTTON_CONFIG
        assert 'check' in BUTTON_CONFIG
        assert 'cleanup' in BUTTON_CONFIG
        
        # Check preprocess button
        preprocess_btn = BUTTON_CONFIG['preprocess']
        assert preprocess_btn['text'] == '🚀 Mulai Preprocessing'
        assert preprocess_btn['style'] == 'primary'
        assert preprocess_btn['order'] == 1
    
    def test_ui_config(self):
        """Test UI configuration"""
        assert UI_CONFIG['title'] == 'Dataset Preprocessing'
        assert UI_CONFIG['module_name'] == 'preprocess'
        assert UI_CONFIG['parent_module'] == 'dataset'


class TestPreprocessIntegration:
    """Integration tests for preprocessing module"""
    
    def test_legacy_function_wrapper(self):
        """Test legacy function wrapper"""
        with patch('smartcash.ui.dataset.preprocess.preprocess_initializer.PreprocessInitializer') as mock_init_class:
            mock_initializer = Mock()
            mock_initializer.initialize.return_value = {'ui': Mock()}
            mock_init_class.return_value = mock_initializer
            
            result = _legacy_initialize_preprocessing_ui()
            
            mock_init_class.assert_called_once()
            mock_initializer.initialize.assert_called_once()
            assert 'ui' in result
    
    @patch('smartcash.ui.core.initializers.display_initializer.create_ui_display_function')
    def test_display_function_creation(self, mock_create_display):
        """Test display function creation"""
        # Import should trigger display function creation
        from smartcash.ui.dataset.preprocess.preprocess_initializer import initialize_preprocessing_ui
        
        mock_create_display.assert_called_once()
        call_args = mock_create_display.call_args
        assert call_args[1]['module_name'] == 'preprocess'
        assert call_args[1]['parent_module'] == 'dataset'
    
    def test_module_imports(self):
        """Test module imports work correctly"""
        # Test main module imports
        from smartcash.ui.dataset.preprocess import (
            initialize_preprocessing_ui, PreprocessInitializer
        )
        
        assert callable(initialize_preprocessing_ui)
        assert PreprocessInitializer is not None
        
        # Test component imports
        from smartcash.ui.dataset.preprocess.components import (
            create_preprocessing_main_ui, create_preprocessing_input_options
        )
        
        assert callable(create_preprocessing_main_ui)
        assert callable(create_preprocessing_input_options)
        
        # Test config imports
        from smartcash.ui.dataset.preprocess.configs import (
            PreprocessConfigHandler, get_default_preprocessing_config
        )
        
        assert PreprocessConfigHandler is not None
        assert callable(get_default_preprocessing_config)


class TestPreprocessErrorHandling:
    """Test error handling in preprocessing module"""
    
    def test_initializer_error_handling(self):
        """Test initializer error handling"""
        initializer = PreprocessInitializer()
        
        # Test with invalid config
        with pytest.raises(Exception):
            initializer.create_ui_components(None)
    
    def test_config_handler_error_handling(self):
        """Test config handler error handling"""
        handler = PreprocessConfigHandler()
        
        # Test with malformed UI components
        ui_components = {'invalid': 'component'}
        
        # Should not raise exception
        config = handler.extract_config_from_ui(ui_components)
        assert 'preprocessing' in config
    
    def test_ui_handler_error_handling(self):
        """Test UI handler error handling"""
        ui_components = {}
        config_handler = Mock()
        
        handler = PreprocessUIHandler(ui_components, config_handler)
        
        # Test handling clicks without proper UI components
        handler.handle_preprocess_click()  # Should not raise
        handler.handle_check_click()  # Should not raise
        handler.handle_cleanup_click()  # Should not raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])