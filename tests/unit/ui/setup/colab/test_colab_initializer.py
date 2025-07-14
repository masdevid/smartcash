"""
Test module for colab initializer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.setup.colab.colab_uimodule import (
    ColabUIModule,
    create_colab_uimodule,
    initialize_colab_ui
)


class TestColabUIModule:
    """Test cases for ColabUIModule."""
    
    @pytest.fixture
    def uimodule(self):
        """Create a ColabUIModule instance for testing."""
        # Mock UIModule base class
        mock_uimodule = Mock()
        mock_uimodule.module_name = 'colab'
        mock_uimodule.parent_module = 'setup'
        mock_uimodule.status = 'pending'
        mock_uimodule.components = {}
        mock_uimodule.operations = {}
        mock_uimodule.get_component = Mock()
        mock_uimodule.update_config = Mock()
        mock_uimodule._update_status = Mock()
        
        # Create ColabUIModule with mocked base class
        with patch('smartcash.ui.setup.colab.colab_uimodule.UIModule', return_value=mock_uimodule):
            return ColabUIModule()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True,
                'project_name': 'SmartCash',
                'gpu_enabled': True
            },
            'setup': {
                'stages': ['environment_detection', 'drive_mount'],
                'auto_start': False
            }
        }
    
    def test_init(self, uimodule):
        """Test ColabUIModule initialization."""
        assert uimodule is not None
        assert uimodule.module_name == 'colab'
        assert uimodule.parent_module == 'setup'
        assert hasattr(uimodule, 'logger')
        assert hasattr(uimodule, '_operation_manager')
        assert hasattr(uimodule, '_config_handler')
        assert hasattr(uimodule, '_environment_detected')
        assert uimodule._environment_detected is False
    
    def test_initialize(self, uimodule, sample_config):
        """Test initialization with config."""
        # Mock UI components
        mock_components = {
            'setup_button': Mock(),
            'header_container': Mock(),
            'operation_container': Mock(),
            'main_container': Mock(),
            'form_container': Mock(),
        }
        
        # Mock operation manager
        mock_operation_manager = Mock()
        
        # Mock config handler
        mock_config_handler = Mock()
        
        # Mock detect_environment
        mock_detect_environment = Mock()
        mock_detect_environment.return_value = True
        
        # Mock create_ui_components
        mock_create_ui_components = Mock()
        mock_create_ui_components.return_value = mock_components
        
        # Mock setup_operation_manager
        mock_setup_operation_manager = Mock()
        mock_setup_operation_manager.return_value = mock_operation_manager
        
        # Mock setup_config_handler
        mock_setup_config_handler = Mock()
        mock_setup_config_handler.return_value = mock_config_handler
        
        # Mock register_operations
        mock_register_operations = Mock()
        
        # Mock setup_event_handlers
        mock_setup_event_handlers = Mock()
        
        # Mock verify_initialization
        mock_verify_initialization = Mock()
        
        # Mock log_initialization_complete
        mock_log_initialization_complete = Mock()
        
        # Mock _update_status
        mock_update_status = Mock()
        
        # Mock _environment_detected
        mock_environment_detected = Mock()
        mock_environment_detected.return_value = True
        
        # Patch all methods
        with patch.multiple(
            'smartcash.ui.setup.colab.colab_uimodule.UIModule',
            initialize=mock_uimodule.initialize,
            get_component=mock_uimodule.get_component,
            update_config=mock_uimodule.update_config,
            _update_status=mock_uimodule._update_status
        ) as uimodule_patches:
            with patch.multiple(
                'smartcash.ui.setup.colab.colab_uimodule.ColabUIModule',
                _detect_environment=mock_detect_environment,
                _create_ui_components=mock_create_ui_components,
                _setup_operation_manager=mock_setup_operation_manager,
                _setup_config_handler=mock_setup_config_handler,
                _register_operations=mock_register_operations,
                _setup_event_handlers=mock_setup_event_handlers,
                _verify_initialization=mock_verify_initialization,
                _log_initialization_complete=mock_log_initialization_complete,
                _update_status=mock_update_status,
                _environment_detected=mock_environment_detected
            ) as colab_patches:
                # Initialize with config
                result = uimodule.initialize(sample_config)
                
                # Verify initialization
                assert result is uimodule
                assert uimodule._environment_detected is True
                assert uimodule._operation_manager is mock_operation_manager
                assert uimodule._config_handler is mock_config_handler
                
                # Verify component creation
                mock_create_ui_components.assert_called_once()
                
                # Verify operation manager setup
                mock_setup_operation_manager.assert_called_once()
                
                # Verify config handler setup
                mock_setup_config_handler.assert_called_once()
                
                # Verify operations registration
                mock_register_operations.assert_called_once()
                
                # Verify event handlers setup
                mock_setup_event_handlers.assert_called_once()
                
                # Verify initialization verification
                mock_verify_initialization.assert_called_once()
                
                # Verify initialization complete logging
                mock_log_initialization_complete.assert_called_once()
                
                # Verify status update
                mock_update_status.assert_called()
                
                # Verify environment detection
                mock_detect_environment.assert_called()
                
                # Verify component access
                for comp_name in mock_components.keys():
                    uimodule.get_component(comp_name)
                    mock_uimodule.get_component.assert_any_call(comp_name)
                
                # Verify operation access
                for op_name in ['full_setup', 'status', 'reset']:
                    uimodule.get_operation(op_name)
                    assert hasattr(mock_operation_manager, op_name)
    
    def test_initialize_no_config(self, uimodule):
        """Test initialization with no config."""
        # Mock UI components
        mock_components = {
            'setup_button': Mock(),
            'header_container': Mock(),
            'operation_container': Mock(),
            'main_container': Mock(),
            'form_container': Mock(),
            'action_container': Mock()
        }
        
        # Mock operation manager
        mock_operation_manager = Mock()
        mock_operation_manager.log = Mock()
        mock_operation_manager.get_operations = Mock(return_value={'full_setup': Mock(), 'status': Mock(), 'reset': Mock()})
        
        # Mock config handler
        mock_config_handler = Mock()
        
        # Mock logger
        mock_logger = Mock()
        
        # Mock UIModule base class methods
        mock_uimodule = Mock()
        mock_uimodule.initialize = Mock()
        mock_uimodule.get_component = Mock(side_effect=lambda name: mock_components.get(name))
        mock_uimodule.update_config = Mock()
        mock_uimodule._update_status = Mock()
        
        # Mock environment detection
        mock_detect_environment = Mock()
        mock_detect_environment.return_value = True
        
        # Mock create_ui_components
        mock_create_ui_components = Mock()
        mock_create_ui_components.return_value = mock_components
        
        # Mock setup_operation_manager
        mock_setup_operation_manager = Mock()
        mock_setup_operation_manager.return_value = mock_operation_manager
        
        # Mock setup_config_handler
        mock_setup_config_handler = Mock()
        mock_setup_config_handler.return_value = mock_config_handler
        
        # Mock register_operations
        mock_register_operations = Mock()
        
        # Mock setup_event_handlers
        mock_setup_event_handlers = Mock()
        
        # Mock verify_initialization
        mock_verify_initialization = Mock()
        
        # Mock log_initialization_complete
        mock_log_initialization_complete = Mock()
        
        # Mock _update_status
        mock_update_status = Mock()
        
        # Mock _environment_detected
        mock_environment_detected = Mock()
        mock_environment_detected.return_value = True
        
        # Patch all methods
        with patch.multiple(
            # Initialize without config
            result = uimodule.initialize()
            
            # Verify initialization
            assert result is uimodule
            assert uimodule._environment_detected is True
            assert uimodule._operation_manager is mock_operation_manager
            assert uimodule._config_handler is mock_config_handler
            
            # Verify component creation
            mock_create_ui_components.assert_called_once()
            
            # Verify operation manager setup
            mock_setup_operation_manager.assert_called_once()
            
            # Verify config handler setup
            mock_setup_config_handler.assert_called_once()
            
            # Verify operations registration
            mock_register_operations.assert_called_once()
            
            # Verify event handlers setup
            mock_setup_event_handlers.assert_called_once()
            
            # Verify initialization verification
            mock_verify_initialization.assert_called_once()
            
            # Verify initialization complete logging
            mock_log_initialization_complete.assert_called_once()
            
            # Verify status update
            mock_update_status.assert_called()
            
            # Verify environment detection
            mock_detect_environment.assert_called()
            
            # Verify component access
            for comp_name in mock_components.keys():
                uimodule.get_component(comp_name)
                mock_uimodule.get_component.assert_any_call(comp_name)
            
            # Verify operation access
            for op_name in ['full_setup', 'status', 'reset']:
                uimodule.get_operation(op_name)
                assert hasattr(mock_operation_manager, op_name)
    
        with patch('sys.modules', {'google': None}) as mock_modules:
            with patch.object(uimodule, 'update_config') as mock_update_config:
                result = uimodule._detect_environment()
                assert result is False
                assert uimodule._environment_detected is False
                mock_update_config.assert_called_with(
                    environment_type='local',
                    is_colab=False
                )
                mock_modules.assert_called()


class TestInitializeColabUI:
    """Test cases for initialize_colab_ui function."""
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_colab_ui(self, mock_initialize):
        """Test colab UI initialization function."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        config = {'test': 'config'}
        result = initialize_colab_ui(config)
        
        mock_initialize.assert_called_once_with(
            module_name='colab',
            parent_module='setup',
            config=config,
            initializer_class=ColabUIModule
        )
        assert result == mock_ui
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_colab_ui_no_config(self, mock_initialize):
        """Test colab UI initialization without config."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        # Initialize without config
        result = initialize_colab_ui()
        
        # Verify initialization
        mock_initialize.assert_called_once_with(
            module_name='colab',
            parent_module='setup',
            initializer_class=ColabUIModule
        )
        assert result == mock_ui
        assert mock_initialize.call_count == 1