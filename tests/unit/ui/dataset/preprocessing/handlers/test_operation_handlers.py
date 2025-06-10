"""
File: tests/unit/ui/dataset/preprocessing/handlers/test_operation_handlers.py
Deskripsi: Unit test untuk operation handlers
"""
import pytest
from unittest.mock import MagicMock, patch
from smartcash.ui.dataset.preprocessing.handlers.operation_handlers import (
    get_operation_config,
    execute_operation,
    execute_preprocessing,
    check_dataset,
    cleanup_dataset
)

# Test data
MOCK_CONFIG = {
    'source_dir': '/path/to/source',
    'output_dir': '/path/to/output',
    'target_size': (640, 640),
    'normalization': 'minmax'
}

MOCK_UI_COMPONENTS = {
    'logger': MagicMock(),
    'progress_callback': MagicMock(),
    'setup_dual_progress_tracker': MagicMock(),
    'complete_progress_tracker': MagicMock(),
    'error_progress_tracker': MagicMock(),
    'show_ui_success': MagicMock(),
    'handle_ui_error': MagicMock(),
    'clear_outputs': MagicMock(),
}

class TestOperationHandlers:
    """Test case untuk operation handlers"""
    
    @pytest.mark.parametrize("op_type,expected_func_name", [
        ('preprocess', 'execute_preprocessing'),
        ('check', 'check_dataset'),
        ('cleanup', 'cleanup_dataset'),
        ('invalid', None)
    ])
    def test_get_operation_config(self, op_type, expected_func_name):
        """Test get_operation_config dengan berbagai tipe operasi"""
        func, _, _ = get_operation_config(op_type)
        
        if expected_func_name is None:
            assert func is None
        else:
            assert func.__name__ == expected_func_name
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.operation_handlers.execute_preprocessing')
    def test_execute_operation_preprocess_success(self, mock_execute):
        """Test execute_operation untuk operasi preprocessing yang berhasil"""
        # Setup
        mock_execute.return_value = (True, "Success")
        
        # Execute
        success, message = execute_operation(
            MOCK_UI_COMPONENTS,
            'preprocess',
            MOCK_CONFIG
        )
        
        # Verify
        assert success is True
        assert "Success" in message
        mock_execute.assert_called_once_with(MOCK_UI_COMPONENTS, MOCK_CONFIG)
    
    def test_execute_preprocessing_success(self):
        """Test execute_preprocessing dengan skenario sukses"""
        # Setup
        ui_components = MOCK_UI_COMPONENTS.copy()
        ui_components.update({
            'validate_dataset_ready': lambda _: (True, "Valid"),
            'check_preprocessed_exists': lambda _: (False, "No existing data"),
            'create_backend_preprocessor': lambda _: MagicMock(preprocess=MagicMock(return_value={'success': True, 'message': 'Done'})),
            '_convert_ui_to_backend_config': lambda _: MOCK_CONFIG,
            'progress_callback': MagicMock()
        })
        
        # Execute
        success, message = execute_preprocessing(ui_components, MOCK_CONFIG)
        
        # Verify
        assert success is True
        assert "berhasil" in message.lower()
    
    def test_check_dataset_failure(self):
        """Test check_dataset dengan skenario gagal"""
        # Setup
        ui_components = MOCK_UI_COMPONENTS.copy()
        ui_components['create_backend_checker'] = lambda _: None
        
        # Execute
        success, message = check_dataset(ui_components, MOCK_CONFIG)
        
        # Verify
        assert success is False
        assert "gagal" in message.lower()
    
    def test_cleanup_dataset_success(self):
        """Test cleanup_dataset dengan skenario sukses"""
        # Setup
        mock_service = MagicMock()
        mock_service.cleanup.return_value = (True, "Cleanup completed")
        
        ui_components = MOCK_UI_COMPONENTS.copy()
        ui_components['create_backend_cleanup_service'] = lambda _: mock_service
        
        # Execute
        success, message = cleanup_dataset(ui_components, MOCK_CONFIG)
        
        # Verify
        assert success is True
        assert "completed" in message
        mock_service.cleanup.assert_called_once()

    def test_execute_operation_invalid_type(self):
        """Test execute_operation dengan tipe operasi tidak valid"""
        # Execute
        success, message = execute_operation(
            MOCK_UI_COMPONENTS,
            'invalid_operation',
            MOCK_CONFIG
        )
        
        # Verify
        assert success is False
        assert "tidak valid" in message.lower()
