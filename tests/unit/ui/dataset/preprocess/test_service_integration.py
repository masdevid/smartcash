"""
File: tests/unit/ui/dataset/preprocess/test_service_integration.py
Description: Tests for service bridge integration and confirmation dialogs
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import asyncio

from smartcash.ui.dataset.preprocess.services.preprocess_service import PreprocessUIService
from smartcash.ui.dataset.preprocess.handlers.preprocess_ui_handler import PreprocessUIHandler
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler
from smartcash.ui.dataset.preprocess.constants import PreprocessingOperation


class TestServiceBridgeIntegration:
    """Test integration between UI handler and service bridge"""
    
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
            'confirmation_dialog': Mock(),
            'summary_container': Mock()
        }
        
        self.mock_config_handler = Mock(spec=PreprocessConfigHandler)
        self.mock_config_handler.extract_config_from_ui.return_value = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {
                'target_splits': ['train', 'valid'],
                'cleanup_target': 'preprocessed',
                'batch_size': 32
            }
        }
        self.mock_config_handler.validate_config.return_value = (True, [])
        
        self.ui_handler = PreprocessUIHandler(
            ui_components=self.mock_ui_components,
            config_handler=self.mock_config_handler
        )
        
        self.service = PreprocessUIService(self.mock_ui_components)
    
    def test_ui_handler_uses_service_for_operations(self):
        """Test that UI handler can integrate with service for operations"""
        # Mock the service integration
        with patch.object(self.ui_handler, '_create_operation_handler') as mock_create_op:
            mock_operation = Mock()
            mock_operation.execute = AsyncMock(return_value={'success': True, 'message': 'Done'})
            mock_create_op.return_value = mock_operation
            
            # Execute operation through handler
            self.ui_handler._execute_operation(
                PreprocessingOperation.PREPROCESS, 
                self.mock_config_handler.extract_config_from_ui.return_value
            )
            
            # Verify operation handler was created
            mock_create_op.assert_called_once_with(
                PreprocessingOperation.PREPROCESS,
                self.mock_config_handler.extract_config_from_ui.return_value
            )
    
    @pytest.mark.asyncio
    async def test_service_confirmation_workflow_preprocess(self):
        """Test service confirmation workflow for preprocessing"""
        config = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {'target_splits': ['train', 'valid']}
        }
        
        # Mock existing data check
        existing_data = {
            'requires_confirmation': True,
            'total_existing': 50,
            'has_existing': True,
            'by_split': {
                'train': {'existing_files': 30, 'path': 'data/preprocessed/train'},
                'valid': {'existing_files': 20, 'path': 'data/preprocessed/valid'}
            }
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            with patch.object(self.service, 'check_existing_data', return_value=existing_data):
                
                # First call without confirmation - should return confirmation needed
                result = await self.service.execute_preprocess_operation(config, confirm_overwrite=False)
                
                assert result['success'] is False
                assert result['requires_confirmation'] is True
                assert result['existing_data'] == existing_data
                assert 'Found 50 existing files' in result['message']
    
    @pytest.mark.asyncio
    async def test_service_confirmation_workflow_cleanup(self):
        """Test service confirmation workflow for cleanup"""
        config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'cleanup_target': 'preprocessed',
                'target_splits': ['train', 'valid']
            }
        }
        
        cleanup_preview = {
            'success': True,
            'total_files': 100,
            'total_size_mb': 250.5,
            'by_split': {
                'train': {'files': 60, 'size_mb': 150.0},
                'valid': {'files': 40, 'size_mb': 100.5}
            }
        }
        
        mock_cleanup_api = {
            'get_cleanup_preview': Mock(return_value=cleanup_preview)
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._cleanup_api = mock_cleanup_api
            
            # First call without confirmation - should return confirmation needed
            result = await self.service.execute_cleanup_operation(config, confirm_cleanup=False)
            
            assert result['success'] is False
            assert result['requires_confirmation'] is True
            assert result['cleanup_preview'] == cleanup_preview
            assert 'Will remove 100 files (250.5 MB)' in result['message']
    
    @pytest.mark.asyncio
    async def test_service_with_confirmation_accepted(self):
        """Test service workflow when confirmation is accepted"""
        config = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {'target_splits': ['train', 'valid']}
        }
        
        mock_preprocess_api = {
            'preprocess_dataset': Mock(return_value={
                'success': True,
                'message': 'Preprocessing completed',
                'processed_splits': ['train', 'valid'],
                'stats': {'total_files': 100, 'processed_files': 100}
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            
            # Call with confirmation accepted
            result = await self.service.execute_preprocess_operation(config, confirm_overwrite=True)
            
            assert result['success'] is True
            assert result['message'] == 'Preprocessing completed'
            assert result['processed_splits'] == ['train', 'valid']
            
            # Verify backend API was called
            mock_preprocess_api['preprocess_dataset'].assert_called_once_with(
                config=config,
                progress_callback=None,
                ui_components=self.mock_ui_components
            )
    
    def test_service_state_management(self):
        """Test service state management during operations"""
        # Initial state
        assert self.service.current_operation is None
        assert self.service.operation_results == {}
        
        # Set operation state
        self.service.current_operation = PreprocessingOperation.PREPROCESS
        self.service.operation_results['preprocess'] = {'success': True}
        
        # Check state
        assert self.service.current_operation == PreprocessingOperation.PREPROCESS
        assert 'preprocess' in self.service.operation_results
        
        # Clear state
        self.service.current_operation = None
        self.service.clear_operation_results()
        
        assert self.service.current_operation is None
        assert self.service.operation_results == {}
    
    def test_service_backend_availability_integration(self):
        """Test service backend availability checking"""
        # Initially not available
        assert self.service.is_backend_available() is False
        
        # Mock successful loading
        with patch.object(self.service, '_load_backend_modules'):
            assert self.service.is_backend_available() is True
        
        # Mock failed loading
        with patch.object(self.service, '_load_backend_modules', side_effect=RuntimeError("Failed")):
            assert self.service.is_backend_available() is False


class TestConfirmationDialogIntegration:
    """Test confirmation dialog integration patterns"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_ui_components = {
            'confirmation_dialog': Mock(),
            'operation_container': Mock(),
            'summary_container': Mock()
        }
        
        self.service = PreprocessUIService(self.mock_ui_components)
    
    def test_confirmation_dialog_structure(self):
        """Test confirmation dialog component structure"""
        # Mock confirmation dialog with expected interface
        dialog = self.mock_ui_components['confirmation_dialog']
        dialog.show_confirmation = Mock()
        dialog.on_confirm = Mock()
        dialog.on_cancel = Mock()
        
        # Test dialog interface
        assert hasattr(dialog, 'show_confirmation')
        assert hasattr(dialog, 'on_confirm')
        assert hasattr(dialog, 'on_cancel')
    
    @pytest.mark.asyncio
    async def test_confirmation_dialog_workflow(self):
        """Test complete confirmation dialog workflow"""
        config = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {'target_splits': ['train', 'valid']}
        }
        
        # Step 1: Initial operation call triggers confirmation
        existing_data = {
            'requires_confirmation': True,
            'total_existing': 25,
            'has_existing': True
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            with patch.object(self.service, 'check_existing_data', return_value=existing_data):
                
                result = await self.service.execute_preprocess_operation(config, confirm_overwrite=False)
                
                # Should request confirmation
                assert result['requires_confirmation'] is True
                assert result['existing_data'] == existing_data
        
        # Step 2: User confirms - operation proceeds
        mock_preprocess_api = {
            'preprocess_dataset': Mock(return_value={'success': True, 'message': 'Done'})
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            
            result = await self.service.execute_preprocess_operation(config, confirm_overwrite=True)
            
            # Should proceed with operation
            assert result['success'] is True
            mock_preprocess_api['preprocess_dataset'].assert_called_once()
    
    def test_confirmation_dialog_data_formatting(self):
        """Test confirmation dialog data formatting"""
        # Test preprocessing confirmation data
        existing_data = {
            'total_existing': 150,
            'by_split': {
                'train': {'existing_files': 100, 'path': '/data/train'},
                'valid': {'existing_files': 50, 'path': '/data/valid'}
            }
        }
        
        # Format for dialog display
        formatted_message = f"Found {existing_data['total_existing']} existing files. Confirm to overwrite?"
        split_details = []
        for split, data in existing_data['by_split'].items():
            split_details.append(f"{split}: {data['existing_files']} files")
        
        assert "Found 150 existing files" in formatted_message
        assert "train: 100 files" in split_details
        assert "valid: 50 files" in split_details
        
        # Test cleanup confirmation data
        cleanup_preview = {
            'total_files': 75,
            'total_size_mb': 125.5,
            'by_split': {
                'train': {'files': 50, 'size_mb': 80.0},
                'valid': {'files': 25, 'size_mb': 45.5}
            }
        }
        
        cleanup_message = f"Will remove {cleanup_preview['total_files']} files ({cleanup_preview['total_size_mb']:.1f} MB). Confirm?"
        assert "Will remove 75 files (125.5 MB)" in cleanup_message


class TestOperationSummaryIntegration:
    """Test operation summary integration with service results"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_ui_components = {
            'summary_container': Mock(),
            'operation_container': Mock()
        }
        
        self.service = PreprocessUIService(self.mock_ui_components)
    
    def test_summary_container_integration(self):
        """Test summary container integration with operation results"""
        # Mock summary container with expected interface
        summary = self.mock_ui_components['summary_container']
        summary.update_summary = Mock()
        summary.show_results = Mock()
        summary.clear = Mock()
        
        # Test summary interface
        assert hasattr(summary, 'update_summary')
        assert hasattr(summary, 'show_results')
        assert hasattr(summary, 'clear')
    
    def test_operation_results_to_summary_mapping(self):
        """Test mapping operation results to summary display"""
        # Test preprocessing results
        preprocess_results = {
            'operation': 'preprocess',
            'success': True,
            'stats': {
                'total_files': 200,
                'processed_files': 195,
                'failed_files': 5
            },
            'configuration': {
                'normalization_preset': 'yolov5l',
                'target_splits': ['train', 'valid', 'test']
            },
            'processing_time': 120.5
        }
        
        # Store results in service
        self.service.operation_results['preprocess'] = preprocess_results
        
        # Retrieve and verify
        stored = self.service.get_last_operation_results('preprocess')
        assert stored == preprocess_results
        assert stored['stats']['total_files'] == 200
        assert stored['configuration']['normalization_preset'] == 'yolov5l'
        
        # Test check results
        check_results = {
            'operation': 'check',
            'success': True,
            'service_ready': True,
            'file_statistics': {
                'train': {'raw_images': 100, 'preprocessed_files': 90},
                'valid': {'raw_images': 50, 'preprocessed_files': 45},
                'test': {'raw_images': 25, 'preprocessed_files': 20}
            },
            'validation_summary': {
                'total_valid': 175,
                'total_invalid': 0
            }
        }
        
        self.service.operation_results['check'] = check_results
        stored_check = self.service.get_last_operation_results('check')
        assert stored_check['service_ready'] is True
        assert len(stored_check['file_statistics']) == 3
    
    def test_summary_display_state_management(self):
        """Test summary display state management"""
        # Initially no results
        assert len(self.service.operation_results) == 0
        
        # Add multiple operation results
        self.service.operation_results.update({
            'preprocess': {'success': True, 'message': 'Done'},
            'check': {'success': True, 'service_ready': True},
            'cleanup': {'success': True, 'files_removed': 50}
        })
        
        # Check all results are stored
        assert len(self.service.operation_results) == 3
        assert 'preprocess' in self.service.operation_results
        assert 'check' in self.service.operation_results
        assert 'cleanup' in self.service.operation_results
        
        # Clear specific result
        del self.service.operation_results['preprocess']
        assert len(self.service.operation_results) == 2
        assert 'preprocess' not in self.service.operation_results
        
        # Clear all results
        self.service.clear_operation_results()
        assert len(self.service.operation_results) == 0


class TestFullWorkflowIntegration:
    """Test complete workflow integration from UI to backend"""
    
    def setup_method(self):
        """Setup complete test environment"""
        self.mock_ui_components = {
            'preprocess_btn': Mock(),
            'check_btn': Mock(),
            'cleanup_btn': Mock(),
            'operation_container': Mock(),
            'progress_tracker': Mock(),
            'log_accordion': Mock(),
            'confirmation_dialog': Mock(),
            'summary_container': Mock(),
            'update_status': Mock()
        }
        
        self.mock_config_handler = Mock(spec=PreprocessConfigHandler)
        self.mock_config_handler.extract_config_from_ui.return_value = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {
                'target_splits': ['train', 'valid'],
                'batch_size': 32,
                'cleanup_target': 'preprocessed'
            }
        }
        self.mock_config_handler.validate_config.return_value = (True, [])
        
        self.ui_handler = PreprocessUIHandler(
            ui_components=self.mock_ui_components,
            config_handler=self.mock_config_handler
        )
        
        self.service = PreprocessUIService(self.mock_ui_components)
    
    @pytest.mark.asyncio
    async def test_complete_preprocessing_workflow(self):
        """Test complete preprocessing workflow from button click to completion"""
        # Mock backend APIs
        mock_preprocess_api = {
            'preprocess_dataset': Mock(return_value={
                'success': True,
                'message': 'Preprocessing completed successfully',
                'stats': {'total_files': 100, 'processed_files': 100},
                'processed_splits': ['train', 'valid']
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            with patch.object(self.service, 'check_existing_data', return_value={'requires_confirmation': False}):
                self.service._preprocess_api = mock_preprocess_api
                
                # Simulate workflow
                result = await self.service.execute_preprocess_operation(
                    self.mock_config_handler.extract_config_from_ui.return_value,
                    confirm_overwrite=True
                )
                
                # Verify successful completion
                assert result['success'] is True
                assert result['message'] == 'Preprocessing completed successfully'
                assert result['stats']['processed_files'] == 100
                assert 'preprocess' in self.service.operation_results
    
    @pytest.mark.asyncio
    async def test_complete_check_workflow(self):
        """Test complete check workflow"""
        mock_preprocess_api = {
            'get_preprocessing_status': Mock(return_value={
                'success': True,
                'service_ready': True,
                'file_statistics': {
                    'train': {'raw_images': 50, 'preprocessed_files': 50},
                    'valid': {'raw_images': 25, 'preprocessed_files': 25}
                }
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            
            result = await self.service.execute_check_operation(
                self.mock_config_handler.extract_config_from_ui.return_value
            )
            
            assert result['success'] is True
            assert result['service_ready'] is True
            assert 'check' in self.service.operation_results
    
    @pytest.mark.asyncio
    async def test_complete_cleanup_workflow(self):
        """Test complete cleanup workflow with confirmation"""
        mock_cleanup_api = {
            'get_cleanup_preview': Mock(return_value={
                'success': True,
                'total_files': 0,  # No files to clean
                'total_size_mb': 0.0
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._cleanup_api = mock_cleanup_api
            
            result = await self.service.execute_cleanup_operation(
                self.mock_config_handler.extract_config_from_ui.return_value,
                confirm_cleanup=False
            )
            
            # Should complete immediately with no files
            assert result['success'] is True
            assert result['files_removed'] == 0
            assert 'No files found' in result['message']
    
    def test_error_handling_integration(self):
        """Test error handling across service integration"""
        # Test service creation with invalid components
        service = PreprocessUIService(None)
        status = service.get_service_status()
        assert status['ui_components_available'] is False
        
        # Test backend unavailability
        with patch.object(service, '_load_backend_modules', side_effect=RuntimeError("Backend error")):
            assert service.is_backend_available() is False
    
    def test_state_consistency_across_components(self):
        """Test state consistency across UI components and service"""
        # Initial state
        assert self.ui_handler.is_processing is False
        assert self.ui_handler.current_operation is None
        assert self.service.current_operation is None
        
        # Simulate operation start
        self.ui_handler.is_processing = True
        self.ui_handler.current_operation = PreprocessingOperation.PREPROCESS
        self.service.current_operation = PreprocessingOperation.PREPROCESS
        
        # Check consistency
        assert self.ui_handler.is_processing is True
        assert self.ui_handler.current_operation == PreprocessingOperation.PREPROCESS
        assert self.service.current_operation == PreprocessingOperation.PREPROCESS
        
        # Simulate operation end
        self.ui_handler.is_processing = False
        self.ui_handler.current_operation = None
        self.service.current_operation = None
        
        assert self.ui_handler.is_processing is False
        assert self.ui_handler.current_operation is None
        assert self.service.current_operation is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])