"""
File: tests/unit/ui/dataset/preprocess/test_preprocess_service.py
Description: Tests for PreprocessUIService
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any
import asyncio

from smartcash.ui.dataset.preprocess.services.preprocess_service import PreprocessUIService
from smartcash.ui.dataset.preprocess.constants import PreprocessingOperation, CleanupTarget


class TestPreprocessUIService:
    """Test PreprocessUIService class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_ui_components = {
            'progress_tracker': Mock(),
            'log_accordion': Mock(),
            'operation_container': Mock()
        }
        
        self.service = PreprocessUIService(self.mock_ui_components)
    
    def test_service_creation(self):
        """Test service creation"""
        assert self.service.ui_components == self.mock_ui_components
        assert self.service.current_operation is None
        assert self.service.operation_results == {}
        assert self.service._backend_loaded is False
    
    def test_backend_loading(self):
        """Test backend modules loading"""
        with patch('smartcash.dataset.preprocessor.preprocess_dataset') as mock_preprocess:
            with patch('smartcash.dataset.preprocessor.get_preprocessing_status') as mock_status:
                with patch('smartcash.dataset.preprocessor.api.cleanup_api.cleanup_preprocessing_files') as mock_cleanup:
                    
                    self.service._load_backend_modules()
                    
                    assert self.service._backend_loaded is True
                    assert self.service._preprocess_api is not None
                    assert self.service._cleanup_api is not None
                    assert self.service._stats_api is not None
    
    def test_backend_loading_failure(self):
        """Test backend loading failure"""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(RuntimeError, match="Backend preprocessing modules not available"):
                self.service._load_backend_modules()
    
    @pytest.mark.asyncio
    async def test_check_existing_data_with_files(self):
        """Test checking existing data with files present"""
        config = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {'target_splits': ['train', 'valid']}
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.glob', return_value=[Mock(), Mock(), Mock()]):  # 3 files
                with patch.object(self.service, '_load_backend_modules'):
                    
                    result = await self.service.check_existing_data(config)
                    
                    assert result['has_existing'] is True
                    assert result['total_existing'] == 6  # 3 files * 2 splits
                    assert result['requires_confirmation'] is True
                    assert 'train' in result['by_split']
                    assert 'valid' in result['by_split']
    
    @pytest.mark.asyncio
    async def test_check_existing_data_no_files(self):
        """Test checking existing data with no files"""
        config = {
            'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
            'preprocessing': {'target_splits': ['train', 'valid']}
        }
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(self.service, '_load_backend_modules'):
                
                result = await self.service.check_existing_data(config)
                
                assert result['has_existing'] is False
                assert result['total_existing'] == 0
                assert result['requires_confirmation'] is False
    
    @pytest.mark.asyncio
    async def test_check_existing_data_error(self):
        """Test checking existing data with error"""
        config = {'data': {}, 'preprocessing': {}}
        
        with patch.object(self.service, '_load_backend_modules', side_effect=Exception("Backend error")):
            
            result = await self.service.check_existing_data(config)
            
            assert result['has_existing'] is False
            assert 'error' in result
            assert result['requires_confirmation'] is False
    
    @pytest.mark.asyncio
    async def test_execute_preprocess_operation_success(self):
        """Test successful preprocessing operation"""
        config = {'data': {}, 'preprocessing': {}}
        progress_callback = Mock()
        
        mock_preprocess_api = {
            'preprocess_dataset': Mock(return_value={'success': True, 'message': 'Done'})
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            with patch.object(self.service, 'check_existing_data', return_value={'requires_confirmation': False}):
                self.service._preprocess_api = mock_preprocess_api
                
                result = await self.service.execute_preprocess_operation(config, progress_callback)
                
                assert result['success'] is True
                assert result['message'] == 'Done'
                assert self.service.current_operation == PreprocessingOperation.PREPROCESS
                assert 'preprocess' in self.service.operation_results
    
    @pytest.mark.asyncio
    async def test_execute_preprocess_operation_needs_confirmation(self):
        """Test preprocessing operation needing confirmation"""
        config = {'data': {}, 'preprocessing': {}}
        
        existing_check = {
            'requires_confirmation': True,
            'total_existing': 10,
            'has_existing': True
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            with patch.object(self.service, 'check_existing_data', return_value=existing_check):
                
                result = await self.service.execute_preprocess_operation(config, confirm_overwrite=False)
                
                assert result['success'] is False
                assert result['requires_confirmation'] is True
                assert result['existing_data'] == existing_check
    
    @pytest.mark.asyncio
    async def test_execute_preprocess_operation_with_confirmation(self):
        """Test preprocessing operation with confirmation"""
        config = {'data': {}, 'preprocessing': {}}
        
        mock_preprocess_api = {
            'preprocess_dataset': Mock(return_value={'success': True, 'message': 'Done'})
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            
            result = await self.service.execute_preprocess_operation(config, confirm_overwrite=True)
            
            assert result['success'] is True
            mock_preprocess_api['preprocess_dataset'].assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_preprocess_operation_error(self):
        """Test preprocessing operation error"""
        config = {'data': {}, 'preprocessing': {}}
        
        with patch.object(self.service, '_load_backend_modules', side_effect=Exception("Backend error")):
            
            result = await self.service.execute_preprocess_operation(config)
            
            assert result['success'] is False
            assert 'error' in result
            assert 'Backend error' in result['message']
    
    @pytest.mark.asyncio
    async def test_execute_check_operation_success(self):
        """Test successful check operation"""
        config = {'data': {'dir': 'data'}, 'preprocessing': {'target_splits': ['train', 'valid']}}
        
        mock_preprocess_api = {
            'get_preprocessing_status': Mock(return_value={
                'success': True,
                'service_ready': True,
                'file_statistics': {'train': {'raw_images': 100, 'preprocessed_files': 50}}
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            
            result = await self.service.execute_check_operation(config)
            
            assert result['success'] is True
            assert result['service_ready'] is True
            assert 'file_statistics' in result
            assert self.service.current_operation == PreprocessingOperation.CHECK
    
    @pytest.mark.asyncio
    async def test_execute_check_operation_with_stats_enhancement(self):
        """Test check operation with statistics enhancement"""
        config = {'data': {'dir': 'data'}, 'preprocessing': {'target_splits': ['train', 'valid']}}
        
        mock_preprocess_api = {
            'get_preprocessing_status': Mock(return_value={
                'success': True,
                'service_ready': True
                # No file_statistics - should be enhanced
            })
        }
        
        mock_stats_api = {
            'get_dataset_stats': Mock(return_value={
                'success': True,
                'by_split': {
                    'train': {
                        'file_counts': {'raw': 100, 'preprocessed': 50},
                        'total_size_mb': 250.5
                    }
                }
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._preprocess_api = mock_preprocess_api
            self.service._stats_api = mock_stats_api
            
            result = await self.service.execute_check_operation(config)
            
            assert result['success'] is True
            assert 'file_statistics' in result
            assert 'train' in result['file_statistics']
            assert result['file_statistics']['train']['raw_images'] == 100
    
    @pytest.mark.asyncio
    async def test_execute_check_operation_error(self):
        """Test check operation error"""
        config = {'data': {}, 'preprocessing': {}}
        
        with patch.object(self.service, '_load_backend_modules', side_effect=Exception("Backend error")):
            
            result = await self.service.execute_check_operation(config)
            
            assert result['success'] is False
            assert 'error' in result
            assert result['service_ready'] is False
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_operation_needs_confirmation(self):
        """Test cleanup operation needing confirmation"""
        config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'cleanup_target': 'preprocessed',
                'target_splits': ['train', 'valid']
            }
        }
        
        mock_cleanup_api = {
            'get_cleanup_preview': Mock(return_value={
                'success': True,
                'total_files': 100,
                'total_size_mb': 500.0
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._cleanup_api = mock_cleanup_api
            
            result = await self.service.execute_cleanup_operation(config, confirm_cleanup=False)
            
            assert result['success'] is False
            assert result['requires_confirmation'] is True
            assert result['cleanup_preview']['total_files'] == 100
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_operation_no_files(self):
        """Test cleanup operation with no files to remove"""
        config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'cleanup_target': 'preprocessed',
                'target_splits': ['train', 'valid']
            }
        }
        
        mock_cleanup_api = {
            'get_cleanup_preview': Mock(return_value={
                'success': True,
                'total_files': 0,
                'total_size_mb': 0.0
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._cleanup_api = mock_cleanup_api
            
            result = await self.service.execute_cleanup_operation(config, confirm_cleanup=False)
            
            assert result['success'] is True
            assert result['files_removed'] == 0
            assert 'No files found' in result['message']
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_operation_with_confirmation(self):
        """Test cleanup operation with confirmation"""
        config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'cleanup_target': 'preprocessed',
                'target_splits': ['train', 'valid']
            }
        }
        
        mock_cleanup_api = {
            'cleanup_preprocessing_files': Mock(return_value={
                'success': True,
                'files_removed': 50,
                'message': 'Cleanup completed'
            })
        }
        
        with patch.object(self.service, '_load_backend_modules'):
            self.service._cleanup_api = mock_cleanup_api
            
            result = await self.service.execute_cleanup_operation(config, confirm_cleanup=True)
            
            assert result['success'] is True
            assert result['files_removed'] == 50
            assert self.service.current_operation == PreprocessingOperation.CLEANUP
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_operation_error(self):
        """Test cleanup operation error"""
        config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'cleanup_target': 'preprocessed',
                'target_splits': ['train', 'valid']
            }
        }
        
        with patch.object(self.service, '_load_backend_modules', side_effect=Exception("Backend error")):
            
            result = await self.service.execute_cleanup_operation(config)
            
            assert result['success'] is False
            assert 'error' in result
            assert result['files_removed'] == 0
    
    def test_get_last_operation_results(self):
        """Test getting last operation results"""
        # Set some test results
        self.service.operation_results = {
            'preprocess': {'success': True, 'message': 'Done'},
            'check': {'success': True, 'service_ready': True}
        }
        
        preprocess_result = self.service.get_last_operation_results('preprocess')
        check_result = self.service.get_last_operation_results('check')
        cleanup_result = self.service.get_last_operation_results('cleanup')
        
        assert preprocess_result['success'] is True
        assert check_result['service_ready'] is True
        assert cleanup_result is None
    
    def test_clear_operation_results(self):
        """Test clearing operation results"""
        self.service.operation_results = {'preprocess': {'success': True}}
        
        self.service.clear_operation_results()
        
        assert self.service.operation_results == {}
    
    def test_is_backend_available_true(self):
        """Test backend availability check - available"""
        with patch.object(self.service, '_load_backend_modules'):
            
            available = self.service.is_backend_available()
            
            assert available is True
    
    def test_is_backend_available_false(self):
        """Test backend availability check - not available"""
        with patch.object(self.service, '_load_backend_modules', side_effect=RuntimeError("Not available")):
            
            available = self.service.is_backend_available()
            
            assert available is False
    
    def test_get_service_status(self):
        """Test getting service status"""
        self.service._backend_loaded = True
        self.service.current_operation = PreprocessingOperation.PREPROCESS
        self.service.operation_results = {'preprocess': {'success': True}}
        
        with patch.object(self.service, 'is_backend_available', return_value=True):
            
            status = self.service.get_service_status()
            
            assert status['backend_available'] is True
            assert status['backend_loaded'] is True
            assert status['current_operation'] == 'preprocess'
            assert status['stored_results'] == ['preprocess']
            assert status['ui_components_available'] is True


class TestPreprocessUIServiceEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_service_creation_with_empty_components(self):
        """Test service creation with empty UI components"""
        service = PreprocessUIService({})
        
        assert service.ui_components == {}
        assert service.current_operation is None
    
    def test_service_creation_with_none_components(self):
        """Test service creation with None UI components"""
        service = PreprocessUIService(None)
        
        assert service.ui_components is None
        status = service.get_service_status()
        assert status['ui_components_available'] is False
    
    @pytest.mark.asyncio
    async def test_operations_with_malformed_config(self):
        """Test operations with malformed configuration"""
        service = PreprocessUIService({})
        
        # Test with None config
        result = await service.check_existing_data(None)
        assert result['has_existing'] is False
        assert 'error' in result
        
        # Test with empty config
        result = await service.execute_preprocess_operation({})
        assert result['success'] is False
        
        # Test with missing keys
        result = await service.execute_cleanup_operation({'data': {}})
        assert result['success'] is False
    
    def test_backend_loading_idempotent(self):
        """Test that backend loading is idempotent"""
        service = PreprocessUIService({})
        
        with patch('smartcash.dataset.preprocessor.preprocess_dataset') as mock_import:
            with patch('smartcash.dataset.preprocessor.get_preprocessing_status'):
                with patch('smartcash.dataset.preprocessor.api.cleanup_api.cleanup_preprocessing_files'):
                    
                    # Load multiple times
                    service._load_backend_modules()
                    service._load_backend_modules()
                    service._load_backend_modules()
                    
                    # Should only import once
                    mock_import  # Just verify it was patched
                    assert service._backend_loaded is True
    
    def test_concurrent_operations(self):
        """Test handling of concurrent operations"""
        service = PreprocessUIService({})
        
        # Set operation state
        service.current_operation = PreprocessingOperation.PREPROCESS
        
        # Should still allow new operations (service manages state)
        assert service.current_operation == PreprocessingOperation.PREPROCESS
        
        # Changing operation should update state
        service.current_operation = PreprocessingOperation.CHECK
        assert service.current_operation == PreprocessingOperation.CHECK


if __name__ == '__main__':
    pytest.main([__file__, '-v'])