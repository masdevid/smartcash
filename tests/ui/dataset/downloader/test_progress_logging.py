"""
File: tests/ui/dataset/downloader/test_progress_logging.py
Description: Tests for progress tracking and logging in the dataset downloader.
"""

import pytest
from unittest.mock import MagicMock, call, ANY
from smartcash.dataset.downloader.progress_tracker import DownloadStage

class TestProgressLogging:
    """Tests for progress tracking and logging functionality."""
    
    def test_progress_tracking_updates(self, downloader_module, mock_progress_tracker):
        """Test that progress tracking is properly updated during operations."""
        # Setup test data
        test_stage = DownloadStage.DOWNLOAD
        test_message = "Downloading dataset files..."
        
        # Execute operation that triggers progress updates
        downloader_module.execute_download()
        
        # Verify progress tracking methods were called
        mock_progress_tracker.start_stage.assert_called_once()
        mock_progress_tracker.update_stage.assert_called()
        mock_progress_tracker.complete_stage.assert_called_once()
    
    def test_log_output_during_download(self, downloader_module, mock_ui_components):
        """Test that log messages are properly displayed during download."""
        # Get the log output widget
        log_output = mock_ui_components['log_output']
        
        # Execute download operation
        downloader_module.execute_download()
        
        # Verify log messages were written
        assert log_output.append_stdout.call_count > 0
        
        # Check for specific log messages
        log_calls = [str(call) for call in log_output.append_stdout.call_args_list]
        assert any("Starting download" in str(call) for call in log_calls)
        assert any("Download completed" in str(call) for call in log_calls)
    
    def test_error_logging(self, downloader_module, mock_download_service, mock_ui_components):
        """Test that errors are properly logged and displayed."""
        # Setup error condition
        error_message = "Test error: Connection failed"
        mock_download_service.download_dataset.side_effect = Exception(error_message)
        
        # Get the log output widget
        log_output = mock_ui_components['log_output']
        
        # Execute operation that will fail
        result = downloader_module.execute_download()
        
        # Verify error was logged
        assert result["success"] is False
        assert error_message in result.get("error", "")
        
        # Check that error was written to log
        log_calls = [str(call) for call in log_output.append_stderr.call_args_list]
        assert any(error_message in str(call) for call in log_calls)
    
    def test_operation_container_updates(self, downloader_module, mock_ui_components):
        """Test that the operation container is properly updated during operations."""
        operation_container = mock_ui_components['operation_container']
        
        # Clear any existing calls from setup
        operation_container.clear_output.reset_mock()
        
        # Execute operation
        downloader_module.execute_download()
        
        # Verify operation container was updated
        operation_container.clear_output.assert_called_once()
        assert operation_container.append_display_data.call_count > 0
    
    def test_progress_bar_updates(self, downloader_module, mock_ui_components):
        """Test that the progress bar is properly updated during operations."""
        progress_bar = mock_ui_components['progress_bar']
        
        # Clear any existing calls from setup
        progress_bar.value = 0
        progress_bar.max = 100
        progress_bar.bar_style = ''
        
        # Execute operation
        downloader_module.execute_download()
        
        # Verify progress bar was updated
        assert progress_bar.value > 0
        assert progress_bar.bar_style == 'success'
        
        # Check for progress updates
        assert any(call[0][0] > 0 for call in progress_bar.__setattr__.call_args_list 
                  if call[0][0] == 'value')
    
    def test_status_updates(self, downloader_module, mock_ui_components):
        """Test that status updates are properly displayed."""
        status_label = mock_ui_components['status_label']
        
        # Clear any existing calls from setup
        status_label.value = ''
        
        # Execute operation
        downloader_module.execute_download()
        
        # Verify status was updated
        assert status_label.value != ''
        assert 'complete' in status_label.value.lower() or 'success' in status_label.value.lower()
        
        # Check for status updates during operation
        status_updates = [call[0][0] for call in status_label.__setattr__.call_args_list 
                         if call[0][0] == 'value']
        assert any('downloading' in str(update).lower() for update in status_updates)
    
    def test_concurrent_operations(self, downloader_module, mock_download_service):
        """Test that concurrent operations are handled correctly."""
        import asyncio
        
        # Setup mock to simulate a long-running download
        async def mock_download():
            await asyncio.sleep(0.1)
            return {"success": True, "message": "Download completed"}
            
        mock_download_service.download_dataset.side_effect = mock_download
        
        # Start multiple operations
        results = []
        
        async def run_operations():
            tasks = [
                downloader_module.execute_download(),
                downloader_module.execute_check(),
                downloader_module.execute_cleanup()
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
            
        # Run the operations
        results = asyncio.run(run_operations())
        
        # Verify all operations completed successfully
        assert len(results) == 3
        assert all(result["success"] for result in results if not isinstance(result, Exception))
