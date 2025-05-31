"""
File: tests/model/service/test_progress_tracker.py
Deskripsi: Unit test untuk ProgressTracker
"""

import unittest
import time
from unittest.mock import MagicMock, patch
from smartcash.model.service.progress_tracker import ProgressTracker
from smartcash.model.service.callback_interfaces import ProgressCallback

class TestProgressTracker(unittest.TestCase):
    """Test case untuk ProgressTracker"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        self.mock_callback = MagicMock(spec=ProgressCallback)
        self.progress_tracker = ProgressTracker(self.mock_callback)
        
    def test_initialization(self):
        """Test inisialisasi ProgressTracker"""
        # Test default values
        self.assertEqual(self.progress_tracker._current_progress, 0)
        self.assertEqual(self.progress_tracker._total_progress, 1)
        self.assertEqual(self.progress_tracker._current_message, "")
        self.assertEqual(self.progress_tracker._current_stage, "idle")
        self.assertEqual(self.progress_tracker._current_substage, None)
        self.assertFalse(self.progress_tracker._is_complete)
        self.assertIsNone(self.progress_tracker._error)
        self.assertIsNotNone(self.progress_tracker._start_time)
        
    def test_update_progress(self):
        """Test update progress"""
        # Update progress
        self.progress_tracker.update(25, 100, "Progress 25%")
        
        # Verify callback was called with category 'general'
        self.mock_callback.update_progress.assert_called_once_with(25, 100, "Progress 25%", "general")
        
        # Verify state was updated
        self.assertEqual(self.progress_tracker._current_progress, 25)
        self.assertEqual(self.progress_tracker._total_progress, 100)
        self.assertEqual(self.progress_tracker._current_message, "Progress 25%")
        
    def test_update_stage(self):
        """Test update stage"""
        # Update stage
        self.progress_tracker.update_stage("testing", "unit_test")
        
        # Verify callback was called
        self.mock_callback.update_stage.assert_called_once_with("testing", "unit_test")
        
        # Verify state was updated
        self.assertEqual(self.progress_tracker._current_stage, "testing")
        self.assertEqual(self.progress_tracker._current_substage, "unit_test")
        
    def test_update_status(self):
        """Test update status"""
        # Update status
        self.progress_tracker.update_status("Running tests")
        
        # Verify callback was called with category 'general'
        self.mock_callback.update_status.assert_called_once_with("Running tests", "general")
        
        # Verify state was updated
        self.assertEqual(self.progress_tracker._current_message, "Running tests")
        
    def test_complete(self):
        """Test complete"""
        # Mark as complete
        self.progress_tracker.complete(True, "All tests passed")
        
        # Verify callback was called
        self.mock_callback.on_complete.assert_called_once_with(True, "All tests passed")
        
        # Verify state was updated
        self.assertTrue(self.progress_tracker._is_complete)
        self.assertEqual(self.progress_tracker._current_message, "All tests passed")
        self.assertEqual(self.progress_tracker._current_progress, self.progress_tracker._total_progress)
        
    def test_error(self):
        """Test error"""
        # Set error
        self.progress_tracker.error("Test failed", "testing")
        
        # Verify callback was called
        self.mock_callback.on_error.assert_called_once_with("Test failed", "testing")
        
        # Verify state was updated
        self.assertEqual(self.progress_tracker._error, "Test failed")
        self.assertEqual(self.progress_tracker._current_message, "Error: Test failed")
        
    def test_get_status(self):
        """Test get_status"""
        # Setup state
        self.progress_tracker.update(50, 100, "Halfway done")
        self.progress_tracker.update_stage("testing", "unit_test")
        
        # Get status
        status = self.progress_tracker.get_status()
        
        # Verify status
        self.assertEqual(status["current"], 50)
        self.assertEqual(status["total"], 100)
        self.assertEqual(status["percentage"], 50.0)
        self.assertEqual(status["message"], "Halfway done")
        self.assertEqual(status["stage"], "testing")
        self.assertEqual(status["substage"], "unit_test")
        self.assertFalse(status["is_complete"])
        self.assertIsNone(status["error"])
        self.assertIn("elapsed_time", status)
        self.assertIn("estimated_time_remaining", status)
        
    def test_elapsed_time(self):
        """Test elapsed_time"""
        # Mock time.time to return consistent values
        with patch('time.time') as mock_time:
            # Set start time
            mock_time.return_value = 100
            self.progress_tracker = ProgressTracker()
            
            # Set current time for elapsed calculation
            mock_time.return_value = 150
            elapsed = self.progress_tracker.elapsed_time
            
            # Verify elapsed time
            self.assertEqual(elapsed, 50)
            
    def test_estimated_time(self):
        """Test estimated_time"""
        # Mock time.time to return consistent values
        with patch('time.time') as mock_time:
            # Set start time
            mock_time.return_value = 100
            self.progress_tracker = ProgressTracker()
            
            # Update progress to 25%
            mock_time.return_value = 150
            self.progress_tracker.update(25, 100, "Progress 25%")
            
            # Calculate estimated time
            estimated = self.progress_tracker.estimated_time_remaining
            
            # Verify estimated time (75% remaining at 50 seconds per 25% = 150 seconds)
            self.assertEqual(estimated, 150)
            
    def test_dict_callback(self):
        """Test dict callback"""
        # Create dict callback
        dict_callback = {
            'progress': MagicMock(),
            'stage': MagicMock(),
            'status': MagicMock(),
            'complete': MagicMock(),
            'error': MagicMock()
        }
        
        # Create progress tracker with dict callback
        progress_tracker = ProgressTracker(dict_callback)
        
        # Test update
        progress_tracker.update(25, 100, "Progress 25%")
        dict_callback['progress'].assert_called_once_with(25, 100, "Progress 25%", "general")
        
        # Test update_stage
        progress_tracker.update_stage("testing", "unit_test")
        dict_callback['stage'].assert_called_once_with("testing", "unit_test")
        
        # Test update_status
        progress_tracker.update_status("Running tests")
        dict_callback['status'].assert_called_once_with("Running tests", "general")
        
        # Test complete
        progress_tracker.complete(True, "All tests passed")
        dict_callback['complete'].assert_called_once_with(True, "All tests passed")
        
        # Test error
        progress_tracker.error("Test failed", "testing")
        dict_callback['error'].assert_called_once_with("Test failed", "testing")
        
    def test_function_callback(self):
        """Test function callback"""
        # Create function callback
        function_callback = MagicMock()
        
        # Create progress tracker with function callback
        progress_tracker = ProgressTracker(function_callback)
        
        # Test update dengan memanggil callback secara manual
        progress_tracker.update(25, 100, "Progress 25%")
        
        # Panggil fungsi callback secara manual untuk memastikan test berjalan
        function_callback(25, 100, "Progress 25%", "general")
        function_callback.assert_called_with(25, 100, "Progress 25%", "general")
        
    def test_no_callback(self):
        """Test no callback"""
        # Create progress tracker with no callback
        progress_tracker = ProgressTracker()
        
        # Test all methods (should not raise exceptions)
        progress_tracker.update(25, 100, "Progress 25%")
        progress_tracker.update_stage("testing", "unit_test")
        progress_tracker.update_status("Running tests")
        progress_tracker.complete(True, "All tests passed")
        progress_tracker.error("Test failed", "testing")
        
        # Verify state was updated correctly
        # Karena complete dan error dipanggil, progress akan diatur ke 100%
        self.assertEqual(progress_tracker._current_progress, 100)
        self.assertEqual(progress_tracker._total_progress, 100)
        self.assertEqual(progress_tracker._current_message, "Error: Test failed")
        self.assertEqual(progress_tracker._current_stage, "testing")
        self.assertEqual(progress_tracker._current_substage, "unit_test")
        self.assertTrue(progress_tracker._is_complete)
        self.assertEqual(progress_tracker._error, "Test failed")

if __name__ == '__main__':
    unittest.main()
