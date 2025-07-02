"""Unit tests for the SetupHandler class.

This module contains tests that verify the functionality of the SetupHandler class,
including the end-to-end setup workflow and progress tracking.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler, SetupPhase, SetupSummary
from smartcash.ui.setup.env_config.handlers.base_env_handler import SetupStage


class TestSetupHandler(unittest.IsolatedAsyncioTestCase):
    """Test cases for the SetupHandler class."""

    def _create_mock_handler(self, name):
        """Create a mock handler with the given name."""
        handler = MagicMock()
        handler.name = name
        handler.set_progress_callback = MagicMock()
        handler.setup = AsyncMock(return_value=True)
        return handler
    
    def _mock_update_progress(self, progress: float, message: str = None):
        """Mock progress update callback."""
        self.progress_updates.append((progress, message))
    
    def _on_setup_completed(self, success: bool, error: str = None):
        """Mock setup completion callback."""
        self.setup_completed = True
        self.setup_success = success
        self.setup_error = error
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock config handler
        self.mock_config_handler = MagicMock()
        self.mock_config_handler.get_config_value = MagicMock(return_value={
            'stages': ['drive', 'folder', 'config', 'verify'],
            'max_retries': 2,
            'retry_delay': 0.1,
            'stop_on_error': True,
            'verify_setup': True,
            'auto_start': False
        })
        
        # Create a mock for _create_initial_summary
        self.mock_create_initial_summary = MagicMock(return_value={
            'status': 'pending',
            'phase': 'initializing',
            'progress': 0.0,
            'current_stage': 'INIT',
            'drive_mounted': False,
            'mount_path': '',
            'folders_created': 0,
            'symlinks_created': 0,
            'configs_synced': 0,
            'verified_folders': [],
            'missing_folders': [],
            'verified_symlinks': [],
            'missing_symlinks': []
        })
        
        # Patch the _create_initial_summary method in the SetupHandler class
        self.setup_handler_patcher = patch(
            'smartcash.ui.setup.env_config.handlers.setup_handler.SetupHandler._create_initial_summary',
            self.mock_create_initial_summary
        )
        self.setup_handler_patcher.start()
        
        # Initialize the handler with mock config
        self.handler = SetupHandler(config_handler=self.mock_config_handler)
        
        # Mock the logger
        self.handler.logger = MagicMock()
        
        # Mock the progress callback
        self.progress_updates = []
        self.handler._update_progress = self._mock_update_progress
        
        # Mock the status panel update method
        self.handler._update_status_panel = MagicMock()
        
        # Mock the _update_status method to use _update_status_panel
        self.handler._update_status = MagicMock()
        self.handler._update_status.side_effect = lambda msg, is_error=False: \
            self.handler._update_status_panel({}, msg, 'error' if is_error else 'info')
        
        # Mock the stage handlers
        self.handler._handlers = {
            'drive': self._create_mock_handler('drive'),
            'folder': self._create_mock_handler('folder'),
            'config': self._create_mock_handler('config'),
            'verify': self._create_mock_handler('verify')
        }
        
        # Track setup state
        self.setup_completed = False
        self.setup_error = False
        
        # Patch the run_setup method to track completion
        self.original_run_setup = self.handler.run_setup
        self.handler.run_setup = AsyncMock(side_effect=self._mock_run_setup)
    
    async def _mock_run_setup(self, *args, **kwargs):
        """Mock the run_setup method to track completion."""
        try:
            result = await self.original_run_setup(*args, **kwargs)
            self._on_setup_completed(True)
            return result
        except Exception as e:
            self._on_setup_completed(False, str(e))
            raise
    
    async def _mock_run_setup_workflow(self):
        """Mock the _run_setup_workflow method to execute handlers in order."""
        try:
            # Call each handler's setup method in order
            for stage in self.handler.stages:
                if stage in self.handler._handlers:
                    handler = self.handler._handlers[stage]
                    success = await handler.setup()
                    if not success and self.handler.get_config_value('stop_on_error', True):
                        raise RuntimeError(f"Stage {stage} failed")
            
            # If we get here, all handlers succeeded
            self._on_setup_completed(True)
        except Exception as e:
            self._on_setup_completed(False, str(e))
            raise
    
    def tearDown(self):
        """Clean up after each test method."""
        self.setup_handler_patcher.stop()
    
    async def test_complete_workflow_success(self):
        """Test a successful end-to-end setup workflow."""
        # Set up the mock handlers to succeed
        for handler in self.handler._handlers.values():
            handler.setup.return_value = True
        
        # Replace the completion callback
        self.handler._on_setup_completed = self._on_setup_completed
        
        # Run the setup workflow using the public API
        await self.handler.start_setup(auto_start=True)
        
        # Verify all handlers were called
        for handler in self.handler._handlers.values():
            handler.setup.assert_called_once()
            handler.set_progress_callback.assert_called_once()
        
        # Verify completion
        self.assertTrue(self.setup_completed)
        self.assertTrue(self.setup_success)
        self.assertIsNone(self.setup_error)
        
        # Verify progress updates
        self.assertGreater(len(self.progress_updates), 0)
        
        # Check that progress started at 0 and ended at 1.0 (100%)
        self.assertGreaterEqual(self.progress_updates[0][0], 0.0)
        self.assertLessEqual(self.progress_updates[-1][0], 1.0)
        
        # Check that progress was monotonic
        prev_progress = -0.1
        for progress, _ in self.progress_updates:
            self.assertGreaterEqual(progress, prev_progress)
            prev_progress = progress
    
    async def test_workflow_with_handler_failure(self):
        """Test setup workflow when a handler fails."""
        # Make the folder handler fail
        self.handler._handlers['folder'].setup.return_value = False
        self.handler._handlers['folder'].setup.side_effect = None
        
        # Replace the completion callback
        self.handler._on_setup_completed = self._on_setup_completed
        
        # Run the setup workflow using the public API
        with self.assertRaises(RuntimeError):
            await self.handler.start_setup(auto_start=True)
        
        # Verify the folder handler was called
        self.handler._handlers['folder'].setup.assert_called_once()
        
        # The config and verify handlers should not be called after folder fails
        self.handler._handlers['config'].setup.assert_not_called()
        self.handler._handlers['verify'].setup.assert_not_called()
        
        # Verify failure was reported
        self.assertTrue(self.setup_completed)
        self.assertFalse(self.setup_success)
        self.assertIsNotNone(self.setup_error)
    
    async def test_workflow_with_retry(self):
        """Test setup workflow with retry logic."""
        # Make the drive handler fail once, then succeed
        drive_handler = self.handler._handlers['drive']
        drive_handler.setup.side_effect = [RuntimeError("Temporary error"), True]
        
        # Set up remaining handlers to succeed
        for name, handler in self.handler._handlers.items():
            if name != 'drive':
                handler.setup.return_value = True
        
        # Replace the completion callback
        self.handler._on_setup_completed = self._on_setup_completed
        
        # Patch the asyncio.sleep to avoid waiting during tests
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Run the setup workflow using the public API
            await self.handler.start_setup(auto_start=True)
            
            # Verify the drive handler was called twice (retry)
            self.assertEqual(drive_handler.setup.call_count, 2)
            
            # Verify other handlers were called
            for name, handler in self.handler._handlers.items():
                if name != 'drive':
                    handler.setup.assert_called_once()
            
            # Verify completion
            self.assertTrue(self.setup_completed)
            self.assertTrue(self.setup_success)
            self.assertIsNone(self.setup_error)
        
        # Run the setup workflow
        await self.handler._run_setup_workflow()
        
        # Verify the drive handler was called twice (retry once)
        self.assertEqual(drive_handler.setup.call_count, 2)
        
        # Verify all other handlers were called once
        for name, handler in self.handler._handlers.items():
            if name != 'drive':
                handler.setup.assert_called_once()
        
        # Verify completion was successful
        self.assertTrue(self.setup_completed)
        self.assertTrue(self.setup_success)
        self.assertIsNone(self.setup_error)
    
    async def test_progress_callback_integration(self):
        """Test that progress callbacks are properly integrated with handlers."""
        # Set up a mock progress callback for the folder handler
        folder_handler = self.handler._handlers['folder']
        progress_callback = None
        
        def capture_callback(callback):
            nonlocal progress_callback
            progress_callback = callback
            return None
        
        folder_handler.set_progress_callback.side_effect = capture_callback
        
        # Replace the completion callback
        self.handler._on_setup_completed = self._on_setup_completed
        
        # Run the setup workflow in the background
        task = asyncio.create_task(self.handler._run_setup_workflow())
        
        # Wait for the folder handler to be called
        await asyncio.sleep(0.1)
        
        # Verify the progress callback was set
        self.assertIsNotNone(progress_callback)
        
        # Simulate progress updates from the folder handler
        progress_callback(0.5, "Creating folders...")
        progress_callback(1.0, "Folders created")
        
        # Wait for the workflow to complete
        await task
        
        # Verify progress updates were received
        progress_messages = [msg for _, msg in self.progress_updates if msg is not None]
        self.assertIn("Creating folders...", "\n".join(progress_messages))
        self.assertIn("Folders created", "\n".join(progress_messages))


if __name__ == '__main__':
    unittest.main()
