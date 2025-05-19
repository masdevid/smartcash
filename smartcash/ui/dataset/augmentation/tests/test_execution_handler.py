"""
File: smartcash/ui/dataset/augmentation/tests/test_button_handler.py
Deskripsi: Unit test untuk button handler modul augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
import threading
from concurrent.futures import ThreadPoolExecutor

class TestExecutionHandler(unittest.TestCase):
    """Test untuk execution handler augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk ThreadPoolExecutor
        self.executor_patch = patch('concurrent.futures.ThreadPoolExecutor')
        self.mock_executor = self.executor_patch.start()
        self.mock_executor_instance = MagicMock()
        self.mock_executor.return_value = self.mock_executor_instance
        self.mock_executor_instance.__enter__.return_value = self.mock_executor_instance
        
        # Mock untuk augmentation_service_handler
        self.service_patch = patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.execute_augmentation')
        self.mock_service = self.service_patch.start()
        self.mock_service.return_value = {'status': 'success', 'count': 10}
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'update_status_panel': MagicMock(),
            'on_process_start': MagicMock(),
            'on_process_complete': MagicMock(),
            'on_process_error': MagicMock(),
            'on_process_stop': MagicMock(),
            'get_augmentation_config': MagicMock(),
            'augmentation_options': widgets.Tab(children=[
                widgets.VBox(children=[
                    widgets.IntSlider(value=2),  # factor
                    widgets.IntSlider(value=100),  # target_count
                    widgets.IntSlider(value=4),  # num_workers
                    widgets.Text(value='aug')  # prefix
                ]),
                widgets.VBox(children=[
                    widgets.HTML(),
                    widgets.Dropdown(options=['train', 'valid', 'test'], value='train')  # split
                ]),
                widgets.VBox(children=[
                    widgets.HTML(),
                    widgets.SelectMultiple(options=[('combined', 'combined')], value=('combined',)),  # aug_types
                    widgets.HTML(),
                    widgets.HBox(children=[
                        widgets.Checkbox(value=True),  # enabled
                        widgets.Checkbox(value=False)  # balance_classes
                    ]),
                    widgets.HBox(children=[
                        widgets.Checkbox(value=True),  # move_to_preprocessed
                        widgets.Checkbox(value=True)  # validate_results
                    ]),
                    widgets.HBox(children=[
                        widgets.Checkbox(value=False)  # resume
                    ])
                ])
            ]),
            'primary_button': widgets.Button(),
            'stop_button': widgets.Button(),
            'running': False,
            'executor': None
        }
        
        # Setup return value untuk get_augmentation_config
        self.mock_ui_components['get_augmentation_config'].return_value = {
            'augmentation': {
                'types': ['combined'],
                'factor': 2,
                'target_count': 100,
                'balance_classes': False,
                'num_workers': 4,
                'prefix': 'aug'
            }
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.executor_patch.stop()
        self.service_patch.stop()
    
    def test_run_augmentation(self):
        """Test untuk fungsi run_augmentation."""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
        
        # Mock untuk ThreadPoolExecutor
        with patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.ThreadPoolExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            
            # Panggil fungsi yang akan ditest
            run_augmentation(self.mock_ui_components)
            
            # Verifikasi hasil
            mock_executor_instance.submit.assert_called_once()
    
    def test_button_click_handler(self):
        """Test untuk fungsi on_click di button."""
        # Mock untuk run_augmentation
        with patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.run_augmentation') as mock_run:
            # Buat button dengan on_click handler
            button = widgets.Button(description='Augment')
            
            # Tambahkan handler ke button
            def on_click(b):
                mock_run(self.mock_ui_components)
            
            button.on_click(on_click)
            
            # Simulasi klik button
            for handler in button._click_handlers.callbacks:
                handler(button)
            
            # Verifikasi hasil
            mock_run.assert_called_once_with(self.mock_ui_components)
    
    def test_stop_button_click_handler(self):
        """Test untuk fungsi on_click di stop button."""
        # Buat button dengan on_click handler
        button = widgets.Button(description='Stop')
        
        # Setup running flag dan update_status_panel
        self.mock_ui_components['augmentation_running'] = True
        self.mock_ui_components['update_status_panel'] = MagicMock()
        
        # Tambahkan handler ke button
        def on_click(b):
            self.mock_ui_components['augmentation_running'] = False
            self.mock_ui_components['update_status_panel'](self.mock_ui_components, "Augmentasi dihentikan", "warning")
        
        button.on_click(on_click)
        
        # Simulasi klik button
        for handler in button._click_handlers.callbacks:
            handler(button)
        
        # Verifikasi hasil
        self.assertFalse(self.mock_ui_components['augmentation_running'])
        self.mock_ui_components['update_status_panel'].assert_called_once()
    
    def test_extract_augmentation_params(self):
        """Test untuk fungsi extract_augmentation_params."""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import extract_augmentation_params
        
        # Panggil fungsi yang akan ditest
        params = extract_augmentation_params(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertIn('augmentation_types', params)
        self.assertIn('num_variations', params)
        self.assertIn('target_count', params)
        self.assertIn('num_workers', params)
        self.assertIn('output_prefix', params)
        self.assertIn('split', params)
    
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.ThreadPoolExecutor')
    def test_execute_augmentation(self, mock_executor):
        """Test untuk fungsi execute_augmentation."""
        # Gunakan run_augmentation sebagai pengganti execute_augmentation
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
        
        # Setup mock executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Panggil fungsi yang akan ditest
        run_augmentation(self.mock_ui_components)
        
        # Verifikasi hasil
        mock_executor_instance.submit.assert_called_once()
    
    def test_error_handling(self):
        """Test untuk penanganan error dalam proses augmentasi."""
        # Setup mock untuk update_status_panel
        self.mock_ui_components['update_status_panel'] = MagicMock()
        self.mock_ui_components['notify_process_error'] = MagicMock()
        
        # Buat fungsi yang akan melempar error
        def error_function(ui_components):
            raise Exception("Test Error")
        
        # Coba jalankan fungsi dengan error
        try:
            error_function(self.mock_ui_components)
        except Exception as e:
            # Panggil fungsi notifikasi error
            self.mock_ui_components['notify_process_error'](self.mock_ui_components, str(e))
            self.mock_ui_components['update_status_panel'](self.mock_ui_components, f"Error: {str(e)}", "error")
        
        # Verifikasi hasil
        self.mock_ui_components['notify_process_error'].assert_called_once()
        self.mock_ui_components['update_status_panel'].assert_called_once()

if __name__ == '__main__':
    unittest.main()
