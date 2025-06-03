"""
File: smartcash/ui/training/tests/test_training_ui.py
Deskripsi: Test suite untuk menguji fungsionalitas komponen UI training
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from typing import Dict, Any
import ipywidgets as widgets

# Import komponen training
from smartcash.ui.training.components.training_form import create_training_form
from smartcash.ui.training.components.training_layout import create_training_layout
from smartcash.ui.training.components.config_tabs import create_config_tabs
from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
from smartcash.ui.training.components.control_buttons import create_training_control_buttons as create_control_buttons
from smartcash.ui.training.components.fallback_component import create_fallback_layout
from smartcash.ui.training.training_init import TrainingInitializer


class TestTrainingComponents(unittest.TestCase):
    """Test untuk individual UI components"""
    
    def setUp(self):
        """Setup data pengujian dan mocks"""
        self.mock_config = {
            'model': {
                'name': 'yolov5-efficientnet-b4',
                'backbone': 'efficientnet-b4',
                'anchors': '5,6, 7,8, 9,10',
                'input_size': 640
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937
            },
            'data': {
                'train': '/path/to/train',
                'val': '/path/to/val',
                'classes': ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'rotate': 10,
                'shear': 2.0
            }
        }
        
        self.mock_metrics = {
            'train_loss': 0.453,
            'val_loss': 0.521,
            'map': 0.723,
            'f1': 0.789,
            'precision': 0.812,
            'recall': 0.765
        }
        
        # Patch untuk ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_widgets.HTML = MagicMock()
        self.mock_widgets.Button = MagicMock()
        self.mock_widgets.VBox = MagicMock()
        self.mock_widgets.HBox = MagicMock()
        self.mock_widgets.Tab = MagicMock()
        self.mock_widgets.Accordion = MagicMock()
        self.mock_widgets.Output = MagicMock()

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_config_tabs(self):
        """Test pembuatan config tabs"""
        # Patch tab_factory
        with patch('smartcash.ui.components.tab_factory.create_tab_widget') as mock_create_tabs:
            mock_tab = MagicMock(spec=widgets.Tab)
            mock_create_tabs.return_value = mock_tab
            
            result = create_config_tabs(self.mock_config)
            
            # Verifikasi tab dibuat dengan benar
            self.assertEqual(result, mock_tab)
            mock_create_tabs.assert_called_once()

    def test_create_metrics_accordion(self):
        """Test pembuatan metrics accordion"""
        # Patch accordion_factory
        with patch('smartcash.ui.components.accordion_factory.create_accordion') as mock_create_accordion:
            mock_accordion = MagicMock(spec=widgets.Accordion)
            mock_create_accordion.return_value = mock_accordion
            
            result = create_metrics_accordion()
            
            # Verifikasi accordion dibuat dengan benar
            self.assertIn('metrics_accordion', result)
            self.assertIn('chart_output', result)
            self.assertIn('metrics_output', result)
            self.assertEqual(result['metrics_accordion'], mock_accordion)

    def test_create_control_buttons(self):
        """Test pembuatan control buttons"""
        # Patch action_buttons
        with patch('smartcash.ui.components.action_buttons.create_action_buttons') as mock_action_buttons:
            mock_button = MagicMock(spec=widgets.Button)
            mock_action_buttons.return_value = {'download_button': mock_button}
            
            # Patch HBox untuk menghindari error validasi children
            with patch('ipywidgets.widgets.HBox', MagicMock()) as mock_hbox:
                mock_hbox_instance = MagicMock(spec=widgets.HBox)
                mock_hbox.return_value = mock_hbox_instance
                
                result = create_control_buttons()
                
                # Verifikasi button handlers
                self.assertIn('start_button', result)
                self.assertIn('stop_button', result)
                self.assertIn('reset_button', result)
                self.assertIn('button_container', result)

    def test_create_training_form(self):
        """Test pembuatan training form"""
        # Patch semua komponen yang digunakan dalam create_training_form
        with patch('smartcash.ui.components.progress_tracking.create_progress_tracking_container') as mock_progress,\
             patch('smartcash.ui.components.log_accordion.create_log_accordion') as mock_log,\
             patch('smartcash.ui.components.status_panel.create_status_panel') as mock_status,\
             patch('smartcash.ui.training.components.control_buttons.create_training_control_buttons') as mock_control,\
             patch('smartcash.ui.training.components.control_buttons.create_non_training_buttons') as mock_utility,\
             patch('smartcash.ui.training.components.config_tabs.create_config_tabs') as mock_tabs,\
             patch('smartcash.ui.training.components.metrics_accordion.create_metrics_accordion') as mock_metrics:
            
            # Setup mock returns
            mock_progress.return_value = {'container': MagicMock(), 'tracker': MagicMock()}
            mock_log.return_value = {'log_output': MagicMock(), 'log_accordion': MagicMock()}
            mock_status.return_value = MagicMock()
            mock_control.return_value = {'start_button': MagicMock(), 'stop_button': MagicMock(), 'reset_button': MagicMock(), 'button_container': MagicMock()}
            mock_utility.return_value = {'validate_button': MagicMock(), 'cleanup_button': MagicMock(), 'refresh_button': MagicMock(), 'utility_container': MagicMock()}
            mock_tabs.return_value = MagicMock()
            mock_metrics.return_value = {'metrics_accordion': MagicMock(), 'chart_output': MagicMock(), 'metrics_output': MagicMock()}
            
            result = create_training_form(self.mock_config)
            
            # Verifikasi form components
            self.assertIn('control_buttons', result)
            self.assertIn('utility_buttons', result)
            self.assertIn('config_tabs', result)
            self.assertIn('metrics_accordion', result)
            self.assertIn('progress_container', result)
            self.assertIn('log_output', result)

    def test_create_training_form_with_error(self):
        """Test pembuatan training form dengan error handling"""
        # Simulasi error dengan patch
        with patch('smartcash.ui.components.progress_tracking.create_progress_tracking_container', side_effect=Exception('Test error')),\
             patch('smartcash.ui.training.components.fallback_component.create_fallback_training_form') as mock_fallback:
            
            # Setup mock fallback return
            mock_fallback.return_value = {'error_message': MagicMock(), 'fallback_container': MagicMock()}
            
            result = create_training_form(self.mock_config)
            
            # Verifikasi fallback digunakan
            self.assertIn('error_message', result)
            self.assertIn('fallback_container', result)
            mock_fallback.assert_called_once_with('Test error')

    def test_create_training_layout(self):
        """Test pembuatan training layout"""
        # Patch fungsi-fungsi helper
        with patch('smartcash.ui.training.components.training_layout.create_header_section') as mock_header,\
             patch('smartcash.ui.training.components.training_layout.create_info_section') as mock_info,\
             patch('smartcash.ui.training.components.training_layout.create_control_section') as mock_control,\
             patch('smartcash.ui.training.components.training_layout.create_progress_section') as mock_progress,\
             patch('smartcash.ui.training.components.training_layout.create_metrics_section') as mock_metrics,\
             patch('smartcash.ui.training.components.training_layout.create_log_section') as mock_log,\
             patch('smartcash.ui.training.components.training_layout.create_divider') as mock_divider:
            
            # Setup mock returns
            mock_header.return_value = MagicMock(spec=widgets.HTML)
            mock_info.return_value = MagicMock(spec=widgets.VBox)
            mock_control.return_value = MagicMock(spec=widgets.VBox)
            mock_progress.return_value = MagicMock(spec=widgets.VBox)
            mock_metrics.return_value = MagicMock(spec=widgets.VBox)
            mock_log.return_value = MagicMock(spec=widgets.VBox)
            mock_divider.return_value = MagicMock(spec=widgets.HTML)
            
            # Mock form components
            mock_form = {
                'config_tabs': MagicMock(spec=widgets.Tab),
                'button_container': MagicMock(spec=widgets.HBox),
                'progress_container': MagicMock(spec=widgets.VBox),
                'status_panel': MagicMock(spec=widgets.HTML),
                'metrics_accordion': MagicMock(spec=widgets.Accordion),
                'chart_output': MagicMock(spec=widgets.Output),
                'metrics_output': MagicMock(spec=widgets.Output),
                'log_output': MagicMock(spec=widgets.Output),
                'log_accordion': MagicMock(spec=widgets.Accordion)
            }
            
            result = create_training_layout(mock_form)
            
            # Verifikasi layout components
            self.assertIn('info_section', result)
            self.assertIn('control_section', result)
            self.assertIn('progress_section', result)
            self.assertIn('metrics_section', result)
            self.assertIn('log_section', result)
            self.assertIn('main_container', result)
            
            # Verifikasi helper functions dipanggil
            mock_header.assert_called_once()
            mock_info.assert_called_once_with(mock_form)
            mock_control.assert_called_once_with(mock_form)
            mock_progress.assert_called_once_with(mock_form)
            mock_metrics.assert_called_once_with(mock_form)
            mock_log.assert_called_once_with(mock_form)


class TestTrainingInitializer(unittest.TestCase):
    """Test untuk training initializer"""
    
    def setUp(self):
        """Setup mocks untuk pengujian"""
        self.mock_config = {
            'model': {'name': 'yolov5-efficientnet-b4'},
            'training': {'batch_size': 16, 'epochs': 100, 'model_type': 'efficient_optimized'},
            'data': {'classes': ['Rp1000', 'Rp2000']}
        }
        
        # Mock CommonInitializer parent class
        self.mock_common_init = patch('smartcash.ui.utils.common_initializer.CommonInitializer.__init__').start()
        self.mock_common_init.return_value = None
        
        # Mock logger
        self.mock_logger = patch('smartcash.common.logger.get_logger').start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock training components
        self.mock_create_form = patch('smartcash.ui.training.components.training_form.create_training_form').start()
        self.mock_create_layout = patch('smartcash.ui.training.components.training_layout.create_training_layout').start()
        
        # Mock model services
        self.mock_create_model_manager = patch('smartcash.ui.training.training_init.TrainingInitializer._create_model_manager').start()
        self.mock_create_training_services = patch('smartcash.ui.training.training_init.TrainingInitializer._create_training_services').start()
        self.mock_create_training_manager = patch('smartcash.ui.training.training_init.TrainingInitializer._create_training_manager').start()
        
        # Mock display
        self.mock_display = patch('IPython.display.display').start()
        
        # Dummy UI components
        self.mock_ui_components = {
            'main_container': MagicMock(spec=widgets.VBox),
            'config_tabs': MagicMock(spec=widgets.Tab),
            'metrics_accordion': MagicMock(spec=widgets.Accordion),
            'progress_container': MagicMock(spec=widgets.VBox),
            'progress_tracker': MagicMock(spec=widgets.HBox),
            'status_panel': MagicMock(spec=widgets.HTML),
            'log_output': MagicMock(spec=widgets.Output),
            'start_button': MagicMock(spec=widgets.Button),
            'stop_button': MagicMock(spec=widgets.Button),
            'reset_button': MagicMock(spec=widgets.Button)
        }
        
        self.mock_create_form.return_value = self.mock_ui_components
        self.mock_create_layout.return_value = self.mock_ui_components

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_initialize(self):
        """Test inisialisasi training UI"""
        # Setup mocks untuk initialize
        with patch('smartcash.ui.training.training_init.TrainingInitializer._create_ui_components') as mock_create_ui,\
             patch('smartcash.ui.training.training_init.TrainingInitializer._setup_module_handlers') as mock_setup_handlers,\
             patch('smartcash.ui.training.training_init.TrainingInitializer._get_merged_config') as mock_get_config,\
             patch('smartcash.ui.training.training_init.TrainingInitializer._validate_setup') as mock_validate,\
             patch('smartcash.ui.training.training_init.TrainingInitializer._finalize_setup') as mock_finalize,\
             patch('smartcash.ui.utils.logger_bridge.create_ui_logger_bridge') as mock_logger_bridge:
            
            # Setup mock returns
            mock_create_ui.return_value = self.mock_ui_components
            mock_setup_handlers.return_value = self.mock_ui_components
            mock_get_config.return_value = self.mock_config
            mock_validate.return_value = {'valid': True, 'message': 'Valid'}
            mock_logger_bridge.return_value = MagicMock()
            
            # Inisialisasi training UI
            initializer = TrainingInitializer()
            initializer.module_name = 'training'
            initializer.logger_namespace = 'smartcash.ui.training'
            initializer.logger = MagicMock()
            
            # Mock _get_return_value
            initializer._get_return_value = lambda ui: ui
            
            result = initializer.initialize(config=self.mock_config)
            
            # Verifikasi UI components
            self.assertEqual(result, self.mock_ui_components)
            mock_create_ui.assert_called_once()
            mock_setup_handlers.assert_called_once()

    def test_config_callback(self):
        """Test callback config update"""
        # Setup mocks untuk config callback
        initializer = TrainingInitializer()
        initializer.module_name = 'training'
        initializer.logger_namespace = 'smartcash.ui.training'
        initializer.logger = MagicMock()
        
        # Tambahkan callback dummy
        mock_callback = MagicMock()
        initializer.config_update_callbacks.append(mock_callback)
        
        # Trigger config update
        new_config = {'model': {'name': 'updated-model'}}
        initializer.trigger_config_update(new_config)
        
        # Verifikasi callback dipanggil
        mock_callback.assert_called_once_with(new_config)

    def test_prevent_duplicate_display(self):
        """Test pencegahan duplicate display"""
        # Buat mock untuk fungsi initialize_training_ui
        with patch('smartcash.ui.training.training_init.initialize_training_ui') as mock_initialize_ui:
            # Panggil fungsi initialize_training_ui dengan flag displayed=True
            from smartcash.ui.training.training_init import initialize_training_ui
            
            # Setup mock untuk _training_initializer
            initializer = MagicMock()
            initializer._ui_displayed = True
            
            with patch('smartcash.ui.training.training_init._training_initializer', initializer):
                # Panggil fungsi dengan flag displayed=True
                initialize_training_ui(config=self.mock_config, displayed=True)
                
                # Verifikasi bahwa initializer.initialize tidak dipanggil
                initializer.initialize.assert_not_called()


def run_tests():
    """Run test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tambahkan test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingInitializer))
    
    # Run tests dengan output lebih detail
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
