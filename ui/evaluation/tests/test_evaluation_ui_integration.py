"""
File: smartcash/tests/integration/test_evaluation_ui_integration.py
Deskripsi: Test integrasi untuk UI components evaluasi dengan handlers
"""

import os
import sys
import unittest
import numpy as np
import ipywidgets as widgets
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, Any, List

# Pastikan path root project ada di sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import UI components dan handlers
from smartcash.ui.evaluation.components.evaluation_form import create_evaluation_form
from smartcash.ui.evaluation.components.evaluation_layout import create_evaluation_layout
from smartcash.ui.evaluation.handlers.evaluation_handler import (
    setup_evaluation_handlers,
    run_evaluation_process
)
from smartcash.ui.utils.button_state_manager import get_button_state_manager

class TestEvaluationUIIntegration(unittest.TestCase):
    """Test integrasi untuk UI components evaluasi dengan handlers"""
    
    def setUp(self):
        """Setup untuk test"""
        # Mock logger
        self.logger = MagicMock()
        
        # Mock config
        self.config = {
            'scenario': {
                'selected_scenario': 'test_scenario',
                'scenarios': {
                    'test_scenario': {
                        'name': 'Skenario Test',
                        'backbone': 'cspdarknet_s',
                        'augmentation_type': 'default'
                    },
                    'scenario_2': {
                        'name': 'Skenario 2',
                        'backbone': 'efficientnet_b4',
                        'augmentation_type': 'position'
                    }
                }
            },
            'model': {
                'img_size': 416,
                'use_custom_checkpoint': False
            },
            'test_data': {
                'batch_size': 4,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'evaluation': {
                'class_names': ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp75000', 'Rp100000']
            }
        }
        
        # Mock test folder
        self.test_folder = '/tmp/smartcash_test'
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(os.path.join(self.test_folder, 'images'), exist_ok=True)
    
    def tearDown(self):
        """Cleanup setelah test"""
        # Hapus dummy test data
        import shutil
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
    
    @patch('ipywidgets.Dropdown')
    @patch('ipywidgets.HTML')
    def test_create_evaluation_form(self, mock_html, mock_dropdown):
        """Test pembuatan form evaluasi dengan komponen UI"""
        # Setup mock
        mock_dropdown_instance = MagicMock()
        mock_html_instance = MagicMock()
        mock_dropdown.return_value = mock_dropdown_instance
        mock_html.return_value = mock_html_instance
        
        # Panggil fungsi
        result = create_evaluation_form(self.config)
        
        # Verifikasi
        self.assertIsNotNone(result)
        self.assertIn('scenario_dropdown', result)
        self.assertIn('evaluate_button', result)
        self.assertIn('checkpoint_selector', result)
        mock_dropdown.assert_called()
        
    def test_create_evaluation_layout(self):
        """Test pembuatan layout untuk evaluation UI"""
        # Buat form components yang valid untuk layout
        form_components = {
            'scenario_dropdown': widgets.Dropdown(),
            'scenario_description': widgets.HTML(),
            'evaluate_button': widgets.Button(),
            'cancel_button': widgets.Button(),
            'checkpoint_selector': widgets.VBox(),
            'progress_tracker': widgets.VBox(),
            'status_panel': widgets.HTML(),
            'metrics_display': widgets.VBox(),
            'test_folder_text': widgets.HTML(),
            'apply_augmentation_checkbox': widgets.Checkbox(),
            'batch_size_slider': widgets.IntSlider(),
            'image_size_dropdown': widgets.Dropdown(),
            'confidence_slider': widgets.FloatSlider(),
            'iou_slider': widgets.FloatSlider(),
            'save_predictions_checkbox': widgets.Checkbox(),
            'save_metrics_checkbox': widgets.Checkbox(),
            'confusion_matrix_checkbox': widgets.Checkbox(),
            'visualize_results_checkbox': widgets.Checkbox(),
            'class_metrics_checkbox': widgets.Checkbox(),
            'inference_time_checkbox': widgets.Checkbox(),
            'save_to_drive_checkbox': widgets.Checkbox(),
            'drive_path_text': widgets.Text(),
            'container': widgets.VBox()
        }
        
        # Panggil fungsi dengan patching
        with patch('smartcash.ui.evaluation.components.evaluation_layout.create_section_title') as mock_section_title:
            with patch('smartcash.ui.evaluation.components.evaluation_form.create_metrics_display'):
                with patch('smartcash.ui.components.progress_tracking.create_progress_tracking_container'):
                    with patch('smartcash.ui.components.log_accordion.create_log_accordion'):
                        mock_section_title.return_value = widgets.HTML()
                        result = create_evaluation_layout(form_components, self.config)
        
        # Verifikasi
        self.assertIsInstance(result, dict)
        self.assertIn('main_container', result)
        self.assertIn('scenario_section', result)
        self.assertIn('checkpoint_section', result)
        self.assertTrue(mock_section_title.called)
    
    def test_setup_evaluation_handlers(self):
        """Test setup handlers untuk evaluation process"""
        # Setup UI components
        ui_components = {
            'evaluate_button': MagicMock(),
            'logger': self.logger,
            'progress_tracker': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Panggil fungsi dengan patching
        with patch('smartcash.ui.evaluation.handlers.evaluation_handler.get_button_state_manager') as mock_get_manager:
            with patch('smartcash.ui.evaluation.handlers.evaluation_handler.run_evaluation_process'):
                # Setup mock
                mock_button_manager = MagicMock()
                mock_get_manager.return_value = mock_button_manager
                
                # Panggil fungsi
                result = setup_evaluation_handlers(ui_components, self.config)
                
                # Verifikasi
                self.assertEqual(result, ui_components)
                mock_get_manager.assert_called_once_with(ui_components)
                
                # Verifikasi button click handler
                on_click_handler = ui_components['evaluate_button'].on_click.call_args[0][0]
                self.assertIsNotNone(on_click_handler)
    
    def test_integration_handlers_with_components(self):
        """Test integrasi antara UI components dan handlers"""
        # Gabungkan components untuk test
        ui_components = {
            'scenario_dropdown': MagicMock(value='test_scenario'),
            'evaluate_button': MagicMock(),
            'checkpoint_selector': {
                'checkpoint_dropdown': MagicMock(),
                'custom_checkpoint_path': MagicMock(value=self.test_folder),
                'browse_button': MagicMock()
            },
            'progress_tracker': MagicMock(),
            'status_panel': MagicMock(),
            'metrics_display': MagicMock(),
            'logger': self.logger,
            'show_for_operation': MagicMock(),
            'error_operation': MagicMock(),
            'success_operation': MagicMock(),
            'warning_operation': MagicMock(),
            'update_progress': MagicMock()
        }
        
        # Setup mocks dan panggil fungsi
        with patch('smartcash.ui.evaluation.handlers.evaluation_handler.get_button_state_manager') as mock_get_manager:
            with patch('smartcash.ui.evaluation.handlers.evaluation_handler.run_evaluation_process') as mock_run_eval:
                with patch('smartcash.model.utils.evaluation_pipeline.run_evaluation_pipeline') as mock_run_pipeline:
                    # Setup mock return values
                    mock_run_pipeline.return_value = {
                        'success': True,
                        'metrics': {
                            'map': 0.85,
                            'overall_metrics': {
                                'precision': 0.9,
                                'recall': 0.8,
                                'f1_score': 0.85
                            },
                            'class_metrics': {
                                'Rp1000': {'precision': 0.9, 'recall': 0.8, 'f1_score': 0.85, 'ap': 0.85}
                            }
                        },
                        'predictions': [{'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]), 'scores': np.array([0.9]), 'classes': np.array([0])}]
                    }
                    
                    # Setup button manager
                    mock_button_manager = MagicMock()
                    mock_button_manager.operation_context.return_value.__enter__ = MagicMock()
                    mock_button_manager.operation_context.return_value.__exit__ = MagicMock()
                    mock_get_manager.return_value = mock_button_manager
                    
                    # Setup handlers
                    setup_evaluation_handlers(ui_components, self.config)
                    
                    # Panggil handler yang terdaftar pada tombol evaluate
                    ui_components['evaluate_button'].on_click.call_args[0][0](None)
                    
                    # Verifikasi
                    mock_get_manager.assert_called_once_with(ui_components)
                    self.assertTrue(mock_run_eval.called)

if __name__ == '__main__':
    unittest.main()
