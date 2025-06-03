"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/evaluation/tests/test_ui_components.py
Deskripsi: Test suite untuk memverifikasi komponen UI evaluasi model berfungsi dengan benar.
"""
import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.evaluation.components.evaluation_form import create_evaluation_form
from smartcash.ui.evaluation.components.evaluation_layout import create_evaluation_layout

class TestEvaluationUIComponents(unittest.TestCase):
    """Test suite untuk komponen UI evaluasi model."""
    
    def setUp(self) -> None:
        """Setup untuk test dengan konfigurasi default."""
        self.default_config = {
            'checkpoint': {
                'auto_select_best': True,
                'custom_checkpoint_path': '',
                'validation_metrics': ['mAP@0.5', 'mAP@0.5:0.95']
            },
            'test_data': {
                'apply_augmentation': False,
                'batch_size': 16,
                'image_size': 640,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'evaluation': {
                'save_predictions': True,
                'save_metrics': True,
                'generate_confusion_matrix': True,
                'visualize_results': True
            },
            'scenario': {
                'selected_scenario': 'scenario_1',
                'save_to_drive': True,
                'drive_path': '/content/drive/MyDrive/SmartCash/evaluation_results',
                'test_folder': '/content/drive/MyDrive/SmartCash/dataset/test'
            }
        }
        
        # Buat komponen form dan layout
        self.form_components = create_evaluation_form(self.default_config)
        self.layout_components = create_evaluation_layout(self.form_components, self.default_config)
    
    def test_form_components_creation(self) -> None:
        """Test bahwa semua komponen form dibuat dengan benar."""
        # Verifikasi komponen utama ada
        essential_components = [
            'scenario_dropdown', 'auto_select_checkbox', 'checkpoint_path_text',
            'test_folder_text', 'apply_augmentation_checkbox', 'batch_size_slider',
            'image_size_dropdown', 'confidence_slider', 'iou_slider',
            'save_predictions_checkbox', 'save_metrics_checkbox',
            'confusion_matrix_checkbox', 'visualize_results_checkbox',
            'save_to_drive_checkbox', 'drive_path_text'
        ]
        
        for component in essential_components:
            self.assertIn(component, self.form_components, f"Komponen {component} tidak ditemukan")
            self.assertIsInstance(self.form_components[component], widgets.Widget, 
                                 f"Komponen {component} bukan instance dari widgets.Widget")
    
    def test_checkbox_style_consistency(self) -> None:
        """Test bahwa semua checkbox memiliki style yang konsisten."""
        checkbox_components = [
            'auto_select_checkbox', 'apply_augmentation_checkbox',
            'save_predictions_checkbox', 'save_metrics_checkbox',
            'confusion_matrix_checkbox', 'visualize_results_checkbox',
            'save_to_drive_checkbox'
        ]
        
        for component in checkbox_components:
            # Verifikasi style description_width adalah 'initial'
            self.assertEqual(self.form_components[component].style.description_width, 'initial',
                            f"Checkbox {component} tidak memiliki style.description_width 'initial'")
            
            # Verifikasi layout width adalah '100%'
            self.assertEqual(self.form_components[component].layout.width, '100%',
                            f"Checkbox {component} tidak memiliki layout.width '100%'")
    
    def test_layout_components_creation(self) -> None:
        """Test bahwa semua komponen layout dibuat dengan benar."""
        # Verifikasi komponen utama layout ada
        essential_layout_components = [
            'main_container', 'main_content', 'scenario_section',
            'checkpoint_section', 'test_section', 'eval_options',
            'actions_section', 'results_section', 'results_tabs'
        ]
        
        for component in essential_layout_components:
            self.assertIn(component, self.layout_components, f"Komponen layout {component} tidak ditemukan")
    
    def test_checkbox_grouping(self) -> None:
        """Test bahwa checkbox dikelompokkan dengan benar dalam layout."""
        # Verifikasi bahwa test_section memiliki struktur yang benar
        test_section = self.layout_components['test_section']
        self.assertEqual(len(test_section.children), 4, "test_section harus memiliki 4 children")
        
        # Verifikasi bahwa checkbox apply_augmentation ada di test_config_checkboxes
        test_config_checkboxes = test_section.children[2]  # Index 2 adalah test_config_checkboxes
        self.assertIsInstance(test_config_checkboxes, widgets.VBox, "test_config_checkboxes harus berupa VBox")
        self.assertEqual(len(test_config_checkboxes.children), 1, "test_config_checkboxes harus memiliki 1 child")
        self.assertIs(test_config_checkboxes.children[0], self.form_components['apply_augmentation_checkbox'],
                      "Child dari test_config_checkboxes harus berupa apply_augmentation_checkbox")
        
        # Verifikasi bahwa eval_options memiliki eval_options_checkboxes
        eval_options = self.layout_components['eval_options']
        self.assertEqual(len(eval_options.children), 2, "eval_options harus memiliki 2 children")
        
        # Verifikasi bahwa eval_options_container memiliki eval_options_checkboxes dan eval_options_drive
        eval_options_container = eval_options.children[1]
        self.assertEqual(len(eval_options_container.children), 2, "eval_options_container harus memiliki 2 children")
        
        # Verifikasi bahwa checkbox dikelompokkan dalam eval_options_checkboxes
        eval_options_checkboxes = eval_options_container.children[0]
        self.assertIsInstance(eval_options_checkboxes, widgets.VBox, "eval_options_checkboxes harus berupa VBox")
        self.assertEqual(len(eval_options_checkboxes.children), 4, "eval_options_checkboxes harus memiliki 4 children (termasuk inference_time_checkbox)")
        
        # Verifikasi bahwa drive_path_text ada di eval_options_drive
        eval_options_drive = eval_options_container.children[1]
        self.assertIsInstance(eval_options_drive, widgets.VBox, "eval_options_drive harus berupa VBox")
        self.assertEqual(len(eval_options_drive.children), 1, "eval_options_drive harus memiliki 1 child")
        self.assertIs(eval_options_drive.children[0], self.form_components['drive_path_text'],
                     "Child dari eval_options_drive harus berupa drive_path_text")

if __name__ == '__main__':
    unittest.main()
