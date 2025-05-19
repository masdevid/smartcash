"""
File: smartcash/ui/dataset/augmentation/tests/test_helper.py
Deskripsi: Helper untuk unit test modul augmentasi dataset
"""

import ipywidgets as widgets
from unittest.mock import MagicMock
import os
import sys
import tempfile
import shutil
from pathlib import Path

def create_mock_ui_components():
    """
    Membuat mock untuk UI components yang digunakan dalam pengujian.
    
    Returns:
        Dict: Dictionary berisi mock UI components
    """
    # Buat mock untuk komponen dasar
    mock_ui_components = {
        'status': widgets.Output(),
        'logger': MagicMock(),
        'update_status_panel': MagicMock(),
        'update_progress': MagicMock(),
        'on_process_start': MagicMock(),
        'on_process_complete': MagicMock(),
        'on_process_error': MagicMock(),
        'on_process_stop': MagicMock(),
        'get_augmentation_config': MagicMock(),
        'sync_config_with_drive': MagicMock(),
        'reset_config_to_default': MagicMock(),
        'progress_bar': widgets.IntProgress(),
        'progress_label': widgets.Label(),
        'running': False,
        'executor': None
    }
    
    # Buat mock untuk augmentation_options
    mock_ui_components['augmentation_options'] = widgets.Tab(children=[
        widgets.VBox(children=[
            widgets.IntText(value=2),  # factor
            widgets.IntText(value=100),  # target_count
            widgets.IntText(value=4),  # num_workers
            widgets.Text(value='aug')  # prefix
        ]),
        widgets.VBox(children=[
            widgets.Dropdown(value='train')  # split
        ]),
        widgets.VBox(children=[
            widgets.SelectMultiple(value=('combined',)),  # aug_types
            widgets.Checkbox(value=False)  # balance_classes
        ])
    ])
    
    # Buat mock untuk tombol
    mock_ui_components['primary_button'] = widgets.Button()
    mock_ui_components['stop_button'] = widgets.Button()
    mock_ui_components['reset_button'] = widgets.Button()
    mock_ui_components['save_button'] = widgets.Button()
    
    # Setup return value untuk get_augmentation_config
    mock_ui_components['get_augmentation_config'].return_value = {
        'augmentation': {
            'types': ['combined'],
            'factor': 2,
            'target_count': 100,
            'balance_classes': False,
            'num_workers': 4,
            'prefix': 'aug',
            'split': 'train'
        }
    }
    
    return mock_ui_components

def create_test_directory_structure():
    """
    Membuat struktur direktori sementara untuk pengujian.
    
    Returns:
        Tuple: (base_dir, dataset_dir, augmented_dir, preprocessed_dir)
    """
    # Buat direktori dasar
    base_dir = tempfile.mkdtemp()
    
    # Buat struktur direktori
    dataset_dir = os.path.join(base_dir, 'dataset')
    augmented_dir = os.path.join(base_dir, 'augmented')
    preprocessed_dir = os.path.join(base_dir, 'preprocessed')
    
    # Buat direktori split
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, split), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_dir, split), exist_ok=True)
    
    # Buat beberapa file dummy di direktori train
    for i in range(5):
        with open(os.path.join(dataset_dir, 'train', f'image_{i}.jpg'), 'w') as f:
            f.write('dummy image content')
    
    return base_dir, dataset_dir, augmented_dir, preprocessed_dir

def cleanup_test_directory(base_dir):
    """
    Membersihkan direktori test setelah pengujian selesai.
    
    Args:
        base_dir (str): Path ke direktori dasar
    """
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

def create_mock_config_manager():
    """
    Membuat mock untuk ConfigManager.
    
    Returns:
        MagicMock: Mock untuk ConfigManager
    """
    mock_config_manager = MagicMock()
    
    # Setup return value untuk get_module_config
    mock_config_manager.get_module_config.return_value = {
        'augmentation': {
            'types': ['combined'],
            'factor': 2,
            'target_count': 100,
            'balance_classes': False,
            'num_workers': 4,
            'prefix': 'aug',
            'split': 'train'
        }
    }
    
    return mock_config_manager

def create_mock_augmentation_service():
    """
    Membuat mock untuk augmentation service.
    
    Returns:
        MagicMock: Mock untuk augmentation service
    """
    mock_service = MagicMock()
    
    # Setup return value untuk augment_dataset
    mock_service.augment_dataset.return_value = {
        'status': 'success',
        'count': 10,
        'message': 'Augmentasi berhasil!'
    }
    
    return mock_service

def run_all_tests():
    """
    Menjalankan semua unit test untuk modul augmentasi.
    """
    import unittest
    from smartcash.ui.dataset.augmentation.tests.test_simple import TestAugmentationInitializer, TestAugmentationUtils
    from smartcash.ui.dataset.augmentation.tests.test_handlers import TestSetupHandlers, TestStateHandler, TestPersistenceHandler
    from smartcash.ui.dataset.augmentation.tests.test_button_handler import TestButtonHandler
    from smartcash.ui.dataset.augmentation.tests.test_status_observer import TestStatusHandler, TestObserverHandler
    from smartcash.ui.dataset.augmentation.tests.test_augmentation_ui import TestAugmentationUI, TestAugmentationOptions
    
    # Buat test suite
    test_suite = unittest.TestSuite()
    
    # Tambahkan test case ke test suite
    test_suite.addTest(unittest.makeSuite(TestAugmentationInitializer))
    test_suite.addTest(unittest.makeSuite(TestAugmentationUtils))
    test_suite.addTest(unittest.makeSuite(TestSetupHandlers))
    test_suite.addTest(unittest.makeSuite(TestStateHandler))
    test_suite.addTest(unittest.makeSuite(TestPersistenceHandler))
    test_suite.addTest(unittest.makeSuite(TestButtonHandler))
    test_suite.addTest(unittest.makeSuite(TestStatusHandler))
    test_suite.addTest(unittest.makeSuite(TestObserverHandler))
    test_suite.addTest(unittest.makeSuite(TestAugmentationUI))
    test_suite.addTest(unittest.makeSuite(TestAugmentationOptions))
    
    # Jalankan test suite
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)

if __name__ == '__main__':
    run_all_tests()
