"""
Test imports for all UI initializers to ensure they can be imported without errors.
"""
import unittest
import importlib
from pathlib import Path

# List of modules to test
INITIALIZER_MODULES = [
    'smartcash.ui.setup.dependency.dependency_initializer',
    'smartcash.ui.dataset.downloader.downloader_initializer',
    'smartcash.ui.dataset.preprocessing.preprocessing_initializer',
    'smartcash.ui.dataset.augmentation.augmentation_initializer',
    'smartcash.ui.dataset.split.split_init',
    'smartcash.ui.pretrained.pretrained_initializer',
    'smartcash.ui.model.backbone.backbone_init',
]

class TestImports(unittest.TestCase):
    """Test that all initializer modules can be imported without errors."""

    def test_import_initializers(self):
        """Test that all initializer modules can be imported."""
        for module_name in INITIALIZER_MODULES:
            with self.subTest(module=module_name):
                try:
                    module = importlib.import_module(module_name)
                    self.assertIsNotNone(module, f"Failed to import {module_name}")
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {str(e)}")
                except Exception as e:
                    self.fail(f"Unexpected error importing {module_name}: {str(e)}")

    def test_initializer_classes_exist(self):
        """Test that expected initializer classes exist in their modules."""
        module_classes = {
            'smartcash.ui.setup.dependency.dependency_initializer': ['DependencyInitializer', 'initialize_dependency_ui'],
            'smartcash.ui.dataset.downloader.downloader_initializer': ['DownloaderInitializer', 'initialize_downloader_ui'],
            'smartcash.ui.dataset.preprocessing.preprocessing_initializer': ['PreprocessingInitializer', 'initialize_preprocessing_ui'],
            'smartcash.ui.dataset.augmentation.augmentation_initializer': ['AugmentationInitializer', 'initialize_augmentation_ui'],
            'smartcash.ui.dataset.split.split_init': ['SplitConfigInitializer', 'create_split_config_cell', 'get_split_config_components'],
            'smartcash.ui.pretrained.pretrained_initializer': ['PretrainedInitializer', 'initialize_pretrained_ui'],
            'smartcash.ui.model.backbone.backbone_init': ['BackboneInitializer']
        }
        
        for module_name, expected_attrs in module_classes.items():
            with self.subTest(module=module_name):
                try:
                    module = importlib.import_module(module_name)
                    for attr in expected_attrs:
                        self.assertTrue(
                            hasattr(module, attr),
                            f"{module_name} is missing expected attribute: {attr}"
                        )
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {str(e)}")

if __name__ == '__main__':
    unittest.main()
