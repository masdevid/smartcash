"""
Integration tests for UI initializers to verify their basic functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
from IPython.display import display

class TestInitializerIntegration(unittest.TestCase):    
    """Integration tests for UI initializers."""
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyConfigHandler')
    @patch('smartcash.ui.setup.dependency.components.ui_components.create_dependency_main_ui')
    def test_dependency_initializer(self, mock_create_ui, mock_handler_class):
        """Test DependencyInitializer initialization and UI creation."""
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer, initialize_dependency_ui
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'dependency': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_ui = {'ui': widgets.VBox(), 'log_output': widgets.Output()}
        mock_create_ui.return_value = mock_ui
        
        # Test class initialization
        initializer = DependencyInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        ui = initializer.initialize_ui()
        self.assertIsNotNone(ui)
        
        # Test factory function
        ui_func = initialize_dependency_ui()
        self.assertIsNotNone(ui_func)
    
    @patch('smartcash.ui.dataset.downloader.downloader_initializer.DownloaderConfigHandler')
    @patch('smartcash.ui.dataset.downloader.components.ui_components.create_downloader_main_ui')
    def test_downloader_initializer(self, mock_create_ui, mock_handler_class):
        """Test DownloaderInitializer initialization and UI creation."""
        from smartcash.ui.dataset.downloader.downloader_initializer import DownloaderInitializer, initialize_downloader_ui
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'downloader': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_ui = {'ui': widgets.VBox(), 'log_output': widgets.Output()}
        mock_create_ui.return_value = mock_ui
        
        # Test class initialization
        initializer = DownloaderInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        ui = initializer.initialize_ui()
        self.assertIsNotNone(ui)
        
        # Test factory function
        ui_func = initialize_downloader_ui()
        self.assertIsNotNone(ui_func)
    
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.PreprocessingConfigHandler')
    @patch('smartcash.ui.dataset.preprocessing.components.ui_components.create_preprocessing_main_ui')
    def test_preprocessing_initializer(self, mock_create_ui, mock_handler_class):
        """Test PreprocessingInitializer initialization and UI creation."""
        from smartcash.ui.dataset.preprocessing.preprocessing_initializer import PreprocessingInitializer, initialize_preprocessing_ui
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'preprocessing': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_ui = {'ui': widgets.VBox(), 'log_output': widgets.Output()}
        mock_create_ui.return_value = mock_ui
        
        # Test class initialization
        initializer = PreprocessingInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        ui = initializer.initialize_ui()
        self.assertIsNotNone(ui)
        
        # Test factory function
        ui_func = initialize_preprocessing_ui()
        self.assertIsNotNone(ui_func)
    
    @patch('smartcash.ui.dataset.augmentation.augmentation_initializer.AugmentationConfigHandler')
    @patch('smartcash.ui.dataset.augmentation.components.ui_components.create_augmentation_main_ui')
    def test_augmentation_initializer(self, mock_create_ui, mock_handler_class):
        """Test AugmentationInitializer initialization and UI creation."""
        from smartcash.ui.dataset.augmentation.augmentation_initializer import AugmentationInitializer, initialize_augmentation_ui
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'augmentation': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_ui = {'ui': widgets.VBox(), 'log_output': widgets.Output()}
        mock_create_ui.return_value = mock_ui
        
        # Test class initialization
        initializer = AugmentationInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        ui = initializer.initialize_ui()
        self.assertIsNotNone(ui)
        
        # Test factory function
        ui_func = initialize_augmentation_ui()
        self.assertIsNotNone(ui_func)
    
    @patch('smartcash.ui.dataset.split.split_init.SplitConfigHandler')
    @patch('smartcash.ui.dataset.split.components.ui_form.create_split_form')
    def test_split_initializer(self, mock_create_form, mock_handler_class):
        """Test SplitConfigInitializer initialization and UI creation."""
        from smartcash.ui.dataset.split.split_init import SplitConfigInitializer, create_split_config_cell, get_split_config_components
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'split': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_form = {'form': widgets.VBox()}
        mock_create_form.return_value = mock_form
        
        # Test class initialization
        initializer = SplitConfigInitializer()
        self.assertIsNotNone(initializer)
        
        # Test components function
        with patch.object(initializer, '_create_ui_components'):
            components = get_split_config_components()
            self.assertIsNotNone(components)
    
    @patch('smartcash.ui.pretrained.pretrained_initializer.PretrainedConfigHandler')
    @patch('smartcash.ui.pretrained.components.ui_components.create_pretrained_main_ui')
    def test_pretrained_initializer(self, mock_create_ui, mock_handler_class):
        """Test PretrainedInitializer initialization and UI creation."""
        from smartcash.ui.pretrained.pretrained_initializer import PretrainedInitializer, initialize_pretrained_ui
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'pretrained': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_ui = {'ui': widgets.VBox(), 'log_output': widgets.Output()}
        mock_create_ui.return_value = mock_ui
        
        # Test class initialization
        initializer = PretrainedInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        ui = initializer.initialize_ui()
        self.assertIsNotNone(ui)
        
        # Test factory function
        ui_func = initialize_pretrained_ui()
        self.assertIsNotNone(ui_func)
    
    @patch('smartcash.ui.model.backbone.backbone_init.BackboneConfigHandler')
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_form')
    def test_backbone_initializer(self, mock_create_form, mock_handler_class):
        """Test BackboneInitializer initialization and UI creation."""
        from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
        
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.get_default_config.return_value = {'model': {}}
        mock_handler_class.return_value = mock_handler
        
        mock_form = {'form': widgets.VBox()}
        mock_create_form.return_value = mock_form
        
        # Test class initialization
        initializer = BackboneInitializer()
        self.assertIsNotNone(initializer)
        
        # Test UI initialization
        with patch.object(initializer, '_create_ui_components'):
            ui = initializer.initialize_ui()
            self.assertIsNotNone(ui)

if __name__ == '__main__':
    unittest.main()
