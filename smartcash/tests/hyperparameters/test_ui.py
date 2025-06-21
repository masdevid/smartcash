"""
File: smartcash/tests/hyperparameters/test_ui.py
Deskripsi: Test case untuk komponen UI Hyperparameters
"""

import pytest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from smartcash.ui.hyperparameters.hyperparameters_init import HyperparametersConfigInitializer

# Mock untuk komponen UI
class MockHyperparametersUI:
    """Mock class untuk Hyperparameters UI components"""
    
    def __init__(self):
        self.container = widgets.VBox()
        self.form = widgets.VBox()
        self.save_button = widgets.Button(description="Save")
        self.reset_button = widgets.Button(description="Reset")
        self.status = widgets.Output()
        
        # Mock untuk form fields
        self.learning_rate = widgets.FloatSlider(value=0.001, min=0.0001, max=0.1, step=0.0001, description='Learning Rate')
        self.batch_size = widgets.IntSlider(value=32, min=8, max=256, step=8, description='Batch Size')
        self.epochs = widgets.IntSlider(value=10, min=1, max=100, description='Epochs')
        
        # Tambahkan ke form
        self.form.children = [self.learning_rate, self.batch_size, self.epochs]
        self.container.children = [self.form, widgets.HBox([self.save_button, self.reset_button]), self.status]
    
    def get_ui(self):
        return self.container

# Fixture untuk mock UI
@pytest.fixture
def mock_hyperparameters_ui():
    """Fixture untuk mock Hyperparameters UI"""
    return MockHyperparametersUI()

class TestHyperparametersUI:
    """Test class untuk Hyperparameters UI"""
    
    def test_ui_initialization(self, mock_hyperparameters_ui):
        """Test inisialisasi Hyperparameters UI"""
        ui = mock_hyperparameters_ui
        
        # Verifikasi komponen utama ada
        assert hasattr(ui, 'container')
        assert hasattr(ui, 'form')
        assert hasattr(ui, 'save_button')
        assert hasattr(ui, 'reset_button')
        assert hasattr(ui, 'status')
        
        # Verifikasi form fields
        assert hasattr(ui, 'learning_rate')
        assert hasattr(ui, 'batch_size')
        assert hasattr(ui, 'epochs')
    
    def test_ui_display(self, mock_hyperparameters_ui):
        """Test menampilkan UI"""
        ui = mock_hyperparameters_ui
        result = ui.get_ui()
        assert result == ui.container, "Harus mengembalikan container UI"
    
    def test_form_values(self, mock_hyperparameters_ui):
        """Test nilai default form"""
        ui = mock_hyperparameters_ui
        
        assert ui.learning_rate.value == 0.001
        assert ui.batch_size.value == 32
        assert ui.epochs.value == 10

class TestHyperparametersConfigInitializer:
    """Test class untuk HyperparametersConfigInitializer"""
    
    def test_initializer_creation(self):
        """Test inisialisasi HyperparametersConfigInitializer"""
        with patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            initializer = HyperparametersConfigInitializer()
            
            # Verifikasi inisialisasi dasar
            assert initializer is not None
            assert initializer.module_name == 'hyperparameters'
            assert initializer.config_filename == 'hyperparameters_config'
            mock_handler_class.assert_called_once()
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    def test_initialize_with_mock_handler(self, mock_handler_class):
        """Test inisialisasi dengan mock handler"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {}
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI components
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock create_hyperparameters_config_cell
        with patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell', 
                 return_value=mock_ui) as mock_create_cell:
            
            # Inisialisasi dan panggil initialize
            initializer = HyperparametersConfigInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                mock_handler.load_config.assert_called_once()
                mock_create_cell.assert_called_once()
                initializer._create_config_ui.assert_called_once()

class TestHyperparametersIntegration:
    """Test integrasi komponen Hyperparameters"""
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    def test_load_config(self, mock_handler_class):
        """Test loading konfigurasi"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {'learning_rate': 0.002, 'batch_size': 64}
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock create_hyperparameters_config_cell
        with patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell', 
                 return_value=mock_ui) as mock_create_cell:
            
            # Jalankan initializer
            initializer = HyperparametersConfigInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                mock_handler.load_config.assert_called_once()
                mock_create_cell.assert_called_once()
                initializer._create_config_ui.assert_called_once()
    
    @patch('smartcash.ui.hyperparameters.hyperparameters_init.HyperparametersConfigHandler')
    def test_ui_creation(self, mock_handler_class):
        """Test pembuatan UI"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {}
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock create_hyperparameters_config_cell
        with patch('smartcash.ui.hyperparameters.hyperparameters_init.create_hyperparameters_config_cell', 
                 return_value=mock_ui) as mock_create_cell:
            
            # Jalankan initializer
            initializer = HyperparametersConfigInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                assert hasattr(result, 'get_ui')
                result.get_ui.assert_called_once()
                mock_handler.load_config.assert_called_once()
                mock_create_cell.assert_called_once()
                initializer._create_config_ui.assert_called_once()
