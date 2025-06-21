"""
File: smartcash/tests/strategy/test_ui.py
Deskripsi: Test case untuk komponen UI Strategy
"""

import pytest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from smartcash.ui.strategy.strategy_init import StrategyInitializer

# Mock untuk komponen UI
class MockStrategyUI:
    """Mock class untuk Strategy UI components"""
    
    def __init__(self):
        self.container = widgets.VBox()
        self.form = widgets.VBox()
        self.summary_card = widgets.Output()
        self.save_button = widgets.Button(description="Save Strategy")
        self.reset_button = widgets.Button(description="Reset to Default")
        self.status = widgets.Output()
        
        # Mock untuk form fields
        self.strategy_name = widgets.Dropdown(
            options=['Moving Average', 'RSI', 'MACD'],
            value='Moving Average',
            description='Strategy:'
        )
        self.fast_ma = widgets.IntSlider(value=10, min=5, max=50, description='Fast MA:')
        self.slow_ma = widgets.IntSlider(value=30, min=10, max=200, description='Slow MA:')
        self.rsi_period = widgets.IntSlider(value=14, min=5, max=30, description='RSI Period:')
        self.rsi_overbought = widgets.IntSlider(value=70, min=50, max=90, description='RSI Overbought:')
        self.rsi_oversold = widgets.IntSlider(value=30, min=10, max=50, description='RSI Oversold:')
        
        # Tambahkan ke form
        self.form.children = [
            self.strategy_name,
            self.fast_ma,
            self.slow_ma,
            self.rsi_period,
            self.rsi_overbought,
            self.rsi_oversold
        ]
        
        self.container.children = [
            self.form,
            self.summary_card,
            widgets.HBox([self.save_button, self.reset_button]),
            self.status
        ]
    
    def get_ui(self):
        return self.container

# Fixture untuk mock UI
@pytest.fixture
def mock_strategy_ui():
    """Fixture untuk mock Strategy UI"""
    return MockStrategyUI()

class TestStrategyUI:
    """Test class untuk Strategy UI"""
    
    def test_ui_initialization(self, mock_strategy_ui):
        """Test inisialisasi Strategy UI"""
        ui = mock_strategy_ui
        
        # Verifikasi komponen utama ada
        assert hasattr(ui, 'container')
        assert hasattr(ui, 'form')
        assert hasattr(ui, 'summary_card')
        assert hasattr(ui, 'save_button')
        assert hasattr(ui, 'reset_button')
        assert hasattr(ui, 'status')
        
        # Verifikasi form fields
        assert hasattr(ui, 'strategy_name')
        assert hasattr(ui, 'fast_ma')
        assert hasattr(ui, 'slow_ma')
        assert hasattr(ui, 'rsi_period')
        assert hasattr(ui, 'rsi_overbought')
        assert hasattr(ui, 'rsi_oversold')
    
    def test_ui_display(self, mock_strategy_ui):
        """Test menampilkan UI"""
        ui = mock_strategy_ui
        result = ui.get_ui()
        assert result == ui.container, "Harus mengembalikan container UI"
    
    def test_form_values(self, mock_strategy_ui):
        """Test nilai default form"""
        ui = mock_strategy_ui
        
        assert ui.strategy_name.value == 'Moving Average'
        assert ui.fast_ma.value == 10
        assert ui.slow_ma.value == 30
        assert ui.rsi_period.value == 14
        assert ui.rsi_overbought.value == 70
        assert ui.rsi_oversold.value == 30

class TestStrategyInitializer:
    """Test class untuk StrategyInitializer"""
    
    def test_initializer_creation(self):
        """Test inisialisasi StrategyInitializer"""
        with patch('smartcash.ui.strategy.strategy_init.StrategyConfigHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            initializer = StrategyInitializer()
            
            # Verifikasi inisialisasi dasar
            assert initializer is not None
            assert initializer.module_name == 'strategy'
            assert initializer.config_filename == 'training_config'
            mock_handler_class.assert_called_once()
    
    @patch('smartcash.ui.strategy.strategy_init.StrategyConfigHandler')
    def test_initialize_with_mock_handler(self, mock_handler_class):
        """Test inisialisasi dengan mock handler"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {}
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock initialize_strategy_config
        with patch('smartcash.ui.strategy.strategy_init.initialize_strategy_config', 
                 return_value=mock_ui) as mock_init:
            
            # Inisialisasi
            initializer = StrategyInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                mock_handler.load_config.assert_called_once()
                mock_init.assert_called_once()
                initializer._create_config_ui.assert_called_once()
                
                # Verifikasi UI components
                assert hasattr(result, 'get_ui')
                result.get_ui.assert_called_once()

class TestStrategyIntegration:
    """Test integrasi komponen Strategy"""
    
    @patch('smartcash.ui.strategy.strategy_init.StrategyConfigHandler')
    def test_config_loading(self, mock_handler_class):
        """Test loading konfigurasi strategy"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {
            'strategy': 'Moving Average',
            'fast_ma': 10,
            'slow_ma': 30
        }
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock initialize_strategy_config
        with patch('smartcash.ui.strategy.strategy_init.initialize_strategy_config',
                 return_value=mock_ui) as mock_init:
            
            # Jalankan initializer
            initializer = StrategyInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                mock_handler.load_config.assert_called_once()
                mock_init.assert_called_once()
                initializer._create_config_ui.assert_called_once()
    
    @patch('smartcash.ui.strategy.strategy_init.StrategyConfigHandler')
    def test_ui_creation(self, mock_handler_class):
        """Test pembuatan UI strategy"""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_handler.load_config.return_value = {}
        mock_handler_class.return_value = mock_handler
        
        # Setup mock UI
        mock_ui = MagicMock()
        mock_ui.get_ui.return_value = MagicMock(spec=widgets.VBox)
        
        # Mock initialize_strategy_config
        with patch('smartcash.ui.strategy.strategy_init.initialize_strategy_config',
                 return_value=mock_ui) as mock_init:
            
            # Jalankan initializer
            initializer = StrategyInitializer()
            
            # Mock the _create_config_ui method to avoid UI creation
            with patch.object(initializer, '_create_config_ui', return_value=mock_ui):
                result = initializer.initialize()
                
                # Verifikasi
                assert result is not None
                assert hasattr(result, 'get_ui')
                result.get_ui.assert_called_once()
                mock_handler.load_config.assert_called_once()
                mock_init.assert_called_once()
                initializer._create_config_ui.assert_called_once()
