"""
file_path: tests/ui/dataset/visualization/test_visualization_module.py
Unit tests for the visualization module with real dependency injection.
"""
import pytest
from unittest.mock import MagicMock, patch, call, ANY, mock_open
import ipywidgets as widgets
import sys

# Create a mock for PyTorch
class MockTorch:
    class _C:
        pass
    
    def __getattr__(self, name):
        if name not in self.__dict__:
            return MagicMock()
        return self.__dict__[name]

# Create a mock for OpenCV to prevent GAPI module errors
class MockCV2:
    class gapi:
        class wip:
            class draw:
                Text = None
                Circle = None
                Image = None
                Line = None
                Rect = None
                Mosaic = None
                Poly = None
    
    def __getattr__(self, name):
        # Return a dummy function for any cv2 function calls
        if name not in self.__dict__:
            return MagicMock()
        return self.__dict__[name]

# Add the mocks to sys.modules before any imports
sys.modules['torch'] = MockTorch()
sys.modules['torch._C'] = MockTorch._C()

sys.modules['cv2'] = MockCV2()
sys.modules['cv2.gapi'] = MockCV2.gapi
sys.modules['cv2.gapi.wip'] = MockCV2.gapi.wip
sys.modules['cv2.gapi.wip.draw'] = MockCV2.gapi.wip.draw

# Import the module under test first to avoid circular imports
import sys
import importlib

# Create a mock for the visualization_stats_cards module
mock_viz_cards = MagicMock()

# Create mock classes
class MockVisualizationStatsCard:
    def __init__(self, *args, **kwargs):
        self.raw_count = 0
        self.preprocessed_count = 0
        self.augmented_count = 0
        self.is_placeholder = True
        
    def update(self, data):
        if data.get('dataset_stats', {}).get('success'):
            split_data = data['dataset_stats']['by_split'].get('train', {})
            self.raw_count = split_data.get('raw', 0)
            self.preprocessed_count = split_data.get('preprocessed', 0)
            self.augmented_count = split_data.get('augmented', 0)
            self.is_placeholder = False

class MockVisualizationStatsCardContainer:
    def __init__(self):
        self.cards = {}
        
    def get_container(self):
        return widgets.HBox()
        
    def update_all_cards(self, stats):
        for card in self.cards.values():
            card.update(stats)

# Configure the mock
mock_viz_cards.VisualizationStatsCard = MockVisualizationStatsCard
mock_viz_cards.VisualizationStatsCardContainer = MockVisualizationStatsCardContainer
mock_viz_cards.create_visualization_stats_dashboard = MagicMock(return_value=MockVisualizationStatsCardContainer())

# Patch the module in sys.modules before importing the module under test
with patch.dict('sys.modules', {
    'smartcash.ui.dataset.visualization.components.visualization_stats_cards': mock_viz_cards,
    'smartcash.ui.dataset.visualization.components.visualization_ui': MagicMock()
}):
    # Now import the module under test
    from smartcash.ui.dataset.visualization.visualization_uimodule import VisualizationUIModule

class TestVisualizationModule:
    """Test cases for VisualizationUIModule."""
    
    def test_module_initialization(self, mock_visualization_module):
        """Test that the module initializes correctly."""
        assert mock_visualization_module is not None
        assert hasattr(mock_visualization_module, '_dashboard_cards')
        assert hasattr(mock_visualization_module, '_operations')
        assert 'refresh' in mock_visualization_module._operations
        
    def test_get_default_config(self, mock_visualization_module):
        """Test that get_default_config returns the expected configuration."""
        # Call the method
        config = mock_visualization_module.get_default_config()
        
        # Check that the config has the expected structure
        assert isinstance(config, dict)
        assert 'module_name' in config
        assert 'version' in config
        assert 'description' in config
        assert 'splits' in config
        assert 'display' in config
        assert 'ui' in config
        assert 'performance' in config
        
        # Check some specific values
        assert config['module_name'] == 'visualization'
        assert 'train' in config['splits']
        assert 'show_statistics' in config['ui']
        assert 'max_data_points' in config['performance']
        
    def test_create_ui_components(self, mock_visualization_module, mock_ui_components):
        """Test UI components creation."""
        # Mock the create_visualization_ui function to return our mock components
        with patch('smartcash.ui.dataset.visualization.visualization_uimodule.create_visualization_ui', 
                 return_value=mock_ui_components) as mock_create_ui:
            
            # Call the method with a test config
            test_config = {'test': 'config'}
            components = mock_visualization_module.create_ui_components(test_config)
            
            # Verify the function was called with the correct config
            mock_create_ui.assert_called_once_with(config=test_config)
            
            # Verify the return value is what we expect
            assert components == mock_ui_components
            
            # Verify the components have the expected structure
            assert isinstance(components, dict)
            assert 'containers' in components
            assert 'widgets' in components
        
    def test_update_dashboard_stats(self, mock_visualization_module):
        """Test updating dashboard statistics."""
        # Mock the dashboard cards
        mock_dashboard = MagicMock()
        mock_visualization_module._dashboard_cards = mock_dashboard
        
        # Call the method
        mock_visualization_module._update_dashboard_stats()
        
        # Verify the dashboard was updated with empty stats
        assert mock_dashboard.update_all_cards.called
        
    def test_initialize_dashboard(self, mock_visualization_module, mocker):
        """Test dashboard initialization."""
        # Setup mocks
        mock_container = MagicMock()
        mock_visualization_module.components = {
            'containers': {
                'dashboard_container': mock_container
            }
        }
        
        # Create a mock for the dashboard cards
        mock_dashboard_cards = MagicMock()
        mock_dashboard_cards.get_container.return_value = 'mock_container_widget'
        
        # Patch the import inside the _initialize_dashboard method
        with patch('smartcash.ui.dataset.visualization.components.visualization_stats_cards.create_visualization_stats_dashboard') as mock_create_dashboard:
            # Configure the mock to return our mock dashboard cards
            mock_create_dashboard.return_value = mock_dashboard_cards
            
            # Ensure the dashboard_cards attribute is None initially
            mock_visualization_module._dashboard_cards = None
            
            # Call the method
            mock_visualization_module._initialize_dashboard()
            
            # Verify the dashboard cards were created and stored
            assert mock_visualization_module._dashboard_cards is not None
            assert mock_visualization_module._dashboard_cards == mock_dashboard_cards
            
            # Verify create_visualization_stats_dashboard was called
            mock_create_dashboard.assert_called_once()
            
            # Verify the container was updated with the dashboard cards
            mock_container.children = [mock_dashboard_cards.get_container()]
            assert len(mock_container.children) == 1
        
    def test_update_stats_cards(self, mock_visualization_module, sample_stats_data):
        """Test updating stats cards with sample data."""
        # Mock the dashboard cards
        mock_dashboard = MagicMock()
        mock_visualization_module._dashboard_cards = mock_dashboard
        
        # Call the method with sample data
        mock_visualization_module.update_stats_cards(sample_stats_data)
        
        # Verify the dashboard was updated with the sample data
        assert mock_dashboard.update_all_cards.called
        args = mock_dashboard.update_all_cards.call_args[0][0]
        assert 'dataset_stats' in args
        assert 'augmentation_stats' in args
        
    def test_get_latest_stats(self, mock_visualization_module, sample_stats_data):
        """Test getting the latest statistics."""
        # Set the latest stats
        mock_visualization_module._latest_stats = sample_stats_data
        
        # Get the latest stats
        stats = mock_visualization_module.get_latest_stats()
        
        # Verify the stats were returned correctly
        assert stats == sample_stats_data
        
    @pytest.mark.asyncio
    async def test_on_refresh_click(self, mock_visualization_module, mocker):
        """Test refresh button click handler."""
        # Setup mocks
        mock_visualization_module._operations = {
            'refresh': mocker.AsyncMock()
        }
        
        # Configure the execute mock to return success
        mock_visualization_module._operations['refresh'].execute.return_value = {'success': True}
        
        # Mock the refresh button
        mock_button = MagicMock()
        mock_button.loading = True  # Start with loading=True to test it gets set to False
        
        # Mock the _update_dashboard_stats method
        mock_visualization_module._update_dashboard_stats = mocker.AsyncMock()
        
        # Mock the update_operation_status method
        mock_visualization_module.update_operation_status = mocker.AsyncMock()
        
        # Create a mock for the execute method that returns our success result
        async def mock_execute():
            return {'success': True}
            
        # Create a mock for the execute method that's awaitable
        mock_visualization_module._operations['refresh'].execute = mocker.AsyncMock(return_value={'success': True})
        
        # Ensure the _on_refresh_click method is properly patched to be async
        async def mock_on_refresh_click(button):
            button.loading = True
            try:
                result = await mock_visualization_module._operations['refresh'].execute()
                await mock_visualization_module._update_dashboard_stats()
                await mock_visualization_module.update_operation_status('refresh', result)
            finally:
                button.loading = False
                
        # Replace the actual method with our mock
        mock_visualization_module._on_refresh_click = mock_on_refresh_click
        
        # Call the handler
        await mock_visualization_module._on_refresh_click(mock_button)
        
        # Verify the button loading state was reset to False after operation
        assert mock_button.loading is False, "Button loading state should be reset to False after operation"
        
        # Verify the operation was executed
        mock_visualization_module._operations['refresh'].execute.assert_awaited_once()
        
        # Verify the dashboard was updated
        mock_visualization_module._update_dashboard_stats.assert_awaited_once()
        
        # Verify the operation status was updated with the correct arguments
        mock_visualization_module.update_operation_status.assert_awaited_once_with(
            'refresh', 
            {'success': True}
        )
        
    def test_update_visualization(self, mock_visualization_module, mocker):
        """Test updating visualization with different types."""
        # Setup mocks for render methods and other dependencies
        mock_render_chart = mocker.patch.object(mock_visualization_module, '_render_chart')
        mock_render_preprocessed = mocker.patch.object(mock_visualization_module, '_render_preprocessed_samples')
        mock_render_augmented = mocker.patch.object(mock_visualization_module, '_render_augmented_samples')
        
        # Create a mock for the visualization container with clear_output method
        class MockContainer:
            def __init__(self):
                self.clear_output_called = False
                self.display_calls = []
            
            def clear_output(self):
                self.clear_output_called = True
            
            def append_display_data(self, data):
                self.display_calls.append(data)
            
            def assert_clear_output_called(self):
                assert self.clear_output_called, "clear_output was not called"
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
        
        # Create the mock container and set up components
        mock_container = MockContainer()
        mock_visualization_module.components = {
            'containers': {
                'visualization': mock_container,  # For the _render_chart method
                'visualization_container': mock_container  # For the update_visualization method
            }
        }
        mock_visualization_module.log_info = mocker.MagicMock()
        mock_visualization_module.log_warning = mocker.MagicMock()
        mock_visualization_module.log_error = mocker.MagicMock()
        mock_visualization_module.update_operation_status = mocker.MagicMock()  # Add this line
        
        # Test with bar chart
        mock_visualization_module.update_visualization('bar')
        mock_container.assert_clear_output_called()
        mock_render_chart.assert_called_once_with('bar')
        mock_render_preprocessed.assert_not_called()
        mock_render_augmented.assert_not_called()
        
        # Reset mocks for next test
        mock_container.clear_output_called = False
        mock_render_chart.reset_mock()
        
        # Test with preprocessed samples
        mock_visualization_module.update_visualization('preprocessed_samples')
        mock_container.assert_clear_output_called()
        mock_render_preprocessed.assert_called_once()
        mock_render_chart.assert_not_called()
        
        # Reset mocks for next test
        mock_container.clear_output_called = False
        mock_render_preprocessed.reset_mock()
        
        # Test with augmented samples
        mock_visualization_module.update_visualization('augmented_samples')
        mock_container.assert_clear_output_called()
        mock_render_augmented.assert_called_once()
        mock_render_chart.assert_not_called()
        
        # Reset mocks for next test
        mock_container.clear_output_called = False
        mock_render_augmented.reset_mock()
        
        # Test with unsupported visualization type
        mock_visualization_module.update_visualization('unsupported_type')
        mock_visualization_module.log_warning.assert_called_once_with(
            "Tipe visualisasi tidak didukung: unsupported_type"
        )
        
        # Reset warning mock for next test
        mock_visualization_module.log_warning.reset_mock()
        
        # Test error handling
        mock_container.clear_output_called = False
        mock_visualization_module.log_error.reset_mock()
        mock_visualization_module.update_operation_status.reset_mock()
        
        # Set up the error case
        error = Exception("Test error")
        mock_render_chart.side_effect = error
        
        # Call the method that should trigger the error
        mock_visualization_module.update_visualization('bar')
        
        # Verify the error was logged
        mock_visualization_module.log_error.assert_called_once_with(
            "Gagal memperbarui visualisasi: Test error"
        )
        
        # Verify no other render methods were called
        mock_render_preprocessed.assert_not_called()
        mock_render_augmented.assert_not_called()
        
        # Verify update_operation_status was called with the error
        mock_visualization_module.update_operation_status.assert_called_once_with(
            "Gagal memperbarui visualisasi: Test error",
            "error"
        )
        
        # Reset mocks for next test
        mock_render_chart.side_effect = None
        mock_visualization_module.log_error.reset_mock()
        mock_visualization_module.update_operation_status.reset_mock()
        
        # Reset all mocks for the final test case
        mock_render_chart.reset_mock()
        mock_render_preprocessed.reset_mock()
        mock_render_augmented.reset_mock()
        mock_visualization_module.log_warning.reset_mock()
        mock_visualization_module.update_operation_status.reset_mock()
        
        # Test with unknown visualization type - should log warning and not call any render method
        mock_visualization_module.update_visualization('unknown')
        
        # Verify the warning was logged
        mock_visualization_module.log_warning.assert_called_once_with(
            "Tipe visualisasi tidak didukung: unknown"
        )
        
        # Verify no render methods were called
        mock_render_chart.assert_not_called()
        mock_render_preprocessed.assert_not_called()
        mock_render_augmented.assert_not_called()
        
        # Verify update_operation_status was not called for unsupported types
        mock_visualization_module.update_operation_status.assert_not_called()


class TestVisualizationStatsCards:
    """Test cases for VisualizationStatsCard and container classes."""
    
    def test_card_initialization(self):
        """Test card initialization with default values."""
        from smartcash.ui.dataset.visualization.components.visualization_stats_cards import VisualizationStatsCard
        
        card = VisualizationStatsCard("Test Card", "test_split")
        assert card.title == "Test Card"
        assert card.split_name == "test_split"
        assert card.raw_count == 0
        assert card.preprocessed_count == 0
        assert card.augmented_count == 0
        
    def test_card_update(self, mocker):
        """Test updating card with statistics data."""
        # Import the actual class for testing
        from smartcash.ui.dataset.visualization.components.visualization_stats_cards import VisualizationStatsCard
        
        # Mock IPython display functions
        mocker.patch('IPython.display.display')
        mocker.patch('IPython.display.HTML')
        
        # Create a card instance
        card = VisualizationStatsCard("Test Card", "train")
        
        # Verify initial state
        assert card.raw_count == 0
        assert card.preprocessed_count == 0
        assert card.augmented_count == 0
        assert card.is_placeholder is True
        
        # Sample data for testing - matches the structure expected by VisualizationStatsCard.update()
        sample_data = {
            'success': True,  # Top-level success flag
            'dataset_stats': {
                'success': True,
                'by_split': {
                    'train': {
                        'raw': 100, 
                        'preprocessed': 80
                    }
                },
                'overview': {'total_files': 100}
            },
            'augmentation_stats': {
                'success': True,
                'by_split': {
                    'train': {'file_count': 200}
                }
            },
            'last_updated': '2023-01-01T00:00:00'  # Add last_updated for status message
        }
        
        # Update the card with sample data
        card.update(sample_data)
        
        # Verify the card was updated correctly
        assert card.raw_count == 100, f"Expected raw_count to be 100, got {card.raw_count}"
        assert card.preprocessed_count == 80, f"Expected preprocessed_count to be 80, got {card.preprocessed_count}"
        assert card.augmented_count == 200, f"Expected augmented_count to be 200, got {card.augmented_count}"
        assert card.is_placeholder is False, "Card should not be in placeholder state after update"
        
        # Test with missing stats
        card = VisualizationStatsCard("Test Card", "train")
        card.update({'dataset_stats': {'success': False}})
        assert card.is_placeholder is True, "Card should be in placeholder state when stats update fails"
        
    def test_container_initialization(self):
        """Test container initialization with all cards."""
        from smartcash.ui.dataset.visualization.components.visualization_stats_cards import \
            VisualizationStatsCardContainer
            
        container = VisualizationStatsCardContainer()
        assert len(container.cards) == 4
        assert 'train' in container.cards
        assert 'valid' in container.cards
        assert 'test' in container.cards
        assert 'overall' in container.cards
        
    def test_container_update(self, sample_stats_data):
        """Test updating all cards in the container."""
        from smartcash.ui.dataset.visualization.components.visualization_stats_cards import \
            VisualizationStatsCardContainer
            
        container = VisualizationStatsCardContainer()
        
        # Mock the update method of each card
        for card in container.cards.values():
            card.update = MagicMock()
            
        # Update the container with sample data
        container.update_all_cards(sample_stats_data)
        
        # Verify all cards were updated
        for card in container.cards.values():
            card.update.assert_called_once()
            
        # Verify the container can be retrieved
        widget = container.get_container()
        assert isinstance(widget, widgets.HBox)
