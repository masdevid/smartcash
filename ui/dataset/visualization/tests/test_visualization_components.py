"""
File: smartcash/ui/dataset/visualization/handlers/tests/test_visualization_components.py
Deskripsi: Test untuk komponen UI visualisasi dataset
"""

import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch

class TestDashboardCards(unittest.TestCase):
    """Test untuk komponen dashboard cards"""
    
    @patch('smartcash.ui.dataset.visualization.components.dashboard_cards.widgets')
    def test_create_preprocessing_cards(self, mock_widgets):
        """Test pembuatan preprocessing cards"""
        from smartcash.ui.dataset.visualization.components.dashboard_cards import create_preprocessing_cards
        
        # Setup mock widgets
        mock_card = MagicMock()
        mock_widgets.VBox.return_value = mock_card
        mock_widgets.HBox.return_value = mock_card
        mock_widgets.HTML.return_value = mock_card
        
        # Panggil fungsi
        result = create_preprocessing_cards()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_card)
        
        # Verifikasi widgets dibuat
        self.assertTrue(mock_widgets.VBox.called)
        self.assertTrue(mock_widgets.HBox.called)
        self.assertTrue(mock_widgets.HTML.called)
    
    @patch('smartcash.ui.dataset.visualization.components.dashboard_cards.widgets')
    def test_create_augmentation_cards(self, mock_widgets):
        """Test pembuatan augmentation cards"""
        from smartcash.ui.dataset.visualization.components.dashboard_cards import create_augmentation_cards
        
        # Setup mock widgets
        mock_card = MagicMock()
        mock_widgets.VBox.return_value = mock_card
        mock_widgets.HBox.return_value = mock_card
        mock_widgets.HTML.return_value = mock_card
        
        # Panggil fungsi
        result = create_augmentation_cards()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_card)
        
        # Verifikasi widgets dibuat
        self.assertTrue(mock_widgets.VBox.called)
        self.assertTrue(mock_widgets.HBox.called)
        self.assertTrue(mock_widgets.HTML.called)

class TestSplitStatsCards(unittest.TestCase):
    """Test untuk komponen split stats cards"""
    
    @patch('smartcash.ui.dataset.visualization.components.split_stats_cards.widgets')
    def test_create_split_stats_cards(self, mock_widgets):
        """Test pembuatan split stats cards"""
        from smartcash.ui.dataset.visualization.components.split_stats_cards import create_split_stats_cards
        
        # Setup mock widgets
        mock_card = MagicMock()
        mock_widgets.VBox.return_value = mock_card
        mock_widgets.HBox.return_value = mock_card
        mock_widgets.HTML.return_value = mock_card
        
        # Panggil fungsi
        result = create_split_stats_cards()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_card)
        
        # Verifikasi widgets dibuat
        self.assertTrue(mock_widgets.VBox.called)
        self.assertTrue(mock_widgets.HBox.called)
        self.assertTrue(mock_widgets.HTML.called)

class TestVisualizationTabs(unittest.TestCase):
    """Test untuk komponen visualization tabs"""
    
    @patch('smartcash.ui.dataset.visualization.components.visualization_tabs.widgets')
    def test_create_visualization_tabs(self, mock_widgets):
        """Test pembuatan visualization tabs"""
        from smartcash.ui.dataset.visualization.components.visualization_tabs import create_visualization_tabs
        
        # Setup mock widgets
        mock_tab = MagicMock()
        mock_widgets.Tab.return_value = mock_tab
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.Output.return_value = MagicMock()
        mock_widgets.Button.return_value = MagicMock()
        
        # Panggil fungsi
        result = create_visualization_tabs()
        
        # Verifikasi hasil
        self.assertEqual(result['tabs'], mock_tab)
        
        # Verifikasi tabs dibuat
        self.assertTrue(mock_widgets.Tab.called)
        self.assertTrue(mock_widgets.VBox.called)
        self.assertTrue(mock_widgets.Output.called)
        self.assertTrue(mock_widgets.Button.called)

class TestDashboardComponent(unittest.TestCase):
    """Test untuk komponen dashboard"""
    
    @patch('smartcash.ui.dataset.visualization.components.dashboard_component.widgets')
    @patch('smartcash.ui.dataset.visualization.components.dashboard_component.create_split_stats_cards')
    @patch('smartcash.ui.dataset.visualization.components.dashboard_component.create_preprocessing_cards')
    @patch('smartcash.ui.dataset.visualization.components.dashboard_component.create_augmentation_cards')
    def test_create_dashboard(self, mock_aug_cards, mock_prep_cards, mock_split_cards, mock_widgets):
        """Test pembuatan dashboard"""
        from smartcash.ui.dataset.visualization.components.dashboard_component import create_dashboard
        
        # Setup mock returns
        mock_split_cards.return_value = MagicMock()
        mock_prep_cards.return_value = MagicMock()
        mock_aug_cards.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HBox.return_value = MagicMock()
        mock_widgets.Button.return_value = MagicMock()
        
        # Panggil fungsi
        result = create_dashboard()
        
        # Verifikasi fungsi dipanggil
        mock_split_cards.assert_called_once()
        mock_prep_cards.assert_called_once()
        mock_aug_cards.assert_called_once()
        
        # Verifikasi komponen dibuat
        self.assertTrue(mock_widgets.VBox.called)
        self.assertTrue(mock_widgets.HBox.called)
        self.assertTrue(mock_widgets.Button.called)

class TestMainLayout(unittest.TestCase):
    """Test untuk main layout"""
    
    @patch('smartcash.ui.dataset.visualization.components.main_layout.widgets')
    @patch('smartcash.ui.dataset.visualization.components.main_layout.create_header')
    @patch('smartcash.ui.dataset.visualization.components.main_layout.create_tabs')
    @patch('smartcash.ui.dataset.visualization.components.main_layout.create_status_indicator')
    @patch('smartcash.ui.dataset.visualization.components.main_layout.display')
    def test_create_visualization_layout(self, mock_display, mock_status, mock_tabs, mock_header, mock_widgets):
        """Test pembuatan main layout"""
        from smartcash.ui.dataset.visualization.components.main_layout import create_visualization_layout
        
        # Setup mock returns
        mock_header.return_value = MagicMock()
        mock_tabs.return_value = MagicMock()
        mock_status.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HBox.return_value = MagicMock()
        mock_widgets.Output.return_value = MagicMock()
        mock_widgets.Button.return_value = MagicMock()
        mock_widgets.IntProgress.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        
        # Panggil fungsi
        result = create_visualization_layout()
        
        # Verifikasi fungsi dipanggil
        mock_header.assert_called_once()
        mock_tabs.assert_called_once()
        mock_status.assert_called_once()
        
        # Verifikasi komponen utama dibuat
        self.assertIn('main_container', result)
        self.assertIn('header', result)
        self.assertIn('status', result)
        self.assertIn('refresh_button', result)
        self.assertIn('visualization_components', result)
        
        # Verifikasi tab visualisasi dibuat
        vis_components = result.get('visualization_components', {})
        self.assertIn('distribution_tab', vis_components)
        self.assertIn('split_tab', vis_components)
        self.assertIn('layer_tab', vis_components)
        self.assertIn('bbox_tab', vis_components)
        self.assertIn('heatmap_tab', vis_components)

if __name__ == '__main__':
    unittest.main() 