"""
File: smartcash/ui/dataset/visualization/tests/test_dashboard_cards.py
Deskripsi: Test untuk komponen dashboard cards visualisasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_status_card, create_split_cards, 
    create_preprocessing_cards, create_augmentation_cards
)
from smartcash.ui.utils.color_utils import get_color_for_status


class TestDashboardCards(unittest.TestCase):
    """Test untuk komponen dashboard cards visualisasi dataset."""
    
    def test_get_color_for_status(self):
        """Test untuk fungsi get_color_for_status."""
        # Test untuk status default
        bg_color, text_color = get_color_for_status('default')
        self.assertEqual(bg_color, '#f5f5f5')
        self.assertEqual(text_color, '#424242')
        
        # Test untuk status preprocessing
        bg_color, text_color = get_color_for_status('preprocessing')
        self.assertEqual(bg_color, '#e3f2fd')
        self.assertEqual(text_color, '#0d47a1')
        
        # Test untuk status augmentation
        bg_color, text_color = get_color_for_status('augmentation')
        self.assertEqual(bg_color, '#e8f5e9')
        self.assertEqual(text_color, '#1b5e20')
    
    def test_create_status_card(self):
        """Test untuk fungsi create_status_card."""
        # Test dengan status default
        card = create_status_card(
            title="Test Card",
            value=10,
            icon="📊",
            status="default",
            description="Test description"
        )
        
        # Verifikasi hasil
        self.assertIsInstance(card, widgets.Box)
        self.assertEqual(len(card.children), 3)
        self.assertIsInstance(card.children[0], widgets.HTML)
        self.assertIsInstance(card.children[1], widgets.HTML)
        self.assertIsInstance(card.children[2], widgets.HTML)
        
        # Verifikasi kelas CSS
        self.assertIn('bg-default', card._dom_classes)
    
    def test_create_split_cards(self):
        """Test untuk fungsi create_split_cards."""
        # Setup data
        split_stats = {
            'train': {'images': 100, 'labels': 90, 'objects': 150},
            'val': {'images': 20, 'labels': 18, 'objects': 30},
            'test': {'images': 10, 'labels': 9, 'objects': 15}
        }
        
        preprocessing_status = {
            'train': True,
            'val': False,
            'test': False
        }
        
        augmentation_status = {
            'train': False,
            'val': False,
            'test': False
        }
        
        # Panggil fungsi
        cards = create_split_cards(split_stats, preprocessing_status, augmentation_status)
        
        # Verifikasi hasil
        self.assertIsInstance(cards, widgets.HBox)
        self.assertEqual(len(cards.children), 3)  # 3 splits
        
        # Verifikasi kelas CSS untuk setiap card
        train_card = cards.children[0]
        val_card = cards.children[1]
        test_card = cards.children[2]
        
        self.assertIn('bg-preprocessing', train_card._dom_classes)  # Preprocessing (biru)
        self.assertIn('bg-default', val_card._dom_classes)    # Default (abu-abu)
        self.assertIn('bg-default', test_card._dom_classes)   # Default (abu-abu)
    
    def test_create_preprocessing_cards(self):
        """Test untuk fungsi create_preprocessing_cards."""
        # Setup data
        preprocessing_stats = {
            'processed_images': 100,
            'filtered_images': 10,
            'normalized_images': 90
        }
        
        # Panggil fungsi
        cards = create_preprocessing_cards(preprocessing_stats)
        
        # Verifikasi hasil
        self.assertIsInstance(cards, widgets.HBox)
        self.assertEqual(len(cards.children), 3)  # 3 cards
        
        # Verifikasi kelas CSS untuk setiap card
        for card in cards.children:
            self.assertIn('bg-preprocessing', card._dom_classes)  # Preprocessing (biru)
    
    def test_create_augmentation_cards(self):
        """Test untuk fungsi create_augmentation_cards."""
        # Setup data
        augmentation_stats = {
            'augmented_images': 50,
            'generated_images': 200,
            'augmentation_types': 5
        }
        
        # Panggil fungsi
        cards = create_augmentation_cards(augmentation_stats)
        
        # Verifikasi hasil
        self.assertIsInstance(cards, widgets.HBox)
        self.assertEqual(len(cards.children), 3)  # 3 cards
        
        # Verifikasi kelas CSS untuk setiap card
        for card in cards.children:
            self.assertIn('bg-augmentation', card._dom_classes)  # Augmentation (hijau)


if __name__ == '__main__':
    unittest.main()
