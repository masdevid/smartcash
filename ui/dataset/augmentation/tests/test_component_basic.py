"""
File: smartcash/ui/dataset/augmentation/tests/test_component_basic.py
Deskripsi: Test dasar untuk komponen UI augmentasi dataset
"""

import unittest
import os

class TestAugmentationComponentBasic(unittest.TestCase):
    """Test dasar untuk komponen UI augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        self.component_path = '/Users/masdevid/Projects/smartcash/smartcash/ui/dataset/augmentation/components/augmentation_component.py'
        self.action_buttons_path = '/Users/masdevid/Projects/smartcash/smartcash/ui/dataset/augmentation/components/action_buttons.py'
        
        # Pastikan file ada
        self.assertTrue(os.path.exists(self.component_path), f"File tidak ditemukan: {self.component_path}")
        self.assertTrue(os.path.exists(self.action_buttons_path), f"File tidak ditemukan: {self.action_buttons_path}")
    
    def test_no_redundant_buttons(self):
        """Test tidak ada redundansi tombol dalam komponen augmentasi"""
        # Baca isi file komponen
        with open(self.component_path, 'r') as f:
            component_content = f.read()
        
        # Verifikasi bahwa hanya mengimpor action_buttons dari komponen khusus augmentasi
        self.assertIn(
            "from smartcash.ui.dataset.augmentation.components.action_buttons import create_action_buttons", 
            component_content
        )
        
        # Verifikasi bahwa tidak mengimpor action_buttons dari komponen umum
        self.assertNotIn(
            "from smartcash.ui.components.action_buttons import create_action_buttons", 
            component_content
        )
        
        # Verifikasi bahwa memanggil create_action_buttons tanpa parameter
        self.assertIn("action_buttons = create_action_buttons()", component_content)
        
        # Verifikasi bahwa menggunakan referensi tombol yang benar
        self.assertIn("'augment_button': action_buttons['augment_button']", component_content)
        self.assertIn("'stop_button': action_buttons['stop_button']", component_content)
        self.assertIn("'reset_button': action_buttons['reset_button']", component_content)
        self.assertIn("'cleanup_button': action_buttons['cleanup_button']", component_content)
        self.assertIn("'save_button': action_buttons['save_button']", component_content)

if __name__ == '__main__':
    unittest.main()
