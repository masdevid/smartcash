"""
File: smartcash/ui/setup/env_config/tests/test_ui_factory.py
Deskripsi: Integration test untuk komponen UI Factory
"""

import unittest
import ipywidgets as widgets
from ipywidgets import Label, Button, HTML, VBox

from smartcash.ui.setup.env_config.components import UIFactory

class TestUIFactory(unittest.TestCase):
    """
    Integration test untuk UIFactory dan komponennya
    """
    
    def test_create_ui_components(self):
        """
        Test integrasi pembuatan komponen UI
        """
        # Buat komponen UI
        ui_components = UIFactory.create_ui_components()
        
        # Test komponen-komponen utama tersedia
        self.assertIn('header', ui_components, "Header tidak tersedia di UI components")
        self.assertIn('setup_button', ui_components, "Setup button tidak tersedia di UI components")
        self.assertIn('status_panel', ui_components, "Status panel tidak tersedia di UI components")
        self.assertIn('log_panel', ui_components, "Log panel tidak tersedia di UI components")
        self.assertIn('log_output', ui_components, "Log output tidak tersedia di UI components")
        self.assertIn('progress_bar', ui_components, "Progress bar tidak tersedia di UI components")
        self.assertIn('progress_message', ui_components, "Progress message tidak tersedia di UI components")
        self.assertIn('ui_layout', ui_components, "UI layout tidak tersedia di UI components")
        
        # Test tipe komponen
        self.assertIsInstance(ui_components['header'], HTML, "Header bukan instance dari HTML")
        self.assertIsInstance(ui_components['setup_button'], Button, "Setup button bukan instance dari Button")
        self.assertIsInstance(ui_components['progress_message'], Label, "Progress message bukan instance dari Label")
        self.assertIsInstance(ui_components['ui_layout'], VBox, "UI layout bukan instance dari VBox")
        
        # Test interaksi antar komponen
        # Verifikasi layout berisi komponen-komponen yang diharapkan
        children = ui_components['ui_layout'].children
        self.assertIn(ui_components['header'], children, "Header tidak termasuk dalam UI layout")
        self.assertIn(ui_components['status_panel'], children, "Status panel tidak termasuk dalam UI layout")
        self.assertIn(ui_components['log_panel'], children, "Log panel tidak termasuk dalam UI layout")
        
    def test_error_ui_components(self):
        """
        Test integrasi pembuatan komponen UI untuk error
        """
        # Buat komponen UI error dengan pesan kustom
        error_message = "Test error message"
        ui_components = UIFactory.create_error_ui_components(error_message)
        
        # Test komponen-komponen utama tersedia
        self.assertIn('header', ui_components, "Header tidak tersedia di UI components")
        self.assertIn('error_alert', ui_components, "Error alert tidak tersedia di UI components")
        self.assertIn('log_panel', ui_components, "Log panel tidak tersedia di UI components")
        self.assertIn('log_output', ui_components, "Log output tidak tersedia di UI components")
        self.assertIn('ui_layout', ui_components, "UI layout tidak tersedia di UI components")
        
        # Test pesan error ada dalam alert
        error_html = ui_components['error_alert'].value
        self.assertIn(error_message, error_html, f"Pesan error '{error_message}' tidak ditemukan dalam error alert")
        
        # Test interaksi antar komponen dalam error layout
        children = ui_components['ui_layout'].children
        self.assertIn(ui_components['header'], children, "Header tidak termasuk dalam error UI layout")
        self.assertIn(ui_components['error_alert'], children, "Error alert tidak termasuk dalam error UI layout")
        self.assertIn(ui_components['log_panel'], children, "Log panel tidak termasuk dalam error UI layout")

if __name__ == '__main__':
    unittest.main() 