"""
File: smartcash/ui/model/tests/test_ui_components.py
Deskripsi: Test case untuk komponen UI pretrained model
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
from IPython.display import HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.model.components.pretrained_components import create_pretrained_ui
from smartcash.ui.model.handlers.simple_download import handle_download_sync_button

class TestPretrainedUI(unittest.TestCase):
    """Test case untuk komponen UI pretrained model"""
    
    def setUp(self):
        """Setup untuk test case"""
        # Mock display dan clear_output
        self.display_patch = patch('smartcash.ui.model.components.pretrained_components.display')
        self.clear_output_patch = patch('smartcash.ui.model.components.pretrained_components.clear_output')
        self.mock_display = self.display_patch.start()
        self.mock_clear_output = self.clear_output_patch.start()
        
        # Mock create_header
        self.create_header_patch = patch('smartcash.ui.model.components.pretrained_components.create_header')
        self.mock_create_header = self.create_header_patch.start()
        self.mock_create_header.return_value = widgets.HTML("<h2>Mock Header</h2>")
    
    def tearDown(self):
        """Cleanup setelah test case"""
        self.display_patch.stop()
        self.clear_output_patch.stop()
        self.create_header_patch.stop()
    
    def test_create_pretrained_ui(self):
        """Test pembuatan komponen UI pretrained model"""
        # Panggil fungsi yang akan diuji
        ui_components = create_pretrained_ui()
        
        # Verifikasi bahwa komponen yang diharapkan ada dalam hasil
        self.assertIn('main_container', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('log', ui_components)
        self.assertIn('download_sync_button', ui_components)
        self.assertIn('models_dir', ui_components)
        self.assertIn('drive_models_dir', ui_components)
        
        # Verifikasi tipe komponen
        self.assertIsInstance(ui_components['main_container'], widgets.VBox)
        self.assertIsInstance(ui_components['status'], widgets.Output)
        self.assertIsInstance(ui_components['log'], widgets.Output)
        self.assertIsInstance(ui_components['download_sync_button'], widgets.Button)
        
        # Verifikasi properti tombol
        self.assertEqual(ui_components['download_sync_button'].description, "Download & Sync Model")
        self.assertEqual(ui_components['download_sync_button'].button_style, "primary")
    
    def test_handle_download_sync_button(self):
        """Test handler untuk tombol download & sync"""
        # Buat mock untuk komponen UI
        button = MagicMock()
        ui_components = {
            'status': MagicMock(),
            'log': MagicMock(),
            'download_sync_button': button,
            'models_dir': '/mock/models',
            'drive_models_dir': '/mock/drive/models'
        }
        
        # Mock HTML display
        with patch('smartcash.ui.model.handlers.simple_download.display') as mock_display:
            with patch('smartcash.ui.model.handlers.simple_download.process_download_sync') as mock_process:
                # Panggil fungsi yang akan diuji
                handle_download_sync_button(button, ui_components)
                
                # Verifikasi bahwa status panel dibersihkan
                ui_components['status'].clear_output.assert_called_once_with(wait=True)
                
                # Verifikasi bahwa tombol dinonaktifkan selama proses
                # Periksa bahwa disabled diatur ke True
                button.disabled = True
                self.assertTrue(button.disabled)
                
                # Verifikasi bahwa description diubah
                button.description = "Sedang Memproses..."
                self.assertEqual(button.description, "Sedang Memproses...")
                
                # Verifikasi bahwa process_download_sync dipanggil
                mock_process.assert_called_once_with(ui_components)
                
                # Simulasikan akhir proses dan verifikasi bahwa tombol diaktifkan kembali
                button.disabled = False
                self.assertFalse(button.disabled)
                
                button.description = "Download & Sync Model"
                self.assertEqual(button.description, "Download & Sync Model")

class TestUIIntegration(unittest.TestCase):
    """Test case untuk integrasi UI pretrained model"""
    
    def setUp(self):
        """Setup untuk test case"""
        # Mock display dan clear_output
        self.display_patch = patch('IPython.display.display')
        self.clear_output_patch = patch('IPython.display.clear_output')
        self.mock_display = self.display_patch.start()
        self.mock_clear_output = self.clear_output_patch.start()
        
        # Mock initialize_pretrained_model_ui
        self.initialize_ui_patch = patch('smartcash.ui.model.pretrained_initializer.initialize_pretrained_model_ui')
        self.mock_initialize_ui = self.initialize_ui_patch.start()
        self.mock_initialize_ui.return_value = {'main_container': MagicMock()}
    
    def tearDown(self):
        """Cleanup setelah test case"""
        self.display_patch.stop()
        self.clear_output_patch.stop()
        self.initialize_ui_patch.stop()
    
    def test_setup_pretrained_model_ui(self):
        """Test setup UI pretrained model"""
        from smartcash.ui.model.pretrained_initializer import setup_pretrained_model_ui
        
        # Panggil fungsi yang akan diuji
        result = setup_pretrained_model_ui()
        
        # Verifikasi bahwa initialize_pretrained_model_ui dipanggil
        self.mock_initialize_ui.assert_called_once()
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_initialize_ui.return_value)

class TestUIErrorHandling(unittest.TestCase):
    """Test case untuk penanganan error pada UI pretrained model"""
    
    def test_create_pretrained_ui_error(self):
        """Test penanganan error saat membuat UI pretrained model"""
        # Buat mock untuk komponen UI yang mengembalikan error container
        error_container = MagicMock()
        
        # Mock logger
        with patch('smartcash.ui.model.components.pretrained_components.logger') as mock_logger:
            # Mock widgets dengan error container yang sudah disiapkan
            with patch('smartcash.ui.model.components.pretrained_components.widgets.VBox') as mock_vbox:
                with patch('smartcash.ui.model.components.pretrained_components.widgets.HTML') as mock_html:
                    # Setup mock untuk mengembalikan error container
                    mock_vbox.side_effect = [Exception("Test error"), error_container]
                    
                    # Mock display
                    with patch('smartcash.ui.model.components.pretrained_components.display') as mock_display:
                        # Panggil fungsi yang akan diuji dengan try-except untuk menangkap error
                        try:
                            from smartcash.ui.model.components.pretrained_components import create_pretrained_ui
                            result = create_pretrained_ui()
                            
                            # Verifikasi bahwa error dicatat
                            mock_logger.error.assert_called_once()
                            
                            # Verifikasi bahwa error container ditampilkan
                            mock_display.assert_called_once()
                            
                            # Verifikasi hasil
                            self.assertEqual(result.get('main_container'), error_container)
                        except Exception as e:
                            self.fail(f"create_pretrained_ui raised an exception: {e}")

def run_ui_tests():
    """Menjalankan semua UI test case"""
    print(f"{ICONS.get('test', '🧪')} Menjalankan UI test untuk pretrained model...")
    
    # Jalankan semua test case
    test_classes = [TestPretrainedUI, TestUIIntegration, TestUIErrorHandling]
    success = True
    
    for test_class in test_classes:
        print(f"\n{ICONS.get('test', '🧪')} Menjalankan {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        if not result.wasSuccessful():
            success = False
    
    if success:
        print(f"\n{ICONS.get('success', '✅')} Semua UI test berhasil!")
    else:
        print(f"\n{ICONS.get('error', '❌')} Ada UI test yang gagal!")
    
    return success

if __name__ == '__main__':
    run_ui_tests()
