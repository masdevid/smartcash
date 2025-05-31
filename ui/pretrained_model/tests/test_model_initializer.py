"""
File: smartcash/ui/pretrained_model/tests/test_model_initializer.py
Deskripsi: Test case untuk model_initializer dengan pendekatan DRY dan one-liner style
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.pretrained_model.pretrained_initializer import PretrainedModelInitializer, initialize_pretrained_model_ui

class TestModelInitializer(unittest.TestCase):
    """Test case untuk model_initializer dengan CommonInitializer"""
    
    def setUp(self): 
        """One-liner setup untuk test case"""
        self.patches = [patch(f'smartcash.ui.pretrained_model.pretrained_initializer.{mod}') for mod in ['display', 'get_environment_manager']]
        self.mocks = {p.attribute.split('.')[-1]: p.start() for p in self.patches}
        self.initializer = PretrainedModelInitializer(); self.env = MagicMock(); self.config = {}
    
    def tearDown(self): 
        """One-liner teardown setelah test case"""
        [p.stop() for p in self.patches]; del self.initializer; del self.env; del self.config
    
    def test_create_ui_components(self):
        """Test _create_ui_components dengan komponen yang diharapkan termasuk progress tracking dan log accordion"""
        with patch('smartcash.ui.pretrained_model.components.pretrained_components.create_pretrained_ui') as mock_create_ui:
            # Mock return value dengan komponen progress tracking dan log accordion
            mock_create_ui.return_value = {
                'main_container': MagicMock(), 'status': MagicMock(), 'log': MagicMock(), 'download_sync_button': MagicMock(),
                'progress_bar': MagicMock(), 'progress_label': MagicMock(), 'log_accordion': MagicMock(),
                'container': MagicMock(), 'reset_progress': MagicMock()
            }
            ui_components = self.initializer._create_ui_components(self.config, self.env)
            mock_create_ui.assert_called_once()
            
            # Verifikasi environment dan config disimpan
            self.assertEqual(ui_components['env'], self.env)
            self.assertEqual(ui_components['config'], self.config)
            
            # Verifikasi komponen progress tracking dan log accordion ada
            self.assertIn('progress_bar', ui_components)
            self.assertIn('progress_label', ui_components)
            self.assertIn('log_accordion', ui_components)
    
    def test_setup_module_handlers(self):
        """Test _setup_module_handlers memanggil fungsi yang benar"""
        ui_components = {'main_container': MagicMock(), 'status': MagicMock(), 'log': MagicMock(), 'download_sync_button': MagicMock()}
        with patch('smartcash.ui.pretrained_model.handlers.setup_handlers.setup_model_handlers') as mock_setup_handlers:
            with patch('smartcash.ui.pretrained_model.handlers.setup_handlers.setup_model_cleanup_handler') as mock_setup_cleanup:
                mock_setup_handlers.return_value = ui_components
                mock_setup_cleanup.return_value = ui_components
                result = self.initializer._setup_module_handlers(ui_components, self.config, self.env)
                mock_setup_handlers.assert_called_once_with(ui_components, self.env, self.config)
                mock_setup_cleanup.assert_called_once()
                self.assertEqual(result, ui_components)
    
    def test_get_default_config(self):
        """Test _get_default_config mengembalikan nilai yang diharapkan"""
        config = self.initializer._get_default_config()
        self.assertIn('models_dir', config); self.assertIn('drive_models_dir', config); self.assertIn('models', config)
        self.assertIn('yolov5s', config['models']); self.assertIn('efficientnet-b4', config['models'])

    def test_get_critical_components(self):
        """Test _get_critical_components mengembalikan daftar yang tepat"""
        components = self.initializer._get_critical_components()
        [self.assertIn(comp, components) for comp in ['main_container', 'status', 'log', 'download_sync_button']]
    
    def test_initialize_model_ui(self):
        """Test fungsi initialize_pretrained_model_ui entry point"""
        with patch('smartcash.ui.pretrained_model.pretrained_initializer.PretrainedModelInitializer.initialize') as mock_init:
            mock_init.return_value = {'main_container': MagicMock()}
            result = initialize_pretrained_model_ui()
            mock_init.assert_called_once(); self.assertEqual(result, mock_init.return_value)

class TestSetupHandlers(unittest.TestCase):
    """Test case untuk setup_handlers dengan one-liner style"""
    
    def test_setup_model_handlers(self):
        """Test setup_model_handlers mendaftarkan handler dengan benar"""
        from smartcash.ui.pretrained_model.handlers.setup_handlers import setup_model_handlers
        ui_components = {'download_sync_button': MagicMock(), 'status': MagicMock(), 'log': MagicMock(),
                         'progress_bar': MagicMock(), 'progress_label': MagicMock()}
        
        # Gunakan implementasi asli, tidak perlu mock handler
        result = setup_model_handlers(ui_components)
        
        # Verifikasi handler download didaftarkan
        ui_components['download_sync_button'].on_click.assert_called_once()
        
        # Verifikasi reset_progress_bar didaftarkan
        self.assertIn('reset_progress_bar', result)
        self.assertTrue(callable(result['reset_progress_bar']))
        
        # Verifikasi komponen UI dikembalikan
        self.assertEqual(result, ui_components)
            
    def test_reset_progress_bar(self):
        """Test reset_progress_bar memperbarui progress bar dan label dengan parameter show_progress"""
        from smartcash.ui.pretrained_model.handlers.setup_handlers import reset_progress_bar
        ui_components = {
            'progress_bar': MagicMock(layout=MagicMock()),
            'progress_label': MagicMock(layout=MagicMock()),
            'status_widget': MagicMock(layout=MagicMock()),
            'tracker': MagicMock(show=MagicMock(), hide=MagicMock())
        }
        
        # Test dengan show_progress=True (default)
        reset_progress_bar(ui_components, 42, "Sedang memproses")
        
        # Verifikasi progress bar dan label diperbarui dengan benar
        self.assertEqual(ui_components['progress_bar'].value, 42)
        self.assertEqual(ui_components['progress_bar'].max, 100)
        self.assertEqual(ui_components['progress_label'].value, "Sedang memproses")
        self.assertEqual(ui_components['progress_bar'].layout.visibility, 'visible')
        
        # Test dengan show_progress=False
        # Reset mock untuk tracker.hide
        ui_components['tracker'].hide.reset_mock()
        
        # Tambahkan reset_all untuk memicu pemanggilan hide
        ui_components['reset_all'] = MagicMock()
        
        reset_progress_bar(ui_components, 0, "Memeriksa model", show_progress=False)
        
        # Verifikasi progress bar disembunyikan tapi label tetap terlihat
        self.assertEqual(ui_components['progress_bar'].layout.visibility, 'hidden')
        self.assertEqual(ui_components['progress_label'].layout.visibility, 'visible')
        self.assertEqual(ui_components['status_widget'].layout.visibility, 'visible')
        
        # Verifikasi reset_all dipanggil dan hide dipanggil
        ui_components['reset_all'].assert_called_once()
        ui_components['tracker'].hide.assert_called_once()
    
    def test_setup_model_cleanup_handler(self):
        """Test setup_model_cleanup_handler mendaftarkan cleanup function"""
        from smartcash.ui.pretrained_model.handlers.setup_handlers import setup_model_cleanup_handler
        ui_components = {}; result = setup_model_cleanup_handler(ui_components)
        self.assertIn('cleanup', result); self.assertTrue(callable(result['cleanup']))

def run_model_initializer_tests():
    """Menjalankan test untuk model_initializer"""
    print(f"{ICONS.get('test', '🧪')} Menjalankan test untuk model_initializer...")
    test_classes = [TestModelInitializer, TestSetupHandlers]
    success = True
    
    for test_class in test_classes:
        print(f"\n{ICONS.get('test', '🧪')} Menjalankan {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class); runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite); success = success and result.wasSuccessful()
    
    print(f"\n{ICONS.get('success' if success else 'error', '✅' if success else '❌')} {'Semua' if success else 'Ada'} test {'berhasil' if success else 'yang gagal'}!")
    return success

if __name__ == '__main__':
    run_model_initializer_tests()
