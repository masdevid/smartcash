import unittest
from unittest.mock import patch, MagicMock
from smartcash.ui.dataset.download.handlers import handle_download_click
from smartcash.ui.dataset.download.components.download_component import create_download_main_ui
from smartcash.ui.dataset.download.utils.download_utils import extract_download_config
from smartcash.common.dataset.manager import AbstractDatasetManager

class TestDownloadIntegration(unittest.TestCase):
    """
    Test integrasi untuk modul dataset download
    """

    @patch('smartcash.common.dataset.manager.DatasetManager')
    def test_download_flow_sukses(self, mock_dataset_manager):
        """
        Test alur download sukses dari awal hingga akhir
        """
        # 1. Setup UI components
        ui_components = create_download_main_ui()
        
        # 2. Setup mock DatasetManager
        mock_manager = MagicMock(spec=AbstractDatasetManager)
        mock_dataset_manager.return_value = mock_manager
        mock_manager.download_from_roboflow.return_value = True
        
        # 3. Simulasikan pengisian form
        ui_components['project_input'].value = 'currency-detection'
        ui_components['version_input'].value = '5'
        ui_components['api_key_input'].value = 'valid_api_key'
        
        # 4. Panggil handler download
        handle_download_click(ui_components, mock_manager)
        
        # 5. Verifikasi
        mock_manager.download_from_roboflow.assert_called_once()
        self.assertIn('✅', ui_components['status_panel'].value)

    @patch('smartcash.common.dataset.manager.DatasetManager')
    def test_download_flow_gagal(self, mock_dataset_manager):
        """
        Test alur download gagal dan penanganan errornya
        """
        # 1. Setup UI components
        ui_components = create_download_main_ui()
        
        # 2. Setup mock DatasetManager
        mock_manager = MagicMock()
        mock_dataset_manager.return_value = mock_manager
        mock_manager.download_from_roboflow.side_effect = Exception('Koneksi error')
        
        # 3. Simulasikan pengisian form
        ui_components['project_input'].value = 'invalid-project'
        ui_components['version_input'].value = '99'
        
        # 4. Panggil handler download
        handle_download_click(ui_components)
        
        # 5. Verifikasi
        self.assertIn('❌', ui_components['status_panel'].value)
        self.assertIn('Koneksi error', ui_components['log_output'].value)

    def test_ui_config_mapping(self):
        """
        Test pemetaan UI ke konfigurasi
        """
        # 1. Setup UI components
        ui_components = create_download_main_ui()
        
        # 2. Set nilai UI
        ui_components['project_input'].value = 'smart-cash'
        ui_components['workers_slider'].value = 8
        
        # 3. Ekstrak konfigurasi
        config = extract_download_config(ui_components)
        
        # 4. Verifikasi pemetaan
        self.assertEqual(config['project'], 'smart-cash')
        self.assertEqual(config['workers'], 8)

    @patch('smartcash.ui.dataset.download.handlers.download_action_handlers.update_status_panel')
    def test_error_handling(self, mock_update):
        """
        Test penanganan error pada komponen kritis
        """
        # 1. Setup UI tanpa komponen kritis
        ui_components = {'log_output': MagicMock()}
        
        # 2. Panggil handler dengan UI tidak lengkap
        with self.assertRaises(KeyError):
            handle_download_click(ui_components)
        
        # 3. Verifikasi error handling
        mock_update.assert_called()

if __name__ == '__main__':
    unittest.main()
