"""
File: smartcash/ui/dataset/preprocessing/tests/test_cell.py
Deskripsi: Test untuk cell preprocessing
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestPreprocessingCell(unittest.TestCase):
    """Test untuk cell preprocessing."""
    
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer._INITIALIZED_COMPONENTS', None)
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.get_config_manager')
    @patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.get_logger')
    def test_cell_execution(self, mock_get_logger, mock_get_config_manager, *args):
        """Test eksekusi cell preprocessing."""
        # Arrange
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {"preprocessing": {"enabled": True}}
        mock_get_config_manager.return_value = mock_config_manager
        
        # Buat mock UI components
        mock_ui = widgets.VBox()
        mock_components = {'ui': mock_ui}
        
        # Mock semua fungsi yang dibutuhkan dengan pendekatan sederhana
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.create_preprocessing_ui', 
                  return_value=mock_components):
            with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.setup_ui_logger', 
                      return_value=mock_components):
                with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer._setup_core_components', 
                          return_value=mock_components):
                    with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.setup_preprocessing_handlers', 
                              return_value=mock_components):
                        with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.update_status_panel'):
                            with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.log_message'):
                                # Act - Simulasi cell
                                from smartcash.ui.dataset.preprocessing.preprocessing_initializer import initialize_preprocessing_ui
                                
                                # Verifikasi bahwa cell preprocessing berjalan
                                try:
                                    # Coba inisialisasi UI
                                    ui = initialize_preprocessing_ui()
                                    
                                    # Jika berhasil, assert hasilnya
                                    self.assertTrue(isinstance(ui, widgets.VBox))
                                    print("\n✅ Simulasi cell preprocessing berhasil!")
                                    
                                except Exception as e:
                                    self.fail(f"Cell preprocessing gagal dijalankan: {str(e)}")

if __name__ == "__main__":
    unittest.main() 