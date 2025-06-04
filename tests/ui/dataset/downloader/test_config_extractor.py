"""
File: tests/ui/dataset/downloader/test_config_extractor.py
Deskripsi: Test untuk config_extractor.py
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
import ipywidgets as widgets

from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
from smartcash.common.logger import get_logger

class TestDownloaderConfigExtractor(unittest.TestCase):
    """Test case untuk DownloaderConfigExtractor."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        self.logger = get_logger('test_config_extractor')
        self.extractor = DownloaderConfigExtractor  # Class method, tidak perlu instance
        
        self.ui_components = {
            # Roboflow
            'workspace_field': widgets.Text(value='test-workspace'),
            'project_field': widgets.Text(value='test-project'),
            'version_field': widgets.Text(value='1'),
            'api_key_field': widgets.Password(value='test-api-key'),
            'format_dropdown': widgets.Dropdown(
                options=['yolov5pytorch', 'coco', 'pascalvoc'],
                value='yolov5pytorch'
            ),
            
            # Local
            'output_dir_field': widgets.Text(value='./data/raw'),
            'backup_dir_field': widgets.Text(value='./data/backup'),
            'organize_checkbox': widgets.Checkbox(value=True),
            'backup_checkbox': widgets.Checkbox(value=False),
            
            # Advanced
            'retry_attempts_field': widgets.IntText(value=3, min=0, max=10, step=1),
            'timeout_field': widgets.IntText(value=30, min=5, max=300, step=5),
            'chunk_size_field': widgets.IntText(value=8, min=1, max=64, step=1)
        }
    
    def test_extract_roboflow_config(self):
        """Test ekstraksi konfigurasi Roboflow."""
        # Pastikan UI components memiliki field yang diperlukan
        if 'workspace_field' not in self.ui_components:
            self.ui_components['workspace_field'] = widgets.Text(value='test-workspace')
        if 'project_field' not in self.ui_components:
            self.ui_components['project_field'] = widgets.Text(value='test-project')
        if 'version_field' not in self.ui_components:
            self.ui_components['version_field'] = widgets.Text(value='1')
        if 'api_key_field' not in self.ui_components:
            self.ui_components['api_key_field'] = widgets.Password(value='test-api-key')
        if 'format_dropdown' not in self.ui_components:
            self.ui_components['format_dropdown'] = widgets.Dropdown(
                options=['yolov5pytorch', 'coco', 'pascal-voc', 'tfrecord'],
                value='yolov5pytorch'
            )
            
        # Atur nilai UI components
        self.ui_components['workspace_field'].value = 'test-workspace'
        self.ui_components['project_field'].value = 'test-project'
        self.ui_components['version_field'].value = '1'
        self.ui_components['api_key_field'].value = 'test-api-key-1234567890'
        self.ui_components['format_dropdown'].value = 'yolov5pytorch'
        
        # Simpan nilai asli
        original_values = {}
        for key in ['workspace_field', 'project_field', 'version_field', 'api_key_field', 'format_dropdown']:
            original_values[key] = self.ui_components[key].value
        
        try:
            # Set nilai yang diharapkan
            self.ui_components['workspace_field'].value = 'test-workspace'
            self.ui_components['project_field'].value = 'test-project'
            self.ui_components['version_field'].value = '1'
            self.ui_components['api_key_field'].value = 'test-api-key'
            self.ui_components['format_dropdown'].value = 'yolov5pytorch'
            
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            
            # Verifikasi hasil ekstraksi
            self.assertIn('roboflow', config)
            self.assertEqual(config['roboflow']['workspace'], 'test-workspace')
            self.assertEqual(config['roboflow']['project'], 'test-project')
            self.assertEqual(config['roboflow']['version'], '1')
            self.assertEqual(config['roboflow']['api_key'], 'test-api-key')
            self.assertEqual(config['roboflow']['format'], 'yolov5pytorch')
            
        finally:
            # Kembalikan nilai asli
            for key, value in original_values.items():
                self.ui_components[key].value = value
    
    def test_extract_local_config(self):
        """Test ekstraksi konfigurasi lokal."""
        # Pastikan UI components memiliki field yang diperlukan
        required_fields = {
            'output_dir_field': widgets.Text(value='./data/raw'),
            'backup_dir_field': widgets.Text(value='./data/backup'),
            'organize_checkbox': widgets.Checkbox(value=True),
            'backup_checkbox': widgets.Checkbox(value=False)
        }
        
        # Inisialisasi field yang belum ada
        original_values = {}
        for key, widget in required_fields.items():
            if key in self.ui_components:
                original_values[key] = self.ui_components[key].value
            else:
                self.ui_components[key] = widget
        
        try:
            # Test 1: Nilai default
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            
            # Verifikasi hasil ekstraksi
            self.assertIn('local', config)
            local = config['local']
            self.assertEqual(local['output_dir'], './data/raw')
            self.assertEqual(local['backup_dir'], './data/backup')
            self.assertTrue(local['organize_dataset'])
            self.assertFalse(local['backup_enabled'])
            
            # Test 2: Backup enabled
            self.ui_components['backup_checkbox'].value = True
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertTrue(config['local']['backup_enabled'])
            
            # Test 3: Organize dataset disabled
            self.ui_components['organize_checkbox'].value = False
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertFalse(config['local']['organize_dataset'])
            
        finally:
            # Kembalikan nilai asli
            for key, value in original_values.items():
                if key in self.ui_components:
                    self.ui_components[key].value = value
    
    def test_extract_advanced_config(self):
        """Test ekstraksi konfigurasi lanjutan."""
        # Pastikan UI components memiliki field yang diperlukan
        required_fields = {
            'retry_attempts_field': widgets.IntText(value=3, min=1, max=10, step=1),
            'timeout_field': widgets.IntText(value=30, min=10, max=300, step=5),
            'chunk_size_field': widgets.IntText(value=8, min=1, max=1024, step=1)
        }
        
        # Inisialisasi field yang belum ada
        original_values = {}
        for key, widget in required_fields.items():
            if key in self.ui_components:
                original_values[key] = self.ui_components[key].value
            else:
                self.ui_components[key] = widget
        
        try:
            # Test 1: Nilai default
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            
            # Verifikasi hasil ekstraksi
            self.assertIn('advanced', config)
            advanced = config['advanced']
            self.assertEqual(advanced['retry_attempts'], 3)
            self.assertEqual(advanced['timeout_seconds'], 30)
            self.assertEqual(advanced['chunk_size_kb'], 8)
            
            # Test 2: Nilai batas bawah
            self.ui_components['retry_attempts_field'].value = 1
            self.ui_components['timeout_field'].value = 10
            self.ui_components['chunk_size_field'].value = 1
            
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertEqual(config['advanced']['retry_attempts'], 1)
            self.assertEqual(config['advanced']['timeout_seconds'], 10)
            self.assertEqual(config['advanced']['chunk_size_kb'], 1)
            
            # Test 3: Nilai batas atas
            self.ui_components['retry_attempts_field'].value = 10
            self.ui_components['timeout_field'].value = 300
            self.ui_components['chunk_size_field'].value = 1024
            
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertEqual(config['advanced']['retry_attempts'], 10)
            self.assertEqual(config['advanced']['timeout_seconds'], 300)
            self.assertEqual(config['advanced']['chunk_size_kb'], 1024)
            
        finally:
            # Kembalikan nilai asli
            for key, value in original_values.items():
                if key in self.ui_components:
                    self.ui_components[key].value = value
    
    def test_extract_with_missing_fields(self):
        """Test ekstraksi dengan field yang hilang."""
        # Simpan UI components asli
        original_ui_components = self.ui_components.copy()
        
        # Field yang akan diuji
        test_fields = {
            'workspace_field': widgets.Text(value='test-workspace'),
            'project_field': widgets.Text(value='test-project'),
            'version_field': widgets.Text(value='1'),
            'api_key_field': widgets.Password(value='test-api-key-1234567890'),
            'format_dropdown': widgets.Dropdown(
                options=['yolov5pytorch', 'coco', 'pascal-voc', 'tfrecord'],
                value='yolov5pytorch'
            ),
            'output_dir_field': widgets.Text(value='./data/raw'),
            'backup_dir_field': widgets.Text(value='./data/backup'),
            'organize_checkbox': widgets.Checkbox(value=True),
            'backup_checkbox': widgets.Checkbox(value=False),
            'retry_attempts_field': widgets.IntText(value=3, min=1, max=10),
            'timeout_field': widgets.IntText(value=30, min=10, max=300),
            'chunk_size_field': widgets.IntText(value=8, min=1, max=1024)
        }
        
        # Set UI components untuk test
        self.ui_components = {k: v for k, v in test_fields.items()}
        
        try:
            # Test 1: Hapus field yang tidak wajib
            optional_fields = ['backup_dir_field', 'backup_checkbox', 'timeout_field']
            for field in optional_fields:
                if field in self.ui_components:
                    del self.ui_components[field]
            
            # Ekstraksi config harus berhasil meskipun ada field opsional yang hilang
            config = DownloaderConfigExtractor.extract_config(self.ui_components)
            
            # Field yang ada harus tetap terekstrak
            self.assertEqual(config['roboflow']['project'], 'test-project')
            self.assertEqual(config['roboflow']['workspace'], 'test-workspace')
            self.assertEqual(config['local']['output_dir'], './data/raw')
            
            # Field opsional yang tidak ada harus menggunakan nilai default
            self.assertFalse(config['local']['backup_enabled'])  # Default False
            self.assertEqual(config['advanced']['timeout_seconds'], 30)  # Default 30
            
            # Test 2: Hapus field wajib (harus error)
            required_fields = ['workspace_field', 'project_field', 'output_dir_field']
            
            # Simpan field yang akan dihapus untuk dikembalikan nanti
            saved_fields = {}
            for field in required_fields:
                if field in self.ui_components:
                    saved_fields[field] = self.ui_components[field]
                    del self.ui_components[field]
            
            # Pastikan field sudah dihapus
            for field in required_fields:
                self.assertNotIn(field, self.ui_components)
            
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            
            # Pastikan pesan error sesuai
            self.assertIn('tidak boleh kosong', str(context.exception))
            
            # Kembalikan field yang dihapus
            for field, value in saved_fields.items():
                self.ui_components[field] = value
                
        finally:
            # Kembalikan UI components ke keadaan semula
            self.ui_components = original_ui_components
    
    def test_extract_with_none_values(self):
        """Test ekstraksi dengan nilai None atau string kosong."""
        # Pastikan field yang diperlukan ada
        required_fields = {
            'workspace_field': widgets.Text(value='test-workspace'),
            'project_field': widgets.Text(value='test-project'),
            'version_field': widgets.Text(value='1'),
            'api_key_field': widgets.Password(value='test-api-key-1234567890'),
            'format_dropdown': widgets.Dropdown(
                options=['yolov5pytorch', 'coco', 'pascal-voc', 'tfrecord'],
                value='yolov5pytorch'
            ),
            'output_dir_field': widgets.Text(value='./data/raw')
        }
        
        # Simpan nilai asli
        original_values = {}
        for key, widget in required_fields.items():
            if key in self.ui_components:
                original_values[key] = self.ui_components[key].value
            else:
                self.ui_components[key] = widget
        
        try:
            # Test 1: Project string kosong (tidak bisa set None ke Text widget)
            self.ui_components['project_field'].value = ''
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertIn('roboflow.project tidak boleh kosong', str(context.exception))
            
            # Test 2: Workspace string kosong
            self.ui_components['project_field'].value = 'test-project'  # Kembalikan nilai project
            self.ui_components['workspace_field'].value = ''
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertIn('roboflow.workspace tidak boleh kosong', str(context.exception))
            
            # Test 3: API key terlalu pendek
            self.ui_components['workspace_field'].value = 'test-workspace'  # Kembalikan nilai workspace
            self.ui_components['api_key_field'].value = 'short'
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertIn('roboflow.api_key minimal 20 karakter', str(context.exception))
            
            # Test 4: Output dir kosong
            self.ui_components['api_key_field'].value = 'test-api-key-1234567890'  # Kembalikan API key
            self.ui_components['output_dir_field'].value = ''
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertIn('local.output_dir tidak boleh kosong', str(context.exception))
            
            # Test 5: Format tidak valid
            self.ui_components['output_dir_field'].value = './data/raw'  # Kembalikan output dir
            self.ui_components['format_dropdown'].value = 'invalid-format'
            with self.assertRaises(ValueError) as context:
                DownloaderConfigExtractor.extract_config(self.ui_components)
            self.assertIn('Nilai tidak valid untuk roboflow.format', str(context.exception))
            
        finally:
            # Kembalikan nilai asli
            for key, value in original_values.items():
                if key in self.ui_components:
                    self.ui_components[key].value = value
    
    def test_get_default_config(self):
        """Test mendapatkan konfigurasi default."""
        # Dapatkan konfigurasi default
        default_config = DownloaderConfigExtractor.get_default_config()
        
        # Verifikasi struktur dasar
        self.assertIsInstance(default_config, dict)
        self.assertIn('roboflow', default_config)
        self.assertIn('local', default_config)
        self.assertIn('advanced', default_config)
        
        # Verifikasi nilai default untuk roboflow
        roboflow = default_config['roboflow']
        self.assertIn('workspace', roboflow)
        self.assertEqual(roboflow['workspace'], '')
        self.assertIn('project', roboflow)
        self.assertEqual(roboflow['project'], '')
        self.assertIn('version', roboflow)
        self.assertEqual(roboflow['version'], '1')
        self.assertIn('api_key', roboflow)
        self.assertEqual(roboflow['api_key'], '')
        self.assertIn('format', roboflow)
        self.assertEqual(roboflow['format'], 'yolov5pytorch')
        
        # Verifikasi nilai default untuk local
        local = default_config['local']
        self.assertIn('output_dir', local)
        self.assertEqual(local['output_dir'], '')
        self.assertIn('backup_dir', local)
        self.assertEqual(local['backup_dir'], '')
        self.assertIn('organize_dataset', local)
        self.assertTrue(local['organize_dataset'])
        self.assertIn('backup_enabled', local)
        self.assertFalse(local['backup_enabled'])
        
        # Verifikasi nilai default untuk advanced
        advanced = default_config['advanced']
        self.assertIn('retry_attempts', advanced)
        self.assertEqual(advanced['retry_attempts'], 3)
        self.assertIn('timeout_seconds', advanced)
        self.assertEqual(advanced['timeout_seconds'], 30)
        self.assertIn('chunk_size_kb', advanced)
        self.assertEqual(advanced['chunk_size_kb'], 8)
        
        # Verifikasi tipe data
        self.assertIsInstance(roboflow['workspace'], str)
        self.assertIsInstance(roboflow['project'], str)
        self.assertIsInstance(roboflow['version'], str)
        self.assertIsInstance(roboflow['api_key'], str)
        self.assertIsInstance(roboflow['format'], str)
        
        self.assertIsInstance(local['output_dir'], str)
        self.assertIsInstance(local['backup_dir'], str)
        self.assertIsInstance(local['organize_dataset'], bool)
        self.assertIsInstance(local['backup_enabled'], bool)
        
        self.assertIsInstance(advanced['retry_attempts'], int)
        self.assertIsInstance(advanced['timeout_seconds'], int)
        self.assertIsInstance(advanced['chunk_size_kb'], int)
        
        # Verifikasi nilai validasi
        self.assertGreaterEqual(len(roboflow['version']), 1)  # Minimal 1 karakter
        self.assertIn(roboflow['format'], ['yolov5pytorch', 'coco', 'pascal-voc', 'tfrecord'])
        self.assertGreaterEqual(advanced['retry_attempts'], 1)  # Minimal 1
        self.assertLessEqual(advanced['retry_attempts'], 10)  # Maksimal 10
        self.assertGreaterEqual(advanced['timeout_seconds'], 10)  # Minimal 10 detik
        self.assertLessEqual(advanced['timeout_seconds'], 300)  # Maksimal 300 detik
        self.assertGreaterEqual(advanced['chunk_size_kb'], 1)  # Minimal 1 KB
        self.assertLessEqual(advanced['chunk_size_kb'], 1024)  # Maksimal 1024 KB (1MB)


if __name__ == '__main__':
    unittest.main()
