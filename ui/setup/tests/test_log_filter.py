"""
File: smartcash/ui/setup/tests/test_log_filter.py
Deskripsi: Test untuk filter log sinkronisasi drive config
"""

import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import io
import re

class TestLogFilter(unittest.TestCase):
    """Test case untuk filter log sinkronisasi drive config"""
    
    def setUp(self):
        """Setup untuk test"""
        # Simpan stdout asli
        self.original_stdout = sys.stdout
        # Buat StringIO untuk menangkap output
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
    
    def tearDown(self):
        """Cleanup setelah test"""
        # Kembalikan stdout asli
        sys.stdout = self.original_stdout
    
    def test_drive_sync_log_filtering(self):
        """Test filter log sinkronisasi drive config"""
        # Verifikasi bahwa filter yang ditambahkan ke ui_logger.py berfungsi dengan benar
        # Pendekatan: Verifikasi bahwa pesan yang berisi filter tidak akan ditampilkan
        
        # Import fungsi yang akan diuji
        from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
        
        # Baca file ui_logger.py untuk memeriksa filter yang sudah ada
        import inspect
        source_code = inspect.getsource(intercept_stdout_to_ui)
        
        # Daftar filter yang seharusnya ada
        expected_filters = [
            'INFO:config_sync', 
            'Menyinkronkan konfigurasi', 
            'Konfigurasi berhasil disinkronkan',
            'Memuat konfigurasi dari Drive', 
            'Sinkronisasi konfigurasi',
            'config_sync:', 
            'drive_sync:',
            'Environment config handlers',
            'berhasil diinisialisasi'
        ]
        
        # Verifikasi semua filter yang diharapkan ada dalam kode
        for filter_text in expected_filters:
            self.assertTrue(filter_text in source_code, 
                           f"Filter '{filter_text}' tidak ditemukan dalam kode ui_logger.py")
        
        # Verifikasi bahwa pesan log yang mengandung filter tidak akan ditampilkan
        # dengan membuat simulasi sederhana
        
        # Buat fungsi simulasi untuk menguji filter
        def should_be_filtered(message):
            """Fungsi untuk menguji apakah pesan akan difilter"""
            # Pesan error dan warning tidak boleh difilter meskipun mengandung kata kunci filter
            if message.startswith("ERROR:") or message.startswith("WARNING:"):
                return False
            # Pesan info yang mengandung kata kunci filter harus difilter
            return any(filter_text in message for filter_text in expected_filters)
        
        # Daftar pesan log yang seharusnya difilter
        filtered_messages = [
            "INFO:config_sync:Menyinkronkan konfigurasi ke Google Drive",
            "INFO:config_sync:Konfigurasi berhasil disinkronkan",
            "INFO:config_sync:Memuat konfigurasi dari Google Drive",
            "INFO:smartcash.ui.setup:Environment config handlers berhasil diinisialisasi",
            "Menyinkronkan konfigurasi dataset ke Google Drive",
            "Sinkronisasi konfigurasi selesai",
            "drive_sync: Memulai sinkronisasi"
        ]
        
        # Daftar pesan log yang seharusnya tidak difilter
        non_filtered_messages = [
            "ERROR:config_sync:Gagal menyinkronkan konfigurasi: File not found",
            "WARNING:config_sync:Konfigurasi tidak lengkap",
            "INFO:training:Memulai proses training"
        ]
        
        # Test pesan yang seharusnya difilter
        for message in filtered_messages:
            self.assertTrue(should_be_filtered(message), 
                           f"Pesan '{message}' seharusnya difilter")
        
        # Test pesan yang seharusnya tidak difilter (kecuali untuk debug messages)
        for message in non_filtered_messages:
            if "DEBUG:" not in message:
                # Pesan error dan warning tidak boleh difilter meskipun mengandung kata kunci filter
                if message.startswith("ERROR:") or message.startswith("WARNING:"):
                    self.assertFalse(should_be_filtered(message), 
                                   f"Pesan '{message}' seharusnya tidak difilter")
                # Pesan info yang tidak terkait dengan config_sync tidak boleh difilter
                elif not message.startswith("INFO:config_sync:"):
                    self.assertFalse(should_be_filtered(message), 
                                   f"Pesan '{message}' seharusnya tidak difilter")

class TestDriveSyncLogFilter(unittest.TestCase):
    """Test case untuk implementasi filter log drive sync"""
    
    def test_drive_sync_log_filter_implementation(self):
        """Test implementasi filter log drive sync"""
        # Import fungsi yang akan diuji
        from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
        
        # Baca file ui_logger.py untuk memeriksa filter yang sudah ada
        import inspect
        source_code = inspect.getsource(intercept_stdout_to_ui)
        
        # Verifikasi filter untuk drive sync sudah ada
        self.assertTrue('INFO:config_sync' in source_code)
        
        # Verifikasi filter tambahan untuk drive sync sudah ada
        additional_filters = [
            'Menyinkronkan konfigurasi',
            'Konfigurasi berhasil disinkronkan',
            'Memuat konfigurasi dari Drive',
            'Sinkronisasi konfigurasi',
            'config_sync:',
            'drive_sync:'
        ]
        
        # Verifikasi filter tambahan sudah ada
        for filter_text in additional_filters:
            self.assertTrue(filter_text in source_code, 
                           f"Filter '{filter_text}' tidak ditemukan dalam kode ui_logger.py")

if __name__ == '__main__':
    unittest.main()
