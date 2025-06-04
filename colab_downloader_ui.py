#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: colab_downloader_ui.py
Deskripsi: Script untuk menampilkan dan menggunakan UI downloader di Google Colab
"""

import sys
import traceback
from IPython.display import display, HTML
import ipywidgets as widgets

def tampilkan_downloader_ui():
    """
    Menampilkan UI downloader dengan penanganan error yang lebih baik.
    
    Returns:
        Dict: UI components jika berhasil, None jika gagal
    """
    print("🔍 Memulai inisialisasi UI downloader...")
    
    try:
        # Import modul yang diperlukan
        from smartcash.common.logger import get_logger
        logger = get_logger('downloader_colab')
        
        # Tampilkan header
        header = HTML(
            """
            <div style="background-color: #4285f4; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h2 style="margin: 0;">📥 SmartCash Dataset Downloader</h2>
                <p style="margin: 5px 0 0 0;">Unduh dataset dari Roboflow untuk pelatihan model deteksi mata uang</p>
            </div>
            """
        )
        display(header)
        
        # Import fungsi initialize_downloader_ui
        from smartcash.ui.dataset.downloader.downloader_init import initialize_downloader_ui
        
        logger.info("📦 Mengimpor fungsi initialize_downloader_ui berhasil")
        
        # Inisialisasi UI downloader
        logger.info("🚀 Menjalankan initialize_downloader_ui...")
        ui_components = initialize_downloader_ui()
        
        if ui_components is None:
            logger.error("❌ initialize_downloader_ui mengembalikan None")
            _tampilkan_error("Gagal menginisialisasi UI downloader")
            return None
            
        # Validasi hasil
        if not isinstance(ui_components, dict):
            logger.error(f"❌ Hasil bukan dictionary, tipe: {type(ui_components)}")
            _tampilkan_error(f"Hasil inisialisasi UI downloader bukan dictionary: {type(ui_components)}")
            return None
            
        # Cek keberadaan komponen penting
        required_keys = ['ui', 'workspace_field', 'project_field', 'version_field', 
                        'api_key_field', 'download_button', 'validate_button']
        
        missing_keys = [key for key in required_keys if key not in ui_components]
        
        if missing_keys:
            logger.warning(f"⚠️ Komponen yang hilang: {', '.join(missing_keys)}")
            
            # Auto-fix untuk komponen yang hilang
            logger.info("🔧 Mencoba auto-fix untuk komponen yang hilang...")
            
            # Import komponen yang diperlukan
            from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
            from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons
            from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
            
            # Buat komponen dasar jika perlu
            if any(key in missing_keys for key in ['workspace_field', 'project_field', 'version_field', 'api_key_field']):
                logger.info("🔧 Membuat form fields...")
                config_handler = DownloaderConfigHandler()
                default_config = config_handler.get_default_config()
                form_components = create_form_fields(default_config)
                
                for key in ['workspace_field', 'project_field', 'version_field', 'api_key_field']:
                    if key in missing_keys and key in form_components:
                        ui_components[key] = form_components[key]
                        logger.info(f"✅ Berhasil menambahkan {key}")
            
            # Buat action buttons jika perlu
            if any(key in missing_keys for key in ['download_button', 'validate_button', 'quick_validate_button']):
                logger.info("🔧 Membuat action buttons...")
                action_components = create_action_buttons()
                
                for key in ['download_button', 'validate_button', 'quick_validate_button', 'save_button', 'reset_button']:
                    if key in missing_keys and key in action_components:
                        ui_components[key] = action_components[key]
                        logger.info(f"✅ Berhasil menambahkan {key}")
        
        # Cek UI component
        if 'ui' not in ui_components:
            logger.error("❌ Komponen UI tidak ditemukan")
            _tampilkan_error("Komponen UI tidak ditemukan dalam hasil inisialisasi")
            return None
            
        logger.success("✅ Semua komponen UI penting tersedia")
        
        # Setup download handlers jika belum ada
        if 'download_handlers' not in ui_components:
            try:
                logger.info("🔧 Mencoba setup download handlers...")
                from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
                
                # Jalankan setup handlers
                handlers_result = setup_download_handlers(ui_components)
                
                if handlers_result and isinstance(handlers_result, dict) and 'download_handlers' in handlers_result:
                    logger.success("✅ Setup handlers berhasil")
                    ui_components.update(handlers_result)
                else:
                    logger.warning(f"⚠️ Hasil setup handlers tidak sesuai format: {handlers_result}")
                    
            except Exception as e:
                logger.error(f"❌ Error saat setup handlers: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Tampilkan UI
        logger.info("🖥️ Menampilkan UI downloader...")
        display(ui_components['ui'])
        
        # Tampilkan instruksi penggunaan
        _tampilkan_instruksi()
        
        # Tampilkan informasi komponen yang tersedia
        komponen = [k for k in ui_components.keys() if k != 'ui']
        logger.info(f"📋 Komponen yang tersedia: {', '.join(komponen)}")
        
        return ui_components
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        _tampilkan_error(f"Error saat menampilkan UI downloader: {str(e)}")
        return None

def _tampilkan_error(pesan):
    """Menampilkan pesan error dengan format yang baik."""
    error_html = HTML(
        f"""
        <div style="background-color: #ffebee; color: #c62828; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="margin-top: 0;">❌ Error</h3>
            <p>{pesan}</p>
            <p>Silakan jalankan cell ini kembali atau hubungi pengembang jika masalah berlanjut.</p>
        </div>
        """
    )
    display(error_html)

def _tampilkan_instruksi():
    """Menampilkan instruksi penggunaan UI downloader."""
    instruksi = HTML(
        """
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #2e7d32; margin-top: 0;">📝 Instruksi Penggunaan</h3>
            <ol>
                <li>Masukkan <b>Workspace</b>, <b>Project</b>, dan <b>Version</b> dari dataset Roboflow</li>
                <li>Masukkan <b>API Key</b> Roboflow Anda</li>
                <li>Klik tombol <b>Validate</b> untuk memvalidasi parameter</li>
                <li>Jika valid, klik tombol <b>Download</b> untuk mengunduh dataset</li>
                <li>Gunakan tombol <b>Reset</b> untuk mengatur ulang form</li>
            </ol>
            <p><b>Catatan:</b> Pastikan API Key Roboflow Anda valid dan memiliki akses ke project yang ditentukan.</p>
        </div>
        """
    )
    display(instruksi)

if __name__ == "__main__":
    # Jalankan fungsi
    ui_components = tampilkan_downloader_ui()
