"""
File: smartcash/ui_handlers/env_config.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI konfigurasi environment SmartCash
"""

import os
import platform
import subprocess
from pathlib import Path
from IPython.display import display, HTML, clear_output

from smartcash.utils.ui_utils import (
    create_info_alert, create_status_indicator, styled_html,
    create_section_title
)
from smartcash.utils.environment_manager import EnvironmentManager

def setup_env_handlers(ui):
    """Setup handler untuk UI konfigurasi environment"""
    
    # Inisialisasi environment manager
    env_manager = EnvironmentManager()
    
    # Deteksi environment
    def detect_environment():
        # Deteksi environment menggunakan EnvironmentManager
        is_colab = env_manager.is_colab
        drive_mounted = env_manager.is_drive_mounted()
        
        # Update UI berdasarkan status
        if is_colab:
            if drive_mounted:
                ui['colab_panel'].value = styled_html(
                    content=f"{create_section_title('☁️ Google Colab Environment', 'Terdeteksi').value}"
                    f"{create_info_alert('Google Drive sudah terhubung di /content/drive/MyDrive', 'success', '✅').value}",
                    style={"padding": "10px", "margin": "10px 0"}
                ).value
                
                # Update tombol
                ui['drive_button'].description = 'Reconnect Google Drive'
                ui['drive_button'].icon = 'refresh'
            else:
                ui['colab_panel'].value = styled_html(
                    content=f"{create_section_title('☁️ Google Colab Environment', 'Terdeteksi').value}"
                    f"{create_info_alert('Project akan dikonfigurasi untuk berjalan di Google Colab. Hubungkan Google Drive untuk menyimpan data.', 'info', 'ℹ️').value}",
                    style={"padding": "10px", "margin": "10px 0"}
                ).value
            
            ui['drive_button'].layout.display = ''
        else:
            ui['colab_panel'].value = styled_html(
                content=f"{create_section_title('💻 Environment Lokal', 'Terdeteksi').value}"
                f"{create_info_alert('Project akan dikonfigurasi untuk berjalan di environment lokal.', 'success', '✅').value}",
                style={"padding": "10px", "margin": "10px 0"}
            ).value
        
        # Display system info
        with ui['status']:
            clear_output()
            
            # Dapatkan informasi environment dari EnvironmentManager
            env_info = env_manager.get_environment_info()
            
            # Tambahkan info system
            system_items = [
                f"<b>Python:</b> {env_info['python_version']}",
                f"<b>OS:</b> {env_info['os_name']} {env_info['os_version']}",
                f"<b>Path:</b> {env_info['cwd']}",
                f"<b>Google Drive:</b> {'Terhubung' if drive_mounted else 'Tidak terhubung'}"
            ]
            
            # Tambahkan info GPU jika tersedia
            if 'gpu_info' in env_info and env_info['gpu_info']:
                system_items.append(f"<b>GPU:</b> {env_info['gpu_info']}")
            
            # Buat HTML untuk system info
            system_info_content = "<ul>" + "".join([f"<li>{item}</li>" for item in system_items]) + "</ul>"
            
            # Tampilkan dengan styled_html
            display(styled_html(
                content=f"{create_section_title('📊 System Information', '').value}{system_info_content}",
                style={"background": "#f8f9fa", "padding": "10px", "margin": "10px 0", "border-radius": "5px"}
            ))
        
        return is_colab
    
    # Google Drive connection handler
    def on_drive_connect(b):
        with ui['status']:
            clear_output()
            # Cek jika Google Drive sudah terhubung
            drive_mounted = env_manager.is_drive_mounted()
            
            if drive_mounted:
                display(create_status_indicator("info", "ℹ️ Google Drive sudah terhubung. Mencoba menghubungkan kembali..."))
            else:
                display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
            
            try:
                # Mount drive menggunakan EnvironmentManager
                mount_result = env_manager.mount_drive()
                
                if not mount_result['success']:
                    display(create_status_indicator("error", f"❌ Gagal menghubungkan ke Google Drive: {mount_result['message']}"))
                    return
                
                # Mendapatkan drive path dari EnvironmentManager
                drive_path = env_manager.get_drive_path('SmartCash')
                
                # Buat direktori SmartCash jika belum ada
                if not drive_path.exists():
                    drive_path.mkdir(parents=True)
                    display(create_status_indicator(
                        "success", f"✅ Direktori {drive_path} berhasil dibuat di Google Drive"
                    ))
                else:
                    display(create_status_indicator(
                        "info", f"ℹ️ Direktori {drive_path} sudah ada di Google Drive"
                    ))
                
                # Membuat symlink dari EnvironmentManager
                symlink_result = env_manager.create_symlink(drive_path, 'SmartCash_Drive')
                if symlink_result['success']:
                    display(create_status_indicator(
                        "success", f"✅ {symlink_result['message']}"
                    ))
                else:
                    display(create_status_indicator(
                        "info", f"ℹ️ {symlink_result['message']}"
                    ))
                
                # Update button
                ui['drive_button'].description = 'Reconnect Google Drive'
                ui['drive_button'].icon = 'refresh'
                
                # Get Drive usage statistics
                drive_stats = env_manager.get_drive_stats()
                if drive_stats['success']:
                    stats = drive_stats['data']
                    storage_info = f"""
                    <h4>💾 Drive Usage Stats</h4>
                    <ul>
                        <li>Total: {stats['total_gb']:.1f} GB</li>
                        <li>Used: {stats['used_gb']:.1f} GB ({stats['usage_percent']:.1f}%)</li>
                        <li>Free: {stats['free_gb']:.1f} GB</li>
                    </ul>
                    """
                    
                    display(styled_html(
                        content=storage_info,
                        style={"background": "#f8f9fa", "padding": "10px", "margin": "10px 0", "border-radius": "5px"}
                    ))
                
                # Success message
                display(create_info_alert(
                    message=f"Google Drive berhasil terhubung. Data akan disimpan di {drive_path}",
                    alert_type="success",
                    icon="✅"
                ))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Gagal terhubung ke Google Drive: {str(e)}"))
    
    # Directory structure setup handler
    def on_dir_setup(b):
        with ui['status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Membuat struktur direktori..."))
            
            # Gunakan EnvironmentManager untuk setup direktori
            setup_result = env_manager.setup_directories(
                base_dirs=['data', 'models', 'runs', 'configs', 'logs', 'results'],
                data_subdirs=['train', 'valid', 'test'],
                runs_subdirs=['train', 'detect'],
                use_drive=env_manager.is_drive_mounted()
            )
            
            if setup_result['success']:
                # Tampilkan info direktori
                display(create_section_title("✅ Struktur Direktori", "Berhasil Dibuat"))
                
                # Daftar direktori utama
                dir_info = """
                <ul>
                    <li><code>data/</code> - Dataset training, validasi, dan testing</li>
                    <li><code>models/</code> - Model yang diexport</li>
                    <li><code>runs/</code> - Hasil training dan deteksi</li>
                    <li><code>configs/</code> - File konfigurasi</li>
                    <li><code>logs/</code> - Log proses</li>
                    <li><code>results/</code> - Hasil evaluasi dan visualisasi</li>
                </ul>
                """
                
                display(styled_html(
                    content=dir_info,
                    style={"padding": "10px", "margin": "10px 0"}
                ))
                
                # Tampilkan daftar direktori yang baru dibuat
                if 'created_dirs' in setup_result and setup_result['created_dirs']:
                    created_dirs = setup_result['created_dirs']
                    dir_list = "".join([f"<li><code>{d}</code></li>" for d in created_dirs])
                    display(create_info_alert(
                        message=f"<b>Direktori baru yang dibuat:</b><ul>{dir_list}</ul>",
                        alert_type="success",
                        icon="📁"
                    ))
            else:
                display(create_status_indicator("error", f"❌ Gagal membuat struktur direktori: {setup_result['message']}"))
    
    # Register handlers
    ui['drive_button'].on_click(on_drive_connect)
    ui['dir_button'].on_click(on_dir_setup)
    
    # Run initial detection
    detect_environment()
    
    return ui