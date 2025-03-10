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
        drive_mounted = env_manager.is_drive_mounted
        
        # Update UI berdasarkan status
        if is_colab:
            if drive_mounted:
                ui['colab_panel'].value = styled_html(
                    content=f"{create_section_title('☁️ Google Colab Environment', 'Terdeteksi').value}"
                    f"{create_info_alert('Google Drive sudah terhubung di ' + str(env_manager.drive_path), 'success', '✅').value}",
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
            sys_info = env_manager.get_system_info()
            
            # Format system items
            system_items = [
                f"<b>Python:</b> {sys_info['python_version'].split()[0]}",
                f"<b>Base Directory:</b> {sys_info['base_dir']}",
                f"<b>Google Drive:</b> {'Terhubung' if sys_info['drive_mounted'] else 'Tidak terhubung'}"
            ]
            
            # Tambahkan info GPU jika tersedia
            if 'cuda_available' in sys_info and sys_info['cuda_available']:
                system_items.append(f"<b>GPU:</b> {sys_info['cuda_device']} ({sys_info['cuda_memory']:.1f} GB)")
            
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
            drive_mounted = env_manager.is_drive_mounted
            
            if drive_mounted:
                display(create_status_indicator("info", "ℹ️ Google Drive sudah terhubung. Mencoba menghubungkan kembali..."))
            else:
                display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
            
            try:
                # Mount drive menggunakan EnvironmentManager
                success, message = env_manager.mount_drive()
                
                if not success:
                    display(create_status_indicator("error", f"❌ Gagal menghubungkan ke Google Drive: {message}"))
                    return
                
                display(create_status_indicator("success", message))
                
                # Create symlinks
                symlink_stats = env_manager.create_symlinks()
                if symlink_stats['created'] > 0:
                    display(create_status_indicator(
                        "success", f"✅ {symlink_stats['created']} symlinks berhasil dibuat"
                    ))
                elif symlink_stats['existing'] > 0:
                    display(create_status_indicator(
                        "info", f"ℹ️ {symlink_stats['existing']} symlinks sudah ada"
                    ))
                
                # Update button
                ui['drive_button'].description = 'Reconnect Google Drive'
                ui['drive_button'].icon = 'refresh'
                
                # Get system info with Drive stats
                sys_info = env_manager.get_system_info()
                if 'cuda_memory' in sys_info:
                    # Format disk usage stats jika tersedia
                    try:
                        import shutil
                        drive_usage = shutil.disk_usage('/content/drive')
                        total_gb = drive_usage.total / (1024**3)
                        used_gb = drive_usage.used / (1024**3)
                        free_gb = drive_usage.free / (1024**3)
                        usage_percent = (used_gb / total_gb) * 100
                        
                        storage_info = f"""
                        <h4>💾 Drive Usage Stats</h4>
                        <ul>
                            <li>Total: {total_gb:.1f} GB</li>
                            <li>Used: {used_gb:.1f} GB ({usage_percent:.1f}%)</li>
                            <li>Free: {free_gb:.1f} GB</li>
                        </ul>
                        """
                        
                        display(styled_html(
                            content=storage_info,
                            style={"background": "#f8f9fa", "padding": "10px", "margin": "10px 0", "border-radius": "5px"}
                        ))
                    except:
                        pass
                
                # Success message
                display(create_info_alert(
                    message=f"Google Drive berhasil terhubung. Data akan disimpan di {env_manager.drive_path}",
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
            dir_stats = env_manager.setup_directories(use_drive=env_manager.is_drive_mounted)
            
            if dir_stats['error'] == 0:
                # Tampilkan info direktori
                display(create_section_title("✅ Struktur Direktori", "Berhasil Dibuat"))
                
                # Daftar direktori utama
                dir_info = """
                <ul>
                    <li><code>data/</code> - Dataset training, validasi, dan testing</li>
                    <li><code>configs/</code> - File konfigurasi</li>
                    <li><code>runs/</code> - Hasil training dan weights</li>
                    <li><code>logs/</code> - Log proses</li>
                    <li><code>exports/</code> - Model yang diexport</li>
                </ul>
                """
                
                display(styled_html(
                    content=dir_info,
                    style={"padding": "10px", "margin": "10px 0"}
                ))
                
                # Tampilkan statistik pembuatan direktori
                stats_message = f"<b>Statistik:</b> {dir_stats['created']} direktori baru dibuat, {dir_stats['existing']} sudah ada sebelumnya."
                display(create_info_alert(
                    message=stats_message,
                    alert_type="success",
                    icon="📁"
                ))
                
                # Tampilkan directory tree jika ada direktori baru yang dibuat
                if dir_stats['created'] > 0:
                    display(HTML("<h4>🌲 Struktur Direktori:</h4>"))
                    tree_html = env_manager.get_directory_tree(max_depth=2)
                    display(HTML(tree_html))
            else:
                display(create_status_indicator("error", f"❌ Terjadi error saat membuat {dir_stats['error']} direktori"))
    
    # Register handlers
    ui['drive_button'].on_click(on_drive_connect)
    ui['dir_button'].on_click(on_dir_setup)
    
    # Run initial detection
    detect_environment()
    
    return ui