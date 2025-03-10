"""
File: smartcash/ui_handlers/env_config.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI konfigurasi environment SmartCash yang menggunakan EnvironmentManager
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from IPython.display import display, HTML, clear_output

def setup_env_handlers(ui):
    """Setup handler untuk UI konfigurasi environment"""
    
    # Coba import EnvironmentManager
    try:
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.logger import get_logger
        logger = get_logger("env_config")
        env_manager = EnvironmentManager(logger=logger)
        using_env_manager = True
    except ImportError:
        print("Info: EnvironmentManager tidak tersedia, menggunakan implementasi default.")
        env_manager = None
        using_env_manager = False
    
    # Deteksi environment
    def detect_environment():
        is_colab = False
        
        if using_env_manager:
            is_colab = env_manager.is_colab
            
            # Display message based on environment
            if is_colab:
                ui['colab_panel'].value = """
                    <div style="padding: 10px; background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #0c5460;">☁️ Google Colab Terdeteksi</h3>
                        <p>Project akan dikonfigurasi untuk berjalan di Google Colab.</p>
                    </div>
                """
                ui['drive_button'].layout.display = ''
            else:
                ui['colab_panel'].value = """
                    <div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #155724;">💻 Environment Lokal Terdeteksi</h3>
                        <p>Project akan dikonfigurasi untuk berjalan di environment lokal.</p>
                    </div>
                """
                
            # Get system info from environment manager
            system_info = env_manager.get_system_info()
            
            with ui['status']:
                clear_output()
                info_html = f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h4>📊 System Information</h4>
                    <ul>
                        <li><b>Python:</b> {system_info.get('python_version', 'Unknown')}</li>
                        <li><b>Base Directory:</b> {system_info.get('base_dir', 'Unknown')}</li>
                        <li><b>CUDA Available:</b> {'Yes' if system_info.get('cuda_available', False) else 'No'}</li>
                    </ul>
                </div>
                """
                display(HTML(info_html))
        else:
            # Fallback implementation
            try:
                import google.colab
                is_colab = True
                ui['colab_panel'].value = """
                    <div style="padding: 10px; background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #0c5460;">☁️ Google Colab Terdeteksi</h3>
                        <p>Project akan dikonfigurasi untuk berjalan di Google Colab.</p>
                    </div>
                """
                ui['drive_button'].layout.display = ''
            except ImportError:
                is_colab = False
                ui['colab_panel'].value = """
                    <div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #155724;">💻 Environment Lokal Terdeteksi</h3>
                        <p>Project akan dikonfigurasi untuk berjalan di environment lokal.</p>
                    </div>
                """
            
            # Display system info
            with ui['status']:
                clear_output()
                system_info = f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h4>📊 System Information</h4>
                    <ul>
                        <li><b>Python:</b> {platform.python_version()}</li>
                        <li><b>OS:</b> {platform.system()} {platform.release()}</li>
                        <li><b>Path:</b> {Path.cwd()}</li>
                    </ul>
                </div>
                """
                display(HTML(system_info))
        
        return is_colab
    
    # Google Drive connection handler
    def on_drive_connect(b):
        with ui['status']:
            clear_output()
            
            if using_env_manager:
                display(HTML('<p>🔄 Menghubungkan ke Google Drive menggunakan EnvironmentManager...</p>'))
                success, message = env_manager.mount_drive()
                
                if success:
                    # Create symlinks
                    display(HTML('<p>🔄 Membuat symlink ke Google Drive...</p>'))
                    symlink_stats = env_manager.create_symlinks()
                    
                    display(HTML(
                        f"""<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #155724;">✅ Google Drive Terhubung</h3>
                            <p>Data akan disimpan di {env_manager.drive_path}</p>
                            <p>Symlinks dibuat: {symlink_stats['created']}, yang sudah ada: {symlink_stats['existing']}</p>
                        </div>"""
                    ))
                    
                    # Display directory tree
                    display(HTML("<h4>📂 Struktur direktori:</h4>"))
                    tree_html = env_manager.get_directory_tree(max_depth=2)
                    display(HTML(tree_html))
                    
                else:
                    display(HTML(
                        f"""<div style="padding: 10px; background: #f8d7da; border-left: 4px solid #721c24; color: #721c24; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #721c24;">❌ Gagal Terhubung ke Google Drive</h3>
                            <p>Error: {message}</p>
                        </div>"""
                    ))
            else:
                # Fallback implementation
                try:
                    display(HTML('<p>🔄 Menghubungkan ke Google Drive...</p>'))
                    from google.colab import drive
                    drive.mount('/content/drive')
                    
                    # Create SmartCash directory in Drive if needed
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    if not drive_path.exists():
                        drive_path.mkdir(parents=True)
                        display(HTML(
                            f'<p>✅ Direktori <code>{drive_path}</code> berhasil dibuat di Google Drive</p>'
                        ))
                    else:
                        display(HTML(
                            f'<p>ℹ️ Direktori <code>{drive_path}</code> sudah ada di Google Drive</p>'
                        ))
                        
                    # Create symlink
                    if not Path('SmartCash_Drive').exists():
                        os.symlink(drive_path, 'SmartCash_Drive')
                        display(HTML(
                            '<p>✅ Symlink <code>SmartCash_Drive</code> berhasil dibuat</p>'
                        ))
                    else:
                        display(HTML(
                            '<p>ℹ️ Symlink <code>SmartCash_Drive</code> sudah ada</p>'
                        ))
                        
                    display(HTML(
                        """<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #155724;">✅ Google Drive Terhubung</h3>
                            <p>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></p>
                        </div>"""
                    ))
                    
                except Exception as e:
                    display(HTML(
                        f"""<div style="padding: 10px; background: #f8d7da; border-left: 4px solid #721c24; color: #721c24; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #721c24;">❌ Gagal Terhubung ke Google Drive</h3>
                            <p>Error: {str(e)}</p>
                        </div>"""
                    ))
    
    # Directory structure setup handler
    def on_dir_setup(b):
        with ui['status']:
            clear_output()
            
            if using_env_manager:
                display(HTML('<p>🔄 Membuat struktur direktori dengan EnvironmentManager...</p>'))
                
                # Setup directories using EnvironmentManager
                use_drive = env_manager.is_drive_mounted
                stats = env_manager.setup_directories(use_drive=use_drive)
                
                # Display stats
                display(HTML(
                    f"""<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #155724;">✅ Struktur Direktori Berhasil Dibuat</h3>
                        <p>Direktori baru: {stats['created']}, sudah ada: {stats['existing']}, error: {stats.get('error', 0)}</p>
                    </div>"""
                ))
                
                # Display directory tree
                display(HTML("<h4>📂 Struktur direktori yang dibuat:</h4>"))
                tree_html = env_manager.get_directory_tree(max_depth=3)
                display(HTML(tree_html))
                
            else:
                # Fallback implementation
                # Create necessary directories
                dirs = [
                    'data/train/images', 'data/train/labels',
                    'data/valid/images', 'data/valid/labels',
                    'data/test/images', 'data/test/labels',
                    'models', 'runs/train', 'runs/detect',
                    'configs', 'logs', 'results'
                ]
                
                display(HTML('<p>🔄 Membuat struktur direktori...</p>'))
                
                created = 0
                existing = 0
                for d in dirs:
                    path = Path(d)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        created += 1
                    else:
                        existing += 1
                    
                display(HTML(
                    f"""<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #155724;">✅ Struktur Direktori Berhasil Dibuat</h3>
                        <p>Direktori baru: {created}, sudah ada: {existing}</p>
                        <ul>
                            <li><code>data/</code> - Dataset training, validasi, dan testing</li>
                            <li><code>models/</code> - Model yang diexport</li>
                            <li><code>runs/</code> - Hasil training dan deteksi</li>
                            <li><code>configs/</code> - File konfigurasi</li>
                            <li><code>logs/</code> - Log proses</li>
                            <li><code>results/</code> - Hasil evaluasi dan visualisasi</li>
                        </ul>
                    </div>"""
                ))
    
    # Register handlers
    ui['drive_button'].on_click(on_drive_connect)
    ui['dir_button'].on_click(on_dir_setup)
    
    # Run initial detection
    detect_environment()
    
    return ui