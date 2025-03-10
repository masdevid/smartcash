"""
File: smartcash/ui_handlers/project_setup.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI setup project SmartCash
"""

import os
import time
import platform
import subprocess
import ipywidgets as widgets
from pathlib import Path
from IPython.display import display, clear_output, HTML
from smartcash.utils.ui_utils import create_info_alert, create_status_indicator

def setup_project_handlers(ui):
    """Setup handler UI project setup SmartCash"""
    
    # Handler untuk perubahan environment
    def on_env_change(change):
        with ui['env_status']:
            clear_output()
            if change['new'] == 'Google Colab':
                ui['colab_connect_button'].layout.display = ''
                display(create_info_alert("Google Colab terdeteksi. Koneksi ke Drive diperlukan.", "info", "☁️"))
            else:
                ui['colab_connect_button'].layout.display = 'none'
                display(create_info_alert(f"Environment lokal terdeteksi. Path: {Path.cwd()}", "info", "💻"))
    
    # Registrasi handler environment
    ui['env_type'].observe(on_env_change, 'value')
    
    # Fungsi deteksi environment
    def detect_env():
        try:
            import google.colab
            ui['env_type'].value = 'Google Colab'
        except ImportError:
            ui['env_type'].value = 'Local'
    
    # Fungsi untuk mengeksekusi command dengan feedback
    def run_command(cmd, status_output, success_msg, error_prefix="Error: "):
        with status_output:
            try:
                display(create_status_indicator("info", f"🔄 Executing: {cmd}"))
                result = subprocess.run(cmd, shell=True, check=True, 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                display(create_status_indicator("success", f"✅ {success_msg}"))
                return True, result.stdout
            except subprocess.CalledProcessError as e:
                display(create_status_indicator("error", f"❌ {error_prefix}{e.stderr}"))
                return False, e.stderr
    
    # Handler untuk tombol clone repository
    def on_clone(b):
        with ui['repo_status']:
            clear_output()
            
            # Clone YOLOv5
            yolo_url = ui['repo_url'].value
            yolo_repo = yolo_url.split("/")[-1].replace(".git", "")
            success = False
            
            if Path(yolo_repo).exists():
                display(create_status_indicator("warning", f"⚠️ Repository {yolo_repo} sudah ada"))
            else:
                success, _ = run_command(
                    f"git clone {yolo_url}",
                    ui['repo_status'],
                    f"YOLOv5 berhasil di-clone ke {yolo_repo}"
                )
            
            # Clone SmartCash
            smartcash_url = ui['smartcash_url'].value
            smartcash_repo = smartcash_url.split("/")[-1].replace(".git", "")
            
            if Path(smartcash_repo).exists():
                display(create_status_indicator("warning", f"⚠️ Repository {smartcash_repo} sudah ada"))
            else:
                success, _ = run_command(
                    f"git clone {smartcash_url}",
                    ui['repo_status'],
                    f"SmartCash berhasil di-clone ke {smartcash_repo}"
                )
            
            # Update overall status
            with ui['overall_status']:
                clear_output()
                display(create_info_alert(
                    "Repository berhasil di-clone! Langkah selanjutnya: setup environment", 
                    "success", "🎉"
                ))
    
    # Registrasi handler tombol clone
    ui['clone_button'].on_click(on_clone)
    
    # Handler untuk koneksi ke Google Drive di Colab
    def on_colab_connect(b):
        with ui['env_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
            
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                
                # Buat symlink ke direktori project di Drive jika ada
                drive_path = Path('/content/drive/MyDrive/SmartCash')
                if drive_path.exists():
                    if not Path('SmartCash_Drive').exists():
                        os.symlink(drive_path, 'SmartCash_Drive')
                    display(create_status_indicator("success", "✅ Symlink ke SmartCash di Drive dibuat"))
                
                # Update status
                display(create_status_indicator("success", "✅ Google Drive berhasil terhubung"))
                
                with ui['overall_status']:
                    clear_output()
                    display(create_info_alert(
                        "Google Drive berhasil terhubung! Langkah selanjutnya: install dependencies", 
                        "success", "🎉"
                    ))
            except Exception as e:
                display(create_status_indicator("error", f"❌ Gagal menghubungkan ke Drive: {str(e)}"))
    
    # Registrasi handler tombol connect
    ui['colab_connect_button'].on_click(on_colab_connect)
    
    # Handler untuk instalasi dependencies
    def on_install(b):
        with ui['deps_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menginstall dependencies..."))
            
            # Parse packages
            pkgs = [p.strip() for p in ui['required_packages'].value.split('\n') if p.strip()]
            
            # Install requirements YOLOv5 jika ada
            yolo_path = Path('yolov5')
            if yolo_path.exists() and (yolo_path / 'requirements.txt').exists():
                success, _ = run_command(
                    "pip install -r yolov5/requirements.txt",
                    ui['deps_status'],
                    "✅ YOLOv5 requirements berhasil diinstall"
                )
            
            # Install additional packages
            for pkg in pkgs:
                success, _ = run_command(
                    f"pip install {pkg}",
                    ui['deps_status'],
                    f"✅ {pkg.split('>=')[0]} berhasil diinstall"
                )
            
            # Update overall status
            with ui['overall_status']:
                clear_output()
                display(create_info_alert("Setup selesai! SmartCash siap digunakan.", "success", "🎉"))
                
                # Display system info
                display(HTML(f"""
                    <div style='background: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 5px;'>
                        <h4>📊 System Information</h4>
                        <ul>
                            <li><b>Python:</b> {platform.python_version()}</li>
                            <li><b>OS:</b> {platform.system()} {platform.release()}</li>
                            <li><b>Environment:</b> {ui['env_type'].value}</li>
                            <li><b>Packages:</b> {len(pkgs)} additional packages</li>
                        </ul>
                    </div>
                """))
    
    # Registrasi handler tombol install
    ui['install_button'].on_click(on_install)
    
    # Auto-detect environment saat inisialisasi
    detect_env()
    
    return ui