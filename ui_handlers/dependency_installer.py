"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI instalasi dependencies SmartCash dengan fitur uncheck all dan SmartCash requirements
"""

import os
import sys
import subprocess
import pkg_resources
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from pathlib import Path

def setup_dependency_handlers(ui):
    """Setup handler untuk UI instalasi dependencies dengan fitur check/uncheck all"""
    
    # Handler untuk check all button
    def on_check_all(b):
        # Dapatkan semua checkbox dari grid
        checkboxes = [child for child in ui['checkbox_grid'].children if isinstance(child, widgets.Checkbox)]
        
        # Set semua ke checked
        for checkbox in checkboxes:
            checkbox.value = True
    
    # Handler untuk uncheck all button
    def on_uncheck_all(b):
        # Dapatkan semua checkbox dari grid
        checkboxes = [child for child in ui['checkbox_grid'].children if isinstance(child, widgets.Checkbox)]
        
        # Set semua ke unchecked
        for checkbox in checkboxes:
            checkbox.value = False
    
    # Handler untuk install packages
    def on_install(b):
        with ui['status']:
            clear_output()
            
            # Collect selected packages
            packages = []
            
            if ui['yolov5_req'].value:
                display(HTML("<p>📋 YOLOv5 requirements ditambahkan ke daftar instalasi</p>"))
                packages.append("yolov5_requirements")
            
            if ui['torch_req'].value:
                display(HTML("<p>📋 PyTorch ditambahkan ke daftar instalasi</p>"))
                packages.append("torch_requirements")
            
            if ui['albumentations_req'].value:
                display(HTML("<p>📋 Albumentations ditambahkan ke daftar instalasi</p>"))
                packages.append("albumentations")
            
            if ui['notebook_req'].value:
                display(HTML("<p>📋 Notebook packages ditambahkan ke daftar instalasi</p>"))
                packages.append("notebook_requirements")
            
            if ui['smartcash_req'].value:
                display(HTML("<p>📋 SmartCash requirements ditambahkan ke daftar instalasi</p>"))
                packages.append("smartcash_requirements")
            
            if ui['opencv_req'].value:
                display(HTML("<p>📋 OpenCV ditambahkan ke daftar instalasi</p>"))
                packages.append("opencv-python")
            
            if ui['matplotlib_req'].value:
                display(HTML("<p>📋 Matplotlib ditambahkan ke daftar instalasi</p>"))
                packages.append("matplotlib")
            
            if ui['pandas_req'].value:
                display(HTML("<p>📋 Pandas ditambahkan ke daftar instalasi</p>"))
                packages.append("pandas")
            
            if ui['seaborn_req'].value:
                display(HTML("<p>📋 Seaborn ditambahkan ke daftar instalasi</p>"))
                packages.append("seaborn")
            
            # Add custom packages
            custom = ui['custom_packages'].value.strip()
            if custom:
                custom_packages = [pkg.strip() for pkg in custom.split('\n') if pkg.strip()]
                for pkg in custom_packages:
                    display(HTML(f"<p>📋 Custom package <code>{pkg}</code> ditambahkan ke daftar instalasi</p>"))
                packages.extend(custom_packages)
            
            if not packages:
                display(HTML(
                    """<div style="padding: 10px; background: #fff3cd; color: #856404; border-left: 4px solid #ffc107;">
                        <p><b>⚠️ Warning:</b> Tidak ada package yang dipilih untuk diinstall.</p>
                    </div>"""
                ))
                return
            
            # Show progress bar
            ui['install_progress'].layout.visibility = 'visible'
            ui['install_progress'].value = 0
            ui['install_progress'].max = len(packages)
            
            # Force reinstall flag
            force_reinstall = ui['force_reinstall'].value
            force_flag = "--force-reinstall" if force_reinstall else ""
            
            # Install packages
            display(HTML("<h3>🚀 Memulai instalasi package</h3>"))
            
            for i, package in enumerate(packages):
                ui['install_progress'].value = i
                
                if package == "yolov5_requirements":
                    display(HTML(f"<p>🔄 Menginstall YOLOv5 requirements... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        # Check if yolov5 folder exists
                        if Path("yolov5").exists() and Path("yolov5/requirements.txt").exists():
                            cmd = f"{sys.executable} -m pip install -r yolov5/requirements.txt {force_flag}"
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                display(HTML("<p>✅ YOLOv5 requirements berhasil diinstall</p>"))
                            else:
                                display(HTML(f"<p>❌ Error saat install YOLOv5 requirements:</p><pre>{result.stderr}</pre>"))
                        else:
                            display(HTML("<p>❌ Folder YOLOv5 tidak ditemukan. Pastikan repository YOLOv5 sudah di-clone.</p>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install YOLOv5 requirements: {str(e)}</p>"))
                
                elif package == "torch_requirements":
                    display(HTML(f"<p>🔄 Menginstall PyTorch... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        # Install PyTorch (CPU or CUDA version based on environment)
                        try:
                            import google.colab
                            is_colab = True
                        except ImportError:
                            is_colab = False
                        
                        if is_colab:
                            # Colab already has PyTorch
                            display(HTML("<p>✅ PyTorch sudah terinstall di Google Colab</p>"))
                        else:
                            # Install PyTorch for local environment
                            cmd = f"{sys.executable} -m pip install torch torchvision torchaudio {force_flag}"
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                display(HTML("<p>✅ PyTorch berhasil diinstall</p>"))
                            else:
                                display(HTML(f"<p>❌ Error saat install PyTorch:</p><pre>{result.stderr}</pre>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install PyTorch: {str(e)}</p>"))
                
                elif package == "albumentations":
                    display(HTML(f"<p>🔄 Menginstall Albumentations... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        cmd = f"{sys.executable} -m pip install albumentations {force_flag}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            display(HTML("<p>✅ Albumentations berhasil diinstall</p>"))
                        else:
                            display(HTML(f"<p>❌ Error saat install Albumentations:</p><pre>{result.stderr}</pre>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install Albumentations: {str(e)}</p>"))
                
                elif package == "notebook_requirements":
                    display(HTML(f"<p>🔄 Menginstall Notebook packages... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        cmd = f"{sys.executable} -m pip install ipywidgets tqdm matplotlib {force_flag}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            display(HTML("<p>✅ Notebook packages berhasil diinstall</p>"))
                        else:
                            display(HTML(f"<p>❌ Error saat install Notebook packages:</p><pre>{result.stderr}</pre>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install Notebook packages: {str(e)}</p>"))

                elif package == "smartcash_requirements":
                    display(HTML(f"<p>🔄 Menginstall SmartCash requirements... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        # SmartCash core requirements
                        cmd = f"{sys.executable} -m pip install pyyaml termcolor python-dotenv roboflow {force_flag}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            display(HTML("<p>✅ SmartCash requirements berhasil diinstall</p>"))
                        else:
                            display(HTML(f"<p>❌ Error saat install SmartCash requirements:</p><pre>{result.stderr}</pre>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install SmartCash requirements: {str(e)}</p>"))
                
                else:
                    # Custom package
                    display(HTML(f"<p>🔄 Menginstall {package}... (package {i+1}/{len(packages)})</p>"))
                    
                    try:
                        cmd = f"{sys.executable} -m pip install {package} {force_flag}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            display(HTML(f"<p>✅ Package <code>{package}</code> berhasil diinstall</p>"))
                        else:
                            display(HTML(f"<p>❌ Error saat install {package}:</p><pre>{result.stderr}</pre>"))
                    except Exception as e:
                        display(HTML(f"<p>❌ Exception saat install {package}: {str(e)}</p>"))
            
            # Complete installation
            ui['install_progress'].value = len(packages)
            
            # Show summary
            display(HTML(
                """<div style="padding: 10px; background: #d4edda; color: #155724; border-left: 4px solid #28a745; margin-top: 20px;">
                    <h3 style="margin-top: 0;">✅ Instalasi Selesai</h3>
                    <p>Semua package telah diproses. Gunakan tombol 'Check Installations' untuk memeriksa hasil instalasi.</p>
                </div>"""
            ))
            
            # Hide progress bar after completion (delay for effect)
            time.sleep(1)
            ui['install_progress'].layout.visibility = 'hidden'
    
    # Handler untuk check installations
    def on_check(b):
        with ui['status']:
            clear_output()
            
            display(HTML("<h3>🔍 Checking installed packages</h3>"))
            
            # Check PyTorch
            try:
                import torch
                import torchvision
                display(HTML(f"<p>✅ PyTorch: v{torch.__version__}, CUDA available: {torch.cuda.is_available()}</p>"))
                display(HTML(f"<p>✅ TorchVision: v{torchvision.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ PyTorch: Not installed</p>"))
            
            # Check OpenCV
            try:
                import cv2
                display(HTML(f"<p>✅ OpenCV: v{cv2.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ OpenCV: Not installed</p>"))
            
            # Check Albumentations
            try:
                import albumentations as A
                display(HTML(f"<p>✅ Albumentations: v{A.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ Albumentations: Not installed</p>"))
            
            # Check Notebook requirements
            try:
                import ipywidgets
                display(HTML(f"<p>✅ ipywidgets: v{ipywidgets.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ ipywidgets: Not installed</p>"))
            
            try:
                import tqdm
                display(HTML(f"<p>✅ tqdm: v{tqdm.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ tqdm: Not installed</p>"))
            
            try:
                import matplotlib
                display(HTML(f"<p>✅ matplotlib: v{matplotlib.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ matplotlib: Not installed</p>"))
            
            # Check SmartCash Requirements
            try:
                import yaml
                display(HTML(f"<p>✅ PyYAML: v{yaml.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ PyYAML: Not installed</p>"))
            
            try:
                import termcolor
                display(HTML("<p>✅ termcolor: installed</p>"))
            except ImportError:
                display(HTML("<p>❌ termcolor: Not installed</p>"))
            
            try:
                import dotenv
                display(HTML(f"<p>✅ python-dotenv:installed</p>"))
            except ImportError:
                display(HTML("<p>❌ python-dotenv: Not installed</p>"))

            try:
                import roboflow
                display(HTML("<p>✅ roboflow: installed</p>"))
            except ImportError:
                display(HTML("<p>❌ roboflow: Not installed</p>"))
            
            # Check YOLOv5 requirements (numpy, scipy, etc)
            try:
                import numpy as np
                display(HTML(f"<p>✅ NumPy: v{np.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ NumPy: Not installed</p>"))
            
            try:
                import pandas as pd
                display(HTML(f"<p>✅ Pandas: v{pd.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ Pandas: Not installed</p>"))
            
            try:
                import scipy
                display(HTML(f"<p>✅ SciPy: v{scipy.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ SciPy: Not installed</p>"))
            
            try:
                import seaborn as sns
                display(HTML(f"<p>✅ Seaborn: v{sns.__version__}</p>"))
            except ImportError:
                display(HTML("<p>❌ Seaborn: Not installed</p>"))
            
            # Check for custom packages
            custom = ui['custom_packages'].value.strip()
            if custom:
                display(HTML("<h4>Custom Packages:</h4>"))
                custom_packages = [pkg.strip() for pkg in custom.split('\n') if pkg.strip()]
                
                for pkg in custom_packages:
                    try:
                        # Extract package name (without version)
                        pkg_name = pkg.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        
                        # Try to get distribution
                        dist = pkg_resources.get_distribution(pkg_name)
                        display(HTML(f"<p>✅ {pkg_name}: v{dist.version}</p>"))
                    except pkg_resources.DistributionNotFound:
                        display(HTML(f"<p>❌ {pkg_name}: Not installed</p>"))
                    except Exception as e:
                        display(HTML(f"<p>❓ {pkg_name}: Error checking - {str(e)}</p>"))
    
    # Register handlers
    ui['install_button'].on_click(on_install)
    ui['check_button'].on_click(on_check)
    ui['check_all_button'].on_click(on_check_all)
    ui['uncheck_all_button'].on_click(on_uncheck_all)
    
    return ui