"""
File: smartcash/handlers/ui_handlers/fallback_repository_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler fallback untuk komponen UI klon repository jika modul utama tidak tersedia
"""

import os
import sys
import shutil
import subprocess
from IPython.display import clear_output

def setup_fallback_repository_handlers(ui_components, logger=None):
    """
    Setup handler untuk UI fallback klon repository.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger: Logger untuk mencatat aktivitas (opsional)
        
    Returns:
        Dictionary komponen UI yang telah ditambahkan handler
    """
    # Handler untuk tombol clone
    def on_clone_button_clicked(b):
        ui_components['clone_button'].disabled = True
        ui_components['clone_button'].description = "Cloning..."
        ui_components['status_indicator'].value = "<p>Status: <span style='color: blue'>Cloning...</span></p>"
        
        with ui_components['output_area']:
            clear_output()
            
            try:
                # Get parameters
                url = ui_components['repo_url'].value
                out_dir = ui_components['output_dir'].value
                selected_branch = ui_components['branch'].value
                install_dependencies = ui_components['install_deps'].value
                
                # Log jika logger tersedia
                if logger:
                    logger.info(f"Mencoba mengkloning repository {url} (branch: {selected_branch})")
                
                # Check if output directory already exists
                if os.path.exists(out_dir):
                    print(f"⚠️ Directory '{out_dir}' already exists.")
                    print("Removing existing directory...")
                    try:
                        shutil.rmtree(out_dir)
                        print(f"✅ Existing directory '{out_dir}' removed successfully.")
                    except Exception as e:
                        print(f"❌ Error removing directory: {str(e)}")
                        ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Error</span></p>"
                        ui_components['clone_button'].disabled = False
                        ui_components['clone_button'].description = "Clone Repository"
                        return
                
                # Clone repository
                print(f"🔄 Cloning repository {url} (branch: {selected_branch}) to {out_dir}...")
                
                try:
                    # Run git clone
                    clone_process = subprocess.run(
                        ["git", "clone", "-b", selected_branch, url, out_dir],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    print(f"✅ Repository cloned successfully to {out_dir}")
                    
                    # Add to Python path
                    repo_path = os.path.abspath(out_dir)
                    if repo_path not in sys.path:
                        sys.path.append(repo_path)
                        print(f"✅ Added {repo_path} to Python path")
                    
                    # Install dependencies if requested
                    if install_dependencies:
                        print("\n📦 Installing dependencies...")
                        requirements_file = os.path.join(out_dir, "requirements.txt")
                        
                        if os.path.exists(requirements_file):
                            try:
                                pip_process = subprocess.run(
                                    [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    check=True
                                )
                                print("✅ Dependencies installed successfully")
                                
                                if logger:
                                    logger.success("Dependencies berhasil diinstal")
                                    
                            except subprocess.CalledProcessError as e:
                                print(f"⚠️ Warning: Error installing dependencies: {e.stderr}")
                                ui_components['status_indicator'].value = "<p>Status: <span style='color: orange'>Partially Completed</span></p>"
                                
                                if logger:
                                    logger.warning(f"Error saat install dependencies: {e.stderr}")
                        else:
                            print("⚠️ requirements.txt not found. Skipping dependency installation.")
                    
                    ui_components['status_indicator'].value = "<p>Status: <span style='color: green'>✅ Completed</span></p>"
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error cloning repository: {e.stderr}")
                    ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Failed</span></p>"
                    
                    if logger:
                        logger.error(f"Error saat clone repository: {e.stderr}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Error</span></p>"
                
                if logger:
                    logger.error(f"Error: {str(e)}")
            
            finally:
                # Re-enable button
                ui_components['clone_button'].disabled = False
                ui_components['clone_button'].description = "Clone Repository"
    
    # Attach handler to button
    ui_components['clone_button'].on_click(on_clone_button_clicked)
    
    return ui_components