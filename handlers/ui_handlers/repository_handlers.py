"""
File: handlers/ui_handlers/repository_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen repository, menangani proses clone dan setup repository.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from IPython.display import clear_output

def on_clone_button_clicked(ui_components):
    """
    Handler untuk tombol clone repository.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
    """
    # Disable tombol selama proses
    ui_components['clone_button'].disabled = True
    ui_components['clone_button'].description = "Cloning..."
    
    # Update status
    ui_components['status_indicator'].value = "<p>Status: <span style='color: blue'>Cloning...</span></p>"
    
    # Clear output area
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Get parameters
            repo_url = ui_components['repo_url_input'].value
            output_dir = ui_components['output_dir_input'].value
            branch = ui_components['branch_dropdown'].value
            install_deps = ui_components['install_deps_checkbox'].value
            
            # Check if output directory already exists
            if os.path.exists(output_dir):
                print(f"⚠️ Directory '{output_dir}' already exists.")
                print("Removing existing directory...")
                try:
                    shutil.rmtree(output_dir)
                    print(f"✅ Existing directory '{output_dir}' removed successfully.")
                except Exception as e:
                    print(f"❌ Error removing directory: {str(e)}")
                    raise
            
            # Clone repository
            print(f"🔄 Cloning repository {repo_url} (branch: {branch}) to {output_dir}...")
            
            # Run git clone command
            clone_command = ["git", "clone", "-b", branch, repo_url, output_dir]
            clone_process = subprocess.run(
                clone_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Show clone output
            if clone_process.stdout:
                print("Git output:")
                print(clone_process.stdout)
                
            print(f"✅ Repository cloned successfully to {output_dir}")
            
            # Setup Python path
            repo_path = os.path.abspath(output_dir)
            if repo_path not in sys.path:
                sys.path.append(repo_path)
                print(f"✅ Added {repo_path} to Python path.")
            
            # Install dependencies if requested
            if install_deps:
                print("\n📦 Installing dependencies...")
                
                # Create requirements.txt file path
                requirements_file = os.path.join(output_dir, "requirements.txt")
                
                # Check if requirements.txt exists
                if os.path.exists(requirements_file):
                    # Run pip install
                    pip_command = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
                    pip_process = subprocess.run(
                        pip_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Check for errors
                    if pip_process.returncode != 0:
                        print("❌ Error installing dependencies:")
                        print(pip_process.stderr)
                    else:
                        print("✅ Dependencies installed successfully.")
                else:
                    print("⚠️ requirements.txt not found. Skipping dependency installation.")
            
            # Update status to success
            ui_components['status_indicator'].value = "<p>Status: <span style='color: green'>✅ Completed</span></p>"
            
        except Exception as e:
            print(f"❌ Error cloning repository: {str(e)}")
            # Update status to error
            ui_components['status_indicator'].value = f"<p>Status: <span style='color: red'>❌ Error: {str(e)}</span></p>"
        
        finally:
            # Re-enable button
            ui_components['clone_button'].disabled = False
            ui_components['clone_button'].description = "Clone Repository"

def on_custom_repo_checkbox_changed(change, ui_components):
    """
    Handler untuk checkbox penggunaan custom repository.
    
    Args:
        change: Change event dari observe
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
    """
    # Enable/disable repository URL input based on checkbox
    ui_components['repo_url_input'].disabled = not change['new']

def setup_repository_handlers(ui_components):
    """
    Setup semua event handler untuk UI repository.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Setup handler untuk tombol clone
    ui_components['clone_button'].on_click(
        lambda b: on_clone_button_clicked(ui_components)
    )
    
    # Setup handler untuk checkbox custom repo
    ui_components['custom_repo_checkbox'].observe(
        lambda change: on_custom_repo_checkbox_changed(change, ui_components),
        names='value'
    )
    
    return ui_components