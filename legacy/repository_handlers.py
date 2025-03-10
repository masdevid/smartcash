"""
File: smartcash/smartcash/ui_handlers/repository_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen repository, menangani proses clone dan setup repository
           dengan pendekatan yang lebih aman untuk subprocesses dan UI decoupling.
"""

import os
import sys
import shutil
import subprocess
from IPython.display import clear_output
from typing import Dict, Any, Optional, Tuple, List

from handlers.ui_handlers.common_utils import validate_ui_components

def run_subprocess_safely(
    command: List[str], 
    logger: Optional[Any] = None, 
    error_message: str = "Error executing command"
) -> Tuple[bool, str]:
    """
    Run a subprocess command safely with proper error handling.
    
    Args:
        command: List of command arguments
        logger: Optional logger object
        error_message: Custom error message prefix
        
    Returns:
        Tuple of (success, output_or_error_message)
    """
    try:
        # Run without check=True to handle errors manually
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check return code manually
        if process.returncode != 0:
            error_detail = f"{error_message}: {process.stderr}"
            if logger:
                logger.error(error_detail)
            return False, error_detail
        
        return True, process.stdout
    except Exception as e:
        error_detail = f"{error_message}: {str(e)}"
        if logger:
            logger.error(error_detail)
        return False, error_detail

def on_clone_button_clicked(ui_components: Dict[str, Any], logger: Optional[Any] = None) -> None:
    """
    Handler untuk tombol clone repository dengan error handling yang lebih baik.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
        logger: Optional logger untuk logging aktivitas
    """
    # Validate required UI components
    required_components = [
        'clone_button', 'repo_url_input', 'output_dir_input', 
        'branch_dropdown', 'install_deps_checkbox', 
        'status_indicator', 'error_area', 'output_area'
    ]
    
    if not validate_ui_components(ui_components, required_components, logger):
        return
    
    # Disable tombol selama proses
    ui_components['clone_button'].disabled = True
    ui_components['clone_button'].description = "Cloning..."
    
    # Update status
    ui_components['status_indicator'].value = "<p>Status: <span style='color: blue'>Cloning...</span></p>"
    ui_components['error_area'].value = ""
    ui_components['error_area'].layout.display = 'none'
    
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
                    error_message = f"❌ Error removing directory: {str(e)}"
                    print(error_message)
                    ui_components['error_area'].value = f"<div style='color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px;'>{error_message}</div>"
                    ui_components['error_area'].layout.display = 'block'
                    ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Error</span></p>"
                    ui_components['clone_button'].disabled = False
                    ui_components['clone_button'].description = "Clone Repository"
                    return
            
            # Clone repository
            print(f"🔄 Cloning repository {repo_url} (branch: {branch}) to {output_dir}...")
            
            # Run git clone command with safer approach
            clone_command = ["git", "clone", "-b", branch, repo_url, output_dir]
            success, result = run_subprocess_safely(
                clone_command,
                logger=logger,
                error_message="Error cloning repository"
            )
            
            if not success:
                print(f"❌ {result}")
                ui_components['error_area'].value = f"<div style='color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px;'>{result}</div>"
                ui_components['error_area'].layout.display = 'block'
                ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Failed</span></p>"
                ui_components['clone_button'].disabled = False
                ui_components['clone_button'].description = "Clone Repository"
                return
                
            # Show clone output
            if result:
                print("Git output:")
                print(result)
                
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
                    # Run pip install safely
                    pip_command = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
                    pip_success, pip_result = run_subprocess_safely(
                        pip_command,
                        logger=logger,
                        error_message="Error installing dependencies"
                    )
                    
                    if not pip_success:
                        print(f"❌ {pip_result}")
                        ui_components['error_area'].value = f"<div style='color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;'>Repository cloned but dependencies installation failed: {pip_result}</div>"
                        ui_components['error_area'].layout.display = 'block'
                        ui_components['status_indicator'].value = "<p>Status: <span style='color: orange'>Partially Completed</span></p>"
                    else:
                        print("✅ Dependencies installed successfully.")
                        ui_components['status_indicator'].value = "<p>Status: <span style='color: green'>✅ Completed</span></p>"
                else:
                    print("⚠️ requirements.txt not found. Skipping dependency installation.")
                    ui_components['status_indicator'].value = "<p>Status: <span style='color: orange'>Completed with Warnings</span></p>"
                    ui_components['error_area'].value = "<div style='color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;'>requirements.txt not found. Skipping dependency installation.</div>"
                    ui_components['error_area'].layout.display = 'block'
            else:
                ui_components['status_indicator'].value = "<p>Status: <span style='color: green'>✅ Completed</span></p>"
            
            pass
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            print(error_message)
            ui_components['error_area'].value = f"<div style='color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px;'>{error_message}</div>"
            ui_components['error_area'].layout.display = 'block'
            ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Error</span></p>"
            if logger:
                logger.error(error_message)
        finally:
            # Re-enable button
            ui_components['clone_button'].disabled = False
            ui_components['clone_button'].description = "Clone Repository"

def on_custom_repo_checkbox_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk checkbox penggunaan custom repository.
    
    Args:
        change: Change event dari observe
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
    """
    if 'repo_url_input' not in ui_components:
        return
        
    # Enable/disable repository URL input based on checkbox
    ui_components['repo_url_input'].disabled = not change['new']

def setup_repository_handlers(ui_components: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Setup semua event handler untuk UI repository dengan validasi komponen yang lebih baik.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_repository_ui()
        logger: Optional logger untuk logging aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Validate required components
    required_components = ['clone_button', 'custom_repo_checkbox']
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        if logger:
            logger.error(f"Missing required UI components: {', '.join(missing_components)}")
        return ui_components
    
    # Setup handler untuk tombol clone
    ui_components['clone_button'].on_click(
        lambda b: on_clone_button_clicked(ui_components, logger)
    )
    
    # Setup handler untuk checkbox custom repo
    ui_components['custom_repo_checkbox'].observe(
        lambda change: on_custom_repo_checkbox_changed(change, ui_components),
        names='value'
    )
    
    return ui_components