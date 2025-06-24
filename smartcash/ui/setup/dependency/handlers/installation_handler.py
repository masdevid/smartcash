"""
File: smartcash/ui/setup/dependency/handlers/installation_handler.py
Deskripsi: Handler untuk instalasi packages dengan parallel processing
"""

from typing import Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from smartcash.ui.setup.dependency.utils import (
    get_selected_packages, install_single_package, 
    update_status_panel, with_button_context,
    show_progress_tracker_safe, complete_operation_with_message
)

def setup_installation_handler(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Setup installation handler dengan parallel processing"""
    
    def handle_installation():
        """Handle package installation dengan progress tracking"""
        logger = ui_components.get('logger')
        
        with with_button_context(ui_components, 'install_button'):
            try:
                # Extract packages to install
                selected_packages = get_selected_packages(ui_components.get('package_selector', {}))
                custom_packages = _get_custom_packages(ui_components)
                all_packages = selected_packages + custom_packages
                
                if not all_packages:
                    update_status_panel(ui_components, "⚠️ Tidak ada packages yang dipilih", "warning")
                    return
                
                # Get installation settings
                config = _extract_installation_config(ui_components)
                
                # Start installation
                update_status_panel(ui_components, f"🚀 Memulai instalasi {len(all_packages)} packages...", "info")
                show_progress_tracker_safe(ui_components, "Package Installation")
                
                if logger:
                    logger.info(f"📦 Installing {len(all_packages)} packages: {', '.join(all_packages[:5])}{'...' if len(all_packages) > 5 else ''}")
                
                # Install packages dengan parallel processing
                results = _install_packages_parallel(all_packages, config, ui_components)
                
                # Process results
                success_count = sum(1 for r in results if r['success'])
                failed_packages = [r['package'] for r in results if not r['success']]
                
                if success_count == len(all_packages):
                    complete_operation_with_message(ui_components, f"✅ Berhasil install {success_count} packages!")
                    if logger:
                        logger.info(f"🎉 Installation completed successfully: {success_count}/{len(all_packages)}")
                else:
                    update_status_panel(ui_components, f"⚠️ {success_count}/{len(all_packages)} berhasil, {len(failed_packages)} gagal", "warning")
                    if logger:
                        logger.warning(f"⚠️ Installation partially failed. Failed packages: {', '.join(failed_packages)}")
                
            except Exception as e:
                update_status_panel(ui_components, f"❌ Installation error: {str(e)}", "error")
                if logger:
                    logger.error(f"❌ Installation failed: {str(e)}")
    
    # Setup button handler
    install_button = ui_components.get('install_button')
    if install_button:
        install_button.on_click(lambda b: handle_installation())
    
    return {
        'handle_installation': handle_installation,
        'install_packages_parallel': lambda packages, config: _install_packages_parallel(packages, config, ui_components)
    }

def _get_custom_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Extract custom packages dari textarea"""
    try:
        widget = ui_components.get('custom_packages')
        if widget and widget.value.strip():
            return [pkg.strip() for pkg in widget.value.strip().split('\n') if pkg.strip()]
        return []
    except:
        return []

def _extract_installation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract installation config dari UI atau gunakan defaults"""
    from ..handlers.config_extractor import extract_dependency_config
    
    try:
        full_config = extract_dependency_config(ui_components)
        return full_config.get('installation', {
            'parallel_workers': 3,
            'timeout': 300,
            'max_retries': 2,
            'retry_delay': 1.0
        })
    except:
        return {
            'parallel_workers': 3,
            'timeout': 300,
            'max_retries': 2,
            'retry_delay': 1.0
        }

def _install_packages_parallel(packages: List[str], config: Dict[str, Any], ui_components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Install packages dengan parallel processing"""
    from smartcash.ui.setup.dependency.utils import update_progress_step
    
    results = []
    max_workers = min(config.get('parallel_workers', 3), len(packages))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_package = {
            executor.submit(install_single_package, pkg, config): pkg 
            for pkg in packages
        }
        
        completed = 0
        for future in as_completed(future_to_package):
            package = future_to_package[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                # Update progress
                progress = int((completed / len(packages)) * 100)
                update_progress_step(ui_components, "overall", progress, f"Installing {package}")
                
            except Exception as e:
                results.append({
                    'package': package,
                    'success': False,
                    'error': str(e)
                })
    
    return results