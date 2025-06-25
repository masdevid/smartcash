"""
File: smartcash/ui/setup/dependency/handlers/installation_handler.py

Package installation handler with parallel processing.

This module provides functionality for installing Python packages with
parallel processing and progress tracking.
"""

# Standard library imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# Absolute imports
from smartcash.common.logger import get_logger
from smartcash.ui.setup.dependency.utils.package.installer import install_single_package
from smartcash.ui.setup.dependency.utils.ui.state import (
    update_status_panel,
    show_progress_tracker_safe,
    complete_operation_with_message,
    update_progress_step,
    with_button_context
)
from smartcash.ui.setup.dependency.utils.ui.utils import get_selected_packages

# Type aliases
UIComponents = Dict[str, Any]
PackageList = List[Dict[str, Any]]
InstallationConfig = Dict[str, Any]
InstallationResult = Dict[str, Any]

def setup_installation_handler(ui_components: UIComponents) -> Dict[str, Callable]:
    """Setup installation handler with parallel processing.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Dictionary mapping button names to their handler functions
    """
    logger = ui_components.get('logger', get_logger(__name__))
    
    @with_button_context('install_button')
    def handle_installation() -> None:
        """Handle package installation with progress tracking.
        
        This function orchestrates the package installation process including:
        - Extracting selected and custom packages
        - Validating the selection
        - Starting the installation process
        - Updating the UI with progress and results
        """
        try:
            # Extract and validate packages
            selected_packages = get_selected_packages(ui_components.get('package_selector', {}))
            custom_packages = _get_custom_packages(ui_components)
            all_packages = selected_packages + custom_packages
            
            if not all_packages:
                update_status_panel(ui_components, "⚠️ No packages selected", "warning")
                return
            
            # Get installation configuration
            config = _extract_installation_config(ui_components)
            
            # Start installation process
            _start_installation(ui_components, all_packages, config, logger)
            
        except Exception as e:
            error_msg = f"❌ Installation error: {str(e)}"
            update_status_panel(ui_components, error_msg, "error")
            logger.error(f"Installation error: {str(e)}", exc_info=True)
    
    # Setup button handler
    install_button = ui_components.get('install_button')
    if install_button:
        install_button.on_click(lambda b: handle_installation())
    
    return {
        'handle_installation': handle_installation,
        'install_packages_parallel': lambda packages, config: _install_packages_parallel(packages, config, ui_components)
    }

def _get_custom_packages(ui_components: UIComponents) -> PackageList:
    """Extract custom packages from textarea.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        List of custom package dictionaries with keys: key, name, optional
    """
    custom_pkg_text = ui_components.get('custom_packages', {}).get('value', '')
    if not custom_pkg_text:
        return []
    
    # Split by newline and clean up
    packages = [pkg.strip() for pkg in custom_pkg_text.split('\n')]
    return [{'key': pkg, 'name': pkg, 'optional': False} 
            for pkg in packages if pkg]

def _extract_installation_config(ui_components: UIComponents) -> InstallationConfig:
    """Extract installation configuration from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Dictionary containing installation configuration
    """
    return {
        'use_venv': ui_components.get('venv_toggle', {}).get('value', True),
        'upgrade': ui_components.get('upgrade_toggle', {}).get('value', False),
        'no_deps': ui_components.get('no_deps_toggle', {}).get('value', False),
        'force_reinstall': ui_components.get('force_reinstall_toggle', {}).get('value', False),
        'extra_index_url': ui_components.get('extra_index_url', {}).get('value', ''),
        'trusted_host': ui_components.get('trusted_host', {}).get('value', ''),
        'timeout': int(ui_components.get('timeout_input', {}).get('value', 300)),
        'retries': int(ui_components.get('retries_input', {}).get('value', 3)),
        'max_workers': int(ui_components.get('workers_input', {}).get('value', 4)),
    }

def _start_installation(
    ui_components: UIComponents,
    packages: PackageList,
    config: InstallationConfig,
    logger: Any
) -> None:
    """Start the package installation process.
    
    Args:
        ui_components: Dictionary containing UI components
        packages: List of packages to install
        config: Installation configuration
        logger: Logger instance for logging
    """
    update_status_panel(ui_components, f"🚀 Starting installation of {len(packages)} packages...", "info")
    show_progress_tracker_safe(ui_components, "Package Installation")
    
    logger.info(f"Starting installation of {len(packages)} packages with config: {config}")
    
    # Install packages in parallel
    results = _install_packages_parallel(
        [pkg['key'] for pkg in packages if 'key' in pkg],
        config,
        ui_components
    )
    
    # Process and display results
    _process_installation_results(ui_components, results, logger)

def _process_installation_results(
    ui_components: UIComponents,
    results: List[InstallationResult],
    logger: Any
) -> None:
    """Process and display installation results.
    
    Args:
        ui_components: Dictionary containing UI components
        results: List of installation results
        logger: Logger instance for logging
    """
    success_count = sum(1 for r in results if r.get('success', False))
    failed_count = len(results) - success_count
    
    if failed_count > 0:
        msg = f"✅ {success_count} succeeded, ❌ {failed_count} failed"
        update_status_panel(ui_components, msg, "warning" if failed_count > 0 else "success")
    else:
        msg = f"✅ All {success_count} packages installed successfully!"
        complete_operation_with_message(ui_components, msg)
    
    logger.info(f"Installation completed: {success_count} success, {failed_count} failed")

def _install_packages_parallel(
    packages: List[str],
    config: InstallationConfig,
    ui_components: UIComponents
) -> List[InstallationResult]:
    """Install packages using parallel processing.
    
    Args:
        packages: List of package names to install
        config: Installation configuration
        ui_components: Dictionary containing UI components
        
    Returns:
        List of installation results
    """
    results = []
    total = len(packages)
    
    if total == 0:
        return results
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        # Start the load operations and mark each future with its package
        future_to_package = {
            executor.submit(install_single_package, package, config): package
            for package in packages
        }
        
        for future in as_completed(future_to_package):
            package = future_to_package[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # Update progress
                completed = len(results)
                progress = int((completed / total) * 100)
                update_progress_step(ui_components, "overall", progress, f"Installing {package}")
                
            except Exception as e:
                results.append({
                    'package': package,
                    'success': False,
                    'error': str(e)
                })
    
    return results