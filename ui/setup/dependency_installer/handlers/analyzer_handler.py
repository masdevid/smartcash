"""
File: smartcash/ui/setup/dependency_installer/handlers/analyzer_handler.py
Deskripsi: Enhanced handler untuk analisis package yang terinstall dengan integrasi observer pattern
"""

from typing import Dict, Any, List
import time

def setup_analyzer_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk analisis package dengan integrasi observer pattern
    
    Args:
        ui_components: Dictionary UI components
    """
    # Import utils
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Wrap analyze_installed_packages dengan progress tracking dan observer notification
    def enhanced_analyze_installed_packages() -> List[str]:
        # Get observer manager
        observer_manager = ui_components.get('observer_manager')
        progress_tracker = ui_components.get('progress_tracker')
        
        # Notify start via observer
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DEPENDENCY_ANALYZE_START', None, {
                    'message': "Memulai analisis packages terinstall",
                    'timestamp': time.time()
                })
            except Exception:
                pass  # Silent fail untuk observer notification
        
        # Tampilkan progress tracker untuk analisis
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('analyze')
        
        # Update progress
        if progress_tracker:
            progress_tracker.update('step', 0, "Menganalisis packages terinstall...", "#007bff")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('step', 0, "Menganalisis packages terinstall...", "#007bff")
        
        # Log start
        log_message(ui_components, "🔍 Memulai analisis packages terinstall...", "info")
        
        try:
            # Jalankan analisis
            start_time = time.time()
            result = analyze_installed_packages(ui_components)
            duration = time.time() - start_time
            
            # Update progress selesai
            if progress_tracker:
                progress_tracker.update('step', 100, "Analisis packages selesai", "#28a745")
            elif 'update_progress' in ui_components:
                ui_components['update_progress']('step', 100, "Analisis packages selesai", "#28a745")
            
            # Log success
            log_message(ui_components, f"✅ Berhasil menganalisis packages terinstall ({duration:.1f}s)", "success")
            
            # Notify complete via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DEPENDENCY_ANALYZE_COMPLETE', None, {
                        'message': f"Analisis packages selesai ({duration:.1f}s)",
                        'timestamp': time.time(),
                        'duration': duration,
                        'result': result
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
            
            return result
            
        except Exception as e:
            # Update progress error
            if progress_tracker:
                progress_tracker.update('step', 100, f"Analisis packages gagal: {str(e)}", "#dc3545")
            elif 'update_progress' in ui_components:
                ui_components['update_progress']('step', 100, f"Analisis packages gagal: {str(e)}", "#dc3545")
            
            # Log error
            log_message(ui_components, f"⚠️ Gagal menganalisis packages: {str(e)}", "error")
            
            # Notify error via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DEPENDENCY_ANALYZE_ERROR', None, {
                        'message': f"Gagal menganalisis packages: {str(e)}",
                        'timestamp': time.time(),
                        'error': str(e)
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
            
            # Re-raise exception
            raise
    
    # Expose enhanced function
    ui_components['analyze_installed_packages'] = enhanced_analyze_installed_packages
