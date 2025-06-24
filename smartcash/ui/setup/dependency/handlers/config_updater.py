"""
File: smartcash/ui/setup/dependency/handlers/config_updater.py
Deskripsi: Config updater untuk dependency installer dengan logging terstandarisasi
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.setup.dependency.utils import LogLevel, with_logging, requires, reset_package_selections

@with_logging("Update Dependency UI", LogLevel.INFO)
@requires('custom_packages', 'auto_analyze_checkbox')
def update_dependency_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari config
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary berisi konfigurasi
    """
    # Update custom packages text
    custom_packages = config.get('custom_packages', '')
    ui_components['custom_packages'].value = str(custom_packages)
    
    # Update auto-analyze checkbox
    auto_analyze = config.get('auto_analyze', True)
    ui_components['auto_analyze_checkbox'].value = bool(auto_analyze)
    
    # Update installation settings
    installation_config = config.get('installation', {})
    _update_installation_settings(ui_components, installation_config)
    
    # Update analysis settings
    analysis_config = config.get('analysis', {})
    _update_analysis_settings(ui_components, analysis_config)
    
    # Update package selections
    selected_packages = config.get('selected_packages', [])
    _update_package_selections(ui_components, selected_packages)

@with_logging(level=LogLevel.DEBUG)
def _update_installation_settings(ui_components: Dict[str, Any], installation_config: Dict[str, Any]) -> None:
    """
    Update installation settings dari konfigurasi
    
    Args:
        ui_components: Dictionary komponen UI
        installation_config: Dictionary konfigurasi instalasi
    """
    settings_mapping = {
        'parallel_workers_slider': 'parallel_workers',
        'timeout_slider': 'timeout',
        'force_reinstall_checkbox': 'force_reinstall',
        'use_cache_checkbox': 'use_cache'
    }
    
    # Update widgets dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', installation_config.get(config_key, 
                                                   _get_installation_default(config_key)))
     for widget_key, config_key in settings_mapping.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

@with_logging(level=LogLevel.DEBUG)
def _update_analysis_settings(ui_components: Dict[str, Any], analysis_config: Dict[str, Any]) -> None:
    """
    Update analysis settings dari konfigurasi
    
    Args:
        ui_components: Dictionary komponen UI
        analysis_config: Dictionary konfigurasi analisis
    """
    settings_mapping = {
        'check_compatibility_checkbox': 'check_compatibility',
        'include_dev_deps_checkbox': 'include_dev_deps'
    }
    
    # Update widgets dengan safe attribute setting - one-liner
    [setattr(ui_components[widget_key], 'value', analysis_config.get(config_key, 
                                                 _get_analysis_default(config_key)))
     for widget_key, config_key in settings_mapping.items() 
     if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]

@with_logging(level=LogLevel.DEBUG)
@requires('package_selector', required=False)
def _update_package_selections(ui_components: Dict[str, Any], selected_packages: list) -> None:
    """
    Update package selections dari daftar paket terpilih
    
    Args:
        ui_components: Dictionary komponen UI
        selected_packages: Daftar nama paket yang dipilih
    """
    # Coba gunakan package selector utils terlebih dahulu
    if hasattr(ui_components['package_selector'], 'select_packages'):
        ui_components['package_selector'].select_packages(selected_packages)
    else:
        # Fallback ke manual update
        _manual_update_package_checkboxes(ui_components, selected_packages)

@with_logging(level=LogLevel.DEBUG)
def _manual_update_package_checkboxes(ui_components: Dict[str, Any], selected_packages: list) -> None:
    """
    Update manual checkbox paket (fallback)
    
    Args:
        ui_components: Dictionary komponen UI
        selected_packages: Daftar nama paket yang dipilih
    """
    if 'package_checkboxes' not in ui_components:
        logger = ui_components.get('logger')
        if logger:
            logger.warning("package_checkboxes not found in UI components")
        return
        
    selected_packages_set = set(selected_packages)
    
    # Reset semua checkbox terlebih dahulu
    for checkbox in ui_components['package_checkboxes'].values():
        if hasattr(checkbox, 'value'):
            checkbox.value = False
    
    # Set nilai checkbox yang dipilih
    for pkg_name, checkbox in ui_components['package_checkboxes'].items():
        if pkg_name in selected_packages_set and hasattr(checkbox, 'value'):
            checkbox.value = True
            selected_packages_set.remove(pkg_name)
    
    # Log warning untuk packages yang tidak ditemukan
    if selected_packages_set:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"Packages not found in UI: {', '.join(selected_packages_set)}")

@with_logging("Reset Dependency UI", LogLevel.INFO)
@requires('package_selector', 'custom_packages')
def reset_dependency_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI ke pengaturan default
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset pengaturan instalasi
    _reset_installation_settings(ui_components)
    
    # Reset pengaturan analisis
    _reset_analysis_settings(ui_components)
    
    # Reset pemilihan paket
    if hasattr(ui_components['package_selector'], 'reset_selection'):
        ui_components['package_selector'].reset_selection()
    
    # Kosongkan custom packages
    ui_components['custom_packages'].value = ''
    
    # Reset auto-analyze checkbox jika ada
    if 'auto_analyze_checkbox' in ui_components and hasattr(ui_components['auto_analyze_checkbox'], 'value'):
        ui_components['auto_analyze_checkbox'].value = True
    
    # Bersihkan state UI
    _clear_ui_state(ui_components)

@with_logging(level=LogLevel.DEBUG)
def _reset_installation_settings(ui_components: Dict[str, Any]) -> None:
    """
    Reset pengaturan instalasi ke nilai default
    
    Args:
        ui_components: Dictionary komponen UI
    """
    default_settings = {
        'parallel_workers_slider': 3,
        'timeout_slider': 300,
        'force_reinstall_checkbox': False,
        'use_cache_checkbox': True
    }
    
    # Reset dengan safe attribute setting
    for widget_key, default_value in default_settings.items():
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = default_value
            except Exception as e:
                logger = ui_components.get('logger')
                if logger:
                    logger.warning(f"Gagal mengatur {widget_key}: {str(e)}")

@with_logging(level=LogLevel.DEBUG)
def _reset_analysis_settings(ui_components: Dict[str, Any]) -> None:
    """
    Reset pengaturan analisis ke nilai default
    
    Args:
        ui_components: Dictionary komponen UI
    """
    default_settings = {
        'check_compatibility_checkbox': True,
        'include_dev_deps_checkbox': False
    }
    
    # Reset dengan safe attribute setting
    for widget_key, default_value in default_settings.items():
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = default_value
            except Exception as e:
                logger = ui_components.get('logger')
                if logger:
                    logger.warning(f"Gagal mengatur {widget_key}: {str(e)}")

def _safe_clear_output(widget) -> bool:
    """
    Bersihkan output widget dengan aman
    
    Args:
        widget: Widget yang akan dibersihkan
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        if hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)
            return True
    except Exception as e:
        pass
    return False

def _safe_reset_progress(widget) -> bool:
    """
    Reset nilai progress widget dengan aman
    
    Args:
        widget: Widget progress yang akan direset
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        if hasattr(widget, 'value'):
            widget.value = 0
            return True
    except Exception as e:
        pass
    return False

@with_logging(level=LogLevel.DEBUG)
def _clear_ui_state(ui_components: Dict[str, Any]) -> None:
    """
    Membersihkan state UI seperti output, status, dan progress bar
    
    Args:
        ui_components: Dictionary komponen UI dengan kemungkinan kunci:
            - output: Widget output utama
            - log_output: Widget untuk log
            - status_output: Widget untuk status
            - progress_bar: Widget progress bar
            - progress_indicator: Widget indikator progress
            - progress_container: Container untuk progress bar
    """
    if not ui_components:
        return
        
    logger = ui_components.get('logger')
    
    # Bersihkan output widgets
    output_widgets = ['output', 'log_output', 'status_output', 'status']
    for widget_key in output_widgets:
        if widget_key in ui_components:
            widget = ui_components[widget_key]
            if not _safe_clear_output(widget) and logger:
                logger.debug(f"Tidak dapat membersihkan output widget: {widget_key}")
    
    # Reset progress widgets
    progress_widgets = ['progress_bar', 'progress_indicator', 'progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            widget = ui_components[widget_key]
            if not _safe_reset_progress(widget) and logger:
                logger.debug(f"Tidak dapat mereset progress widget: {widget_key}")
    
    # Sembunyikan container progress jika ada
    if 'progress_container' in ui_components:
        container = ui_components['progress_container']
        try:
            if hasattr(container, 'layout') and hasattr(container.layout, 'visibility'):
                container.layout.visibility = 'hidden'
        except Exception as e:
            if logger:
                logger.debug(f"Gagal menyembunyikan progress container: {str(e)}")

@with_logging(level=LogLevel.DEBUG, with_status=False)
def _get_installation_default(config_key: str) -> Any:
    """
    Dapatkan nilai default untuk konfigurasi instalasi
    
    Args:
        config_key: Kunci konfigurasi yang dicari
        
    Returns:
        Nilai default untuk kunci yang diminta, atau None jika tidak ditemukan
    """
    defaults = {
        'parallel_workers': 3,
        'timeout': 300,
        'force_reinstall': False,
        'use_cache': True
    }
    
    if config_key not in defaults:
        logger = logging.getLogger(__name__)
        logger.warning(f"Kunci konfigurasi tidak valid: {config_key}")
        return None
        
    return defaults[config_key]

@with_logging(level=LogLevel.DEBUG, with_status=False)
def _get_analysis_default(config_key: str) -> Any:
    """
    Dapatkan nilai default untuk konfigurasi analisis
    
    Args:
        config_key: Kunci konfigurasi yang dicari
        
    Returns:
        Nilai default untuk kunci yang diminta, atau None jika tidak ditemukan
    """
    defaults = {
        'check_compatibility': True,
        'include_dev_deps': False
    }
    
    if config_key not in defaults:
        logger = logging.getLogger(__name__)
        logger.warning(f"Kunci konfigurasi analisis tidak valid: {config_key}")
        return None
        
    return defaults[config_key]

@with_logging(level=LogLevel.DEBUG, with_status=False)
def _get_update_summary(config: Dict[str, Any]) -> str:
    """
    Dapatkan ringkasan konfigurasi untuk logging
    
    Args:
        config: Dictionary konfigurasi dengan keys:
            - selected_packages: List paket yang dipilih
            - custom_packages: String paket kustom (dipisahkan newline)
    
    Returns:
        String ringkasan konfigurasi yang diformat
    """
    try:
        selected = len(config.get('selected_packages', []))
        custom_pkgs = config.get('custom_packages', '')
        custom_count = len([p for p in custom_pkgs.split('\n') if p.strip()])
        
        # Format ringkasan
        parts = []
        if selected > 0:
            parts.append(f"{selected} paket terpilih")
        if custom_count > 0:
            parts.append(f"{custom_count} paket kustom")
            
        if not parts:
            return "Tidak ada paket yang dipilih"
            
        return ", ".join(parts)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Gagal membuat ringkasan konfigurasi: {str(e)}")
        return "Ringkasan konfigurasi tidak tersedia"

# Alias untuk update_dependency_ui dengan logging yang sesuai
apply_config_to_ui = with_logging("Apply Config to UI", LogLevel.INFO)(update_dependency_ui)

@with_logging(level=LogLevel.DEBUG)
def get_ui_state_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan ringkasan state UI saat ini untuk debugging
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict berisi ringkasan state UI dengan format:
        {
            'selected_packages': List[str],
            'custom_packages': str,
            'auto_analyze': bool,
            'installation': Dict[str, Any],
            'analysis': Dict[str, bool],
            'ui_components': List[str]  # Daftar komponen UI yang tersedia
        }
    """
    def safe_get_value(component, attr='value', default=None):
        """Helper untuk mendapatkan nilai attribute dengan aman"""
        if not component:
            return default
            
        try:
            return getattr(component, attr, default)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.debug(f"Gagal mendapatkan nilai dari komponen: {str(e)}")
            return default
    
    # Dapatkan daftar paket terpilih dengan aman
    try:
        selected_pkgs = get_selected_packages(ui_components)
    except Exception as e:
        selected_pkgs = []
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"Gagal mendapatkan daftar paket terpilih: {str(e)}")
    
    # Buat ringkasan state
    summary = {
        'selected_packages': selected_pkgs,
        'custom_packages': safe_get_value(ui_components.get('custom_packages'), default=''),
        'auto_analyze': safe_get_value(ui_components.get('auto_analyze_checkbox'), default=True),
        'installation': {
            'parallel_workers': safe_get_value(ui_components.get('parallel_workers_slider')),
            'timeout': safe_get_value(ui_components.get('timeout_slider')),
            'force_reinstall': safe_get_value(ui_components.get('force_reinstall_checkbox')),
            'use_cache': safe_get_value(ui_components.get('use_cache_checkbox'))
        },
        'analysis': {
            'check_compatibility': safe_get_value(ui_components.get('check_compatibility_checkbox')),
            'include_dev_deps': safe_get_value(ui_components.get('include_dev_deps_checkbox'))
        },
        'ui_components': list(ui_components.keys())  # Daftar semua komponen UI yang tersedia
    }
    
    return summary