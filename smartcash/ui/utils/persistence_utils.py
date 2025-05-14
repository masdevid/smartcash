"""
File: smartcash/ui/utils/persistence_utils.py
Deskripsi: Utilitas untuk memastikan persistensi UI dan konfigurasi antar eksekusi cell
"""

from typing import Dict, Any, Optional, List, Union, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def ensure_ui_persistence(ui_components: Dict[str, Any], module_name: str, logger = None) -> Dict[str, Any]:
    """
    Memastikan persistensi UI components dengan mendaftarkannya ke ConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul untuk identifikasi
        logger: Logger untuk logging
        
    Returns:
        Dictionary UI components yang sudah terdaftar
    """
    # Import ConfigManager
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan instance ConfigManager
    config_manager = get_config_manager()
    
    # Dapatkan UI components yang tersimpan (jika ada)
    saved_components = config_manager.get_ui_components(module_name)
    
    # Gabungkan komponen yang tersimpan dengan yang baru
    if saved_components:
        # Hanya salin komponen yang kompatibel (widgets dan nilai sederhana)
        for key, value in saved_components.items():
            if key not in ui_components:
                # Skip komponen yang tidak ada di UI baru
                continue
                
            if isinstance(value, widgets.Widget) and isinstance(ui_components[key], widgets.Widget):
                # Untuk widget, coba salin nilai jika memiliki atribut yang sama
                try:
                    if hasattr(value, 'value') and hasattr(ui_components[key], 'value'):
                        ui_components[key].value = value.value
                except Exception as e:
                    if logger:
                        logger.debug(f"⚠️ Tidak dapat menyalin nilai widget {key}: {str(e)}")
            elif not callable(value) and not isinstance(value, widgets.Widget):
                # Untuk nilai sederhana (bukan fungsi atau widget), salin langsung
                ui_components[key] = value
    
    # Daftarkan UI components yang baru/diupdate
    config_manager.register_ui_components(module_name, ui_components)
    
    if logger:
        logger.debug(f"🔄 UI components untuk {module_name} berhasil didaftarkan untuk persistensi")
    
    return ui_components

def validate_ui_param(value: Any, default_value: Any, 
                     valid_types: Optional[Union[type, List[type]]] = None, 
                     valid_values: Optional[List[Any]] = None,
                     logger = None) -> Any:
    """
    Validasi parameter UI dengan logging yang informatif.
    
    Args:
        value: Nilai yang akan divalidasi
        default_value: Nilai default jika validasi gagal
        valid_types: Tipe yang valid (single atau list)
        valid_values: List nilai yang valid
        logger: Logger untuk logging
        
    Returns:
        Nilai yang valid atau default
    """
    # Import ConfigManager
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan instance ConfigManager
    config_manager = get_config_manager()
    
    # Gunakan fungsi validasi dari ConfigManager
    return config_manager.validate_param(value, default_value, valid_types, valid_values)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], 
                         config_path: str, logger = None) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi dengan penanganan error yang lebih baik.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan
        config_path: Path dalam config untuk mengakses nilai (format: "section.subsection.key")
        logger: Logger untuk logging
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Parse config path
    path_parts = config_path.split('.')
    
    # Navigasi ke bagian config yang diinginkan
    current = config
    for part in path_parts:
        if part in current:
            current = current[part]
        else:
            if logger:
                logger.warning(f"⚠️ Path {config_path} tidak ditemukan dalam config")
            return ui_components
    
    # Update UI components berdasarkan tipe
    for key, value in ui_components.items():
        if not isinstance(value, widgets.Widget) or not hasattr(value, 'value'):
            continue
            
        try:
            # Cek apakah key ada di config
            if isinstance(current, dict) and key in current:
                # Update nilai widget
                ui_components[key].value = current[key]
                if logger:
                    logger.debug(f"✅ Berhasil update widget {key} dengan nilai dari config")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat update widget {key}: {str(e)}")
    
    return ui_components

def extract_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any], 
                          config_path: str, logger = None) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan diupdate
        config_path: Path dalam config untuk menyimpan nilai (format: "section.subsection")
        logger: Logger untuk logging
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    # Parse config path
    path_parts = config_path.split('.')
    
    # Pastikan struktur config ada
    current = config
    for i, part in enumerate(path_parts):
        if i == len(path_parts) - 1:
            # Bagian terakhir, tidak perlu membuat dict baru
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            # Buat dict baru jika belum ada
            if part not in current:
                current[part] = {}
            current = current[part]
    
    # Ekstrak nilai dari UI components
    for key, value in ui_components.items():
        if not isinstance(value, widgets.Widget) or not hasattr(value, 'value'):
            continue
            
        try:
            # Simpan nilai widget ke config
            current[key] = value.value
            if logger:
                logger.debug(f"✅ Berhasil ekstrak nilai widget {key} ke config")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat ekstrak nilai widget {key}: {str(e)}")
    
    return config

def register_config_observer(module_name: str, ui_components: Dict[str, Any], 
                           update_func: Callable[[Dict[str, Any]], None], 
                           logger = None) -> None:
    """
    Register observer untuk update UI saat konfigurasi berubah.
    
    Args:
        module_name: Nama modul
        ui_components: Dictionary komponen UI
        update_func: Fungsi untuk update UI dari config
        logger: Logger untuk logging
    """
    # Import ConfigManager
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan instance ConfigManager
    config_manager = get_config_manager()
    
    # Register observer
    config_manager.register_observer(module_name, update_func)
    
    if logger:
        logger.debug(f"👁️ Observer berhasil diregister untuk {module_name}")
