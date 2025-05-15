"""
File: smartcash/ui/training_config/backbone/handlers/ui_handlers.py
Deskripsi: Handler untuk interaksi UI backbone model
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Pastikan struktur konfigurasi benar
    if not config:
        if logger: logger.debug(f"{ICONS['warning']} Konfigurasi kosong, menggunakan default")
        from smartcash.ui.training_config.backbone.handlers.config_handlers import load_default_config
        config = load_default_config()
    
    if 'model' not in config:
        config['model'] = {}
    
    # Import ModelManager untuk mendapatkan model yang dioptimalkan
    try:
        from smartcash.model.manager import ModelManager
        
        # Update model_type jika tersedia
        if 'type' in config['model'] and 'model_type' in ui_components:
            try:
                model_type = config['model']['type']
                if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
                    ui_components['model_type'].value = model_type
                    if logger: logger.debug(f"{ICONS['success']} Model type diperbarui ke: {model_type}")
                else:
                    if logger: logger.warning(f"{ICONS['warning']} Model type '{model_type}' tidak tersedia dalam OPTIMIZED_MODELS")
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update model_type: {str(e)}")
        
        # Update backbone_type jika tersedia
        if 'backbone' in config['model'] and 'backbone_type' in ui_components:
            try:
                from smartcash.model.config.backbone_config import BackboneConfig
                backbone = config['model']['backbone']
                available_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
                
                # Pastikan backbone_options tidak kosong
                if not available_options:
                    available_options = ['efficientnet_b4', 'cspdarknet_s']
                
                # Update nilai backbone di UI
                if backbone in available_options:
                    ui_components['backbone_type'].value = backbone
                    if logger: logger.debug(f"{ICONS['success']} Backbone type diperbarui ke: {backbone}")
                else:
                    if logger: logger.warning(f"{ICONS['warning']} Backbone '{backbone}' tidak tersedia dalam opsi")
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update backbone_type: {str(e)}")
        
        # Update informasi backbone
        try:
            if 'model_type' in ui_components and 'backbone_type' in ui_components:
                model_type = ui_components['model_type'].value
                backbone = ui_components['backbone_type'].value
                
                # Dapatkan informasi model
                if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
                    model_config = ModelManager.OPTIMIZED_MODELS[model_type].copy()
                    model_config['backbone'] = backbone
                    
                    # Dapatkan nilai fitur optimasi
                    use_attention = model_config.get('use_attention', False)
                    use_residual = model_config.get('use_residual', False)
                    use_ciou = model_config.get('use_ciou', False)
                    
                    # Update informasi backbone
                    backbone_info = f"""
                    <div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;'>
                        <h4>{model_type.replace('_', ' ').title()}</h4>
                        <p><strong>Deskripsi:</strong> {model_config.get('description', 'Tidak ada deskripsi')}</p>
                        <p><strong>Backbone:</strong> {backbone}</p>
                        <p><strong>Fitur Optimasi:</strong></p>
                        <ul>
                            <li>FeatureAdapter (Attention): {'✅ Aktif' if use_attention else '❌ Tidak aktif'}</li>
                            <li>ResidualAdapter: {'✅ Aktif' if use_residual else '❌ Tidak aktif'}</li>
                            <li>CIoU Loss: {'✅ Aktif' if use_ciou else '❌ Tidak aktif'}</li>
                        </ul>
                    </div>
                    """
                    ui_components['backbone_info'].value = backbone_info
                    if logger: logger.debug(f"{ICONS['success']} Informasi backbone diperbarui")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update informasi backbone: {str(e)}")
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error umum saat update UI: {str(e)}")
    
    return ui_components

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None) -> None:
    """
    Pastikan UI components terdaftar untuk persistensi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
    """
    try:
        from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
        ensure_ui_persistence(ui_components, 'backbone_model', logger)
        if logger: logger.debug(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error saat mendaftarkan UI components untuk persistensi: {str(e)}")

def initialize_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inisialisasi UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Update UI dari konfigurasi
    ui_components = update_ui_from_config(ui_components, config)
    
    # Pastikan UI components terdaftar untuk persistensi
    ensure_ui_persistence(ui_components, config, logger)
    
    return ui_components
