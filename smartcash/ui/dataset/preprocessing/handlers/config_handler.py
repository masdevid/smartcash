"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk operasi konfigurasi preprocessing dengan save/reset dan sinkronisasi
"""

from typing import Dict, Any
from smartcash.ui.components.status_panel import update_status_panel


def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handlers untuk operasi konfigurasi preprocessing."""
    logger = ui_components.get('logger')
    config_manager = ui_components.get('config_manager')
    
    def _get_current_config() -> Dict[str, Any]:
        """Ambil konfigurasi saat ini dari UI components."""
        resolution = ui_components['resolution_dropdown'].value.split('x')
        
        return {
            'preprocessing': {
                'img_size': [int(resolution[0]), int(resolution[1])],
                'normalization': ui_components['normalization_dropdown'].value,
                'num_workers': ui_components['worker_slider'].value,
                'split': ui_components['split_dropdown'].value,
                'normalize': ui_components['normalization_dropdown'].value != 'none',
                'preserve_aspect_ratio': True
            }
        }
    
    def _apply_config_to_ui(config: Dict[str, Any]):
        """Terapkan konfigurasi ke UI components."""
        preprocessing_config = config.get('preprocessing', {})
        
        # Resolution
        img_size = preprocessing_config.get('img_size', [640, 640])
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            resolution_str = f"{img_size[0]}x{img_size[1]}"
            if resolution_str in ui_components['resolution_dropdown'].options:
                ui_components['resolution_dropdown'].value = resolution_str
        
        # Normalization
        normalization = preprocessing_config.get('normalization', 'minmax')
        if normalization in ui_components['normalization_dropdown'].options:
            ui_components['normalization_dropdown'].value = normalization
        
        # Workers
        num_workers = preprocessing_config.get('num_workers', 4)
        ui_components['worker_slider'].value = max(1, min(num_workers, 10))
        
        # Split
        split = preprocessing_config.get('split', 'all')
        if split in ui_components['split_dropdown'].options:
            ui_components['split_dropdown'].value = split
    
    def _on_save_click(b):
        """Handler untuk tombol save konfigurasi."""
        try:
            if not config_manager:
                logger.error("❌ Config manager tidak tersedia")
                return
            
            current_config = _get_current_config()
            
            # Save ke config manager
            success = config_manager.save_config(current_config, 'preprocessing')
            
            if success:
                logger.success("💾 Konfigurasi preprocessing berhasil disimpan")
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi preprocessing berhasil disimpan", "success")
            else:
                logger.error("❌ Gagal menyimpan konfigurasi")
                update_status_panel(ui_components['status_panel'], 
                                  "Gagal menyimpan konfigurasi", "error")
                
        except Exception as e:
            logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            update_status_panel(ui_components['status_panel'], 
                              f"Error menyimpan konfigurasi: {str(e)}", "error")
    
    def _on_reset_click(b):
        """Handler untuk tombol reset konfigurasi."""
        try:
            if not config_manager:
                logger.error("❌ Config manager tidak tersedia")
                return
            
            # Load config dari file
            saved_config = config_manager.load_config('preprocessing')
            
            if saved_config:
                _apply_config_to_ui(saved_config)
                logger.success("🔄 Konfigurasi berhasil direset")
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi berhasil direset", "success")
            else:
                # Reset ke default
                default_config = {
                    'preprocessing': {
                        'img_size': [640, 640],
                        'normalization': 'minmax',
                        'num_workers': 4,
                        'split': 'all'
                    }
                }
                _apply_config_to_ui(default_config)
                logger.info("🔄 Konfigurasi direset ke default")
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi direset ke default", "info")
                
        except Exception as e:
            logger.error(f"❌ Error saat reset konfigurasi: {str(e)}")
            update_status_panel(ui_components['status_panel'], 
                              f"Error reset konfigurasi: {str(e)}", "error")
    
    # Setup event handlers
    ui_components['save_button'].on_click(_on_save_click)
    ui_components['reset_button'].on_click(_on_reset_click)
    
    # Apply initial config
    if config:
        _apply_config_to_ui(config)
    
    if logger:
        logger.debug("✅ Config handler preprocessing setup selesai")
    
    return ui_components