"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk operasi konfigurasi preprocessing yang disederhanakan
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.preprocessing.components.config_manager import (
    get_config_from_ui, update_ui_from_config
)
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

class ConfigHandler:
    """Handler untuk operasi konfigurasi preprocessing."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi config handler."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.config')
        self.config_manager = get_config_manager()
    
    def handle_save_click(self, button: Any) -> None:
        """Handler untuk tombol save konfigurasi."""
        button.disabled = True
        
        try:
            self.logger.info("💾 Menyimpan konfigurasi preprocessing...")
            self._update_status("info", "Menyimpan konfigurasi...")
            
            # Extract dan save config
            config = get_config_from_ui(self.ui_components)
            success = self._save_config(config)
            
            if success:
                self.logger.success("✅ Konfigurasi berhasil disimpan")
                self._update_status("success", "Konfigurasi berhasil disimpan")
            else:
                raise Exception("Gagal menyimpan konfigurasi")
                
        except Exception as e:
            error_msg = f"Error save konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._update_status("error", error_msg)
        finally:
            button.disabled = False
    
    def handle_reset_click(self, button: Any) -> None:
        """Handler untuk tombol reset konfigurasi."""
        button.disabled = True
        
        try:
            self.logger.info("🔄 Reset konfigurasi ke default...")
            self._update_status("info", "Reset ke default...")
            
            # Reset UI ke default
            self._reset_ui_to_defaults()
            
            # Save default config
            default_config = self._get_default_config()
            success = self._save_config(default_config)
            
            if success:
                self.logger.success("✅ Konfigurasi direset ke default")
                self._update_status("success", "Konfigurasi direset ke default")
            else:
                self.logger.warning("⚠️ UI direset tapi gagal simpan ke file")
                
        except Exception as e:
            error_msg = f"Error reset konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._update_status("error", error_msg)
        finally:
            button.disabled = False
    
    def load_config_to_ui(self) -> bool:
        """Load konfigurasi dari file ke UI."""
        try:
            self.logger.info("📂 Memuat konfigurasi...")
            
            # Load config dari manager
            full_config = self.config_manager.get_config()
            
            # Update UI
            update_ui_from_config(self.ui_components, full_config)
            
            self.logger.success("✅ Konfigurasi berhasil dimuat")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal load config: {str(e)}, gunakan default")
            self._reset_ui_to_defaults()
            return False
    
    def _save_config(self, config: Dict[str, Any]) -> bool:
        """Simpan konfigurasi ke file."""
        try:
            # Load existing config
            current_config = self.config_manager.get_config()
            
            # Update preprocessing section
            if 'preprocessing' not in current_config:
                current_config['preprocessing'] = {}
            
            current_config['preprocessing'].update(config.get('preprocessing', {}))
            
            # Update data section
            if 'data' not in current_config:
                current_config['data'] = {}
            
            current_config['data'].update(config.get('data', {}))
            
            # Save to manager
            return self.config_manager.save_config(current_config)
            
        except Exception as e:
            self.logger.error(f"❌ Error save config: {str(e)}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default."""
        return {
            'preprocessing': {
                'img_size': DEFAULT_IMG_SIZE,
                'normalization': 'minmax',
                'normalize': True,
                'preserve_aspect_ratio': True,
                'augmentation': False,
                'force_reprocess': True,
                'num_workers': 4,
                'split': 'all',
                'output_dir': 'data/preprocessed'
            },
            'data': {
                'dir': 'data'
            }
        }
    
    def _reset_ui_to_defaults(self) -> None:
        """Reset UI ke nilai default."""
        try:
            # Reset resolution dropdown
            if 'resolution_dropdown' in self.ui_components:
                self.ui_components['resolution_dropdown'].value = '640x640'
            
            # Reset normalization dropdown
            if 'normalization_dropdown' in self.ui_components:
                self.ui_components['normalization_dropdown'].value = 'minmax'
            
            # Reset worker slider
            if 'worker_slider' in self.ui_components:
                self.ui_components['worker_slider'].value = 4
            
            # Reset split dropdown
            if 'split_dropdown' in self.ui_components:
                self.ui_components['split_dropdown'].value = 'all'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error reset UI: {str(e)}")
    
    def _update_status(self, status: str, message: str) -> None:
        """Update status panel."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
            except ImportError:
                self.ui_components['status_panel'].value = f"<div class='alert alert-{status}'>{message}</div>"


def setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup config handlers untuk save dan reset buttons."""
    # Create config handler
    config_handler = ConfigHandler(ui_components)
    ui_components['config_handler'] = config_handler
    
    # Setup save button
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: config_handler.handle_save_click(b)
        )
    
    # Setup reset button
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: config_handler.handle_reset_click(b)
        )
    
    # Load config awal
    config_handler.load_config_to_ui()