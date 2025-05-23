"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler terpusat untuk operasi konfigurasi preprocessing (save, load, reset) dengan SimpleConfigManager
"""

from typing import Dict, Any, Tuple, Optional
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

class ConfigHandler:
    """Handler terpusat untuk operasi konfigurasi preprocessing."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi config handler dengan komponen UI."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.config')
        self.config_manager = get_config_manager()
    
    def handle_save_click(self, button: Any) -> None:
        """
        Handler untuk tombol save konfigurasi.
        
        Args:
            button: Widget tombol yang diklik
        """
        self._disable_button(button)
        
        try:
            self.logger.info("💾 Menyimpan konfigurasi preprocessing...")
            self._update_status("info", "Menyimpan konfigurasi preprocessing...")
            
            # Extract dan save config
            config = self._extract_config_from_ui()
            success = self._save_config_to_manager(config)
            
            if success:
                self.logger.success("✅ Konfigurasi preprocessing berhasil disimpan")
                self._update_status("success", "Konfigurasi berhasil disimpan dan disinkronkan")
                self._try_sync_to_drive()
            else:
                raise Exception("Gagal menyimpan konfigurasi ke file")
                
        except Exception as e:
            error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._update_status("error", error_msg)
        
        finally:
            self._enable_button(button)
    
    def handle_reset_click(self, button: Any) -> None:
        """
        Handler untuk tombol reset konfigurasi.
        
        Args:
            button: Widget tombol yang diklik
        """
        self._disable_button(button)
        
        try:
            self.logger.info("🔄 Reset konfigurasi preprocessing ke default...")
            self._update_status("info", "Reset konfigurasi ke default...")
            
            # Reset UI ke default
            self._reset_ui_to_defaults()
            
            # Save default config
            default_config = self._get_default_config()
            success = self._save_config_to_manager(default_config)
            
            if success:
                self.logger.success("✅ Konfigurasi berhasil direset ke default")
                self._update_status("success", "Konfigurasi direset ke default")
                self._try_sync_to_drive()
            else:
                self.logger.warning("⚠️ UI direset tapi gagal simpan ke file")
                self._update_status("warning", "UI direset tapi gagal simpan ke file")
                
        except Exception as e:
            error_msg = f"Error saat reset konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._update_status("error", error_msg)
        
        finally:
            self._enable_button(button)
    
    def load_config_to_ui(self) -> bool:
        """
        Load konfigurasi dari config manager ke UI.
        
        Returns:
            True jika berhasil load, False jika error
        """
        try:
            self.logger.info("📂 Memuat konfigurasi preprocessing dari file...")
            
            # Get preprocessing config dari config manager
            config = self._load_config_from_manager()
            
            # Update UI dengan config
            self._update_ui_from_config(config)
            
            self.logger.success("✅ Konfigurasi preprocessing berhasil dimuat ke UI")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}, menggunakan default")
            self._reset_ui_to_defaults()
            return False
    
    def _extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract konfigurasi preprocessing dari UI components."""
        config = {'resolution': DEFAULT_IMG_SIZE}  # Default untuk mencegah error
        
        # Extract dari preprocessing options
        if 'preprocess_options' in self.ui_components:
            options = self.ui_components['preprocess_options']
            
            # Resolution dropdown
            if hasattr(options, 'resolution_dropdown'):
                resolution_str = getattr(options.resolution_dropdown, 'value', '640x640')
                config['resolution'] = self._parse_resolution(resolution_str)
            
            # Normalization dropdown
            config['normalization'] = getattr(
                getattr(options, 'normalization_dropdown', None), 'value', 'minmax'
            )
            
            # Checkboxes
            config['preserve_aspect_ratio'] = getattr(
                getattr(options, 'preserve_aspect_ratio_checkbox', None), 'value', True
            )
            config['augmentation'] = getattr(
                getattr(options, 'augmentation_checkbox', None), 'value', False
            )
            config['force_reprocess'] = getattr(
                getattr(options, 'force_reprocess_checkbox', None), 'value', False
            )
        
        # Extract worker slider dan split selector dari UI utama
        config['num_workers'] = getattr(
            self.ui_components.get('worker_slider'), 'value', 4
        )
        
        split_value = getattr(self.ui_components.get('split_selector'), 'value', 'All Splits')
        config['split'] = self._map_split_value(split_value)
        
        # Extract validation options
        validation_options = self.ui_components.get('validation_options')
        if validation_options and hasattr(validation_options, 'get_selected'):
            try:
                config['validation_items'] = validation_options.get_selected()
            except:
                config['validation_items'] = self._get_default_validation_items()
        else:
            config['validation_items'] = self._get_default_validation_items()
        
        # Direktori paths
        config['data_dir'] = self.ui_components.get('data_dir', 'data')
        config['preprocessed_dir'] = self.ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        return config
    
    def _save_config_to_manager(self, ui_config: Dict[str, Any]) -> bool:
        """
        Simpan konfigurasi ke config manager dengan format yang benar.
        
        Args:
            ui_config: Konfigurasi dari UI
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Load existing config
            current_config = self.config_manager.get_config()
            
            # Format preprocessing config sesuai struktur backend
            preprocessing_config = {
                'img_size': ui_config.get('resolution', DEFAULT_IMG_SIZE),
                'normalize': ui_config.get('normalization', 'minmax') != 'none',
                'normalization': ui_config.get('normalization', 'minmax'),
                'preserve_aspect_ratio': ui_config.get('preserve_aspect_ratio', True),
                'augmentation': ui_config.get('augmentation', False),
                'force_reprocess': ui_config.get('force_reprocess', False),
                'num_workers': ui_config.get('num_workers', 4),
                'split': ui_config.get('split', 'all'),
                'validation_items': ui_config.get('validation_items', []),
                'output_dir': ui_config.get('preprocessed_dir', 'data/preprocessed')
            }
            
            # Update config
            if 'preprocessing' not in current_config:
                current_config['preprocessing'] = {}
            current_config['preprocessing'].update(preprocessing_config)
            
            # Simpan ke config manager
            return self.config_manager.save_config(current_config)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat save config ke manager: {str(e)}")
            return False
    
    def _load_config_from_manager(self) -> Dict[str, Any]:
        """Load konfigurasi preprocessing dari config manager."""
        try:
            # Load full config
            full_config = self.config_manager.get_config()
            preprocessing_config = full_config.get('preprocessing', {})
            
            # Convert ke format UI
            ui_config = {
                'resolution': preprocessing_config.get('img_size', DEFAULT_IMG_SIZE),
                'normalization': preprocessing_config.get('normalization', 'minmax'),
                'preserve_aspect_ratio': preprocessing_config.get('preserve_aspect_ratio', True),
                'augmentation': preprocessing_config.get('augmentation', False),
                'force_reprocess': preprocessing_config.get('force_reprocess', False),
                'num_workers': preprocessing_config.get('num_workers', 4),
                'split': preprocessing_config.get('split', 'all'),
                'validation_items': preprocessing_config.get('validation_items', self._get_default_validation_items()),
                'data_dir': full_config.get('data', {}).get('dir', 'data'),
                'preprocessed_dir': preprocessing_config.get('output_dir', 'data/preprocessed')
            }
            
            return ui_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error load config dari manager: {str(e)}")
            return self._get_default_config()
    
    def _update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components berdasarkan konfigurasi."""
        try:
            # Update preprocessing options
            if 'preprocess_options' in self.ui_components:
                options = self.ui_components['preprocess_options']
                
                # Resolution dropdown
                if hasattr(options, 'resolution_dropdown'):
                    resolution = config.get('resolution', DEFAULT_IMG_SIZE)
                    resolution_str = self._format_resolution(resolution)
                    if hasattr(options.resolution_dropdown, 'options') and resolution_str in options.resolution_dropdown.options:
                        options.resolution_dropdown.value = resolution_str
                
                # Normalization dropdown
                if hasattr(options, 'normalization_dropdown'):
                    normalization = config.get('normalization', 'minmax')
                    if hasattr(options.normalization_dropdown, 'options') and normalization in options.normalization_dropdown.options:
                        options.normalization_dropdown.value = normalization
                
                # Checkboxes
                for checkbox_name, config_key in [
                    ('preserve_aspect_ratio_checkbox', 'preserve_aspect_ratio'),
                    ('augmentation_checkbox', 'augmentation'),
                    ('force_reprocess_checkbox', 'force_reprocess')
                ]:
                    if hasattr(options, checkbox_name):
                        setattr(getattr(options, checkbox_name), 'value', config.get(config_key, False))
            
            # Update worker slider
            if 'worker_slider' in self.ui_components:
                self.ui_components['worker_slider'].value = config.get('num_workers', 4)
            
            # Update split selector
            if 'split_selector' in self.ui_components:
                split_ui_value = self._map_split_to_ui(config.get('split', 'all'))
                if hasattr(self.ui_components['split_selector'], 'options') and split_ui_value in self.ui_components['split_selector'].options:
                    self.ui_components['split_selector'].value = split_ui_value
            
            # Update validation options
            validation_items = config.get('validation_items', [])
            validation_options = self.ui_components.get('validation_options')
            if validation_options and hasattr(validation_options, 'set_selected'):
                try:
                    validation_options.set_selected(validation_items)
                except:
                    pass  # Ignore errors
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat update UI dari config: {str(e)}")
    
    def _reset_ui_to_defaults(self) -> None:
        """Reset UI ke nilai default."""
        default_config = self._get_default_config()
        self._update_ui_from_config(default_config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default untuk preprocessing."""
        return {
            'resolution': DEFAULT_IMG_SIZE,
            'normalization': 'minmax',
            'preserve_aspect_ratio': True,
            'augmentation': False,
            'force_reprocess': False,
            'num_workers': 4,
            'split': 'all',
            'validation_items': self._get_default_validation_items(),
            'data_dir': 'data',
            'preprocessed_dir': 'data/preprocessed'
        }
    
    def _get_default_validation_items(self) -> list:
        """Dapatkan daftar default validation items."""
        return [
            'validate_image_format',
            'validate_label_format',
            'validate_image_dimensions',
            'validate_bounding_box'
        ]
    
    def _try_sync_to_drive(self) -> None:
        """Coba sinkronisasi konfigurasi ke Google Drive."""
        try:
            if hasattr(self.config_manager, 'is_symlink_active') and self.config_manager.is_symlink_active():
                self.logger.info("☁️ Konfigurasi otomatis tersinkronisasi ke Google Drive")
            else:
                self.logger.debug("📁 Konfigurasi disimpan secara lokal (Drive tidak aktif)")
        except Exception as e:
            self.logger.warning(f"⚠️ Info sinkronisasi tidak tersedia: {str(e)}")
    
    # Helper methods untuk parsing dan formatting
    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse string resolusi ke tuple."""
        try:
            if isinstance(resolution_str, str) and 'x' in resolution_str:
                width, height = map(int, resolution_str.split('x'))
                return (width, height)
        except:
            pass
        return DEFAULT_IMG_SIZE
    
    def _format_resolution(self, resolution: Tuple[int, int]) -> str:
        """Format tuple resolusi ke string."""
        if isinstance(resolution, tuple) and len(resolution) == 2:
            return f"{resolution[0]}x{resolution[1]}"
        return f"{DEFAULT_IMG_SIZE[0]}x{DEFAULT_IMG_SIZE[1]}"
    
    def _map_split_value(self, ui_value: str) -> str:
        """Map UI split value ke backend format."""
        split_map = {
            'All Splits': 'all',
            'Train Only': 'train',
            'Validation Only': 'val',
            'Test Only': 'test'
        }
        return split_map.get(ui_value, 'all')
    
    def _map_split_to_ui(self, backend_value: str) -> str:
        """Map backend split value ke UI format."""
        ui_map = {
            'all': 'All Splits',
            'train': 'Train Only',
            'val': 'Validation Only',
            'test': 'Test Only'
        }
        return ui_map.get(backend_value, 'All Splits')
    
    # UI helper methods
    def _disable_button(self, button: Any) -> None:
        """Disable button untuk mencegah multiple click."""
        if button and hasattr(button, 'disabled'):
            button.disabled = True
    
    def _enable_button(self, button: Any) -> None:
        """Enable button kembali."""
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    def _update_status(self, status: str, message: str) -> None:
        """Update status panel UI."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
            except ImportError:
                # Fallback manual update
                self.ui_components['status_panel'].value = f"<div class='alert alert-{status}'>{message}</div>"

# Factory functions untuk membuat config handler
def create_config_handler(ui_components: Dict[str, Any]) -> ConfigHandler:
    """
    Factory function untuk membuat config handler.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance ConfigHandler yang siap digunakan
    """
    return ConfigHandler(ui_components)

def setup_config_buttons(ui_components: Dict[str, Any]) -> None:
    """
    Setup config buttons (save & reset) dengan handler.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Buat config handler
    config_handler = create_config_handler(ui_components)
    
    # Simpan reference untuk cleanup nanti
    ui_components['config_handler'] = config_handler
    
    # Setup save button handler
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: config_handler.handle_save_click(b)
        )
    
    # Setup reset button handler  
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: config_handler.handle_reset_click(b)
        )
    
    # Load config ke UI saat setup
    config_handler.load_config_to_ui()