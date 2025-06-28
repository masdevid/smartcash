# File: smartcash/ui/pretrained/pretrained_initializer.py
"""
File: smartcash/ui/pretrained/pretrained_initializer.py
Deskripsi: Pretrained initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedInitializer(CommonInitializer):
    """🤖 Pretrained models initializer dengan CommonInitializer pattern terbaru"""
    
    def __init__(self, module_name: str = 'pretrained', 
                 config_handler_class: Type[ConfigHandler] = PretrainedConfigHandler,
                 **kwargs):
        """Initialize pretrained initializer dengan fail-fast validation
        
        Args:
            module_name: Nama modul (default: 'pretrained')
            config_handler_class: Kelas handler konfigurasi
            **kwargs: Argumen tambahan untuk parent class
        """
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid
            
        Raises:
            ValueError: Jika UI components tidak valid atau kosong
        """
        from smartcash.ui.pretrained.components.ui_components import create_pretrained_main_ui
        
        # Create UI components dengan immediate validation
        ui_components = create_pretrained_main_ui(config, **kwargs)
        
        if not ui_components or not isinstance(ui_components, dict):
            raise ValueError(f"❌ Gagal membuat UI components untuk {self.module_name}")
        
        # Mark sebagai initialized untuk lifecycle management
        ui_components['pretrained_initialized'] = True
        
        logger.info(f"✅ UI components berhasil dibuat untuk {self.module_name}")
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup module-specific handlers dengan error handling
        
        Args:
            ui_components: Dictionary UI components
            config: Konfigurasi sistem
            **kwargs: Parameter tambahan
            
        Returns:
            Updated UI components dengan handlers
        """
        try:
            from smartcash.ui.pretrained.handlers.event_handlers import setup_all_handlers
            
            # Setup semua handlers untuk pretrained module
            updated_components = setup_all_handlers(ui_components, config, **kwargs)
            
            logger.info("✅ Module handlers berhasil disetup")
            return updated_components
            
        except ImportError as e:
            logger.warning(f"⚠️ Event handlers belum tersedia: {str(e)}")
            return ui_components
        except Exception as e:
            logger.error(f"❌ Error setup module handlers: {str(e)}")
            return ui_components
    
    def _after_init_checks(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> None:
        """Post-initialization checks untuk validasi model pretrained
        
        Args:
            ui_components: Komponen UI yang telah diinisialisasi
            config: Konfigurasi sistem
            **kwargs: Arguments tambahan
            
        Raises:
            RuntimeError: Jika post-init validation gagal
        """
        try:
            # Check critical UI components
            critical_widgets = ['download_sync_button', 'save_button', 'reset_button']
            missing_widgets = [w for w in critical_widgets if not ui_components.get(w)]
            if missing_widgets:
                raise RuntimeError(f"Widget kritis tidak ditemukan: {missing_widgets}")
            
            # Check pretrained model existence
            self._check_pretrained_model_exists(ui_components, config)
            
            # Test logger bridge functionality
            if self._logger_bridge:
                try:
                    self._logger_bridge.info("🤖 Pretrained module initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Logger bridge test warning: {str(e)}")
                    
        except Exception as e:
            logger.error(f"❌ Post-initialization checks failed: {str(e)}")
            raise
    
    def _check_pretrained_model_exists(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """🔍 Check apakah pretrained model sudah ada di direktori
        
        Args:
            ui_components: Dictionary UI components untuk update status
            config: Konfigurasi sistem untuk mendapatkan path model
        """
        try:
            from smartcash.ui.pretrained.services.model_checker import check_model_exists
            
            pretrained_config = config.get('pretrained_models', {})
            models_dir = pretrained_config.get('models_dir', '/content/models')
            drive_dir = pretrained_config.get('drive_models_dir', '/data/pretrained')
            model_type = pretrained_config.get('pretrained_type', 'yolov5s')
            
            # Check model existence
            local_exists = check_model_exists(models_dir, model_type)
            drive_exists = check_model_exists(drive_dir, model_type)
            
            # Update UI status based on model availability
            if local_exists and drive_exists:
                status_msg = f"✅ Model {model_type} tersedia di local dan drive"
                status_style = "success"
                logger.info(f"✅ Model {model_type} found in both locations")
                
            elif local_exists:
                status_msg = f"⚠️ Model {model_type} hanya tersedia di local"
                status_style = "warning"
                logger.warning(f"⚠️ Model {model_type} only found locally")
                
            elif drive_exists:
                status_msg = f"⚠️ Model {model_type} hanya tersedia di drive"
                status_style = "warning"
                logger.warning(f"⚠️ Model {model_type} only found in drive")
                
            else:
                status_msg = f"❌ Model {model_type} tidak ditemukan - perlu download"
                status_style = "error"
                logger.warning(f"❌ Model {model_type} not found in any location")
            
            # Update status di UI
            if 'status' in ui_components:
                ui_components['status'].value = status_msg
            
            # Update button state based on availability
            if 'download_sync_button' in ui_components:
                if local_exists and drive_exists:
                    ui_components['download_sync_button'].description = '🔄 Re-sync Models'
                    ui_components['download_sync_button'].button_style = 'info'
                else:
                    ui_components['download_sync_button'].description = '📥 Download & Sync'
                    ui_components['download_sync_button'].button_style = 'primary'
            
            # Store model status dalam UI components untuk referensi - menggunakan parent method
            ui_components['model_status'] = {
                'local_exists': local_exists,
                'drive_exists': drive_exists,
                'model_type': model_type,
                'last_checked': self._get_timestamp()  # Using parent method
            }
            
        except Exception as e:
            error_msg = f"❌ Error checking pretrained model: {str(e)}"
            logger.error(error_msg)
            
            # Update UI dengan error status
            if 'status' in ui_components:
                ui_components['status'].value = error_msg


def initialize_pretrained_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """🚀 Factory function untuk inisialisasi pretrained UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Parameter tambahan
        
    Returns:
        Dictionary berisi komponen UI yang sudah diinisialisasi
        
    Raises:
        RuntimeError: Jika inisialisasi gagal
    """
    try:
        initializer = PretrainedInitializer()
        return initializer.initialize(config or {}, **kwargs)
    except Exception as e:
        logger.error(f"❌ Gagal inisialisasi pretrained UI: {str(e)}")
        raise RuntimeError(f"Pretrained UI initialization failed: {str(e)}") from e