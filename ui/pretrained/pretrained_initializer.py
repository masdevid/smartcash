# File: smartcash/ui/pretrained/pretrained_initializer.py
"""
File: smartcash/ui/pretrained/pretrained_initializer.py
Deskripsi: Pretrained initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.ui.pretrained.handlers.model_handlers import ModelOperationHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler

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
        self._last_models_dir = None
        # Initialize parent with module name and config handler
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
        )
        # Now self.logger and self._logger_bridge are available from parent
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for pretrained models
        
        Returns:
            Dictionary berisi konfigurasi default untuk pretrained models
        """
        return {
            'model_name': 'yolov5s',
            'version': 'latest',
            'device': 'cuda:0',  # Default to GPU if available
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'classes': None,  # None means all classes
            'max_detections': 1000,
            'agnostic_nms': False,
            'augment': False,
            'models_dir': '/data/pretrained',
            'model_urls': {
                'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
                'efficientnet': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-64a7f98d.pth'
            }
        }
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid dengan minimal keys:
            - 'ui': Komponen UI utama
            - 'log_output': Output log widget
            - 'status_panel': Panel status
            
        Raises:
            ValueError: Jika UI components tidak valid atau komponen penting tidak ada
        """
        try:
            from smartcash.ui.pretrained.components.ui_components import create_pretrained_main_ui
            
            # Add logger_bridge to kwargs for UI components
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                kwargs['logger_bridge'] = self._logger_bridge
            
            # Create UI components dengan immediate validation
            ui_components = create_pretrained_main_ui(config, **kwargs)
            
            if not ui_components or not isinstance(ui_components, dict):
                error_msg = f"❌ Gagal membuat UI components untuk {self.module_name}"
                self._log_error(error_msg)
                raise ValueError(error_msg)
            
            # Validate critical components exist
            required_components = ['ui', 'log_output', 'status_panel']
            missing = [comp for comp in required_components if comp not in ui_components]
            if missing:
                error_msg = f"Komponen UI kritis tidak ditemukan: {missing}"
                self._log_error(error_msg)
                raise ValueError(error_msg)
            
            # Add logger_bridge to ui_components if not already present
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                ui_components['logger_bridge'] = self._logger_bridge
            
            # Mark sebagai initialized untuk lifecycle management
            ui_components['pretrained_initialized'] = True
            
            self._log_info(f"✅ UI components berhasil dibuat untuk {self.module_name}")
            return ui_components
            
        except Exception as e:
            self._log_error(f"Gagal membuat komponen UI: {str(e)}", exc_info=True)
            raise
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge if available"""
        if hasattr(self, '_logger_bridge') and self._logger_bridge and hasattr(self._logger_bridge, 'debug'):
            self._logger_bridge.debug(message, **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge if available"""
        if hasattr(self, '_logger_bridge') and self._logger_bridge and hasattr(self._logger_bridge, 'info'):
            self._logger_bridge.info(message, **kwargs)
            
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using logger_bridge if available"""
        if hasattr(self, '_logger_bridge') and self._logger_bridge and hasattr(self._logger_bridge, 'warning'):
            self._logger_bridge.warning(message, **kwargs)
            
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using logger_bridge if available"""
        if hasattr(self, '_logger_bridge') and self._logger_bridge and hasattr(self._logger_bridge, 'error'):
            self._logger_bridge.error(message, exc_info=exc_info, **kwargs)
    
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
            # Inisialisasi model handler dengan logger bridge
            model_handler = ModelOperationHandler(ui_components, self._logger_bridge)
            
            # Setup button handlers
            if 'download_button' in ui_components:
                def on_download_clicked(b):
                    try:
                        model_handler.check_and_download_model(config.get('pretrained_models', {}))
                    except Exception as e:
                        self.handle_error(f"Gagal memproses download: {str(e)}", exc_info=True)
                        self._update_status(ui_components, f"❌ Gagal memproses download", 'error')
                
                ui_components['download_button'].on_click(on_download_clicked)
            
            # Setup sync button if exists
            if 'sync_button' in ui_components:
                def on_sync_clicked(b):
                    try:
                        model_handler.check_and_download_model(config.get('pretrained_models', {}))
                    except Exception as e:
                        self.handle_error(f"Gagal melakukan sinkronisasi: {str(e)}", exc_info=True)
                        self._update_status(ui_components, f"❌ Gagal melakukan sinkronisasi", 'error')
                
                ui_components['sync_button'].on_click(on_sync_clicked)
            
            self._logger_bridge.info("Module handlers berhasil disetup")
            return ui_components
            
        except Exception as e:
            self.handle_error(f"Gagal setup module handlers: {str(e)}", exc_info=True)
            return ui_components
    
    def post_initialization_checks(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> None:
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
            critical_widgets = {
                'download_btn': 'Download button',
                'save_button': 'Save button',
                'reset_button': 'Reset button'
            }
            missing_widgets = [name for key, name in critical_widgets.items() if not ui_components.get(key)]
            if missing_widgets:
                raise RuntimeError(f"Widget kritis tidak ditemukan: {', '.join(missing_widgets)}")
            
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
    
    def _cleanup_old_models_dir(self, old_dir: str, new_dir: str) -> None:
        """🧹 Membersihkan direktori model lama jika berbeda dengan yang baru
        
        Args:
            old_dir: Path direktori model lama
            new_dir: Path direktori model baru
        """
        if not old_dir or old_dir == new_dir:
            return
            
        try:
            import shutil
            from pathlib import Path
            
            old_path = Path(old_dir)
            if old_path.exists() and old_path.is_dir():
                logger.info(f"🧹 Membersihkan direktori model lama: {old_dir}")
                shutil.rmtree(old_dir, ignore_errors=True)
                logger.info(f"✅ Direktori model lama berhasil dibersihkan")
        except Exception as e:
            logger.warning(f"⚠️ Gagal membersihkan direktori model lama: {str(e)}")
    

    def _check_pretrained_model_exists(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """🔍 Check dan download otomatis pretrained model jika tidak ditemukan
        
        Args:
            ui_components: Dictionary UI components untuk update status
            config: Konfigurasi sistem untuk mendapatkan path model
            
        Returns:
            bool: True jika model tersedia atau berhasil didownload
        """
        try:
            # Inisialisasi handler dengan logger bridge
            model_handler = ModelOperationHandler(ui_components, self._logger_bridge)
            
            # Lakukan pengecekan dan download jika diperlukan
            success = model_handler.check_and_download_model(config.get('pretrained_models', {}))
            
            # Update status di UI
            if success:
                self._update_status(ui_components, "✅ Model siap digunakan", 'success')
            else:
                self._update_status(ui_components, "❌ Gagal memuat model", 'error')
            
            return success
            
        except Exception as e:
            error_msg = f"Gagal memeriksa model: {str(e)}"
            self.handle_error(error_msg, exc_info=True)
            self._update_status(ui_components, f"❌ {error_msg}", 'error')
            return False


def initialize_pretrained_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Factory function untuk inisialisasi pretrained UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Parameter tambahan yang akan diteruskan ke initializer
        
    Returns:
        Widget UI utama yang siap ditampilkan atau dictionary dengan 'ui' key
        
    Example:
        ```python
        ui = initialize_pretrained_ui(config=my_config)
        display(ui)  # or display(ui['ui']) if it's a dict
        ```
    """
    try:
        initializer = PretrainedInitializer()
        result = initializer.initialize(config=config, **kwargs)
        
        # Handle error response
        if isinstance(result, dict) and result.get('error'):
            return result['ui']
        return result
    except Exception as e:
        error_msg = f"❌ Gagal menginisialisasi pretrained UI: {str(e)}"
        return {'ui': create_error_component(error_msg, str(e), "Pretrained Model Error"), 'error': True}