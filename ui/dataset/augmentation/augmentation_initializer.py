"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer yang mengimplementasikan CommonInitializer
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler

class AugmentationInitializer(CommonInitializer):
    """Initializer untuk modul augmentasi yang mengimplementasikan CommonInitializer"""
    
    def __init__(self):
        super().__init__(
            module_name='dataset.augmentation',
            config_handler_class=AugmentationConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Buat komponen UI untuk augmentasi.
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan (mengandung 'env' untuk environment)
            
        Returns:
            Dictionary berisi komponen UI
        """
        from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
        
        ui_components = create_augmentation_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError("UI components harus berupa dictionary")
            
        if not ui_components:
            raise ValueError("UI components tidak boleh kosong")
        
        # Tambahkan metadata dan konfigurasi
        ui_components.update({
            'module_name': self.module_name.split('.')[-1],
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'env': kwargs.get('env'),
            'backend_ready': True,
            'service_integration': True
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup event handlers untuk komponen UI.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
        """
        from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers
        
        # Setup handlers
        handlers = setup_augmentation_handlers(ui_components, config, kwargs.get('env'))
        ui_components['handlers'] = handlers
        
        # Load and update UI with config
        self._load_and_update_ui(ui_components)
        
        return ui_components
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]) -> None:
        """
        Muat konfigurasi dan perbarui UI.
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            return
            
        try:
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            loaded_config = config_handler.load_config()
            if hasattr(config_handler, 'update_ui'):
                config_handler.update_ui(ui_components, loaded_config)
            
            ui_components['config'] = loaded_config
            
        except Exception as e:
            self.logger.warning(f"Gagal memuat konfigurasi: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi default untuk modul augmentasi.
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        return get_default_augmentation_config()

def initialize_augmentation_ui(env: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Inisialisasi UI untuk modul augmentasi.
    
    Args:
        env: Konfigurasi environment opsional
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Komponen UI utama
    """
    return AugmentationInitializer().initialize(env=env, config=config, **kwargs)