"""
File: smartcash/ui/setup/dependency/handlers/dependency_handler_refactored.py
Deskripsi: Refactored dependency handler dengan peningkatan struktur dan manajemen state

Fitur Utama:
- Manajemen state UI yang lebih baik
- Error handling yang lebih kuat
- Kode yang lebih modular dan mudah dipelihara
- Dukungan untuk operasi asinkron
"""

from dataclasses import dataclass
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum, auto

# Core imports
from smartcash.common import get_logger
from smartcash.ui.setup.dependency.utils import (
    LogLevel, with_logging, requires, log_to_ui_safe, with_button_context,
    create_operation_context, update_status_panel
)

# Setup logger
logger = get_logger(__name__)

class OperationType(Enum):
    """Jenis operasi konfigurasi"""
    SAVE = auto()
    LOAD = auto()
    RESET = auto()

@dataclass
class OperationResult:
    """Hasil operasi dengan status dan pesan"""
    success: bool
    message: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        self.details = self.details or {}

class ProgressTracker:
    """Kelas pembantu untuk melacak progress"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.tracker = ui_components.get('progress_tracker')
        self.logger = ui_components.get('logger', logger)
    
    def update(self, level: str, progress: int, message: str, **kwargs) -> None:
        """Update progress dengan error handling"""
        if not self.tracker:
            return
            
        try:
            if hasattr(self.tracker, 'update'):
                self.tracker.update(level, progress, message, **kwargs)
            elif hasattr(self.tracker, f'update_{level}'):
                getattr(self.tracker, f'update_{level}')(progress, message, **kwargs)
        except Exception as e:
            self.logger.warning(f"Gagal update progress: {str(e)}")
    
    def complete(self, message: str = None) -> None:
        """Tandai proses selesai"""
        if hasattr(self.tracker, 'complete'):
            try:
                self.tracker.complete(message or "✅ Proses selesai")
            except Exception as e:
                self.logger.warning(f"Gagal menandai selesai: {str(e)}")

class DependencyHandler:
    """Kelas utama untuk menangani dependensi"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        self.ui = ui_components
        self.config = config
        self.progress = ProgressTracker(ui_components)
        self.logger = ui_components.get('logger', logger)
    
    @classmethod
    def setup(cls, ui_components: Dict[str, Any], config: Dict[str, Any]) -> 'DependencyHandler':
        """Factory method untuk inisialisasi handler"""
        handler = cls(ui_components, config)
        handler._initialize()
        return handler
    
    def _initialize(self) -> None:
        """Inisialisasi handler dan komponen terkait"""
        self._cleanup_existing_generators()
        self._setup_handlers()
        self._setup_config_handlers()
        
        if self.ui.get('auto_analyze_on_render', True):
            self._setup_auto_analyze()
    
    def _cleanup_existing_generators(self) -> int:
        """Bersihkan generator yang ada"""
        count = 0
        for name, comp in list(self.ui.items()):
            if hasattr(comp, 'close') and hasattr(comp, '__next__'):
                try:
                    comp.close()
                    del self.ui[name]
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Gagal menutup generator {name}: {str(e)}")
        return count
    
    def _setup_handlers(self) -> None:
        """Setup semua handler yang diperlukan"""
        try:
            self._setup_installation_handler()
            self._setup_analysis_handler()
            self._setup_status_check_handler()
            self.logger.info("Semua handler berhasil diinisialisasi")
        except Exception as e:
            self.logger.error("Gagal menginisialisasi handler", exc_info=True)
            raise
    
    def _setup_installation_handler(self) -> None:
        """Setup installation handler"""
        from .installation_handler import setup_installation_handler
        setup_installation_handler(self.ui, self.config)
    
    def _setup_analysis_handler(self) -> None:
        """Setup analysis handler"""
        from .analysis_handler import setup_analysis_handler
        setup_analysis_handler(self.ui, self.config)
    
    def _setup_status_check_handler(self) -> None:
        """Setup status check handler"""
        from .status_check_handler import setup_status_check_handler
        setup_status_check_handler(self.ui, self.config)
    
    def _setup_config_handlers(self) -> None:
        """Setup handler untuk operasi konfigurasi"""
        config_handler = self.ui.get('config_handler')
        if not config_handler:
            self.logger.warning("Config handler tidak ditemukan")
            return
        
        self._setup_button('load_config_button', self._handle_load_config)
        self._setup_button('save_config_button', self._handle_save_config)
        self._setup_button('reset_button', self._handle_reset)
    
    def _setup_button(self, button_name: str, handler: Callable) -> None:
        """Setup handler untuk tombol dengan validasi"""
        if button_name not in self.ui:
            self.logger.warning(f"Tombol {button_name} tidak ditemukan")
            return
            
        button = self.ui[button_name]
        if hasattr(button, 'on_click'):
            button.on_click(
                lambda b, h=handler: with_button_context(self.ui, button_name)(lambda: h(b))
            )
    
    def _handle_load_config(self, button) -> None:
        """Handle load config"""
        with create_operation_context(self.ui, 'load_config'):
            config = self.ui['config_handler'].load_config()
            result = OperationResult(
                success=bool(config),
                message="Konfigurasi berhasil dimuat" if config else "Gagal memuat konfigurasi"
            )
            self._handle_operation_result(result, OperationType.LOAD)
    
    def _handle_save_config(self, button) -> None:
        """Handle save config"""
        with create_operation_context(self.ui, 'save_config'):
            current_config = self._extract_current_config()
            success = self.ui['config_handler'].save_config(current_config)
            result = OperationResult(
                success=success,
                message="Konfigurasi berhasil disimpan" if success else "Gagal menyimpan konfigurasi"
            )
            self._handle_operation_result(result, OperationType.SAVE)
    
    def _handle_reset(self, button) -> None:
        """Handle reset UI"""
        with create_operation_context(self.ui, 'reset'):
            self._reset_ui()
            result = OperationResult(
                success=True,
                message="UI berhasil direset"
            )
            self._handle_operation_result(result, OperationType.RESET)
    
    def _handle_operation_result(self, result: OperationResult, operation: OperationType) -> None:
        """Handle hasil operasi dengan feedback yang sesuai"""
        log_func = self.logger.info if result.success else self.logger.error
        log_func(result.message)
        
        # Update UI status
        update_status_panel(
            self.ui,
            result.message,
            'success' if result.success else 'error'
        )
        
        # Log tambahan jika diperlukan
        if result.details:
            self.logger.debug(f"Detail operasi: {result.details}")
    
    def _extract_current_config(self) -> Dict[str, Any]:
        """Ekstrak konfigurasi saat ini"""
        from . import config_extractor
        return config_extractor.extract_dependency_config(self.ui)
    
    def _reset_ui(self) -> None:
        """Reset UI ke pengaturan default"""
        from . import config_updater
        config_updater.reset_dependency_ui(self.ui)
    
    def _setup_auto_analyze(self) -> None:
        """Setup analisis otomatis"""
        # Hanya set flag untuk analisis nanti
        # Komponen UI yang memanggil setup ini harus mengeksekusi analisis
        # setelah UI selesai dirender
        self.ui['_pending_auto_analyze'] = True

# Fungsi kompatibilitas
def setup_dependency_handlers(ui_components: Dict[str, Any], 
                           config: Dict[str, Any], 
                           env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fungsi utama untuk setup handler (kompatibilitas ke belakang)
    
    Args:
        ui_components: Komponen UI
        config: Konfigurasi
        env: Environment (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    return DependencyHandler.setup(ui_components, config).ui
