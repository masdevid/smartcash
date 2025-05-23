"""
File: smartcash/ui/dataset/preprocessing/handlers/main_handler.py
Deskripsi: Handler utama yang terintegrasi dengan stop functionality
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.components.config_manager import get_config_from_ui

logger = get_logger(__name__)

class PreprocessingHandler:
    """Handler utama untuk operasi preprocessing dengan stop functionality."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi preprocessing handler."""
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', logger)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_future: Future = None
        self.is_running = False
        
    def handle_button_click(self, button: Any) -> None:
        """Handler untuk tombol preprocessing utama."""
        if self.is_running:
            self.stop_processing()
            return
            
        button.disabled = True
        
        try:
            self.ui_components['stop_requested'] = False
            config = get_config_from_ui(self.ui_components)
            
            self._log_config(config)
            self._update_status("info", "⚙️ Mempersiapkan preprocessing...")
            self._start_preprocessing(config)
            
        except Exception as e:
            self.logger.error(f"❌ Error persiapan: {str(e)}")
            self._handle_error(str(e))
        finally:
            button.disabled = False
    
    def _start_preprocessing(self, config: Dict[str, Any]) -> None:
        """Start preprocessing secara async."""
        self.is_running = True
        self._update_ui_for_start()
        
        self.current_future = self.executor.submit(self._run_preprocessing, config)
        self.current_future.add_done_callback(self._on_complete)
        
        self.logger.info("🚀 Preprocessing dimulai")
    
    def _run_preprocessing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run preprocessing di background thread."""
        try:
            from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
            
            manager = PreprocessingManager(config, self.logger)
            
            if hasattr(manager, 'register_progress_callback'):
                manager.register_progress_callback(self._progress_callback)
            
            result = manager.preprocess_dataset(
                split=config.get('preprocessing', {}).get('split', 'all'),
                force_reprocess=True,
                show_progress=True
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error preprocessing: {str(e)}")
            raise
    
    def _progress_callback(self, **kwargs) -> bool:
        """Callback untuk progress updates dengan stop check."""
        if self.ui_components.get('stop_requested', False):
            return False
        
        try:
            progress = kwargs.get('progress', 0)
            total = kwargs.get('total', 100)
            message = kwargs.get('message', '')
            
            if 'progress_bar' in self.ui_components:
                percentage = (progress / total * 100) if total > 0 else 0
                self.ui_components['progress_bar'].value = percentage
            
            if 'overall_label' in self.ui_components and message:
                self.ui_components['overall_label'].value = f"🔄 {message} ({progress}/{total})"
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error progress: {str(e)}")
            return True
    
    def stop_processing(self) -> None:
        """Stop preprocessing yang sedang berjalan."""
        if not self.is_running:
            return
            
        self.ui_components['stop_requested'] = True
        
        if self.current_future and not self.current_future.done():
            self.current_future.cancel()
        
        self.logger.warning("⏹️ Preprocessing dihentikan")
        self._update_status("warning", "Preprocessing dihentikan")
        self._reset_ui()
    
    def _on_complete(self, future: Future) -> None:
        """Callback saat preprocessing selesai."""
        try:
            if not future.cancelled():
                result = future.result()
                self._handle_success(result)
        except Exception as e:
            if not future.cancelled():
                self._handle_error(str(e))
        finally:
            self._reset_ui()
    
    def _handle_success(self, result: Dict[str, Any]) -> None:
        """Handle preprocessing berhasil."""
        total_processed = result.get('total_images', 0)
        
        self.logger.info(f"✅ Preprocessing berhasil: {total_processed} file")
        self._update_status("success", f"✅ Selesai - {total_processed} file diproses")
    
    def _handle_error(self, error_message: str) -> None:
        """Handle error preprocessing."""
        self.logger.error(f"❌ Error: {error_message}")
        self._update_status("error", f"❌ Error: {error_message}")
    
    def _update_ui_for_start(self) -> None:
        """Update UI saat mulai preprocessing."""
        # Change button text to "Stop"
        if 'preprocess_button' in self.ui_components:
            self.ui_components['preprocess_button'].description = "Stop Preprocessing"
            self.ui_components['preprocess_button'].button_style = 'danger'
            self.ui_components['preprocess_button'].icon = 'stop'
        
        # Disable other buttons
        for btn in ['save_button', 'reset_button', 'cleanup_button']:
            if btn in self.ui_components:
                self.ui_components[btn].disabled = True
        
        # Show progress
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'visible'
    
    def _reset_ui(self) -> None:
        """Reset UI setelah selesai/stop."""
        self.is_running = False
        
        # Reset button
        if 'preprocess_button' in self.ui_components:
            self.ui_components['preprocess_button'].description = "Mulai Preprocessing" 
            self.ui_components['preprocess_button'].button_style = 'success'
            self.ui_components['preprocess_button'].icon = 'play'
            self.ui_components['preprocess_button'].disabled = False
        
        # Enable other buttons
        for btn in ['save_button', 'reset_button', 'cleanup_button']:
            if btn in self.ui_components:
                self.ui_components[btn].disabled = False
        
        # Reset progress
        if 'progress_bar' in self.ui_components:
            self.ui_components['progress_bar'].value = 0
        if 'overall_label' in self.ui_components:
            self.ui_components['overall_label'].value = ""
    
    def _log_config(self, config: Dict[str, Any]) -> None:
        """Log konfigurasi preprocessing."""
        pc = config.get('preprocessing', {})
        self.logger.info("🔧 Konfigurasi:")
        self.logger.info(f"  • Resolusi: {pc.get('img_size')}")
        self.logger.info(f"  • Normalisasi: {pc.get('normalization')}")
        self.logger.info(f"  • Workers: {pc.get('num_workers')}")
        self.logger.info(f"  • Split: {pc.get('split')}")
    
    def _update_status(self, status: str, message: str) -> None:
        """Update status panel."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
            except ImportError:
                self.ui_components['status_panel'].value = f"<div class='alert alert-{status}'>{message}</div>"


def setup_main_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol preprocessing dengan stop functionality."""
    if 'preprocess_button' not in ui_components:
        return
    
    handler = PreprocessingHandler(ui_components)
    ui_components['main_handler'] = handler
    
    ui_components['preprocess_button'].on_click(
        lambda b: handler.handle_button_click(b)
    )