"""
File: smartcash/ui/dataset/preprocessing/handlers/operation_handlers.py
Deskripsi: Operation handlers dengan minimal logging dan proper button handling
"""

from typing import Dict, Any
from .base_handler import BasePreprocessingHandler
from smartcash.ui.dataset.preprocessing import utils as ui_utils

class OperationHandler(BasePreprocessingHandler):
    """Handler untuk main operations dengan minimal init logging"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup operation button handlers"""
        handlers = {}
        
        # Setup handlers dengan minimal logging
        preprocess_handler = self.setup_button_handler(
            'preprocess_btn', self._handle_preprocess_operation, 'preprocess'
        )
        if preprocess_handler:
            handlers['preprocess'] = preprocess_handler
        
        check_handler = self.setup_button_handler(
            'check_btn', self._handle_check_operation, 'check'
        )
        if check_handler:
            handlers['check'] = check_handler
        
        cleanup_handler = self.setup_button_handler(
            'cleanup_btn', self._handle_cleanup_operation, 'cleanup'
        )
        if cleanup_handler:
            handlers['cleanup'] = cleanup_handler
        
        # SUPPRESSED: minimal success logging
        if handlers:
            self.log_debug("Operation handlers ready")
        
        return handlers
    
    def _handle_preprocess_operation(self) -> None:
        """Handle preprocessing operation"""
        ui_utils.clear_outputs(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            return
        
        self._show_preprocessing_confirmation()
    
    def _handle_check_operation(self) -> None:
        """Handle dataset check operation"""
        self.log_info("🔍 Memeriksa dataset...")
        ui_utils.clear_outputs(self.ui_components)
        
        config = self.extract_config()
        if not config:
            raise ValueError("Konfigurasi tidak valid")
        
        progress_callback = self.create_progress_callback()
        ui_utils.setup_progress(self.ui_components, "🔍 Memeriksa dataset...")
        
        from smartcash.dataset.preprocessor.api import get_preprocessing_status
        result = get_preprocessing_status(config=config)
        
        self.process_operation_result(result, 'check')
        self.update_status_panel("Pemeriksaan selesai", 'success')
    
    def _handle_cleanup_operation(self) -> None:
        """Handle cleanup operation"""
        ui_utils.clear_outputs(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            return
        
        self._show_cleanup_confirmation()
    
    def _show_preprocessing_confirmation(self) -> None:
        """Show preprocessing confirmation"""
        self.show_confirmation_dialog(
            title="Konfirmasi Preprocessing",
            message="Proses dataset dengan YOLO normalization?",
            on_confirm=self._set_preprocessing_confirmed,
            on_cancel=lambda: self.handle_operation_cancel('preprocessing', '_preprocessing_confirmed'),
            confirm_text="Ya, Proses",
            cancel_text="Batal"
        )
        self.log_info("⏳ Menunggu konfirmasi preprocessing...")
    
    def _show_cleanup_confirmation(self) -> None:
        """Show cleanup confirmation"""
        config = self.extract_config()
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        target_descriptions = {
            'preprocessed': 'file preprocessing (pre_*.npy + pre_*.txt)',
            'samples': 'sample images (sample_*.jpg)',
            'both': 'file preprocessing dan sample images'
        }
        target_desc = target_descriptions.get(cleanup_target, cleanup_target)
        
        self.show_confirmation_dialog(
            title="🧹 Konfirmasi Cleanup",
            message=f"Hapus {target_desc}?\n\nTindakan ini akan menghapus file yang sudah diproses.",
            on_confirm=self._set_cleanup_confirmed,
            on_cancel=lambda: self.handle_operation_cancel('cleanup', '_cleanup_confirmed'),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        self.log_info(f"⏳ Konfirmasi cleanup: {target_desc}")
    
    def _set_preprocessing_confirmed(self) -> None:
        """Confirm dan execute preprocessing"""
        self.log_info("✅ Mulai preprocessing...")
        self.ui_components['_preprocessing_confirmed'] = True
        self._execute_preprocessing_with_api()
    
    def _set_cleanup_confirmed(self) -> None:
        """Confirm dan execute cleanup"""
        self.log_info("✅ Mulai cleanup...")
        self.ui_components['_cleanup_confirmed'] = True
        self._execute_cleanup_with_api()
        
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(self.ui_components)
    
    def _execute_preprocessing_with_api(self) -> None:
        """Execute preprocessing dengan minimal progress logs"""
        self.log_info("🚀 Starting preprocessing...")
        ui_utils.disable_buttons(self.ui_components)
        ui_utils.setup_progress(self.ui_components, "🚀 Preprocessing...")
        
        try:
            config = self.extract_config()
            progress_callback = self.create_progress_callback()
            
            from smartcash.dataset.preprocessor.api import preprocess_dataset
            result = preprocess_dataset(
                config=config,
                progress_callback=progress_callback,
                ui_components=self.ui_components
            )
            
            self.process_operation_result(result, 'preprocessing')
            
        finally:
            ui_utils.enable_buttons(self.ui_components)
    
    def _execute_cleanup_with_api(self) -> None:
        """Execute cleanup dengan minimal progress logs"""
        self.log_info("🧹 Starting cleanup...")
        ui_utils.disable_buttons(self.ui_components)
        ui_utils.setup_progress(self.ui_components, "🗑️ Cleanup...")
        
        try:
            config = self.extract_config()
            cleanup_config = config.get('preprocessing', {}).get('cleanup', {})
            
            cleanup_target = cleanup_config.get('target', 'preprocessed')
            target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
            data_dir = config.get('data', {}).get('dir', 'data')
            
            progress_callback = self.create_progress_callback()
            
            from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files
            result = cleanup_preprocessing_files(
                data_dir=data_dir,
                target=cleanup_target,
                splits=target_splits,
                confirm=True,
                progress_callback=progress_callback,
                ui_components=self.ui_components
            )
            
            self.process_operation_result(result, 'cleanup')
            
        finally:
            ui_utils.enable_buttons(self.ui_components)

# Factory function
def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup operation handlers"""
    handler = OperationHandler(ui_components)
    return handler.setup_handlers()