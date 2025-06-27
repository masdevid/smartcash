"""
File: smartcash/ui/dataset/preprocessing/handlers/operation_handlers.py
Deskripsi: Operation handlers menggunakan BasePreprocessingHandler untuk DRY implementation
"""

from typing import Dict, Any
from .base_handler import BasePreprocessingHandler
from smartcash.ui.dataset.preprocessing import utils as ui_utils

class OperationHandler(BasePreprocessingHandler):
    """Handler untuk main operations (preprocess/check/cleanup) menggunakan base class"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup operation button handlers"""
        handlers = {}
        
        # Setup preprocess handler
        preprocess_handler = self.setup_button_handler(
            'preprocess_btn',
            self._handle_preprocess_operation,
            'preprocess'
        )
        if preprocess_handler:
            handlers['preprocess'] = preprocess_handler
        
        # Setup check handler
        check_handler = self.setup_button_handler(
            'check_btn',
            self._handle_check_operation,
            'check'
        )
        if check_handler:
            handlers['check'] = check_handler
        
        # Setup cleanup handler
        cleanup_handler = self.setup_button_handler(
            'cleanup_btn',
            self._handle_cleanup_operation,
            'cleanup'
        )
        if cleanup_handler:
            handlers['cleanup'] = cleanup_handler
        
        self.log_debug("✅ Operation handlers setup completed")
        return handlers
    
    def _handle_preprocess_operation(self) -> None:
        """Handle preprocessing operation dengan confirmation"""
        self.log_debug("🚀 Starting preprocessing operation...")
        ui_utils.clear_outputs(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            return
        
        self._show_preprocessing_confirmation()
    
    def _handle_check_operation(self) -> None:
        """Handle dataset check operation"""
        self.log_info("🔍 Memulai pemeriksaan dataset...")
        ui_utils.clear_outputs(self.ui_components)
        
        config = self.extract_config()
        if not config:
            raise ValueError("Konfigurasi tidak valid atau kosong")
        
        progress_callback = self.create_progress_callback()
        ui_utils.setup_progress(self.ui_components, "🔍 Memeriksa dataset...")
        
        from smartcash.dataset.preprocessor.api import get_preprocessing_status
        result = get_preprocessing_status(config=config)
        
        self.process_operation_result(result, 'check')
        self.log_success("✅ Pemeriksaan dataset selesai")
        self.update_status_panel("Pemeriksaan selesai", 'success')
    
    def _handle_cleanup_operation(self) -> None:
        """Handle cleanup operation dengan confirmation"""
        self.log_debug("🧹 Starting cleanup operation...")
        ui_utils.clear_outputs(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            return
        
        self._show_cleanup_confirmation()
    
    def _show_preprocessing_confirmation(self) -> None:
        """Show preprocessing confirmation dialog"""
        self.show_confirmation_dialog(
            title="Konfirmasi Preprocessing",
            message="Apakah Anda yakin ingin memproses dataset dengan YOLO normalization?",
            on_confirm=self._set_preprocessing_confirmed,
            on_cancel=lambda: self.handle_operation_cancel('preprocessing', '_preprocessing_confirmed'),
            confirm_text="Ya, Proses",
            cancel_text="Batal"
        )
        self.log_info("⏳ Menunggu konfirmasi preprocessing...")
    
    def _show_cleanup_confirmation(self) -> None:
        """Show cleanup confirmation dialog"""
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
            message=f"Hapus {target_desc}?\n\nTindakan ini akan menghapus file-file yang sudah diproses.",
            on_confirm=self._set_cleanup_confirmed,
            on_cancel=lambda: self.handle_operation_cancel('cleanup', '_cleanup_confirmed'),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        self.log_info(f"⏳ Menunggu konfirmasi cleanup untuk: {target_desc}...")
    
    def _set_preprocessing_confirmed(self) -> None:
        """Set preprocessing confirmation dan execute"""
        self.log_info("✅ Konfirmasi diterima, memulai preprocessing...")
        self.ui_components['_preprocessing_confirmed'] = True
        self._execute_preprocessing_with_api()
    
    def _set_cleanup_confirmed(self) -> None:
        """Set cleanup confirmation dan execute"""
        self.log_info("✅ Konfirmasi cleanup diterima, memulai pembersihan...")
        self.ui_components['_cleanup_confirmed'] = True
        self._execute_cleanup_with_api()
        
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(self.ui_components)
    
    def _execute_preprocessing_with_api(self) -> None:
        """Execute preprocessing menggunakan API"""
        self.log_info("🚀 Starting preprocessing pipeline...")
        ui_utils.disable_buttons(self.ui_components)
        ui_utils.setup_progress(self.ui_components, "🚀 Starting preprocessing...")
        
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
            self.log_success("✅ Preprocessing completed successfully")
            
        finally:
            ui_utils.enable_buttons(self.ui_components)
    
    def _execute_cleanup_with_api(self) -> None:
        """Execute cleanup menggunakan API"""
        self.log_info("🧹 Starting cleanup with API...")
        ui_utils.disable_buttons(self.ui_components)
        ui_utils.setup_progress(self.ui_components, "🗑️ Starting cleanup...")
        
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
            self.log_success("✅ Cleanup completed successfully")
            
        finally:
            ui_utils.enable_buttons(self.ui_components)

# Factory function untuk backward compatibility
def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function untuk setup operation handlers"""
    handler = OperationHandler(ui_components)
    return handler.setup_handlers()