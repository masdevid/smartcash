"""
File: smartcash/ui/dataset/preprocessing/handlers/operation_handlers.py
Deskripsi: Enhanced operation handlers dengan log accordion control dan proper status management
"""

from typing import Dict, Any
from .base_handler import BasePreprocessingHandler
from smartcash.ui.dataset.preprocessing.utils.ui_utils import (
    start_operation_flow, complete_operation_flow,
    reset_and_expand_log_accordion, update_status_panel_enhanced
)

class OperationHandler(BasePreprocessingHandler):
    """Enhanced handler untuk operations dengan log accordion control"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup operation button handlers dengan enhanced UI control"""
        handlers = {}
        
        # Setup handlers tanpa init logs
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
        
        return handlers
    
    def _handle_preprocess_operation(self) -> None:
        """Handle preprocessing operation dengan enhanced flow"""
        # Reset dan expand log accordion
        reset_and_expand_log_accordion(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            update_status_panel_enhanced(self.ui_components, "⚠️ Konfirmasi pending", 'warning')
            return
        
        self._show_preprocessing_confirmation()
    
    def _handle_check_operation(self) -> None:
        """Handle dataset check operation dengan enhanced flow"""
        # Start operation flow dengan log accordion control
        start_operation_flow(self.ui_components, "Dataset Check")
        
        try:
            config = self.extract_config()
            if not config:
                raise ValueError("Konfigurasi tidak valid")
            
            progress_callback = self.create_progress_callback()
            
            # Suppress output selama API call
            from smartcash.ui.utils.logging_utils import suppress_all_outputs
            suppress_all_outputs()
            
            from smartcash.dataset.preprocessor.api import get_preprocessing_status
            result = get_preprocessing_status(config=config)
            
            self.process_operation_result(result, 'check')
            complete_operation_flow(self.ui_components, "Dataset Check", True, "Pemeriksaan selesai")
            
        except Exception as e:
            error_msg = f"Gagal memeriksa dataset: {str(e)}"
            self.log_error(error_msg)
            complete_operation_flow(self.ui_components, "Dataset Check", False, error_msg)
    
    def _handle_cleanup_operation(self) -> None:
        """Handle cleanup operation dengan enhanced flow"""
        # Reset dan expand log accordion
        reset_and_expand_log_accordion(self.ui_components)
        
        if self.is_confirmation_pending():
            self.log_warning("⚠️ Ada operasi konfirmasi yang sedang menunggu")
            update_status_panel_enhanced(self.ui_components, "⚠️ Konfirmasi pending", 'warning')
            return
        
        self._show_cleanup_confirmation()
    
    def _show_preprocessing_confirmation(self) -> None:
        """Show preprocessing confirmation dengan status update"""
        self.show_confirmation_dialog(
            title="Konfirmasi Preprocessing",
            message="Proses dataset dengan YOLO normalization?",
            on_confirm=self._set_preprocessing_confirmed,
            on_cancel=lambda: self._handle_operation_cancel('preprocessing'),
            confirm_text="Ya, Proses",
            cancel_text="Batal"
        )
        self.log_info("⏳ Menunggu konfirmasi preprocessing...")
        update_status_panel_enhanced(self.ui_components, "⏳ Menunggu konfirmasi...", 'info')
    
    def _show_cleanup_confirmation(self) -> None:
        """Show cleanup confirmation dengan target info dan status update"""
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
            on_cancel=lambda: self._handle_operation_cancel('cleanup'),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        self.log_info(f"⏳ Konfirmasi cleanup: {target_desc}")
        update_status_panel_enhanced(self.ui_components, f"⏳ Konfirmasi cleanup...", 'info')
    
    def _set_preprocessing_confirmed(self) -> None:
        """Confirm dan execute preprocessing dengan enhanced flow"""
        self.ui_components['_preprocessing_confirmed'] = True
        
        # Start operation flow
        start_operation_flow(self.ui_components, "Preprocessing")
        
        self._execute_preprocessing_with_api()
    
    def _set_cleanup_confirmed(self) -> None:
        """Confirm dan execute cleanup dengan enhanced flow"""
        self.ui_components['_cleanup_confirmed'] = True
        
        # Start operation flow
        start_operation_flow(self.ui_components, "Cleanup")
        
        self._execute_cleanup_with_api()
        
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(self.ui_components)
    
    def _execute_preprocessing_with_api(self) -> None:
        """Execute preprocessing dengan enhanced error handling"""
        try:
            config = self.extract_config()
            progress_callback = self.create_progress_callback()
            
            # Suppress output selama API call
            from smartcash.ui.utils.logging_utils import suppress_all_outputs
            suppress_all_outputs()
            
            from smartcash.dataset.preprocessor.api import preprocess_dataset
            result = preprocess_dataset(
                config=config,
                progress_callback=progress_callback,
                ui_components=self.ui_components
            )
            
            self.process_operation_result(result, 'preprocessing')
            complete_operation_flow(self.ui_components, "Preprocessing", True, "Preprocessing berhasil")
            
        except Exception as e:
            error_msg = f"Gagal preprocessing: {str(e)}"
            self.log_error(error_msg)
            complete_operation_flow(self.ui_components, "Preprocessing", False, error_msg)
    
    def _execute_cleanup_with_api(self) -> None:
        """Execute cleanup dengan enhanced error handling"""
        try:
            config = self.extract_config()
            cleanup_config = config.get('preprocessing', {}).get('cleanup', {})
            
            cleanup_target = cleanup_config.get('target', 'preprocessed')
            target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
            data_dir = config.get('data', {}).get('dir', 'data')
            
            progress_callback = self.create_progress_callback()
            
            # Suppress output selama API call
            from smartcash.ui.utils.logging_utils import suppress_all_outputs
            suppress_all_outputs()
            
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
            complete_operation_flow(self.ui_components, "Cleanup", True, "Cleanup berhasil")
            
        except Exception as e:
            error_msg = f"Gagal cleanup: {str(e)}"
            self.log_error(error_msg)
            complete_operation_flow(self.ui_components, "Cleanup", False, error_msg)
    
    def _handle_operation_cancel(self, operation: str) -> None:
        """Handle operation cancellation dengan proper cleanup"""
        flag_key = f'_{operation}_confirmed'
        self.ui_components[flag_key] = False
        
        cancel_msg = f"{operation.title()} dibatalkan"
        self.log_info(f"❌ {cancel_msg}")
        
        # Update status panel dengan enhanced method
        update_status_panel_enhanced(self.ui_components, f"❌ {cancel_msg}", 'warning', force_update=True)
        
        # Clear dialog untuk cleanup
        if operation == 'cleanup':
            from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
            clear_dialog_area(self.ui_components)

# Factory function
def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup operation handlers dengan enhanced control"""
    handler = OperationHandler(ui_components)
    return handler.setup_handlers()