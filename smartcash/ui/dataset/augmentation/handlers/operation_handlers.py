"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Fixed handlers dengan proper communicator setup dan service integration
"""

from typing import Dict, Any

class AugmentationOperationHandler:
    """Fixed augmentation handler dengan proper communicator integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self._setup_communicator()
    
    def _setup_communicator(self):
        """Setup communicator dari UI components"""
        try:
            from smartcash.dataset.augmentor.communicator import create_communicator
            self.comm = create_communicator(self.ui_components)
            self.ui_components['communicator'] = self.comm
            self.ui_components['communicator_ready'] = True
        except ImportError:
            self.comm = None
            self.ui_components['communicator_ready'] = False
    
    def execute(self):
        """Execute augmentation dengan proper service integration"""
        try:
            # Ensure communicator setup
            if not self.comm:
                self._setup_communicator()
            
            # Create service dengan UI components
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start operation
            self._start_operation('augmentation')
            
            # Execute pipeline dengan progress callback
            result = service.run_full_augmentation_pipeline(
                target_split='train',
                progress_callback=self._create_progress_callback()
            )
            
            # Handle result
            self._handle_result(result)
            return result.get('message', 'Augmentation completed')
            
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
        except Exception as e:
            self._error_operation(f"Augmentation error: {str(e)}")
            raise
    
    def _create_progress_callback(self):
        """Create progress callback untuk service integration"""
        def callback(step: str, current: int, total: int, message: str):
            percentage = min(100, int((current / max(1, total)) * 100))
            self._update_progress(step, percentage, message)
        return callback
    
    # One-liner operation methods
    _start_operation = lambda self, op: (
        self.ui_components.get('show_for_operation', lambda x: None)(op) or
        getattr(self.ui_components.get('tracker'), 'show', lambda x: None)(op)
    )
    
    _update_progress = lambda self, step, pct, msg: (
        self.ui_components.get('update_progress', lambda s, p, m: None)(step, pct, msg) or
        getattr(self.ui_components.get('tracker'), 'update', lambda s, p, m: None)(step, pct, msg)
    )
    
    _complete_operation = lambda self, msg: (
        self.ui_components.get('complete_operation', lambda m: None)(msg) or  
        getattr(self.ui_components.get('tracker'), 'complete', lambda m: None)(msg)
    )
    
    _error_operation = lambda self, msg: (
        self.ui_components.get('error_operation', lambda m: None)(msg) or
        getattr(self.ui_components.get('tracker'), 'error', lambda m: None)(msg)
    )
    
    def _handle_result(self, result: Dict[str, Any]):
        """Handle operation result"""
        if result['status'] == 'success':
            success_msg = f"✅ Pipeline berhasil: {result.get('total_files', 0)} file dalam {result.get('processing_time', 0):.1f}s"
            self._complete_operation(success_msg)
        else:
            error_msg = f"❌ Pipeline gagal: {result.get('message', 'Unknown error')}"
            self._error_operation(error_msg)
            raise Exception(error_msg)

class CheckOperationHandler:
    """Fixed check handler dengan communicator integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def execute(self):
        """Execute check dengan service integration"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start progress
            self._start_operation('check')
            
            # Get status
            status = service.get_augmentation_status()
            
            # Format result dengan better info
            raw_info = status.get('raw_dataset', {})
            aug_info = status.get('augmented_dataset', {})
            prep_info = status.get('preprocessed_dataset', {})
            
            status_lines = [
                f"📁 Raw: {'✅' if raw_info.get('exists') else '❌'} ({raw_info.get('total_images', 0)} img, {raw_info.get('total_labels', 0)} lbl)",
                f"🔄 Aug: {'✅' if aug_info.get('exists') else '❌'} ({aug_info.get('total_images', 0)} files)",
                f"📊 Prep: {'✅' if prep_info.get('exists') else '❌'} ({prep_info.get('total_files', 0)} files)",
                f"🎯 Ready: {'✅' if status.get('ready_for_augmentation') else '❌'}"
            ]
            
            result_message = "📊 Dataset Status:\n" + "\n".join(status_lines)
            self._complete_operation("Dataset check completed")
            
            return result_message
            
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
    
    # One-liner methods
    _start_operation = lambda self, op: getattr(self.ui_components.get('tracker'), 'show', lambda x: None)(op)
    _complete_operation = lambda self, msg: getattr(self.ui_components.get('tracker'), 'complete', lambda m: None)(msg)

class CleanupOperationHandler:
    """Fixed cleanup handler dengan communicator integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def execute(self):
        """Execute cleanup dengan service integration"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start progress
            self._start_operation('cleanup')
            
            # Execute cleanup dengan progress
            result = service.cleanup_augmented_data(
                include_preprocessed=True,
                progress_callback=self._create_progress_callback()
            )
            
            # Handle result
            if result['status'] == 'success':
                success_msg = f"✅ Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus"
                self._complete_operation(success_msg)
                return success_msg
            else:
                error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
                self._error_operation(error_msg)
                raise Exception(error_msg)
                
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
    
    def _create_progress_callback(self):
        """Create progress callback"""
        def callback(step: str, current: int, total: int, message: str):
            percentage = min(100, int((current / max(1, total)) * 100))
            getattr(self.ui_components.get('tracker'), 'update', lambda s, p, m: None)(step, percentage, message)
        return callback
    
    # One-liner methods
    _start_operation = lambda self, op: getattr(self.ui_components.get('tracker'), 'show', lambda x: None)(op)
    _complete_operation = lambda self, msg: getattr(self.ui_components.get('tracker'), 'complete', lambda m: None)(msg)
    _error_operation = lambda self, msg: getattr(self.ui_components.get('tracker'), 'error', lambda m: None)(msg)

class ConfigOperationHandler:
    """Fixed config handler dengan proper integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def save_config(self):
        """Execute save config dengan train split enforcement"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
            handler = create_config_handler(self.ui_components)
            result = handler.save_config()
            
            if result['status'] == 'error':
                raise Exception(result['message'])
            
            return "💾 Konfigurasi berhasil disimpan (Train split only)"
            
        except ImportError as e:
            raise Exception(f"Config handler import error: {str(e)}")
    
    def reset_config(self):
        """Execute reset config dengan train split enforcement"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
            handler = create_config_handler(self.ui_components)
            result = handler.reset_to_default()
            
            if result['status'] == 'error':
                raise Exception(result['message'])
            
            return "🔄 Konfigurasi direset ke default (Train split only)"
            
        except ImportError as e:
            raise Exception(f"Config handler import error: {str(e)}")

# Factory functions dengan proper communicator integration
create_augmentation_handler = lambda ui_components: AugmentationOperationHandler(ui_components)
create_check_handler = lambda ui_components: CheckOperationHandler(ui_components)
create_cleanup_handler = lambda ui_components: CleanupOperationHandler(ui_components)
create_config_handler_ops = lambda ui_components: ConfigOperationHandler(ui_components)