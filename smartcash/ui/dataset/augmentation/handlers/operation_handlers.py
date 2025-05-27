"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: SRP handlers untuk setiap operasi augmentation dengan one-liner style
"""

from typing import Dict, Any

class AugmentationOperationHandler:
    """SRP handler untuk augmentation operation"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def execute(self):
        """Execute augmentation pipeline dengan service integration"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start progress tracking
            self._start_progress('augmentation')
            
            # Execute pipeline
            result = service.run_full_augmentation_pipeline(
                target_split='train', 
                progress_callback=self._get_progress_callback()
            )
            
            # Handle result
            if result['status'] == 'success':
                success_msg = f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}"
                self._complete_progress(success_msg)
                return success_msg
            else:
                error_msg = result.get('message', 'Unknown augmentation error')
                self._error_progress(error_msg)
                raise Exception(error_msg)
                
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
    
    def _start_progress(self, operation: str):
        """Start progress tracking"""
        try:
            if 'show_for_operation' in self.ui_components:
                self.ui_components['show_for_operation'](operation)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'show'):
                self.ui_components['tracker'].show(operation)
        except Exception:
            pass
    
    def _complete_progress(self, message: str):
        """Complete progress tracking"""
        try:
            if 'complete_operation' in self.ui_components:
                self.ui_components['complete_operation'](message)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'complete'):
                self.ui_components['tracker'].complete(message)
        except Exception:
            pass
    
    def _error_progress(self, message: str):
        """Error progress tracking"""
        try:
            if 'error_operation' in self.ui_components:
                self.ui_components['error_operation'](message)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'error'):
                self.ui_components['tracker'].error(message)
        except Exception:
            pass
    
    def _get_progress_callback(self):
        """Get progress callback untuk service"""
        def callback(step: str, current: int, total: int, message: str):
            try:
                percentage = int((current / max(1, total)) * 100) if total > 0 else current
                if 'update_progress' in self.ui_components:
                    self.ui_components['update_progress'](step, percentage, message)
                elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'update'):
                    self.ui_components['tracker'].update(step, percentage, message)
            except Exception:
                pass
        return callback

class CheckOperationHandler:
    """SRP handler untuk check dataset operation"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def execute(self):
        """Execute check dataset status"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start progress
            self._start_progress('check')
            
            # Get status
            status = service.get_augmentation_status()
            
            # Format result
            status_lines = [
                f"📁 Raw: {'✅' if status.get('raw_dataset', {}).get('exists') else '❌'} ({status.get('raw_dataset', {}).get('total_images', 0)} img)",
                f"🔄 Aug: {'✅' if status.get('augmented_dataset', {}).get('exists') else '❌'} ({status.get('augmented_dataset', {}).get('total_images', 0)} files)",
                f"📊 Prep: {'✅' if status.get('preprocessed_dataset', {}).get('exists') else '❌'} ({status.get('preprocessed_dataset', {}).get('total_files', 0)} files)"
            ]
            
            result_message = "📊 Status Dataset:\n" + "\n".join(status_lines)
            self._complete_progress("Check dataset selesai")
            
            return result_message
            
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
    
    def _start_progress(self, operation: str):
        """Start progress tracking"""
        try:
            if 'show_for_operation' in self.ui_components:
                self.ui_components['show_for_operation'](operation)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'show'):
                self.ui_components['tracker'].show(operation)
        except Exception:
            pass
    
    def _complete_progress(self, message: str):
        """Complete progress tracking"""
        try:
            if 'complete_operation' in self.ui_components:
                self.ui_components['complete_operation'](message)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'complete'):
                self.ui_components['tracker'].complete(message)
        except Exception:
            pass

class CleanupOperationHandler:
    """SRP handler untuk cleanup operation"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def execute(self):
        """Execute cleanup operation"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(self.ui_components)
            
            # Start progress
            self._start_progress('cleanup')
            
            # Execute cleanup
            result = service.cleanup_augmented_data(
                include_preprocessed=True, 
                progress_callback=self._get_progress_callback()
            )
            
            # Handle result
            if result['status'] == 'success':
                success_msg = f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus"
                self._complete_progress(success_msg)
                return success_msg
            else:
                error_msg = result.get('message', 'Unknown cleanup error')
                self._error_progress(error_msg)
                raise Exception(error_msg)
                
        except ImportError as e:
            raise Exception(f"Service import error: {str(e)}")
    
    def _start_progress(self, operation: str):
        """Start progress tracking"""
        try:
            if 'show_for_operation' in self.ui_components:
                self.ui_components['show_for_operation'](operation)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'show'):
                self.ui_components['tracker'].show(operation)
        except Exception:
            pass
    
    def _complete_progress(self, message: str):
        """Complete progress tracking"""
        try:
            if 'complete_operation' in self.ui_components:
                self.ui_components['complete_operation'](message)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'complete'):
                self.ui_components['tracker'].complete(message)
        except Exception:
            pass
    
    def _error_progress(self, message: str):
        """Error progress tracking"""
        try:
            if 'error_operation' in self.ui_components:
                self.ui_components['error_operation'](message)
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'error'):
                self.ui_components['tracker'].error(message)
        except Exception:
            pass
    
    def _get_progress_callback(self):
        """Get progress callback untuk service"""
        def callback(step: str, current: int, total: int, message: str):
            try:
                percentage = int((current / max(1, total)) * 100) if total > 0 else current
                if 'update_progress' in self.ui_components:
                    self.ui_components['update_progress'](step, percentage, message)
                elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'update'):
                    self.ui_components['tracker'].update(step, percentage, message)
            except Exception:
                pass
        return callback

class ConfigOperationHandler:
    """SRP handler untuk config operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def save_config(self):
        """Execute save config"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
            result = create_config_handler(self.ui_components).save_config()
            
            if result['status'] == 'error':
                raise Exception(result['message'])
            
            return result['message']
            
        except ImportError as e:
            raise Exception(f"Config handler import error: {str(e)}")
    
    def reset_config(self):
        """Execute reset config"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
            result = create_config_handler(self.ui_components).reset_to_default()
            
            if result['status'] == 'error':
                raise Exception(result['message'])
            
            return result['message']
            
        except ImportError as e:
            raise Exception(f"Config handler import error: {str(e)}")

# Factory functions
create_augmentation_handler = lambda ui_components: AugmentationOperationHandler(ui_components)
create_check_handler = lambda ui_components: CheckOperationHandler(ui_components)
create_cleanup_handler = lambda ui_components: CleanupOperationHandler(ui_components)
create_config_handler_ops = lambda ui_components: ConfigOperationHandler(ui_components)