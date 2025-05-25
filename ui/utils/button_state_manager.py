"""
File: smartcash/ui/utils/button_state_manager.py
Deskripsi: Fixed SRP manager untuk button states dengan missing methods implementation
"""

from contextlib import contextmanager
from typing import Dict, Any, List

class ButtonStateManager:
    """
    Comprehensive shared button state manager untuk semua UI domains SmartCash.
    
    Supports:
    - Dataset: preprocessing, download, augmentation, cleanup, check
    - Model: training, evaluation, detection  
    - Common: save, reset, export, import
    
    Features:
    - Operation context management dengan automatic button disable/enable
    - Button style preservation (tidak mengubah warna button)
    - Multi-domain operation conflict detection
    - Comprehensive logging untuk debugging
    """
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_operations: List[str] = []
        self.button_original_states: Dict[str, Dict[str, Any]] = {}
        
        # Comprehensive button mapping untuk all UI domains
        self.button_groups = {
            # Dataset Preprocessing
            'preprocessing': ['preprocess_button', 'save_button', 'reset_button'],
            'check': ['check_button', 'save_button', 'reset_button'],
            'cleanup': ['cleanup_button', 'preprocess_button', 'check_button'],
            
            # Dataset Download
            'download': ['download_button', 'save_button', 'reset_button'],
            'download_check': ['check_button', 'save_button', 'reset_button'],
            'download_cleanup': ['cleanup_button', 'download_button', 'check_button'],
            
            # Dataset Augmentation
            'augmentation': ['augment_button', 'save_button', 'reset_button'],
            'augment_check': ['check_button', 'save_button', 'reset_button'],
            'augment_cleanup': ['cleanup_button', 'augment_button', 'check_button'],
            
            # Model Training
            'training': ['train_button', 'save_button', 'reset_button'],
            'train_check': ['check_button', 'save_button', 'reset_button'],
            
            # Model Evaluation
            'evaluation': ['evaluate_button', 'save_button', 'reset_button'],
            'eval_check': ['check_button', 'save_button', 'reset_button'],
            
            # Detection/Inference
            'detection': ['detect_button', 'save_button', 'reset_button'],
            'detect_check': ['check_button', 'save_button', 'reset_button']
        }
    
    @contextmanager
    def operation_context(self, operation: str):
        """Context manager untuk operation dengan automatic button state management."""
        try:
            self._start_operation(operation)
            yield
        except Exception as e:
            self._handle_operation_error(operation, str(e))
            raise
        finally:
            self._end_operation(operation)
    
    def disable_other_buttons(self, current_operation: str, button_keys: List[str] = None) -> None:
        """Disable other buttons without changing their style colors - supports all UI domains."""
        if button_keys is None:
            # Comprehensive default button list untuk all domains
            button_keys = [
                # Dataset operations
                'preprocess_button', 'download_button', 'augment_button',
                'cleanup_button', 'check_button',
                # Model operations  
                'train_button', 'evaluate_button', 'detect_button',
                # Common controls
                'save_button', 'reset_button'
            ]
        
        current_button_key = self._get_button_key_for_operation(current_operation)
        disabled_count = 0
        
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and not button.disabled:
                # Store ONLY the disabled state, preserve colors
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled
                        # Intentionally NOT storing button_style to avoid color changes
                    }
                
                # Only disable, don't change colors
                button.disabled = True
                disabled_count += 1
        
        logger = self.ui_components.get('logger')
        if logger:
            logger.debug(f"🔒 Disabled {disabled_count} buttons during {current_operation}")
    
    def enable_other_buttons(self, current_operation: str, button_keys: List[str] = None) -> None:
        """Enable other buttons dan restore original states - supports all UI domains."""
        if button_keys is None:
            # Comprehensive default button list untuk all domains
            button_keys = [
                # Dataset operations
                'preprocess_button', 'download_button', 'augment_button',
                'cleanup_button', 'check_button',
                # Model operations
                'train_button', 'evaluate_button', 'detect_button', 
                # Common controls
                'save_button', 'reset_button'
            ]
        
        current_button_key = self._get_button_key_for_operation(current_operation)
        enabled_count = 0
        
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                button.disabled = original['disabled']
                
                # Clear stored state
                del self.button_original_states[button_key]
                enabled_count += 1
        
        logger = self.ui_components.get('logger')
        if logger:
            logger.debug(f"🔓 Enabled {enabled_count} buttons after {current_operation}")
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running."""
        return operation_name in self.active_operations
    
    def get_running_operations(self) -> List[str]:
        """Get list of currently running operations."""
        return self.active_operations.copy()
    
    def can_start_operation(self, operation_name: str, exclude_operations: List[str] = None) -> tuple:
        """Check apakah operation bisa dimulai."""
        exclude_operations = exclude_operations or []
        
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        running_ops = [op for op in self.active_operations 
                      if op not in exclude_operations and op != operation_name]
        
        if running_ops:
            return False, f"Tidak dapat memulai {operation_name}, {running_ops[0]} sedang berjalan"
        
        return True, "Ready to start"
    
    def force_reset_all_states(self) -> None:
        """Force reset semua button states."""
        self.active_operations.clear()
        self._restore_all_button_states()
    
    def _start_operation(self, operation: str) -> None:
        """Start operation dengan button state changes."""
        self.active_operations.append(operation)
        
        # Disable relevant buttons
        buttons_to_disable = self.button_groups.get(operation, [])
        for button_key in buttons_to_disable:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
        
        # Set processing state untuk primary button
        self._set_processing_state(operation)
    
    def _end_operation(self, operation: str) -> None:
        """End operation dengan restore button states."""
        if operation in self.active_operations:
            self.active_operations.remove(operation)
        
        # Re-enable buttons jika tidak ada operation lain yang aktif
        if not self.active_operations:
            self._restore_all_button_states()
        
        # Restore primary button state
        self._restore_button_state(operation)
    
    def _handle_operation_error(self, operation: str, error_message: str) -> None:
        """Handle operation error dengan button state management."""
        # Set error state untuk primary button
        primary_button = self._get_primary_button(operation)
        if primary_button:
            self._set_button_error_state(primary_button, error_message)
    
    def _set_processing_state(self, operation: str) -> None:
        """Set processing state untuk operation button - supports all UI domains."""
        primary_button = self._get_primary_button(operation)
        if primary_button:
            # Comprehensive processing texts untuk all domains
            processing_texts = {
                # Dataset operations
                'preprocessing': 'Processing...', 'download': 'Downloading...', 'augmentation': 'Augmenting...',
                'check': 'Checking...', 'cleanup': 'Cleaning...',
                # Model operations
                'training': 'Training...', 'evaluation': 'Evaluating...', 'detection': 'Detecting...',
                # Generic operations
                'save': 'Saving...', 'reset': 'Resetting...', 'export': 'Exporting...', 'import': 'Importing...'
            }
            
            # Store original state jika belum ada
            if not hasattr(primary_button, '_original_description'):
                primary_button._original_description = primary_button.description
                primary_button._original_style = getattr(primary_button, 'button_style', '')
            
            primary_button.description = processing_texts.get(operation, 'Processing...')
            primary_button.button_style = 'warning'
            primary_button.disabled = True
    
    def _restore_button_state(self, operation: str) -> None:
        """Restore button ke original state."""
        primary_button = self._get_primary_button(operation)
        if primary_button and hasattr(primary_button, '_original_description'):
            primary_button.description = primary_button._original_description
            primary_button.button_style = getattr(primary_button, '_original_style', '')
            primary_button.disabled = False
    
    def _restore_all_button_states(self) -> None:
        """Restore semua button states."""
        all_buttons = set()
        for button_list in self.button_groups.values():
            all_buttons.update(button_list)
        
        for button_key in all_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
                
                # Restore original state jika ada
                if hasattr(button, '_original_description'):
                    button.description = button._original_description
                if hasattr(button, '_original_style'):
                    button.button_style = button._original_style
    
    def _set_button_error_state(self, button, error_message: str) -> None:
        """Set error state untuk button."""
        button.description = "Error!"
        button.button_style = 'danger'
        button.disabled = False
        
        # Auto-restore setelah delay
        import threading
        import time
        
        def restore_after_delay():
            time.sleep(3)
            if hasattr(button, '_original_description'):
                button.description = button._original_description
                button.button_style = getattr(button, '_original_style', '')
        
        threading.Thread(target=restore_after_delay, daemon=True).start()
    
    def _get_primary_button(self, operation: str):
        """Get primary button untuk operation - supports all UI domains."""
        # Comprehensive primary button mapping untuk all domains
        primary_mapping = {
            # Dataset operations
            'preprocessing': 'preprocess_button', 'download': 'download_button', 'augmentation': 'augment_button',
            'check': 'check_button', 'cleanup': 'cleanup_button',
            # Model operations  
            'training': 'train_button', 'evaluation': 'evaluate_button', 'detection': 'detect_button',
            # Common operations
            'save': 'save_button', 'reset': 'reset_button', 'export': 'export_button', 'import': 'import_button'
        }
        
        button_key = primary_mapping.get(operation)
        return self.ui_components.get(button_key) if button_key else None
    
    def _get_button_key_for_operation(self, operation: str) -> str:
        """Map operation ke button key - supports all UI domains."""
        # Comprehensive operation to button mapping
        operation_to_button = {
            # Dataset operations
            'preprocessing': 'preprocess_button', 'download': 'download_button', 'augmentation': 'augment_button',
            'cleanup': 'cleanup_button', 'check': 'check_button',
            # Model operations
            'training': 'train_button', 'evaluation': 'evaluate_button', 'detection': 'detect_button',
            # Common operations
            'save': 'save_button', 'reset': 'reset_button', 'export': 'export_button', 'import': 'import_button'
        }
        return operation_to_button.get(operation, f"{operation}_button")
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get comprehensive operation status untuk debugging dan monitoring."""
        return {
            'active_operations': self.active_operations.copy(),
            'has_active_operations': len(self.active_operations) > 0,
            'available_buttons': list(self.ui_components.keys()),
            'stored_states': len(self.button_original_states),
            'supported_domains': ['dataset', 'model', 'common'],
            'supported_operations': list(self._get_all_supported_operations()),
            'button_groups': {k: v for k, v in self.button_groups.items()}
        }
    
    def _get_all_supported_operations(self) -> set:
        """Get all supported operations dari button mappings."""
        operations = set()
        
        # From button groups
        operations.update(self.button_groups.keys())
        
        # From primary mappings  
        primary_operations = {
            'preprocessing', 'download', 'augmentation', 'check', 'cleanup',
            'training', 'evaluation', 'detection', 'save', 'reset', 'export', 'import'
        }
        operations.update(primary_operations)
        
        return operations
    
    def debug_button_states(self) -> Dict[str, Any]:
        """Debug method untuk inspect current button states."""
        debug_info = {
            'active_operations': self.active_operations,
            'stored_original_states': self.button_original_states,
            'current_button_states': {}
        }
        
        # Check current state of all known buttons
        all_buttons = [
            'preprocess_button', 'download_button', 'augment_button',
            'cleanup_button', 'check_button', 'train_button', 
            'evaluate_button', 'detect_button', 'save_button', 'reset_button'
        ]
        
        for button_key in all_buttons:
            button = self.ui_components.get(button_key)
            if button:
                debug_info['current_button_states'][button_key] = {
                    'disabled': getattr(button, 'disabled', None),
                    'description': getattr(button, 'description', None),
                    'button_style': getattr(button, 'button_style', None),
                    'has_original_state': hasattr(button, '_original_description')
                }
        
        return debug_info

def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Factory untuk mendapatkan shared button state manager dengan caching."""
    if '_button_state_manager' not in ui_components:
        ui_components['_button_state_manager'] = ButtonStateManager(ui_components)
        
        # Debug info logging
        logger = ui_components.get('logger')
        if logger:
            available_buttons = [k for k in ui_components.keys() if k.endswith('_button')]
            logger.debug(f"🔧 ButtonStateManager initialized dengan {len(available_buttons)} buttons: {available_buttons}")
    
    return ui_components['_button_state_manager']