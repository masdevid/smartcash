"""
File: smartcash/ui/dataset/preprocessing/handlers/progress_handlers.py
Deskripsi: SRP progress handlers untuk integrasi dengan service layer tqdm progress
"""

from typing import Dict, Any

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress handlers dengan tqdm integration untuk service layer."""
    
    # Enhance existing progress components dengan service-friendly callbacks
    if 'show_for_operation' in ui_components:
        original_show = ui_components['show_for_operation']
        
        def enhanced_show_operation(operation: str):
            """Enhanced show operation dengan service layer mapping."""
            # Map service operations ke UI operations
            operation_mapping = {
                'preprocessing': 'download',
                'validation': 'check', 
                'cleanup': 'cleanup',
                'dataset_check': 'check'
            }
            
            ui_operation = operation_mapping.get(operation, operation)
            return original_show(ui_operation)
        
        ui_components['show_for_operation'] = enhanced_show_operation
    
    # Enhance progress update dengan service layer compatibility
    if 'update_progress' in ui_components:
        original_update = ui_components['update_progress']
        
        def service_compatible_update(progress_type: str, value: int, message: str = ""):
            """Service-compatible progress update dengan bounds checking."""
            # Validate inputs
            value = max(0, min(100, int(value))) if value is not None else 0
            message = str(message) if message else ""
            
            # Handle service-specific progress types
            if progress_type in ['overall', 'step', 'current']:
                return original_update(progress_type, value, message)
            elif progress_type == 'batch':
                # Map batch progress ke current progress
                return original_update('current', value, f"Batch: {message}")
            elif progress_type == 'split':
                # Map split progress ke step progress  
                return original_update('step', value, f"Split: {message}")
            else:
                # Default ke overall progress
                return original_update('overall', value, message)
        
        ui_components['update_progress'] = service_compatible_update
    
    # Add service layer progress callback creator
    def create_service_progress_callback():
        """Create callback yang compatible dengan service layer progress notifications."""
        def service_callback(**kwargs):
            try:
                # Extract common progress parameters
                progress = kwargs.get('progress', kwargs.get('overall_progress', 0))
                message = kwargs.get('message', 'Processing...')
                step = kwargs.get('step', 0)
                current = kwargs.get('current_progress', 0)
                
                # Update multi-level progress
                if progress > 0:
                    ui_components.get('update_progress', lambda *a: None)('overall', progress, message)
                
                if step > 0:
                    step_message = kwargs.get('split_step', f"Step {step}")
                    ui_components.get('update_progress', lambda *a: None)('step', step * 33, step_message)
                
                if current > 0:
                    current_message = kwargs.get('split', 'Current operation')
                    ui_components.get('update_progress', lambda *a: None)('current', current, current_message)
                
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.debug(f"🔧 Service progress callback error: {str(e)}")
        
        return service_callback
    
    ui_components['create_service_callback'] = create_service_progress_callback
    
    # Add service completion handlers
    def handle_service_completion(operation: str, message: str = "Operasi selesai"):
        """Handle completion dari service operations."""
        complete_fn = ui_components.get('complete_operation')
        if complete_fn:
            complete_fn(f"{operation}: {message}")
    
    def handle_service_error(operation: str, error_message: str):
        """Handle error dari service operations.""" 
        error_fn = ui_components.get('error_operation')
        if error_fn:
            error_fn(f"{operation} gagal: {error_message}")
    
    ui_components['handle_service_completion'] = handle_service_completion
    ui_components['handle_service_error'] = handle_service_error
    
    return ui_components