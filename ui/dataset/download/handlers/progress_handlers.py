"""
File: smartcash/ui/dataset/preprocessing/handlers/progress_handlers.py
Deskripsi: SRP progress handlers untuk integrasi dengan service layer menggunakan new ProgressTracker
"""

from typing import Dict, Any

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress handlers dengan ProgressTracker integration untuk service layer."""
    
    # Get tracker instance from ui_components
    tracker = ui_components.get('tracker')
    if not tracker:
        return ui_components
    
    # Enhance existing progress components dengan service-friendly callbacks
    if 'show_for_operation' in ui_components:
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
            return tracker.show(ui_operation)
        
        ui_components['show_for_operation'] = enhanced_show_operation
    
    # Enhance progress update dengan service layer compatibility
    if 'update_progress' in ui_components:
        def service_compatible_update(progress_type: str, value: int, message: str = ""):
            """Service-compatible progress update dengan bounds checking."""
            # Validate inputs
            value = max(0, min(100, int(value))) if value is not None else 0
            message = str(message) if message else ""
            
            # Handle service-specific progress types
            if progress_type in ['overall', 'step', 'current']:
                return tracker.update(progress_type, value, message)
            elif progress_type == 'batch':
                # Map batch progress ke current progress
                return tracker.update('current', value, f"Batch: {message}")
            elif progress_type == 'split':
                # Map split progress ke step progress  
                return tracker.update('step', value, f"Split: {message}")
            else:
                # Default ke overall progress
                return tracker.update('overall', value, message)
        
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
                
                # Update multi-level progress using tracker
                if progress > 0:
                    tracker.update('overall', progress, message)
                
                if step > 0:
                    step_message = kwargs.get('split_step', f"Step {step}")
                    tracker.update('step', step * 33, step_message)
                
                if current > 0:
                    current_message = kwargs.get('split', 'Current operation')
                    tracker.update('current', current, current_message)
                
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.debug(f"🔧 Service progress callback error: {str(e)}")
        
        return service_callback
    
    ui_components['create_service_callback'] = create_service_progress_callback
    
    # Add service completion handlers using tracker
    def handle_service_completion(operation: str, message: str = "Operasi selesai"):
        """Handle completion dari service operations."""
        tracker.complete(f"{operation}: {message}")
    
    def handle_service_error(operation: str, error_message: str):
        """Handle error dari service operations.""" 
        tracker.error(f"{operation} gagal: {error_message}")
    
    ui_components['handle_service_completion'] = handle_service_completion
    ui_components['handle_service_error'] = handle_service_error
    
    # Add backward compatibility wrappers to maintain existing API
    def legacy_complete_operation(message: str = "Selesai"):
        """Legacy wrapper for complete_operation."""
        return tracker.complete(message)
    
    def legacy_error_operation(message: str = "Error"):
        """Legacy wrapper for error_operation."""
        return tracker.error(message)
    
    def legacy_show_container():
        """Legacy wrapper for show_container."""
        return tracker.show()
    
    def legacy_hide_container():
        """Legacy wrapper for hide_container."""
        return tracker.hide()
    
    def legacy_reset_all():
        """Legacy wrapper for reset_all."""
        return tracker.reset()
    
    # Maintain backward compatibility
    ui_components.update({
        'complete_operation': legacy_complete_operation,
        'error_operation': legacy_error_operation,
        'show_container': legacy_show_container,
        'hide_container': legacy_hide_container,
        'reset_all': legacy_reset_all
    })
    
    return ui_components