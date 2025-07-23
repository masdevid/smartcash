"""
File: smartcash/ui/model/backbone/operations/backbone_validate_operation.py
Description: Operation handler for backbone configuration validation.
"""

from typing import Dict, Any
from .backbone_base_operation import BaseBackboneOperation


class BackboneValidateOperationHandler(BaseBackboneOperation):
    """
    Orchestrates the backbone configuration validation by calling the backend API.
    """

    def execute(self) -> Dict[str, Any]:
        """Executes the validation check by calling the backend API."""
        # Clear previous operation logs
        self.clear_operation_logs()
        
        self.log_operation("🔍 Memulai validasi konfigurasi backbone...", level='info')
        
        # Start dual progress tracking: 3 overall steps
        self.start_dual_progress("Validasi Backbone", total_steps=3)
        
        try:
            # Step 1: Initialize backend API
            self.update_dual_progress(
                current_step=1, 
                current_percent=0,
                message="Menginisialisasi backend API..."
            )
            
            from smartcash.model.api.core import create_model_api
            
            # Create progress callback for backend (flexible signature)
            def progress_callback(*args, **kwargs):
                # Handle different callback signatures from backend
                if len(args) >= 2:
                    percentage = args[0] if isinstance(args[0], (int, float)) else 0
                    message = args[1] if isinstance(args[1], str) else ""
                elif len(args) == 1:
                    percentage = args[0] if isinstance(args[0], (int, float)) else 0
                    message = ""
                else:
                    percentage = kwargs.get('percentage', 0)
                    message = kwargs.get('message', "")
                
                # Update current step progress
                self.update_dual_progress(
                    current_step=self._current_step if hasattr(self, '_current_step') else 1,
                    current_percent=percentage,
                    message=message
                )
            
            # Initialize API with config
            api = create_model_api(progress_callback=progress_callback)
            
            self.update_dual_progress(
                current_step=1,
                current_percent=100,
                message="Backend API siap"
            )
            
            # Step 2: Validate backbone configuration
            self.update_dual_progress(
                current_step=2,
                current_percent=0,
                message="Memvalidasi konfigurasi backbone..."
            )
            self._current_step = 2  # Store for progress callback
            
            backbone_config = self.config.get('backbone', {})
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            self.log_operation(f"🧬 Memvalidasi backbone {model_type}", level='info')
            
            # Build model to validate (lightweight operation)
            try:
                model_info = api.build_model(model={'backbone': model_type})
                validation_result = {
                    'valid': True,
                    'model_type': model_type,
                    'model_info': model_info
                }
            except Exception as e:
                validation_result = {
                    'valid': False,
                    'errors': [str(e)],
                    'model_type': model_type
                }
            
            self.update_dual_progress(
                current_step=2,
                current_percent=100,
                message="Validasi selesai"
            )
            
            # Step 3: Process results
            self.update_dual_progress(
                current_step=3,
                current_percent=50,
                message="Memproses hasil validasi..."
            )
            
            if validation_result.get('valid', False):
                # Format summary using markdown HTML formatter
                markdown_summary = self._format_validation_summary(validation_result)
                
                # Convert markdown to HTML using the new formatter
                from smartcash.ui.core.utils import format_summary_to_html
                html_summary = format_summary_to_html(
                    markdown_summary, 
                    title="🔍 Backbone Validation Results", 
                    module_name="backbone"
                )
                
                self.log_operation("✅ Validasi konfigurasi backbone berhasil", level='success')
                self._execute_callback('on_success', html_summary)
                
                self.complete_dual_progress("Validasi berhasil diselesaikan")
                return {'success': True, 'message': 'Validation completed successfully'}
            else:
                errors = validation_result.get('errors', [])
                error_msg = '; '.join(errors) if errors else 'Unknown validation error'
                self.log_operation(f"❌ Validasi gagal: {error_msg}", level='error')
                self._execute_callback('on_failure', error_msg)
                
                self.error_dual_progress(f"Validasi gagal: {error_msg}")
                return {'success': False, 'message': f'Validation failed: {error_msg}'}

        except Exception as e:
            error_message = f"Failed to validate backbone configuration: {e}"
            self.log_operation(f"❌ {error_message}", level='error')
            self._execute_callback('on_failure', error_message)
            self.error_dual_progress(error_message)
            return {'success': False, 'message': f'Error: {e}'}
        finally:
            self._execute_callback('on_complete')

    def _format_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """Formats the validation result into a user-friendly markdown summary."""
        model_type = validation_result.get('model_type', 'Unknown')
        model_info = validation_result.get('model_info', {})
        
        total_params = model_info.get('total_params', 0)
        trainable_params = model_info.get('trainable_params', 0)
        memory_usage = model_info.get('memory_usage_mb', 0)
        
        return f"""
## 🔍 Backbone Validation Results

**Model Type**: {model_type}
**Status**: ✅ Valid

### Model Information
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}
- **Estimated Memory Usage**: {memory_usage:.1f} MB

### Configuration
- **Input Size**: {model_info.get('input_size', 'N/A')}
- **Number of Classes**: {model_info.get('num_classes', 'N/A')}
- **Pretrained**: {'✅ Yes' if model_info.get('pretrained', False) else '❌ No'}
        """