"""
File: smartcash/ui/model/backbone/operations/backbone_build_operation.py
Description: Operation handler for backbone model building.
"""

from typing import Dict, Any
from .backbone_base_operation import BaseBackboneOperation


class BackboneBuildOperationHandler(BaseBackboneOperation):
    """
    Orchestrates the backbone model building by calling the backend API.
    """

    def execute(self) -> Dict[str, Any]:
        """Executes the model build by calling the backend API."""
        self.log_operation("🏗️ Memulai pembangunan model backbone...", level='info')
        
        # Start dual progress tracking: 4 overall steps
        self.start_dual_progress("Pembangunan Model", total_steps=4)
        
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
            
            # Step 2: Prepare model configuration
            self.update_dual_progress(
                current_step=2,
                current_percent=0,
                message="Menyiapkan konfigurasi model..."
            )
            
            backbone_config = self.config.get('backbone', {})
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Prepare full model config
            model_config = {
                'backbone': model_type,
                'num_classes': backbone_config.get('num_classes', 7),
                'img_size': backbone_config.get('input_size', 640),
                'feature_optimization': {'enabled': backbone_config.get('feature_optimization', True)},
                'model_name': f'smartcash_{model_type}'
            }
            
            self.log_operation(f"🧬 Membangun model backbone {model_type}", level='info')
            
            self.update_dual_progress(
                current_step=2,
                current_percent=100,
                message="Konfigurasi siap"
            )
            
            # Step 3: Build model
            self.update_dual_progress(
                current_step=3,
                current_percent=0,
                message="Membangun model..."
            )
            self._current_step = 3  # Store for progress callback
            
            # Build model using backend API
            build_result = api.build_model(model=model_config)
            
            self.update_dual_progress(
                current_step=3,
                current_percent=100,
                message="Model berhasil dibangun"
            )
            
            # Step 4: Process and save results
            self.update_dual_progress(
                current_step=4,
                current_percent=50,
                message="Memproses hasil pembangunan..."
            )
            
            if build_result.get('status') == 'built':
                summary = self._format_build_summary(build_result)
                self.log_operation("✅ Model backbone berhasil dibangun", level='success')
                self._execute_callback('on_success', summary)
                
                # Optional: Save checkpoint
                try:
                    checkpoint_path = api.save_checkpoint()
                    self.log_operation(f"💾 Model disimpan: {checkpoint_path}", level='info')
                except Exception as save_error:
                    self.log_operation(f"⚠️ Warning: Could not save checkpoint: {save_error}", level='warning')
                
                self.complete_dual_progress("Pembangunan model berhasil diselesaikan")
                return {'success': True, 'message': 'Model build completed successfully'}
            else:
                error_msg = build_result.get('message', 'Unknown build error')
                self.log_operation(f"❌ Pembangunan model gagal: {error_msg}", level='error')
                self._execute_callback('on_failure', error_msg)
                
                self.error_dual_progress(f"Pembangunan gagal: {error_msg}")
                return {'success': False, 'message': f'Build failed: {error_msg}'}

        except Exception as e:
            error_message = f"Failed to build backbone model: {e}"
            self.log_operation(f"❌ {error_message}", level='error')
            self._execute_callback('on_failure', error_message)
            self.error_dual_progress(error_message)
            return {'success': False, 'message': f'Error: {e}'}
        finally:
            self._execute_callback('on_complete')

    def _format_build_summary(self, build_result: Dict[str, Any]) -> str:
        """Formats the build result into a user-friendly markdown summary."""
        model_info = build_result.get('model_info', {})
        build_stats = build_result.get('build_stats', {})
        
        model_type = model_info.get('model_type', 'Unknown')
        total_params = model_info.get('total_params', 0)
        trainable_params = model_info.get('trainable_params', 0)
        model_size_mb = model_info.get('model_size_mb', 0)
        build_time = build_stats.get('build_time_seconds', 0)
        
        return f"""
## 🏗️ Backbone Build Results

**Model Type**: {model_type}
**Status**: ✅ Successfully Built

### Model Statistics
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}
- **Model Size**: {model_size_mb:.1f} MB
- **Build Time**: {build_time:.2f} seconds

### Configuration Used
- **Input Size**: {model_info.get('input_size', 'N/A')}
- **Number of Classes**: {model_info.get('num_classes', 'N/A')}
- **Pretrained**: {'✅ Yes' if model_info.get('pretrained', False) else '❌ No'}
- **Feature Optimization**: {'✅ Enabled' if model_info.get('feature_optimization', False) else '❌ Disabled'}

### Model Path
- **Saved Location**: {build_result.get('model_path', 'Not saved')}
        """