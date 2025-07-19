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
            
            # Step 3: Build model with enhanced multi-layer support
            self.update_dual_progress(
                current_step=3,
                current_percent=0,
                message="Membangun model dengan dukungan multi-layer..."
            )
            self._current_step = 3  # Store for progress callback
            
            # Enhanced configuration with multi-layer support
            use_multi_layer = backbone_config.get('multi_layer_heads', False)
            
            # Use the new YOLO model builder for enhanced capabilities
            from smartcash.model.core.yolo_model_builder import build_banknote_detection_model
            
            # Build model using enhanced builder
            build_result = build_banknote_detection_model(
                backbone_type=model_type,
                multi_layer=use_multi_layer,
                testing_mode=backbone_config.get('testing_mode', False),
                backbone={'pretrained': backbone_config.get('pretrained', True)},
                head={'use_attention': backbone_config.get('use_attention', True)},
                model={'img_size': backbone_config.get('input_size', 640)}
            )
            
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
            
            if build_result.get('success', False):
                # Extract enhanced model information
                model = build_result['model']
                build_info = build_result['build_info']
                
                # Log multi-layer information
                head_info = build_info['head_info']
                if use_multi_layer and 'layer_specs' in head_info:
                    layer_specs = head_info['layer_specs']
                    self.log_operation(f"🔀 Multi-layer detection aktif dengan {len(layer_specs)} layer:", level='info')
                    for layer_name, layer_spec in layer_specs.items():
                        classes = layer_spec.get('classes', [])
                        desc = layer_spec.get('description', '')
                        self.log_operation(f"   • {layer_name}: {len(classes)} kelas - {desc}", level='info')
                
                # Format summary with enhanced information
                summary = self._format_enhanced_build_summary(build_result, use_multi_layer)
                self.log_operation("✅ Model backbone berhasil dibangun dengan dukungan multi-layer", level='success')
                self._execute_callback('on_success', summary)
                
                # Log parameter count
                total_params = model.count_parameters()
                trainable_params = model.count_trainable_parameters()
                self.log_operation(f"📊 Parameter: {total_params:,} total, {trainable_params:,} trainable", level='info')
                
                self.complete_dual_progress("Pembangunan model dengan multi-layer berhasil diselesaikan")
                return {
                    'success': True, 
                    'message': 'Enhanced model build completed successfully',
                    'model': model,
                    'build_info': build_info,
                    'multi_layer': use_multi_layer
                }
            else:
                error_msg = build_result.get('error', 'Unknown build error')
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
    
    def _format_enhanced_build_summary(self, build_result: Dict[str, Any], use_multi_layer: bool = False) -> str:
        """Formats the enhanced build result with multi-layer information."""
        model = build_result['model']
        build_info = build_result['build_info']
        model_info = build_info['model_info']
        backbone_info = build_info['backbone_info']
        head_info = build_info['head_info']
        
        total_params = model.count_parameters()
        trainable_params = model.count_trainable_parameters()
        model_size_mb = total_params * 4 / (1024 * 1024)  # Rough estimate
        
        summary = f"""
## 🏗️ Enhanced YOLOv5 Build Results

**Architecture**: SmartCashYOLOv5 dengan Multi-Layer Detection
**Status**: ✅ Successfully Built

### Model Architecture
- **Backbone**: {backbone_info.get('type', 'Unknown')} ({backbone_info.get('variant', 'Unknown')})
- **Neck**: FPN-PAN
- **Head**: {'Multi-Layer Detection' if use_multi_layer else 'Single Layer Detection'}
- **Multi-Layer Support**: {'✅ Aktif' if use_multi_layer else '❌ Tidak Aktif'}

### Model Statistics
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}
- **Model Size**: {model_size_mb:.1f} MB
- **Feature Maps**: {len(backbone_info.get('out_channels', []))} (P3, P4, P5)

### Configuration Details
- **Input Size**: {model_info.get('input_size', (640, 640))}
- **Backbone Pretrained**: {'✅ Yes' if backbone_info.get('pretrained', False) else '❌ No'}
- **Channel Attention**: {'✅ Enabled' if head_info.get('use_attention', False) else '❌ Disabled'}
- **FPN Compatible**: {'✅ Yes' if backbone_info.get('fpn_compatible', False) else '❌ No'}
"""
        
        # Add layer specifications if multi-layer
        if use_multi_layer and 'layer_specs' in head_info:
            layer_specs = head_info['layer_specs']
            summary += f"""
### Detection Layers ({len(layer_specs)} layers)
"""
            for layer_name, layer_spec in layer_specs.items():
                classes = layer_spec.get('classes', [])
                desc = layer_spec.get('description', '')
                summary += f"- **{layer_name}**: {len(classes)} kelas - {desc}\n"
            
            summary += f"""
### Training Strategy
- **Phase 1**: Freeze backbone, train detection heads only (Learning Rate: 1e-3)
- **Phase 2**: Unfreeze entire model for fine-tuning (Learning Rate: 1e-5)
- **Loss Function**: Uncertainty-based Multi-Task Loss dengan dynamic weighting
"""
        else:
            summary += f"""
### Single Layer Detection
- **Classes**: 7 (IDR denominations: 001, 002, 005, 010, 020, 050, 100)
- **Detection Type**: Standard YOLO detection head
"""
        
        return summary