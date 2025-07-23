"""
File: smartcash/ui/model/backbone/operations/backbone_build_operation.py
Description: Operation handler for backbone model building.
"""

from typing import Dict, Any
import os
from .backbone_base_operation import BaseBackboneOperation
from smartcash.ui.core.mixins.colab_secrets_mixin import ColabSecretsMixin


class BackboneBuildOperationHandler(BaseBackboneOperation, ColabSecretsMixin):
    """
    Orchestrates the backbone model building by calling the backend API.
    
    Inherits from ColabSecretsMixin to handle HuggingFace token authentication.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with secrets management."""
        BaseBackboneOperation.__init__(self, *args, **kwargs)
        ColabSecretsMixin.__init__(self)

    def execute(self) -> Dict[str, Any]:
        """Executes the model build by calling the backend API."""
        # Clear previous operation logs
        self.clear_operation_logs()
        
        self.log_operation("🏗️ Memulai pembangunan model backbone...", level='info')
        
        # Start dual progress tracking: 4 overall steps
        self.start_dual_progress("Pembangunan Model", total_steps=4)
        
        # Initialize variables at function scope to avoid UnboundLocalError
        use_multi_layer = False
        model_type = 'efficientnet_b4'
        
        try:
            # Step 1: Setup HuggingFace authentication and initialize backend API
            self.update_dual_progress(
                current_step=1, 
                current_percent=0,
                message="Menyiapkan autentikasi HuggingFace..."
            )
            
            # Setup HuggingFace token authentication
            hf_token = self._setup_huggingface_auth()
            if hf_token:
                self.log_operation("✅ HuggingFace token berhasil diatur", level='success')
            else:
                self.log_operation("⚠️ HuggingFace token tidak ditemukan - menggunakan model publik", level='warning')
            
            self.update_dual_progress(
                current_step=1, 
                current_percent=25,
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
            
            layer_mode_text = "Multi-Layer Detection" if use_multi_layer else "Single Layer Detection"
            self.log_operation(f"🧬 Membangun model backbone {model_type} dengan {layer_mode_text}", level='info')
            
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
            
            # Enhanced configuration with multi-layer support (default enabled)
            layer_mode = backbone_config.get('layer_mode', 'multi')
            use_multi_layer = (layer_mode == 'multi') or backbone_config.get('multi_layer_heads', True)
            
            # Use the new YOLO model builder for enhanced capabilities
            from smartcash.model.core.yolo_model_builder import build_banknote_detection_model
            
            # Create enhanced progress callback that routes all logs to operation container
            def enhanced_progress_callback(*args, **kwargs):
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
                
                # Route all backend messages through operation container logging
                if message:
                    self.log_operation(f"🔧 {message}", level='info')
                
                # Update current step progress
                self.update_dual_progress(
                    current_step=self._current_step if hasattr(self, '_current_step') else 3,
                    current_percent=percentage,
                    message=message
                )
            
            # Build model using enhanced builder with proper log routing
            build_result = build_banknote_detection_model(
                backbone_type=model_type,
                multi_layer=use_multi_layer,
                testing_mode=backbone_config.get('testing_mode', False),
                backbone={'pretrained': backbone_config.get('pretrained', True)},
                head={'use_attention': backbone_config.get('use_attention', True)},
                model={'img_size': backbone_config.get('input_size', 640)},
                progress_callback=enhanced_progress_callback
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
                
                # Save model using CheckpointManager
                model_path = None
                try:
                    self.log_operation("💾 Menyimpan model...", level='info')
                    from smartcash.model.core.checkpoint_manager import CheckpointManager
                    from smartcash.model.utils.progress_bridge import ModelProgressBridge
                    
                    # Create progress bridge for checkpoint manager
                    def checkpoint_progress_callback(*args, **kwargs):
                        # Route progress to our logging system
                        if len(args) >= 2:
                            message = args[1] if isinstance(args[1], str) else ""
                            if message:
                                self.log_operation(f"💾 {message}", level='info')
                    
                    progress_bridge = ModelProgressBridge(progress_callback=checkpoint_progress_callback)
                    
                    # Prepare checkpoint config
                    checkpoint_config = {
                        'checkpoint': {
                            'save_dir': self.config.get('save_dir', 'data/models'),
                            'format': 'backbone_{model_name}_{backbone}_{date:%Y%m%d_%H%M}.pt',
                            'max_checkpoints': 10,
                            'auto_cleanup': False  # Keep all backbone builds
                        },
                        'model': {
                            'model_name': 'smartcash',
                            'backbone': model_type,
                            'layer_mode': 'multi' if use_multi_layer else 'single'
                        }
                    }
                    
                    # Create checkpoint manager and save model
                    checkpoint_manager = CheckpointManager(checkpoint_config, progress_bridge)
                    model_path = checkpoint_manager.save_checkpoint(
                        model=model,
                        metrics={'build_success': True},
                        model_name='smartcash',
                        backbone=model_type,
                        layer_mode='multi' if use_multi_layer else 'single'
                    )
                    
                    self.log_operation(f"✅ Model berhasil disimpan di: {model_path}", level='success')
                    
                except Exception as save_error:
                    self.log_operation(f"⚠️ Gagal menyimpan model: {save_error}", level='warning')
                    model_path = None
                
                # Add model path to build result for summary
                build_result['model_path'] = model_path
                
                # Format summary with enhanced information using markdown HTML formatter
                markdown_summary = self._format_enhanced_build_summary(build_result, use_multi_layer)
                
                # Convert markdown to HTML using the new formatter
                from smartcash.ui.core.utils import format_summary_to_html
                html_summary = format_summary_to_html(
                    markdown_summary, 
                    title="🏗️ Backbone Build Results", 
                    module_name="backbone"
                )
                
                location_info = f" dan disimpan di: {model_path}" if model_path else ""
                self.log_operation(f"✅ Model backbone berhasil dibangun dengan dukungan multi-layer{location_info}", level='success')
                self._execute_callback('on_success', html_summary)
                
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
### Model Storage
- **Saved Location**: {build_result.get('model_path', 'Model tidak disimpan - terjadi error saat save')}

### Training Strategy
- **Phase 1**: Freeze backbone, train detection heads only (Learning Rate: 1e-3)
- **Phase 2**: Unfreeze entire model for fine-tuning (Learning Rate: 1e-5)
- **Loss Function**: Uncertainty-based Multi-Task Loss dengan dynamic weighting
"""
        else:
            summary += f"""
### Model Storage
- **Saved Location**: {build_result.get('model_path', 'Model tidak disimpan - terjadi error saat save')}

### Single Layer Detection
- **Classes**: 7 (IDR denominations: 001, 002, 005, 010, 020, 050, 100)
- **Detection Type**: Standard YOLO detection head
"""
        
        return summary

    def _setup_huggingface_auth(self) -> str:
        """Setup HuggingFace authentication using token from secrets or environment."""
        try:
            # Try to get HF_TOKEN from Colab secrets first
            hf_token = None
            
            # Method 1: Try Colab secrets using the mixin
            if hasattr(self, 'get_secret'):
                try:
                    hf_token = self.get_secret('HF_TOKEN')
                    if hf_token:
                        self.log_operation("🔐 HF_TOKEN ditemukan di Colab secrets", level='info')
                except Exception as e:
                    self.log_operation(f"⚠️ Gagal mengakses Colab secrets: {e}", level='warning')
            
            # Method 2: Try environment variable as fallback
            if not hf_token:
                hf_token = os.environ.get('HF_TOKEN')
                if hf_token:
                    self.log_operation("🔐 HF_TOKEN ditemukan di environment variables", level='info')
            
            # Method 3: Try alternative secret names
            if not hf_token and hasattr(self, 'get_secret'):
                alternative_names = ['HUGGINGFACE_TOKEN', 'huggingface_token', 'HF_API_TOKEN']
                for token_name in alternative_names:
                    try:
                        hf_token = self.get_secret(token_name)
                        if hf_token:
                            self.log_operation(f"🔐 Token ditemukan dengan nama: {token_name}", level='info')
                            break
                    except Exception:
                        continue
            
            # Setup the token if found
            if hf_token:
                # Set environment variable for huggingface_hub
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token  # Alternative name
                
                # Try to login to HuggingFace Hub
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                    self.log_operation("✅ Berhasil login ke HuggingFace Hub", level='success')
                except ImportError:
                    self.log_operation("📦 huggingface_hub tidak tersedia - token diatur di environment", level='info')
                except Exception as e:
                    self.log_operation(f"⚠️ Gagal login ke HuggingFace Hub: {e}", level='warning')
                    # Continue anyway, token is still set in environment
                
                return hf_token
            else:
                self.log_operation("❌ HF_TOKEN tidak ditemukan di secrets atau environment", level='warning')
                self.log_operation("💡 Tip: Tambahkan HF_TOKEN di Colab secrets untuk akses model private", level='info')
                return ""
                
        except Exception as e:
            self.log_operation(f"❌ Error saat setup HuggingFace auth: {e}", level='error')
            return ""