#!/usr/bin/env python3
"""
Model Utilities

This module handles model-related operations including configuration inference,
model rebuilding, and backbone management.
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ModelUtils:
    """Utility class for model-related operations."""
    
    @staticmethod
    def infer_model_config_from_checkpoint(checkpoint_data: Dict[str, Any], 
                                         checkpoint_path: str = '') -> Dict[str, Any]:
        """
        Infer model configuration from checkpoint state dict structure.
        
        Args:
            checkpoint_data: Checkpoint data containing state dict
            checkpoint_path: Optional path to checkpoint for additional inference
            
        Returns:
            Inferred model configuration
        """
        try:
            state_dict = checkpoint_data.get('model_state_dict', {})
            if not state_dict:
                return {}
            
            # Analyze detection head output size to infer configuration
            detection_head_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
            
            if detection_head_keys:
                # Get output channels from first detection head layer
                first_head_key = detection_head_keys[0]
                output_channels = state_dict[first_head_key].shape[0]
                logger.info(f"🔍 Detected output channels: {output_channels}")
                
                # Infer configuration based on output channels
                inferred_config = {}
                
                # Map output channels to known configurations
                if output_channels == 36:  # 6 classes * 6 anchors
                    inferred_config = {
                        'num_classes': 6,
                        'layer_mode': 'single',
                        'detection_layers': ['layer_1']
                    }
                elif output_channels == 42:  # 7 classes * 6 anchors  
                    inferred_config = {
                        'num_classes': 7,
                        'layer_mode': 'single',
                        'detection_layers': ['layer_1']
                    }
                elif output_channels == 66:  # 11 classes * 6 anchors OR multi-layer (7+7+3) 
                    # Check if this is multi-layer by looking for hierarchical structure clues
                    inferred_config = {
                        'num_classes': 17,  # 7+7+3 multi-layer configuration
                        'layer_mode': 'multi',
                        'detection_layers': ['layer_1', 'layer_2', 'layer_3']
                    }
                elif output_channels == 102:  # 17 classes * 6 anchors
                    inferred_config = {
                        'num_classes': 17,
                        'layer_mode': 'single',
                        'detection_layers': ['layer_1']
                    }
                else:
                    # Generic inference based on output channels
                    num_classes = output_channels // 6  # Assume 6 anchors per class
                    inferred_config = {
                        'num_classes': num_classes,
                        'layer_mode': 'multi',  # Default to multi
                        'detection_layers': ['layer_1', 'layer_2', 'layer_3']
                    }
                
                # Try to infer backbone from parameter structure and checkpoint context
                backbone_keys = [k for k in state_dict.keys() if 'backbone' in k.lower() or 'model.0' in k or 'model.1' in k]
                if any('efficientnet' in k.lower() for k in backbone_keys):
                    inferred_config['backbone'] = 'efficientnet_b4'
                elif any('resnet' in k.lower() for k in backbone_keys):
                    inferred_config['backbone'] = 'resnet50'
                else:
                    # Try to infer from checkpoint filename if available
                    if checkpoint_path and ('efficientnet_b4' in checkpoint_path.lower()):
                        inferred_config['backbone'] = 'efficientnet_b4'
                    elif checkpoint_path and ('efficientnet_b0' in checkpoint_path.lower()):
                        inferred_config['backbone'] = 'efficientnet_b0'
                    elif checkpoint_path and ('resnet50' in checkpoint_path.lower()):
                        inferred_config['backbone'] = 'resnet50'
                    else:
                        inferred_config['backbone'] = 'cspdarknet'  # Default
                
                # Additional metadata if available
                if 'model_info' in checkpoint_data:
                    model_info = checkpoint_data['model_info']
                    if isinstance(model_info, dict):
                        if 'backbone' in model_info:
                            inferred_config['backbone'] = model_info['backbone']
                        if 'layer_mode' in model_info:
                            inferred_config['layer_mode'] = model_info['layer_mode']
                        if 'detection_layers' in model_info:
                            inferred_config['detection_layers'] = model_info['detection_layers']
                
                # Set defaults
                inferred_config.setdefault('pretrained', True)
                inferred_config.setdefault('img_size', 640)
                
                logger.info(f"🔍 Inferred model configuration from checkpoint structure: {inferred_config}")
                return inferred_config
            else:
                logger.warning("⚠️ Could not find detection head keys in checkpoint")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to infer model config from checkpoint: {e}")
            return {}
    
    @staticmethod
    def rebuild_model_for_phase2(model_api, current_model, config: Dict[str, Any]):
        """
        Rebuild model with unfrozen backbone configuration for Phase 2.
        
        Args:
            model_api: Model API instance for building models
            current_model: Current model instance
            config: Configuration dictionary
            
        Returns:
            Rebuilt model for Phase 2
        """
        try:
            logger.info("🏗️ Building new model with Phase 2 configuration (unfrozen backbone)")
            
            # Create a modified config for Phase 2 with unfrozen backbone
            phase2_config = config.copy()
            
            # CRITICAL: Preserve the multi-layer configuration from Phase 1
            # The original model config must be preserved to maintain architecture compatibility
            original_model_config = config.get('model', {})
            
            # CRITICAL FIX: If model config is empty, extract configuration from current model
            if not original_model_config or len(original_model_config) == 0:
                logger.warning("⚠️ Original model config is empty, extracting from current model")
                if current_model and hasattr(current_model, 'get_model_config'):
                    try:
                        extracted_config = current_model.get_model_config()
                        logger.info(f"🔧 Extracted model config: {extracted_config}")
                        original_model_config = extracted_config
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract model config: {e}")
                
                # If still empty, infer from the current model API config
                if (not original_model_config or len(original_model_config) == 0) and model_api:
                    try:
                        api_config = getattr(model_api, 'config', {})
                        api_model_config = api_config.get('model', {})
                        if api_model_config:
                            logger.info(f"🔧 Using model API config: {api_model_config}")
                            original_model_config = api_model_config
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get API model config: {e}")
                
                # Final fallback: infer from top-level config
                if not original_model_config or len(original_model_config) == 0:
                    logger.warning("⚠️ Using top-level config as fallback for model config")
                    # Build minimal model config from top-level config
                    original_model_config = {
                        'backbone': config.get('backbone', 'cspdarknet'),
                        'num_classes': config.get('num_classes', 17), 
                        'layer_mode': config.get('layer_mode', 'multi'),
                        'detection_layers': config.get('detection_layers', ['layer_1', 'layer_2', 'layer_3']),
                        'pretrained': config.get('pretrained', False),
                        'img_size': config.get('img_size', 640)
                    }
                    logger.info(f"🔧 Fallback model config: {original_model_config}")
            
            # Update model config to ensure backbone is unfrozen while preserving multi-layer setup
            if 'model' not in phase2_config:
                phase2_config['model'] = {}
            
            # Preserve all original model configuration parameters
            phase2_config['model'].update(original_model_config)
            
            # Only override the freeze_backbone setting for Phase 2
            phase2_config['model']['freeze_backbone'] = False
            
            # CRITICAL: Disable pretrained weights for Phase 2 model rebuild since we'll load Phase 1 weights
            phase2_config['model']['pretrained'] = False
            logger.info("🚫 Disabled pretrained weights for Phase 2 rebuild - will load Phase 1 trained weights instead")
            
            # Log the preserved configuration
            model_config = phase2_config['model']
            logger.info("🔥 Phase 2 model config: freeze_backbone = False")
            logger.info(f"🔧 Preserved layer_mode: {model_config.get('layer_mode', 'N/A')}")
            logger.info(f"🔧 Preserved detection_layers: {model_config.get('detection_layers', 'N/A')}")
            logger.info(f"🔧 Preserved num_classes: {model_config.get('num_classes', 'N/A')}")
            
            # Rebuild model using the model API
            if model_api:
                logger.info("🔧 Using model API to rebuild model for Phase 2")
                
                # Build new model with unfrozen backbone and preserved multi-layer config
                build_result = model_api.build_model(model_config=model_config)
                
                # Log the build result for debugging
                if build_result['success']:
                    built_model = build_result['model']
                    # Check if the model has the expected architecture
                    if hasattr(built_model, 'yolov5_model') and hasattr(built_model.yolov5_model.model, 'model'):
                        # Get the detection head to verify the output size
                        detection_head = built_model.yolov5_model.model.model[-1]
                        if hasattr(detection_head, 'm') and len(detection_head.m) > 0:
                            head_output_size = detection_head.m[0].weight.shape[0] if hasattr(detection_head.m[0], 'weight') else 'unknown'
                            logger.info(f"🔧 Phase 2 model detection head output size: {head_output_size}")
                            logger.info(f"🔧 Expected: 66 for multi-layer (7+7+3)*6, or similar multi-layer config")
                        else:
                            logger.warning("⚠️ Could not inspect detection head structure")
                    else:
                        logger.warning("⚠️ Could not inspect model architecture for compatibility")
                
                if build_result['success']:
                    new_model = build_result['model']
                    logger.info("✅ Phase 2 model rebuilt successfully with unfrozen backbone and preserved configuration")
                    return new_model
                else:
                    logger.error(f"❌ Failed to rebuild model for Phase 2: {build_result.get('error')}")
                    logger.info("🔄 Falling back to manual backbone unfreezing (preserves architecture)")
                    # Fallback to current model with manual unfreezing - this preserves the architecture
                    ModelUtils.unfreeze_backbone_for_phase2(current_model)
                    return current_model
            else:
                logger.warning("⚠️ No model API available - using manual backbone unfreezing")
                # Fallback to current model with manual unfreezing
                ModelUtils.unfreeze_backbone_for_phase2(current_model)
                return current_model
                
        except Exception as e:
            logger.error(f"❌ Error rebuilding model for Phase 2: {e}")
            logger.info("🔄 Falling back to manual backbone unfreezing")
            # Fallback to current model with manual unfreezing
            ModelUtils.unfreeze_backbone_for_phase2(current_model)
            return current_model
    
    @staticmethod
    def unfreeze_backbone_for_phase2(model):
        """
        Unfreeze backbone parameters for Phase 2 training.
        
        Args:
            model: Model instance to unfreeze backbone parameters
        """
        try:
            if hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone()
                logger.info("🔥 Backbone unfrozen for Phase 2 fine-tuning")
            else:
                # Manual backbone unfreezing
                backbone_params_unfrozen = 0
                for name, param in model.named_parameters():
                    if any(backbone_part in name.lower() for backbone_part in ['backbone', 'model.0', 'model.1', 'model.2', 'model.3', 'model.4']):
                        param.requires_grad = True
                        backbone_params_unfrozen += 1
                logger.info(f"🔥 Manually unfroze {backbone_params_unfrozen} backbone parameters for Phase 2")
        except Exception as e:
            logger.warning(f"⚠️ Failed to unfreeze backbone: {e}")