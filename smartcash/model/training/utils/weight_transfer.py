#!/usr/bin/env python3
"""
Weight Transfer Utilities for Single→Multi Architecture Transition

This module handles the transfer of weights from single-layer Phase 1 models
to multi-layer Phase 2 models in the two-phase training pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from collections import OrderedDict

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class WeightTransferManager:
    """Manages weight transfer between different model architectures."""
    
    def __init__(self):
        """Initialize weight transfer manager."""
        self.logger = logger
    
    def transfer_single_to_multi_weights(
        self, 
        single_checkpoint: Dict[str, Any], 
        multi_model: nn.Module,
        transfer_mode: str = 'expand'
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Transfer weights from single-layer checkpoint to multi-layer model.
        
        Args:
            single_checkpoint: Phase 1 single-layer checkpoint data
            multi_model: Phase 2 multi-layer model
            transfer_mode: Transfer mode ('expand', 'copy', 'initialize')
            
        Returns:
            Tuple of (success, transfer_info)
        """
        try:
            if 'model_state_dict' not in single_checkpoint:
                return False, {'error': 'No model_state_dict in checkpoint'}
            
            single_state_dict = single_checkpoint['model_state_dict']
            multi_state_dict = multi_model.state_dict()
            
            # Analyze architectures
            single_info = self._analyze_architecture(single_state_dict, "single-layer")
            multi_info = self._analyze_architecture(multi_state_dict, "multi-layer")
            
            self.logger.info(f"🔄 Weight Transfer: {single_info['channels']} → {multi_info['channels']} channels")
            
            # Perform weight transfer
            if transfer_mode == 'expand':
                success, transfer_info = self._expand_transfer(single_state_dict, multi_state_dict)
            elif transfer_mode == 'copy':
                success, transfer_info = self._copy_transfer(single_state_dict, multi_state_dict)
            else:
                success, transfer_info = self._initialize_transfer(single_state_dict, multi_state_dict)
            
            if success:
                # Load transferred weights into model
                multi_model.load_state_dict(transfer_info['transferred_state_dict'], strict=False)
                self.logger.info("✅ Weight transfer completed successfully")
                
                return True, {
                    'transfer_mode': transfer_mode,
                    'single_info': single_info,
                    'multi_info': multi_info,
                    'transferred_layers': transfer_info.get('transferred_layers', []),
                    'initialized_layers': transfer_info.get('initialized_layers', [])
                }
            else:
                return False, transfer_info
                
        except Exception as e:
            self.logger.error(f"❌ Weight transfer failed: {e}")
            return False, {'error': str(e)}
    
    def _analyze_architecture(self, state_dict: Dict[str, torch.Tensor], arch_type: str) -> Dict[str, Any]:
        """Analyze model architecture from state dict."""
        # Find detection head layers
        detection_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
        
        if detection_keys:
            first_head_key = detection_keys[0]
            output_channels = state_dict[first_head_key].shape[0]
            input_channels = state_dict[first_head_key].shape[1]
            
            return {
                'type': arch_type,
                'channels': output_channels,
                'input_channels': input_channels,
                'detection_heads': len([k for k in detection_keys if '.m.' in k and '.weight' in k]),
                'head_keys': detection_keys[:3]  # First 3 for logging
            }
        else:
            return {'type': arch_type, 'channels': 0, 'error': 'No detection heads found'}
    
    def _expand_transfer(
        self, 
        single_state_dict: Dict[str, torch.Tensor], 
        multi_state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Expand single-layer weights to multi-layer architecture.
        
        This method transfers Layer 1 weights and initializes Layer 2 & 3 weights.
        """
        try:
            transferred_state_dict = OrderedDict()
            transferred_layers = []
            initialized_layers = []
            
            # Copy all backbone and non-detection layers directly
            for key, tensor in single_state_dict.items():
                if not self._is_detection_head_layer(key):
                    if key in multi_state_dict:
                        transferred_state_dict[key] = tensor.clone()
                        transferred_layers.append(key)
                    else:
                        self.logger.debug(f"Skipping non-matching layer: {key}")
            
            # Handle detection head expansion: 36 channels → 66 channels
            single_head_keys = [k for k in single_state_dict.keys() if self._is_detection_head_layer(k)]
            multi_head_keys = [k for k in multi_state_dict.keys() if self._is_detection_head_layer(k)]
            
            self.logger.info(f"🔄 Expanding detection heads: {len(single_head_keys)} → {len(multi_head_keys)} layers")
            
            for multi_key in multi_head_keys:
                if multi_key in single_state_dict:
                    # Direct transfer for matching keys (Layer 1)
                    single_tensor = single_state_dict[multi_key]
                    multi_tensor = multi_state_dict[multi_key].clone()
                    
                    if single_tensor.shape[0] == 36 and multi_tensor.shape[0] == 66:
                        # Expand 36 → 66 channels (YOLOv5 single-layer to multi-layer)
                        multi_tensor[:36] = single_tensor  # Layer 1 (7 classes)
                        # Layer 2 & 3 (additional classes) initialized with small random values
                        multi_tensor[36:66] = torch.randn_like(multi_tensor[36:66]) * 0.01
                        
                        transferred_state_dict[multi_key] = multi_tensor
                        transferred_layers.append(f"{multi_key}[0:36]")
                        initialized_layers.append(f"{multi_key}[36:66]")
                    elif single_tensor.shape[0] == 42 and multi_tensor.shape[0] == 66:
                        # Legacy support: Expand 42 → 66 channels  
                        multi_tensor[:42] = single_tensor  # Layer 1 (classes 0-6)
                        # Layer 2 & 3 (classes 7-16) initialized with small random values
                        multi_tensor[42:66] = torch.randn_like(multi_tensor[42:66]) * 0.01
                        
                        transferred_state_dict[multi_key] = multi_tensor
                        transferred_layers.append(f"{multi_key}[0:42]")
                        initialized_layers.append(f"{multi_key}[42:66]")
                        
                    elif single_tensor.shape == multi_tensor.shape:
                        # Direct copy for same shapes
                        transferred_state_dict[multi_key] = single_tensor.clone()
                        transferred_layers.append(multi_key)
                    else:
                        # Initialize with small random values for mismatched shapes
                        transferred_state_dict[multi_key] = torch.randn_like(multi_tensor) * 0.01
                        initialized_layers.append(multi_key)
                        
                else:
                    # Initialize new layers not present in single model
                    multi_tensor = multi_state_dict[multi_key].clone()
                    transferred_state_dict[multi_key] = torch.randn_like(multi_tensor) * 0.01
                    initialized_layers.append(multi_key)
            
            # Ensure all multi-model layers are present
            for key, tensor in multi_state_dict.items():
                if key not in transferred_state_dict:
                    transferred_state_dict[key] = tensor.clone()
            
            self.logger.info(f"✅ Expanded transfer: {len(transferred_layers)} transferred, {len(initialized_layers)} initialized")
            
            return True, {
                'transferred_state_dict': transferred_state_dict,
                'transferred_layers': transferred_layers,
                'initialized_layers': initialized_layers
            }
            
        except Exception as e:
            self.logger.error(f"❌ Expand transfer failed: {e}")
            return False, {'error': str(e)}
    
    def _copy_transfer(
        self, 
        single_state_dict: Dict[str, torch.Tensor], 
        multi_state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Copy compatible layers and initialize incompatible ones."""
        try:
            transferred_state_dict = multi_state_dict.copy()
            transferred_layers = []
            
            for key, single_tensor in single_state_dict.items():
                if key in multi_state_dict:
                    multi_tensor = multi_state_dict[key]
                    if single_tensor.shape == multi_tensor.shape:
                        transferred_state_dict[key] = single_tensor.clone()
                        transferred_layers.append(key)
            
            return True, {
                'transferred_state_dict': transferred_state_dict,
                'transferred_layers': transferred_layers
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _initialize_transfer(
        self, 
        single_state_dict: Dict[str, torch.Tensor], 
        multi_state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Initialize multi-layer model with proper initialization."""
        try:
            # Use multi-model's existing initialization
            return True, {
                'transferred_state_dict': multi_state_dict.copy(),
                'transferred_layers': [],
                'note': 'Used fresh initialization for multi-layer model'
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _is_detection_head_layer(self, layer_name: str) -> bool:
        """Check if layer is a detection head layer."""
        detection_patterns = [
            'head.m.',  # Direct head layers
            'model.24.m.',  # YOLOv5 detection layers
            '.m.0.weight', '.m.0.bias',  # Detection head weights/biases
            '.m.1.weight', '.m.1.bias',
            '.m.2.weight', '.m.2.bias'
        ]
        return any(pattern in layer_name for pattern in detection_patterns)
    
    def validate_transfer_compatibility(
        self, 
        single_checkpoint: Dict[str, Any], 
        multi_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Validate if weight transfer is compatible between architectures.
        
        Args:
            single_checkpoint: Phase 1 checkpoint
            multi_model: Phase 2 model
            
        Returns:
            Compatibility analysis
        """
        try:
            if 'model_state_dict' not in single_checkpoint:
                return {'compatible': False, 'error': 'No model_state_dict in checkpoint'}
            
            single_state_dict = single_checkpoint['model_state_dict']
            multi_state_dict = multi_model.state_dict()
            
            single_info = self._analyze_architecture(single_state_dict, "single")
            multi_info = self._analyze_architecture(multi_state_dict, "multi")
            
            # Expected configurations
            expected_single_channels = 36  # 7 classes, but YOLOv5 uses (nc + 5) * 3 = (7 + 5) * 3 = 36
            expected_multi_channels = 66   # 17 classes * 6 anchors (YOLOv5 calculation)
            
            compatibility = {
                'compatible': True,
                'single_info': single_info,
                'multi_info': multi_info,
                'transfer_strategy': 'expand',
                'issues': []
            }
            
            # Check channel counts
            if single_info['channels'] != expected_single_channels:
                compatibility['issues'].append(f"Unexpected single-layer channels: {single_info['channels']} (expected {expected_single_channels})")
                
            if multi_info['channels'] != expected_multi_channels:
                compatibility['issues'].append(f"Unexpected multi-layer channels: {multi_info['channels']} (expected {expected_multi_channels})")
            
            # Determine best transfer strategy
            if single_info['channels'] == 36 and multi_info['channels'] == 66:
                compatibility['transfer_strategy'] = 'expand'
            elif single_info['channels'] == 42 and multi_info['channels'] == 66:
                compatibility['transfer_strategy'] = 'expand'  # Legacy support
            elif single_info['channels'] == multi_info['channels']:
                compatibility['transfer_strategy'] = 'copy'
            else:
                compatibility['transfer_strategy'] = 'initialize'
                compatibility['issues'].append("Architecture mismatch requires initialization")
            
            if compatibility['issues']:
                self.logger.warning(f"⚠️ Transfer compatibility issues: {compatibility['issues']}")
            
            return compatibility
            
        except Exception as e:
            return {'compatible': False, 'error': str(e)}


def create_weight_transfer_manager():
    """Factory function to create a WeightTransferManager instance."""
    return WeightTransferManager()


