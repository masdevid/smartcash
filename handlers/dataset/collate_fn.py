# File: smartcash/handlers/dataset/collate_fn.py
# Author: Alfrida Sabar
# Deskripsi: Collate functions untuk dataset multilayer (versi ultra-ringkas)

import torch
from typing import Dict, List
from smartcash.utils.layer_config_manager import get_layer_config

def multilayer_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function untuk data multilayer."""
    # Filter invalid items
    batch = [b for b in batch if b and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if not batch:
        layer_config = get_layer_config()
        return {
            'images': torch.zeros((0, 3, 640, 640)),
            'targets': {layer: torch.zeros((0, len(layer_config.get_layer_config(layer)['classes']), 5)) 
                      for layer in layer_config.get_layer_names()},
            'metadata': []
        }
    
    # Stack images
    images_tensor = torch.stack([item['image'] for item in batch])
    metadata = [item.get('metadata', {}) for item in batch]
    
    # Gabung targets per layer
    layer_config = get_layer_config()
    combined_targets = {}
    
    for layer in layer_config.get_layer_names():
        layer_targets = [targets[layer] for targets in [item['targets'] for item in batch] if layer in targets]
        
        if layer_targets:
            combined_targets[layer] = torch.stack(layer_targets)
        else:
            num_classes = len(layer_config.get_layer_config(layer)['classes'])
            combined_targets[layer] = torch.zeros((len(batch), num_classes, 5))
    
    return {'images': images_tensor, 'targets': combined_targets, 'metadata': metadata}


def flat_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function untuk model dengan output flat (non-multilayer)."""
    batch = [b for b in batch if b and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if not batch:
        return {'images': torch.zeros((0, 3, 640, 640)), 'targets': torch.zeros((0, 0, 5)), 'metadata': []}
    
    images_tensor = torch.stack([item['image'] for item in batch])
    metadata = [item.get('metadata', {}) for item in batch]
    layer_config = get_layer_config()
    flat_targets_list = []
    
    for item in batch:
        all_targets = []
        
        for layer_name, layer_tensor in item['targets'].items():
            valid_idx = layer_tensor[:, 4] > 0
            
            if valid_idx.sum() > 0:
                valid_targets = layer_tensor[valid_idx]
                class_ids = layer_config.get_layer_config(layer_name)['class_ids']
                
                for i, target in enumerate(valid_targets):
                    global_target = torch.zeros(5)
                    global_target[0] = class_ids[i % len(class_ids)]
                    global_target[1:] = target[:4]  # x, y, w, h
                    all_targets.append(global_target)
        
        flat_targets_list.append(torch.stack(all_targets) if all_targets else torch.zeros((0, 5)))
    
    return {'images': images_tensor, 'targets': flat_targets_list, 'metadata': metadata}