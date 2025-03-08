# File: smartcash/handlers/dataset/collate_functions.py
# Author: Alfrida Sabar
# Deskripsi: Collate functions untuk pengelolaan batch dataset multilayer

import torch
from typing import Dict, List

from smartcash.utils.layer_config_manager import get_layer_config


def multilayer_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function untuk data multilayer.
    
    Args:
        batch: List item dataset
        
    Returns:
        Dict berisi data batch yang telah digabungkan
    """
    # Filter invalid items
    batch = [b for b in batch if b is not None and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if len(batch) == 0:
        # Return empty batch dengan struktur dasar
        layer_config = get_layer_config()
        layer_names = layer_config.get_layer_names()
        dummy_targets = {
            layer: torch.zeros((0, len(layer_config.get_layer_config(layer)['classes']), 5))
            for layer in layer_names
        }
        return {
            'images': torch.zeros((0, 3, 640, 640)),
            'targets': dummy_targets,
            'metadata': []
        }
    
    # Extract komponen batch
    images = [item['image'] for item in batch]
    targets_list = [item['targets'] for item in batch]
    metadata = [item.get('metadata', {}) for item in batch]
    
    # Stack images
    images_tensor = torch.stack(images)
    
    # Gabung targets per layer
    combined_targets = {}
    layer_config = get_layer_config()
    layer_names = layer_config.get_layer_names()
    
    for layer in layer_names:
        layer_targets = []
        
        for targets in targets_list:
            if layer in targets:
                layer_targets.append(targets[layer])
        
        if layer_targets:
            # Stack jika ada target untuk layer ini
            combined_targets[layer] = torch.stack(layer_targets)
        else:
            # Buat tensor kosong jika tidak ada target
            num_classes = len(layer_config.get_layer_config(layer)['classes'])
            combined_targets[layer] = torch.zeros((len(batch), num_classes, 5))
    
    # Return hasil
    return {
        'images': images_tensor,
        'targets': combined_targets,
        'metadata': metadata
    }


def flat_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function yang mempersiapkan data untuk model dengan output flat (non-multilayer).
    Menggabungkan semua target dari berbagai layer menjadi satu tensor.
    
    Args:
        batch: List item dataset
        
    Returns:
        Dict berisi data batch yang telah digabungkan
    """
    # Filter invalid items
    batch = [b for b in batch if b is not None and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if len(batch) == 0:
        # Return empty batch
        return {
            'images': torch.zeros((0, 3, 640, 640)),
            'targets': torch.zeros((0, 0, 5)),
            'metadata': []
        }
    
    # Extract komponen batch
    images = [item['image'] for item in batch]
    metadata = [item.get('metadata', {}) for item in batch]
    
    # Gabungkan semua target dari berbagai layer menjadi satu tensor flat
    flat_targets_list = []
    for item in batch:
        targets_dict = item['targets']
        all_targets = []
        
        # Gabungkan semua layer targets dengan mengonversi local class ID ke global
        layer_config = get_layer_config()
        for layer_name, layer_tensor in targets_dict.items():
            layer_info = layer_config.get_layer_config(layer_name)
            class_ids = layer_info['class_ids']
            
            # Ambil baris yang memiliki objek (confidence > 0)
            valid_idx = layer_tensor[:, 4] > 0
            
            # Jika ada objek valid
            if valid_idx.sum() > 0:
                valid_targets = layer_tensor[valid_idx]
                
                # Convert semua target valid ke format global [class_id, x, y, w, h]
                for i, target in enumerate(valid_targets):
                    # Konversi local idx ke global class_id
                    local_idx = i % len(class_ids)
                    global_class_id = class_ids[local_idx]
                    
                    # Buat target dengan class_id global
                    global_target = torch.zeros(5)
                    global_target[0] = global_class_id
                    global_target[1:] = target[:4]  # x, y, w, h
                    
                    all_targets.append(global_target)
        
        # Tambahkan ke list
        if all_targets:
            flat_targets = torch.stack(all_targets)
            flat_targets_list.append(flat_targets)
        else:
            # Dummy tensor kosong untuk batch item tanpa objek
            flat_targets_list.append(torch.zeros((0, 5)))
    
    # Stack images
    images_tensor = torch.stack(images)
    
    # Return hasil
    return {
        'images': images_tensor,
        'targets': flat_targets_list,
        'metadata': metadata
    }