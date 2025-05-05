"""
File: smartcash/dataset/components/collate/yolo_collate.py
Deskripsi: Fungsi collate untuk format YOLO standar
"""

import torch
from typing import Dict, List, Tuple, Any


def yolo_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function untuk data dalam format YOLO.
    
    Args:
        batch: Batch data dari dataset
        
    Returns:
        Batch data yang sudah di-collate
    """
    # Filter item yang tidak valid
    batch = [b for b in batch if b and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if not batch:
        return {
            'images': torch.zeros((0, 3, 640, 640)),
            'targets': torch.zeros((0, 0, 5)),
            'metadata': []
        }
    
    # Stack images
    images_tensor = torch.stack([item['image'] for item in batch])
    metadata = [item.get('metadata', {}) for item in batch]
    
    # Collate targets
    targets_list = []
    
    for i, item in enumerate(batch):
        targets = item['targets']
        
        if isinstance(targets, torch.Tensor) and targets.size(0) > 0:
            # Tambahkan batch_idx di awal kolom
            batch_idx = torch.full((targets.size(0), 1), i, device=targets.device, dtype=targets.dtype)
            targets_with_idx = torch.cat((batch_idx, targets), dim=1)
            targets_list.append(targets_with_idx)
    
    # Gabungkan semua targets
    if targets_list:
        targets = torch.cat(targets_list, dim=0)
    else:
        targets = torch.zeros((0, 6))  # [batch_idx, class_id, x, y, w, h]
    
    return {'images': images_tensor, 'targets': targets, 'metadata': metadata}


def yolo_detection_collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Collate function yang menghasilkan format untuk model deteksi.
    
    Args:
        batch: Batch data dari dataset
        
    Returns:
        Tuple (images, targets) yang siap untuk model deteksi YOLOv5
    """
    # Filter item yang tidak valid
    batch = [b for b in batch if b and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if not batch:
        return torch.zeros((0, 3, 640, 640)), []
    
    # Stack images
    images_tensor = torch.stack([item['image'] for item in batch])
    
    # List of targets per image
    targets_list = []
    
    for item in batch:
        targets = item['targets']
        
        if isinstance(targets, torch.Tensor) and targets.size(0) > 0:
            targets_list.append(targets)
        else:
            targets_list.append(torch.zeros((0, 5)))
    
    return images_tensor, targets_list


def yolo_mosaic_collate_fn(batch: List[Dict], mosaic_border: float = 0.5) -> Dict:
    """
    Collate function dengan augmentasi mosaic di level batch.
    
    Args:
        batch: Batch data dari dataset
        mosaic_border: Batas mosaic (0-1)
        
    Returns:
        Batch data yang sudah di-mosaic
    """
    # Filter item yang tidak valid
    batch = [b for b in batch if b and isinstance(b, dict) and 'image' in b and 'targets' in b]
    
    if not batch or len(batch) < 4:
        # Fallback ke collate function biasa jika tidak cukup gambar
        return yolo_collate_fn(batch)
    
    # Ambil ukuran gambar dari batch pertama
    img_size = batch[0]['image'].shape[1:]  # [C, H, W] -> [H, W]
    
    # Buat mosaic untuk setiap 4 gambar
    mosaic_images = []
    mosaic_targets = []
    metadata = []
    
    for i in range(0, len(batch), 4):
        # Ambil 4 gambar, pad dengan yang terakhir jika kurang
        chunk = batch[i:i + 4]
        if len(chunk) < 4:
            while len(chunk) < 4:
                chunk.append(batch[-1])
        
        # Buat mosaic untuk 4 gambar
        mosaic_img, mosaic_target = _create_mosaic(chunk, img_size, mosaic_border)
        
        mosaic_images.append(mosaic_img)
        mosaic_targets.append(mosaic_target)
        metadata.append({'mosaic': True, 'source_images': [item.get('metadata', {}) for item in chunk]})
    
    # Stack semua mosaic images
    images_tensor = torch.stack(mosaic_images)
    
    # Collate mosaic targets
    targets_list = []
    
    for i, targets in enumerate(mosaic_targets):
        if targets.size(0) > 0:
            # Tambahkan batch_idx di awal kolom
            batch_idx = torch.full((targets.size(0), 1), i, device=targets.device, dtype=targets.dtype)
            targets_with_idx = torch.cat((batch_idx, targets), dim=1)
            targets_list.append(targets_with_idx)
    
    # Gabungkan semua targets
    if targets_list:
        targets = torch.cat(targets_list, dim=0)
    else:
        targets = torch.zeros((0, 6))  # [batch_idx, class_id, x, y, w, h]
    
    return {'images': images_tensor, 'targets': targets, 'metadata': metadata}


def _create_mosaic(images: List[Dict], img_size: Tuple[int, int], mosaic_border: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Buat gambar mosaic dari 4 gambar.
    
    Args:
        images: List 4 gambar dengan targetnya
        img_size: Ukuran gambar (height, width)
        mosaic_border: Batas mosaic (0-1)
        
    Returns:
        Tuple (mosaic_image, mosaic_targets)
    """
    h, w = img_size
    
    # Buat mosaic canvas
    mosaic_img = torch.zeros((3, h, w), dtype=images[0]['image'].dtype)
    
    # Tentukan titik pusat mosaic (random)
    cx = int(w * (0.5 + (torch.rand(1) - 0.5) * 2 * mosaic_border))
    cy = int(h * (0.5 + (torch.rand(1) - 0.5) * 2 * mosaic_border))
    
    # Koordinat grid untuk 4 gambar
    grids = [
        [cx, cy],      # top-left
        [w - cx, cy],  # top-right
        [cx, h - cy],  # bottom-left
        [w - cx, h - cy]  # bottom-right
    ]
    
    mosaic_targets = []
    
    # Terapkan 4 gambar ke mosaic
    for i, item in enumerate(images):
        img = item['image']
        targets = item['targets']
        
        # Koordinat untuk gambar ini
        x1, y1 = grids[i][0] - cx if i % 2 == 0 else 0, grids[i][1] - cy if i < 2 else 0
        x2, y2 = w if i % 2 else cx, h if i >= 2 else cy
        
        # Terapkan gambar ke mosaic
        img_h, img_w = img.shape[1:3]
        
        # Hitung area yang akan dimasukkan
        h_src = min(img_h, y2 - y1)
        w_src = min(img_w, x2 - x1)
        
        # Terapkan gambar
        mosaic_img[:, y1:y1 + h_src, x1:x1 + w_src] = img[:, :h_src, :w_src]
        
        # Transformasi target
        if targets.size(0) > 0:
            # Skalakan bbox sesuai ukuran gambar sumber
            bbox = targets[:, 1:]  # [class_id, x, y, w, h] -> [x, y, w, h]
            
            # Konversi ke [x1, y1, x2, y2]
            box_xy = bbox[:, :2]
            box_wh = bbox[:, 2:4]
            box_xy1 = box_xy - box_wh / 2
            box_xy2 = box_xy + box_wh / 2
            
            # Transformasi ke koordinat mosaic
            if i % 2 == 0:
                box_xy1[:, 0] = (box_xy1[:, 0] * img_w + x1) / w
                box_xy2[:, 0] = (box_xy2[:, 0] * img_w + x1) / w
            else:
                box_xy1[:, 0] = (box_xy1[:, 0] * img_w) / w
                box_xy2[:, 0] = (box_xy2[:, 0] * img_w) / w
                
            if i < 2:
                box_xy1[:, 1] = (box_xy1[:, 1] * img_h + y1) / h
                box_xy2[:, 1] = (box_xy2[:, 1] * img_h + y1) / h
            else:
                box_xy1[:, 1] = (box_xy1[:, 1] * img_h) / h
                box_xy2[:, 1] = (box_xy2[:, 1] * img_h) / h
            
            # Kembali ke [x, y, w, h]
            new_box_xy = (box_xy1 + box_xy2) / 2
            new_box_wh = box_xy2 - box_xy1
            
            # Filter bbox yang masih dalam gambar
            mask = (new_box_wh > 0).all(1) & (new_box_xy > 0).all(1) & (new_box_xy < 1).all(1)
            
            if mask.any():
                # [class_id, x, y, w, h]
                new_targets = torch.cat([
                    targets[mask, 0:1],  # class_id
                    new_box_xy[mask],    # x, y
                    new_box_wh[mask]     # w, h
                ], 1)
                
                mosaic_targets.append(new_targets)
    
    # Gabungkan targets dari 4 gambar
    if mosaic_targets:
        mosaic_targets = torch.cat(mosaic_targets, 0)
    else:
        mosaic_targets = torch.zeros((0, 5))
    
    return mosaic_img, mosaic_targets