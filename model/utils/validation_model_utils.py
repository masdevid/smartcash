"""
File: smartcash/model/utils/validation_model_utils.py
Deskripsi: Modul utilitas untuk validasi model deteksi objek
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.model.config.model_config import ModelConfig
from smartcash.model.config.backbone_config import BackboneConfig


class ModelValidator:
    """
    Validator untuk model dan kompatibilitas konfigurasi SmartCash.
    Memeriksa dependensi, validitas parameter, dan kompatibilitas.
    """
    
    @staticmethod
    def check_model_compatibility(config: ModelConfig) -> List[str]:
        """
        Periksa kompatibilitas konfigurasi model.
        
        Args:
            config: Konfigurasi model
            
        Returns:
            List pesan kesalahan jika tidak kompatibel, list kosong jika kompatibel
        """
        errors = []
        
        # Periksa backbone
        backbone_type = config.get('model.backbone')
        if backbone_type:
            try:
                backbone_config = BackboneConfig(config, backbone_type)
            except ValueError as e:
                errors.append(f"Backbone tidak kompatibel: {str(e)}")
        else:
            errors.append("Backbone tidak ditentukan dalam konfigurasi")
        
        # Periksa layers
        layers = config.get('layers')
        if not layers:
            errors.append("Tidak ada layer yang ditentukan dalam konfigurasi")
        
        # Periksa parameter image size
        img_size = config.get('model.img_size')
        if not img_size:
            errors.append("Parameter 'model.img_size' tidak ditemukan")
        elif not isinstance(img_size, list) or len(img_size) != 2:
            errors.append("Parameter 'model.img_size' harus berupa list [width, height]")
        elif any(size % 32 != 0 for size in img_size):
            errors.append("Ukuran gambar harus kelipatan 32 untuk kompatibilitas dengan stride backbone")
        
        # Periksa parameter optimasi
        if not config.get('optimizer.type'):
            errors.append("Tipe optimizer tidak ditentukan")
        
        # Periksa parameter scheduler
        if not config.get('scheduler.type'):
            errors.append("Tipe scheduler tidak ditentukan")
        
        return errors
    
    @staticmethod
    def check_environment_compatibility() -> List[str]:
        """
        Periksa kompatibilitas environment untuk menjalankan model.
        
        Returns:
            List pesan kesalahan jika ada dependensi yang tidak terpenuhi
        """
        errors = []
        
        # Periksa PyTorch
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA tidak tersedia, performa model akan lebih lambat")
        except ImportError:
            errors.append("PyTorch tidak terinstall")
        
        # Periksa dependensi utama
        dependencies = [
            ('numpy', 'numpy'),
            ('cv2', 'opencv-python'),
            ('yaml', 'pyyaml'),
            ('pathlib', 'pathlib'),
            ('efficientnet_pytorch', 'efficientnet-pytorch')
        ]
        
        for module_name, package_name in dependencies:
            try:
                __import__(module_name)
            except ImportError:
                errors.append(f"Dependensi '{package_name}' tidak terinstall")
        
        return errors
    
    @staticmethod
    def check_checkpoint_compatibility(checkpoint: Dict, model) -> List[str]:
        """
        Periksa kompatibilitas checkpoint dengan model.
        
        Args:
            checkpoint: Checkpoint yang dimuat
            model: Model yang akan dimuat
            
        Returns:
            List pesan kesalahan jika tidak kompatibel
        """
        errors = []
        
        # Periksa state_dict
        if 'model_state_dict' not in checkpoint:
            errors.append("Checkpoint tidak berisi 'model_state_dict'")
            return errors
        
        model_state_dict = checkpoint['model_state_dict']
        
        # Cek parameter yang ada di checkpoint tapi tidak di model
        model_dict = model.state_dict()
        missing_in_model = []
        shape_mismatch = []
        
        for key, value in model_state_dict.items():
            if key not in model_dict:
                missing_in_model.append(key)
            elif model_dict[key].shape != value.shape:
                shape_mismatch.append((key, model_dict[key].shape, value.shape))
        
        # Cek parameter yang ada di model tapi tidak di checkpoint
        missing_in_checkpoint = [key for key in model_dict.keys() if key not in model_state_dict]
        
        if missing_in_model:
            errors.append(f"Ada {len(missing_in_model)} parameter di checkpoint yang tidak ada di model")
        
        if shape_mismatch:
            errors.append(f"Ada {len(shape_mismatch)} parameter dengan ukuran yang tidak sesuai")
            for key, model_shape, checkpoint_shape in shape_mismatch[:5]:
                errors.append(f"  - {key}: model {model_shape} vs checkpoint {checkpoint_shape}")
        
        if missing_in_checkpoint:
            errors.append(f"Ada {len(missing_in_checkpoint)} parameter di model yang tidak ada di checkpoint")
        
        return errors
    
    @classmethod
    def validate_training_config(cls, config: ModelConfig) -> List[str]:
        """
        Validasi konfigurasi untuk training.
        
        Args:
            config: Konfigurasi model
            
        Returns:
            List pesan kesalahan
        """
        errors = []
        
        # Cek parameter training dasar
        if config.get('training.epochs', 0) <= 0:
            errors.append("Jumlah epoch harus lebih dari 0")
        
        if config.get('training.lr', 0) <= 0:
            errors.append("Learning rate harus lebih dari 0")
        
        if config.get('model.batch_size', 0) <= 0:
            errors.append("Batch size harus lebih dari 0")
        
        # Cek parameter data
        if not config.get('data.train'):
            errors.append("Path dataset training tidak ditentukan")
        
        if not config.get('data.val'):
            errors.append("Path dataset validation tidak ditentukan")
        
        # Cek parameter checkpoint
        if not config.get('checkpoint.save_dir'):
            errors.append("Direktori penyimpanan checkpoint tidak ditentukan")
        
        return errors
    
    @classmethod
    def validate_inference_config(cls, config: ModelConfig) -> List[str]:
        """
        Validasi konfigurasi untuk inferensi.
        
        Args:
            config: Konfigurasi model
            
        Returns:
            List pesan kesalahan
        """
        errors = []
        
        # Cek parameter inference
        if not config.get('model.img_size'):
            errors.append("Ukuran gambar tidak ditentukan")
        
        return errors


def check_img_size(img_size: Union[int, List[int]], stride: int = 32) -> Tuple[int, int]:
    """
    Verifikasi dan sesuaikan ukuran gambar agar sesuai dengan stride.
    
    Args:
        img_size: Ukuran gambar (int atau [width, height])
        stride: Stride backbone
        
    Returns:
        Ukuran gambar yang disesuaikan [width, height]
    """
    # Konversi ke tuple jika int
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    elif isinstance(img_size, list) and len(img_size) == 1:
        img_size = (img_size[0], img_size[0])
    elif isinstance(img_size, list) and len(img_size) == 2:
        img_size = (img_size[0], img_size[1])
    else:
        raise ValueError(f"Format img_size tidak valid: {img_size}, harus berupa int atau [width, height]")
    
    # Sesuaikan agar kelipatan stride
    new_size = []
    for dim in img_size:
        if dim % stride != 0:
            new_dim = (dim // stride + 1) * stride
            print(f"⚠️ Ukuran gambar {dim} bukan kelipatan {stride}, disesuaikan menjadi {new_dim}")
            new_size.append(new_dim)
        else:
            new_size.append(dim)
    
    return tuple(new_size)


def check_anchors(dataset, model, thr: float = 4.0, imgsz: List[int] = [640, 640]) -> None:
    """
    Periksa apakah anchors perlu diperbarui berdasarkan dataset.
    
    Args:
        dataset: Dataset untuk memeriksa anchors
        model: Model yang berisi anchors
        thr: Threshold untuk memutuskan apakah perlu memperbarui
        imgsz: Ukuran gambar
    """
    # Ekstrak anchors dari model
    try:
        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # ambil Detect()
        shapes = imgsz * dataset.wh0 / dataset.wh0.max()  # ukuran dataset yang diskalakan
        scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augmentasi scale
        wh = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

        # Filter
        i = wh.std(0) < 0.01  # filter bad wh
        if any(i):
            wh = wh[:, ~i]

        # Sesuaikan anchors ke dataset
        ratio = wh[:, 0] / wh[:, 1]  # aspek rasio
        attr = 'anchors'
        a = getattr(m, attr)  # anchors saat ini
        na = a.shape[0]  # jumlah anchors
        new_anchors = kmeans_anchors(dataset, n=na, img_size=imgsz)  # anchors baru

        # Compute metrik
        new_bpr, old_bpr = compute_anchor_metrics(new_anchors, wh), compute_anchor_metrics(a.copy(), wh)
        
        # Tampilkan hasil
        print(f'⚖️ Anchors/Target = {na}/{wh.shape[0]}, '
              f'Anchor terkecil: {a.min()}, terbesar: {a.max()}')
        print(f'📊 Anchor lama: bpr={old_bpr:.3f}')
        print(f'📊 Anchor baru: bpr={new_bpr:.3f}')
        
        # Update anchors jika perlu
        if new_bpr > old_bpr + thr:  # metrik threshold
            setattr(m, attr, new_anchors)
            print(f'✅ Berhasil memperbarui anchors! Peningkatan: {(new_bpr - old_bpr):.3f}')
        else:
            print(f'ℹ️ Mempertahankan anchors saat ini, perbedaan dengan yang baru: {abs(new_bpr - old_bpr):.3f}')
            
    except Exception as e:
        print(f'❌ Pemeriksaan anchors gagal: {e}')


def kmeans_anchors(dataset, n: int = 9, img_size: List[int] = [640, 640], thr: float = 4.0) -> np.ndarray:
    """
    Hitung anchors menggunakan K-means clustering.
    
    Args:
        dataset: Dataset untuk menghitung anchors
        n: Jumlah anchors (default: 9)
        img_size: Ukuran gambar
        thr: Threshold untuk memutuskan apakah perlu memperbarui
        
    Returns:
        Anchors yang dihitung dengan K-means
    """
    # Implementasi sederhana kmeans untuk anchors
    from scipy.cluster.vq import kmeans
    
    # Ekstrak wh dari dataset
    shapes = img_size * dataset.wh0 / dataset.wh0.max()  # ukuran dataset yang diskalakan
    wh = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh
    
    # Filter
    i = wh.std(0) < 0.01  # filter bad wh
    if any(i):
        wh = wh[:, ~i]
    
    # K-means clustering
    print(f'⚙️ Menjalankan k-means untuk {n} anchors dengan {len(wh)} bentuk...')
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    anchors = k * s  # unwhiten
    
    # Sort by area
    area = anchors[:, 0] * anchors[:, 1]
    anchors = anchors[area.argsort()]
    
    return anchors


def compute_anchor_metrics(anchors: np.ndarray, wh: np.ndarray) -> float:
    """
    Hitung metrik best possible recall untuk anchors tertentu.
    
    Args:
        anchors: Anchors untuk dievaluasi
        wh: Width-height dari dataset
        
    Returns:
        Best possible recall
    """
    r = wh[:, None] / anchors[None]
    x = np.minimum(r, 1/r).min(2).max(1)
    bpr = (x > 1/4.0).sum() / x.shape[0]  # best possible recall
    return bpr