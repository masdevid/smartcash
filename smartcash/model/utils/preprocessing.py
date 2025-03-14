"""
File: smartcash/model/utils/preprocessing.py
Deskripsi: Utilitas preprocessing untuk model deteksi mata uang dengan EfficientNet
"""

import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any


class ModelPreprocessor:
    """
    Preprocessing gambar untuk model deteksi objek dengan backbone EfficientNet.
    Mendukung berbagai ukuran input, normalisasi, dan transformasi.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        pad_to_square: bool = True
    ):
        """
        Inisialisasi preprocessor.
        
        Args:
            img_size: Ukuran gambar target (width, height)
            mean: Nilai mean untuk normalisasi RGB
            std: Nilai standard deviation untuk normalisasi RGB
            pad_to_square: Apakah padding ke bentuk persegi sebelum resize
        """
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.pad_to_square = pad_to_square
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray],
        return_tensors: bool = True,
        batch_dim: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess gambar untuk model.
        
        Args:
            image: Path gambar atau numpy array (BGR format jika dari OpenCV)
            return_tensors: Jika True, kembalikan torch.Tensor, jika False, numpy array
            batch_dim: Apakah menambahkan dimensi batch (N,C,H,W) jika return_tensors=True
            
        Returns:
            Gambar yang preprocessed dalam format tensor atau array
        """
        # Muat gambar jika diberikan path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"❌ Gagal membaca gambar dari {image}")
        else:
            img = image.copy()
        
        # Konversi BGR ke RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Padding ke persegi jika diperlukan
        if self.pad_to_square:
            img = self._pad_to_square(img)
        
        # Resize ke target size
        img = cv2.resize(img, self.img_size)
        
        # Normalisasi gambar (0-255 -> 0-1)
        img = img.astype(np.float32) / 255.0
        
        # Normalisasi dengan mean dan std
        img = (img - self.mean) / self.std
        
        if return_tensors:
            # Konversi ke PyTorch tensor dan ubah ke format channel-first (C,H,W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            # Tambahkan dimensi batch jika diperlukan
            if batch_dim:
                img_tensor = img_tensor.unsqueeze(0)
                
            return img_tensor
        else:
            return img
    
    def preprocess_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> torch.Tensor:
        """
        Preprocess batch gambar untuk model.
        
        Args:
            images: List path gambar atau numpy array
            
        Returns:
            Batch tensor gambar yang preprocessed
        """
        batch_tensors = []
        
        for img in images:
            # Preprocess tanpa dimensi batch
            tensor = self.preprocess_image(img, return_tensors=True, batch_dim=False)
            batch_tensors.append(tensor)
        
        # Stack semua tensor menjadi batch
        return torch.stack(batch_tensors)
    
    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        """
        Padding gambar menjadi bentuk persegi.
        
        Args:
            img: Gambar dalam format numpy array
            
        Returns:
            Gambar persegi dengan padding
        """
        height, width = img.shape[:2]
        
        # Hitung dimensi persegi
        max_dim = max(height, width)
        
        # Buat gambar persegi dengan padding
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Letakkan gambar asli di tengah
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
        
        return square_img
    
    def denormalize_image(
        self,
        tensor: torch.Tensor,
        to_uint8: bool = True
    ) -> np.ndarray:
        """
        Denormalisasi tensor gambar ke numpy array.
        
        Args:
            tensor: Tensor gambar (C,H,W) atau (N,C,H,W)
            to_uint8: Jika True, konversi ke format uint8 (0-255)
            
        Returns:
            Gambar denormalisasi dalam format numpy array
        """
        # Konversi tensor ke numpy array
        if tensor.dim() == 4:  # (N,C,H,W)
            # Ambil gambar pertama jika batch
            img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        elif tensor.dim() == 3:  # (C,H,W)
            img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        else:
            raise ValueError(f"❌ Format tensor tidak valid dengan dimensi {tensor.dim()}")
        
        # Denormalisasi dengan mean dan std
        img = img * self.std + self.mean
        
        # Clip nilai ke range [0, 1]
        img = np.clip(img, 0, 1)
        
        # Konversi ke uint8 jika diperlukan
        if to_uint8:
            img = (img * 255).astype(np.uint8)
        
        return img


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize dan pad gambar sementara mempertahankan aspect ratio.
    
    Args:
        img: Gambar dalam format numpy array
        new_shape: Target shape (height, width)
        color: Warna padding
        auto: True untuk minimum rectangular, False untuk specific rectangular
        scale_fill: Stretch gambar
        scale_up: Juga scale up (default: True)
        stride: Stride untuk alignment
        
    Returns:
        Tuple (gambar yang diletterboxed, rasio, padding)
    """
    # Convert new_shape ke tuple height, width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Shape saat ini [height, width]
    shape = img.shape[:2]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # Hanya scale down, tidak scale up
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # Minimum rectangular
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # Stretch
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    # Distribute padding ke kiri/kanan atau atas/bawah
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # Resize
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


def scale_coords(
    img1_shape: Tuple[int, int],
    coords: Union[np.ndarray, torch.Tensor],
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Rescale coords (xyxy) dari img1_shape ke img0_shape.
    
    Args:
        img1_shape: Shape (height, width) setelah letterboxing
        coords: Koordinat deteksi dalam format xyxy
        img0_shape: Shape asli gambar
        ratio_pad: Output dari letterbox untuk img1 (optional)
        
    Returns:
        Koordinat yang direscale
    """
    if ratio_pad is None:  # Calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # Koordinat
    if isinstance(coords, torch.Tensor):
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        # Clip koordinat
        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    else:  # numpy
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        # Clip koordinat
        coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])  # y1, y2
    
    return coords