"""
File: smartcash/ui/dataset/visualization/utils/visualization_utils.py
Deskripsi: Utility functions untuk visualisasi dataset
"""

from typing import Dict, List, Tuple, Optional, Union
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_class_distribution_plot(class_data: Dict[str, Dict[str, int]], 
                                colors: Dict[str, str]) -> go.Figure:
    """Buat plot distribusi kelas
    
    Args:
        class_data: Data distribusi kelas per split
        colors: Warna untuk setiap split
        
    Returns:
        go.Figure: Plot distribusi kelas
    """
    try:
        splits = ["train", "val", "test"]
        
        # Buat figure dengan subplot untuk setiap split
        fig = make_subplots(
            rows=1, 
            cols=len(splits),
            subplot_titles=[f"Distribusi Kelas - {split.upper()}" for split in splits]
        )
        
        for i, split in enumerate(splits, 1):
            if split in class_data and class_data[split]:
                classes = list(class_data[split].keys())
                counts = list(class_data[split].values())
                
                fig.add_trace(
                    go.Bar(
                        x=classes,
                        y=counts,
                        name=f"{split.upper()}",
                        marker_color=colors.get(split, "gray"),
                        text=counts,
                        textposition='auto'
                    ),
                    row=1, col=i
                )
        
        # Update layout
        fig.update_layout(
            title_text="Distribusi Kelas per Split Dataset",
            showlegend=False,
            height=500,
            template="plotly_white",
            margin=dict(t=100, b=100)
        )
        
        # Update x-axis untuk rotasi label
        for i in range(1, len(splits) + 1):
            fig.update_xaxes(tickangle=45, row=1, col=i)
        
        return fig
        
    except Exception as e:
        logger.error(f"Gagal membuat plot distribusi kelas: {e}")
        return None

def create_image_size_plot(size_data: Dict[str, List[Tuple[int, int]]], 
                          colors: Dict[str, str]) -> go.Figure:
    """Buat plot distribusi ukuran gambar
    
    Args:
        size_data: Data ukuran gambar per split
        colors: Warna untuk setiap split
        
    Returns:
        go.Figure: Plot distribusi ukuran gambar
    """
    try:
        fig = go.Figure()
        
        for split, sizes in size_data.items():
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                
                fig.add_trace(
                    go.Scatter(
                        x=widths,
                        y=heights,
                        mode='markers',
                        name=split.upper(),
                        marker=dict(
                            color=colors.get(split, "gray"),
                            size=8,
                            opacity=0.7
                        )
                    )
                )
        
        # Hitung aspect ratio umum
        common_ratios = {
            '1:1': (1, 1),
            '4:3': (4, 3),
            '16:9': (16, 9),
            '3:2': (3, 2)
        }
        
        # Update layout
        fig.update_layout(
            title="Distribusi Ukuran Gambar",
            xaxis_title="Lebar (px)",
            yaxis_title="Tinggi (px)",
            legend_title="Split",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        # Tambahkan garis aspek rasio umum
        max_dim = max(
            max([w for sizes in size_data.values() for w, _ in sizes or []] or [0]),
            max([h for sizes in size_data.values() for _, h in sizes or []] or [0])
        )
        
        if max_dim > 0:
            for ratio_name, (w, h) in common_ratios.items():
                max_side = max(w, h)
                scale = max_dim / max_side
                x_end = w * scale
                y_end = h * scale
                
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=x_end, y1=y_end,
                    line=dict(color="gray", width=1, dash="dot"),
                    name=f"{ratio_name}"
                )
                
                # Tambahkan label aspek rasio
                fig.add_annotation(
                    x=x_end*0.9, y=y_end*0.9,
                    text=ratio_name,
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Gagal membuat plot ukuran gambar: {e}")
        return None

def draw_bounding_boxes(image: np.ndarray, 
                       bboxes: List[List[float]], 
                       class_names: List[str] = None,
                       class_colors: Dict[int, str] = None) -> np.ndarray:
    """Gambar bounding boxes pada gambar
    
    Args:
        image: Gambar dalam format numpy array (H, W, C)
        bboxes: List bounding boxes dalam format [x1, y1, x2, y2, class_id, confidence]
        class_names: Daftar nama kelas
        class_colors: Warna untuk setiap kelas
        
    Returns:
        np.ndarray: Gambar dengan bounding boxes
    """
    try:
        # Konversi ke float32 jika belum
        img = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image
        
        # Konversi ke RGB jika grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        # Default colors
        if class_colors is None:
            class_colors = {
                i: tuple(np.random.randint(0, 255, 3).tolist())
                for i in range(len(class_names) if class_names else 100)
            }
            
        # Default class names
        if class_names is None:
            class_names = [f"Class {i}" for i in range(100)]
            
        # Gambar setiap bounding box
        for box in bboxes:
            if len(box) >= 6:  # Format: [x1, y1, x2, y2, class_id, confidence]
                x1, y1, x2, y2, class_id, confidence = box[:6]
                class_id = int(class_id)
                
                # Dapatkan warna untuk kelas ini
                color = class_colors.get(class_id, (0, 255, 0))
                
                # Konversi koordinat ke integer
                h, w = img.shape[:2]
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                # Gambar bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Tambahkan label
                label = f"{class_names[class_id]}: {confidence:.2f}" if class_id < len(class_names) else f"{class_id}: {confidence:.2f}"
                
                # Hitung ukuran teks
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Gambar background untuk teks
                cv2.rectangle(img, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                
                # Gambar teks
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
        
        return (img * 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Gagal menggambar bounding boxes: {e}")
        return image

def create_augmentation_comparison(original_img: np.ndarray, 
                                 augmented_imgs: List[Dict[str, Union[np.ndarray, str]]],
                                 n_cols: int = 3) -> plt.Figure:
    """Buat perbandingan antara gambar asli dan hasil augmentasi
    
    Args:
        original_img: Gambar asli
        augmented_imgs: List kamus berisi gambar hasil augmentasi dan nama augmentasi
        n_cols: Jumlah kolom dalam grid
        
    Returns:
        plt.Figure: Figure matplotlib
    """
    try:
        n_imgs = 1 + len(augmented_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        if n_imgs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Tampilkan gambar asli
        ax = axes.flat[0]
        ax.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax.set_title("Original")
        ax.axis('off')
        
        # Tampilkan gambar hasil augmentasi
        for i, aug in enumerate(augmented_imgs, 1):
            if i >= n_rows * n_cols:
                break
                
            ax = axes.flat[i]
            ax.imshow(cv2.cvtColor(aug['image'], cv2.COLOR_BGR2RGB))
            ax.set_title(aug.get('name', f'Augmentation {i}'))
            ax.axis('off')
        
        # Matikan sumbu untuk subplot yang tidak digunakan
        for j in range(i + 1, n_rows * n_cols):
            axes.flat[j].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Gagal membuat perbandingan augmentasi: {e}")
        return None
