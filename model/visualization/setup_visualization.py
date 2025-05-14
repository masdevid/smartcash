"""
File: smartcash/model/visualization/setup_visualization.py
Deskripsi: Setup visualisasi untuk model SmartCash
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import base64
from IPython.display import display, HTML

def setup_visualization(theme: str = 'default', figsize: Tuple[int, int] = (10, 6)):
    """
    Setup visualisasi untuk model SmartCash.
    
    Args:
        theme: Tema visualisasi ('default', 'dark', 'light')
        figsize: Ukuran default untuk figure
        
    Returns:
        Dict berisi konfigurasi visualisasi
    """
    # Setup style sesuai tema
    if theme == 'dark':
        plt.style.use('dark_background')
        color_palette = 'viridis'
    elif theme == 'light':
        plt.style.use('seaborn-v0_8-whitegrid')
        color_palette = 'muted'
    else:  # default
        plt.style.use('seaborn-v0_8')
        color_palette = 'deep'
    
    # Setup seaborn
    sns.set_palette(color_palette)
    
    # Setup default figure size
    plt.rcParams['figure.figsize'] = figsize
    
    # Setup font
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Fungsi utilitas untuk visualisasi
    def create_metrics_chart(metrics: Dict[str, List[float]], title: str = 'Training Metrics'):
        """
        Buat chart untuk visualisasi metrik training.
        
        Args:
            metrics: Dictionary berisi metrik training
            title: Judul chart
            
        Returns:
            HTML widget dengan chart
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        
        # Plot metrik
        for name, values in metrics.items():
            if name != 'epochs':
                plt.plot(metrics.get('epochs', range(len(values))), values, label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Konversi plot ke gambar untuk ditampilkan di widget
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return HTML(f'<img src="data:image/png;base64,{img_str}" width="100%">')
    
    def create_confusion_matrix_chart(cm, class_names, title='Confusion Matrix'):
        """
        Buat chart untuk visualisasi confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Nama kelas
            title: Judul chart
            
        Returns:
            HTML widget dengan chart
        """
        plt.figure(figsize=figsize)
        
        # Normalisasi matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        cm_norm = np.nan_to_num(cm_norm)
        
        # Plot heatmap
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Konversi plot ke gambar
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return HTML(f'<img src="data:image/png;base64,{img_str}" width="100%">')
    
    # Konfigurasi visualisasi
    config = {
        'theme': theme,
        'figsize': figsize,
        'color_palette': color_palette,
        'utils': {
            'create_metrics_chart': create_metrics_chart,
            'create_confusion_matrix_chart': create_confusion_matrix_chart
        }
    }
    
    print(f"✨ Visualisasi berhasil disetup dengan tema '{theme}'")
    return config
