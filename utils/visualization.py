# File: smartcash/utils/visualization.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk visualisasi hasil training, evaluasi, dan deteksi
# dengan dukungan khusus untuk Google Colab

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import cv2
from matplotlib.figure import Figure
from IPython.display import display, HTML, clear_output

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def configure_for_colab():
    """Konfigurasi matplotlib untuk lingkungan Colab."""
    try:
        from google.colab import output
        # Aktifkan mode inline
        get_ipython().run_line_magic('matplotlib', 'inline')
        # Tingkatkan resolusi gambar
        get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
        return True
    except:
        return False

# Coba deteksi dan konfigurasi untuk Colab
is_colab = configure_for_colab()

def plot_training_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Metrik Training",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot metrik training seperti loss, accuracy, dll.
    
    Args:
        metrics: Dictionary berisi metrik {'metrik_name': [nilai_per_epoch]}
        title: Judul plot
        figsize: Ukuran gambar (width, height)
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        
    Returns:
        Figure matplotlib
    """
    num_metrics = len(metrics)
    
    if num_metrics == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "❌ Tidak ada metrik yang tersedia", 
                ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Tentukan jumlah subplot berdasarkan jumlah metrik
    fig = plt.figure(figsize=figsize)
    
    # Jumlah baris subplot (maksimal 3 subplot per baris)
    n_rows = int(np.ceil(num_metrics / 2))
    
    # Plot untuk setiap metrik
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = fig.add_subplot(n_rows, min(num_metrics, 2), i+1)
        
        # Plot metrik
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'o-', linewidth=2, markersize=5)
        
        # Beri label
        ax.set_title(f"{metric_name.replace('_', ' ').title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        
        # Tambahkan grid untuk mempermudah pembacaan
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tambahkan nilai terbaik
        if len(values) > 0:
            if 'loss' in metric_name.lower():
                best_epoch = np.argmin(values)
                best_value = values[best_epoch]
                marker = 'min'
            else:
                best_epoch = np.argmax(values)
                best_value = values[best_epoch]
                marker = 'max'
                
            ax.plot(best_epoch + 1, best_value, 'r*', markersize=15)
            ax.annotate(f"{best_value:.4f} ({marker})", 
                       (best_epoch + 1, best_value),
                       xytext=(10, 0),
                       textcoords="offset points",
                       fontsize=12,
                       color='red')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = "Blues"
) -> Figure:
    """
    Plot confusion matrix dengan anotasi.
    
    Args:
        cm: Confusion matrix array
        class_names: List nama kelas
        normalize: Normalisasi nilai ke persentase
        title: Judul plot
        figsize: Ukuran gambar (width, height)
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        cmap: Colormap untuk heatmap
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalisasi nilai jika diminta
    if normalize:
        cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        data_to_plot = cm_norm
        fmt = '.1%'
    else:
        data_to_plot = cm
        fmt = 'd'
    
    # Plot heatmap
    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 10}
    )
    
    # Set judul dan label
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Label Sebenarnya', fontsize=14)
    ax.set_xlabel('Label Prediksi', fontsize=14)
    
    # Rotasi label sumbu X untuk kemudahan pembacaan
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_detections(
    images: Union[np.ndarray, torch.Tensor, List],
    detections: List[Dict],
    class_names: List[str],
    conf_threshold: float = 0.5,
    figsize: Tuple[int, int] = (20, 20),
    max_images: int = 9,
    title: str = "Hasil Deteksi",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Visualisasi hasil deteksi pada gambar.
    
    Args:
        images: Tensor atau array gambar [B, C, H, W] atau list gambar
        detections: List detection results {'boxes': tensor, 'scores': tensor, 'labels': tensor}
        class_names: List nama kelas untuk label
        conf_threshold: Confidence threshold untuk menampilkan deteksi
        figsize: Ukuran gambar (width, height)
        max_images: Jumlah maksimal gambar yang ditampilkan
        title: Judul plot
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        
    Returns:
        Figure matplotlib
    """
    # Konversi images ke numpy array
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
        
        # [B, C, H, W] -> [B, H, W, C]
        if images.shape[1] == 3 and len(images.shape) == 4:
            images = np.transpose(images, (0, 2, 3, 1))
    
    # Batasi jumlah gambar
    n_images = min(len(images), len(detections), max_images)
    
    # Tentukan grid layout
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Buat figure
    fig, axes = plt.subplots(
        grid_size, grid_size, 
        figsize=figsize, 
        squeeze=False
    )
    
    # Color palette untuk bounding box
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    # Plot setiap gambar dan deteksinya
    for i in range(n_images):
        ax = axes[i // grid_size, i % grid_size]
        
        # Ambil gambar
        img = images[i].copy()
        
        # Normalisasi nilai pixel jika perlu
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Tampilkan gambar
        ax.imshow(img)
        
        # Tambahkan deteksi
        det = detections[i]
        
        if 'boxes' in det and len(det['boxes']) > 0:
            boxes = det['boxes'].cpu().numpy()
            scores = det['scores'].cpu().numpy()
            labels = det['labels'].cpu().numpy()
            
            h, w = img.shape[:2]
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= conf_threshold:
                    # YOLO format (xcenter, ycenter, width, height) ke pixel
                    if len(box) == 4 and max(box) <= 1.0:
                        # Normalisasi koordinat
                        x_center, y_center, width, height = box
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                    else:
                        # Koordinat pixel absolut
                        x1, y1, x2, y2 = map(int, box)
                    
                    # Label kelas
                    try:
                        class_name = class_names[int(label)]
                    except:
                        class_name = f"Class {label}"
                    
                    # Gambar bounding box
                    color = colors[int(label) % len(colors)]
                    color = (color[0], color[1], color[2])
                    
                    # Rectangle
                    rect = plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor=color,
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    
                    # Label
                    ax.text(
                        x1, y1 - 5,
                        f"{class_name} {score:.2f}",
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor=color, alpha=0.7, pad=2)
                    )
        
        ax.set_title(f"Gambar {i+1}")
        ax.axis('off')
    
    # Sembunyikan subplot kosong
    for i in range(n_images, grid_size * grid_size):
        axes[i // grid_size, i % grid_size].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_learning_curves(
    train_loss: List[float],
    val_loss: List[float],
    metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Kurva Learning",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot kurva learning (loss dan metrik tambahan).
    
    Args:
        train_loss: List nilai loss training per epoch
        val_loss: List nilai loss validasi per epoch
        metrics: Dictionary metrik tambahan (opsional)
        title: Judul plot
        figsize: Ukuran gambar (width, height)
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        
    Returns:
        Figure matplotlib
    """
    # Tentukan jumlah subplot
    n_plots = 1 + (1 if metrics else 0)
    
    # Buat figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Jika hanya 1 subplot, convert ke list untuk konsistensi
    if n_plots == 1:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    epochs = range(1, len(train_loss) + 1)
    
    ax.plot(epochs, train_loss, 'b-o', label='Training Loss')
    ax.plot(epochs, val_loss, 'r-^', label='Validation Loss')
    
    # Tandai nilai loss terbaik
    best_epoch = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch]
    
    ax.plot(best_epoch + 1, best_val_loss, 'r*', markersize=12)
    ax.annotate(f"Best: {best_val_loss:.4f}", 
               (best_epoch + 1, best_val_loss),
               xytext=(5, -20),
               textcoords="offset points",
               fontsize=10,
               color='red')
    
    ax.set_title("Loss per Epoch", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot metrik tambahan jika ada
    if metrics and n_plots > 1:
        ax = axes[1]
        
        for name, values in metrics.items():
            if len(values) == len(epochs):  # Pastikan jumlah nilai sesuai
                ax.plot(epochs, values, 'o-', linewidth=2, label=name.replace('_', ' ').title())
        
        ax.set_title("Metrik per Epoch", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Nilai")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_realtime_training(
    train_loss: List[float],
    val_loss: List[float],
    metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Progres Training (Real-time)",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot progres training secara real-time, khusus untuk notebook/Colab.
    
    Args:
        train_loss: List nilai loss training per epoch
        val_loss: List nilai loss validasi per epoch
        metrics: Dictionary metrik tambahan (opsional)
        title: Judul plot
        figsize: Ukuran gambar (width, height)
    """
    # Bersihkan output cell
    clear_output(wait=True)
    
    # Plot learning curves
    fig = plot_learning_curves(
        train_loss=train_loss,
        val_loss=val_loss,
        metrics=metrics,
        title=title,
        figsize=figsize,
        show=False
    )
    
    # Tampilkan plot
    display(fig)
    plt.close(fig)
    
    # Tampilkan nilai terakhir dalam format tabel
    if len(train_loss) > 0:
        last_epoch = len(train_loss)
        last_train_loss = train_loss[-1]
        last_val_loss = val_loss[-1]
        
        table_html = f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;">
        <h3>📊 Epoch {last_epoch}</h3>
        <table style="width:60%; text-align:left;">
          <tr>
            <th style="padding:5px;">Metrik</th>
            <th style="padding:5px;">Nilai</th>
          </tr>
          <tr>
            <td style="padding:5px;">Training Loss</td>
            <td style="padding:5px;">{last_train_loss:.4f}</td>
          </tr>
          <tr>
            <td style="padding:5px;">Validation Loss</td>
            <td style="padding:5px;">{last_val_loss:.4f}</td>
          </tr>
        """
        
        # Tambahkan metrik tambahan jika ada
        if metrics:
            for name, values in metrics.items():
                if len(values) > 0:
                    table_html += f"""
                    <tr>
                      <td style="padding:5px;">{name.replace('_', ' ').title()}</td>
                      <td style="padding:5px;">{values[-1]:.4f}</td>
                    </tr>
                    """
        
        table_html += """
        </table>
        </div>
        """
        
        display(HTML(table_html))

def plot_backbone_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'inference_time'],
    title: str = "Perbandingan Backbone",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Visualisasi perbandingan metrik antar backbone.
    
    Args:
        results: Dictionary hasil perbandingan {model_name: {metric: value}}
        metrics: List metrik yang akan divisualisasikan
        title: Judul plot
        figsize: Ukuran gambar (width, height)
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        
    Returns:
        Figure matplotlib
    """
    # Filter metrik yang ada di semua hasil
    available_metrics = []
    for metric in metrics:
        # Periksa apakah metrik ada di semua model
        if all(metric in results[model] for model in results):
            available_metrics.append(metric)
    
    if not available_metrics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "❌ Tidak ada metrik yang tersedia untuk perbandingan", 
                ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Jumlah model dan metrik
    n_models = len(results)
    n_metrics = len(available_metrics)
    
    # Setup figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Jika hanya 1 metrik, konversi axes ke list
    if n_metrics == 1:
        axes = [axes]
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    # Plot setiap metrik
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Kumpulkan nilai metrik untuk setiap model
        model_names = list(results.keys())
        metric_values = [results[model][metric] for model in model_names]
        
        # Jika inference time, konversi ke ms
        if 'time' in metric.lower():
            metric_values = [val * 1000 for val in metric_values]  # ke ms
            y_label = f"{metric.replace('_', ' ').title()} (ms)"
        else:
            y_label = metric.replace('_', ' ').title()
        
        # Plot bar chart
        bars = ax.bar(model_names, metric_values, color=colors)
        
        # Label
        ax.set_title(y_label, fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel(y_label)
        
        # Tambahkan nilai pada bar
        for bar, val in zip(bars, metric_values):
            if 'time' in metric.lower():
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05 * max(metric_values),
                    f"{val:.1f} ms",
                    ha='center',
                    fontsize=9
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha='center',
                    fontsize=9
                )
        
        # Rotasi label sumbu X untuk kemudahan pembacaan
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Tambahkan grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_training_summary(
    history: Dict,
    title: str = "Ringkasan Training",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Visualisasi ringkasan proses training yang lengkap.
    
    Args:
        history: Dictionary riwayat training {'train_loss': [], 'val_loss': [], 'metrics': {}, ...}
        title: Judul plot
        figsize: Ukuran gambar (width, height)
        save_path: Path untuk menyimpan hasil plot (opsional)
        show: Tampilkan plot jika True
        
    Returns:
        Figure matplotlib
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    learning_rates = history.get('learning_rates', [])
    metrics = history.get('metrics', {})
    
    if not train_loss or not val_loss:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "❌ Data training tidak cukup untuk visualisasi", 
                ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Jumlah subplot
    n_plots = 3  # Loss, LR, dan Metrics
    
    # Setup figure
    fig = plt.figure(figsize=figsize)
    
    # 1. Plot Loss
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    epochs = range(1, len(train_loss) + 1)
    
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, val_loss, 'r-^', label='Validation Loss', markersize=4)
    
    # Tandai nilai loss terbaik
    best_epoch = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch]
    
    ax1.plot(best_epoch + 1, best_val_loss, 'r*', markersize=12)
    ax1.annotate(f"Best: {best_val_loss:.4f}", 
               (best_epoch + 1, best_val_loss),
               xytext=(5, -20),
               textcoords="offset points",
               fontsize=10,
               color='red')
    
    ax1.set_title("Loss per Epoch", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot Learning Rate
    if learning_rates:
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax2.plot(epochs, learning_rates, 'g-')
        ax2.set_title("Learning Rate", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot Metrics
    if metrics:
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        
        for name, values in metrics.items():
            if len(values) == len(epochs):  # Pastikan jumlah nilai sesuai
                ax3.plot(epochs, values, '-o', linewidth=2, markersize=4, 
                        label=name.replace('_', ' ').title())
        
        ax3.set_title("Metrik per Epoch", fontsize=14)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Nilai")
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig