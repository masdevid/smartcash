"""
File: smartcash/ui/utils/visualization_utils.py
Deskripsi: Utilitas visualisasi untuk komponen UI
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from smartcash.ui.utils.constants import COLORS

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')

def create_class_distribution_plot(
    class_counts: Dict[str, int],
    title: str = "Distribusi Kelas",
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    sort_by: str = 'value'  # 'value', 'name', 'none'
) -> plt.Figure:
    """
    Buat plot distribusi kelas dengan styling yang konsisten.
    
    Args:
        class_counts: Dictionary berisi {class_name: count}
        title: Judul plot
        figsize: Ukuran gambar
        colors: List warna kustom
        sort_by: Kriteria pengurutan data
        
    Returns:
        Matplotlib Figure
    """
    # Create data frame
    df = pd.DataFrame({
        'class': list(class_counts.keys()),
        'count': list(class_counts.values())
    })
    
    # Sort data
    if sort_by == 'value':
        df = df.sort_values('count', ascending=False)
    elif sort_by == 'name':
        df = df.sort_values('class')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    sns.barplot(
        x='class', 
        y='count', 
        data=df, 
        palette=colors, 
        ax=ax
    )
    
    # Add labels and styling
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Kelas', fontsize=12)
    ax.set_ylabel('Jumlah', fontsize=12)
    
    # Rotate x labels if more than 8 classes
    if len(class_counts) > 8:
        plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(df['count']):
        ax.text(
            i, 
            v + 0.1, 
            str(v), 
            ha='center',
            fontweight='bold'
        )
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Buat visualisasi confusion matrix dengan annotasi.
    
    Args:
        cm: Confusion matrix 2D numpy array
        class_names: List nama kelas
        normalize: Normalisasi nilai 0-1
        title: Judul plot
        figsize: Ukuran gambar
        cmap: Colormap
        
    Returns:
        Matplotlib Figure
    """
    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap,
        linewidths=.5, 
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Add labels and styling
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_metrics_history_plot(
    metrics_history: Dict[str, List[float]],
    title: str = "Training Metrics",
    figsize: Tuple[int, int] = (12, 6),
    include_lr: bool = False,
    log_lr: bool = True
) -> plt.Figure:
    """
    Buat plot history metrik training.
    
    Args:
        metrics_history: Dictionary berisi {metric_name: [values]}
        title: Judul plot
        figsize: Ukuran gambar
        include_lr: Sertakan learning rate
        log_lr: Gunakan skala log untuk learning rate
        
    Returns:
        Matplotlib Figure
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Color maps
    colors = {
        'train_loss': COLORS['danger'],
        'val_loss': COLORS['primary'],
        'precision': COLORS['success'],
        'recall': COLORS['info'],
        'mAP': COLORS['warning'],
        'f1': COLORS['highlight'],
        'accuracy': COLORS['secondary'],
    }
    
    # Plot each metric
    lr_plotted = False
    
    for metric_name, values in metrics_history.items():
        if metric_name == 'lr' and include_lr:
            # Plot learning rate on secondary y-axis
            ax2 = ax1.twinx()
            epochs = range(1, len(values) + 1)
            ax2.plot(epochs, values, 'g--', color='gray', label='Learning Rate')
            ax2.set_ylabel('Learning Rate', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            if log_lr:
                ax2.set_yscale('log')
            lr_plotted = True
        elif metric_name != 'lr':
            # Plot other metrics on primary y-axis
            epochs = range(1, len(values) + 1)
            color = colors.get(metric_name, 'black')
            ax1.plot(epochs, values, '-', color=color, label=metric_name.capitalize())
    
    # Add grid, labels, and legend
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(title, fontsize=16, pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    if lr_plotted:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    
    ax1.legend(handles, labels, loc='best')
    
    plt.tight_layout()
    return fig

def create_model_comparison_plot(
    comparison_data: pd.DataFrame,
    x_col: str,
    metric_cols: List[str],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = 'bar'  # 'bar', 'line', 'radar'
) -> plt.Figure:
    """
    Buat visualisasi perbandingan model.
    
    Args:
        comparison_data: DataFrame berisi data perbandingan
        x_col: Kolom untuk sumbu x (biasanya nama model)
        metric_cols: List kolom metrik yang akan divisualisasikan
        title: Judul plot
        figsize: Ukuran gambar
        plot_type: Jenis plot ('bar', 'line', 'radar')
        
    Returns:
        Matplotlib Figure
    """
    if plot_type == 'radar':
        return _create_radar_comparison(comparison_data, x_col, metric_cols, title, figsize)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get number of models and metrics
    n_models = len(comparison_data[x_col].unique())
    n_metrics = len(metric_cols)
    
    # Set width for bars
    width = 0.8 / n_metrics
    
    # For each metric, create bars for all models
    for i, metric in enumerate(metric_cols):
        # Calculate position for this set of bars
        pos = np.arange(n_models) - (0.4 - width * (i + 0.5))
        
        # Create bars
        if plot_type == 'bar':
            ax.bar(
                pos, 
                comparison_data[metric], 
                width=width, 
                label=metric,
                alpha=0.8
            )
        elif plot_type == 'line':
            ax.plot(
                comparison_data[x_col],
                comparison_data[metric],
                'o-',
                label=metric
            )
    
    # Add labels and styling
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    
    if plot_type == 'bar':
        ax.set_xticks(np.arange(n_models))
        ax.set_xticklabels(comparison_data[x_col])
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def _create_radar_comparison(
    comparison_data: pd.DataFrame,
    x_col: str,
    metric_cols: List[str],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Buat plot radar untuk perbandingan model.
    
    Args:
        comparison_data: DataFrame berisi data perbandingan
        x_col: Kolom untuk label (biasanya nama model)
        metric_cols: List kolom metrik yang akan divisualisasikan
        title: Judul plot
        figsize: Ukuran gambar
        
    Returns:
        Matplotlib Figure
    """
    # Number of variables
    N = len(metric_cols)
    
    # Create angles for radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Set first axis on top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metric_cols)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    
    # Plot data
    for i, row in comparison_data.iterrows():
        model_name = row[x_col]
        values = [row[col] for col in metric_cols]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=16, pad=20)
    
    return fig

def create_metric_display(
    label: str, 
    value: Union[int, float, str],
    unit: Optional[str] = None,
    is_good: Optional[bool] = None
) -> widgets.HTML:
    """
    Buat widget display untuk metrik.
    
    Args:
        label: Label metrik
        value: Nilai metrik
        unit: Unit opsional (%, detik, dll)
        is_good: Menunjukkan apakah nilai baik (hijau), buruk (merah), atau netral
        
    Returns:
        Widget HTML
    """
    # Tentukan warna berdasarkan nilai is_good
    if is_good is None:
        color = COLORS['dark']  # Neutral
    elif is_good:
        color = COLORS['success']  # Green for good
    else:
        color = COLORS['danger']  # Red for bad
    
    # Format nilai
    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)
        
    # Tambahkan unit jika ada
    if unit:
        formatted_value = f"{formatted_value} {unit}"
    
    # Buat HTML
    metric_html = f"""
    <div style="margin: 10px 5px; padding: 8px; background-color: {COLORS['light']}; 
                border-radius: 5px; text-align: center; min-width: 120px;">
        <div style="font-size: 0.9em; color: {COLORS['muted']};">{label}</div>
        <div style="font-size: 1.3em; font-weight: bold; color: {color};">{formatted_value}</div>
    </div>
    """
    
    return widgets.HTML(value=metric_html)

def create_metrics_dashboard(
    metrics: Dict[str, Union[int, float, str]],
    title: Optional[str] = None,
    description: Optional[str] = None
) -> widgets.VBox:
    """
    Buat dashboard metrik dengan layout grid responsive.
    
    Args:
        metrics: Dictionary berisi {metric_name: value}
        title: Judul opsional
        description: Deskripsi opsional
        
    Returns:
        Widget container berisi dashboard
    """
    # Create header if provided
    header_widgets = []
    if title:
        header_widgets.append(
            widgets.HTML(f"<h3 style='margin-bottom:0'>{title}</h3>")
        )
    if description:
        header_widgets.append(
            widgets.HTML(f"<p style='color:{COLORS['muted']}'>{description}</p>")
        )
    
    # Create metrics grid
    metrics_widgets = []
    
    for label, value in metrics.items():
        # Determine if value is good (for coloring)
        is_good = None
        if isinstance(value, (int, float)):
            if 'accuracy' in label.lower() or 'precision' in label.lower() or 'recall' in label.lower():
                is_good = value > 0.7
            elif 'error' in label.lower() or 'loss' in label.lower():
                is_good = value < 0.3
        
        # Determine unit
        unit = None
        if 'time' in label.lower() or 'duration' in label.lower():
            unit = 's'
        elif 'percentage' in label.lower() or any(m in label.lower() for m in ['accuracy', 'recall', 'precision']):
            # Convert to percentage if between 0-1
            if isinstance(value, float) and 0 <= value <= 1:
                value = value * 100
                unit = '%'
        
        # Add metric widget
        metrics_widgets.append(
            create_metric_display(label, value, unit, is_good)
        )
    
    # Create metrics grid layout
    metrics_grid = widgets.HBox(
        metrics_widgets,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='stretch',
            width='100%'
        )
    )
    
    # Combine all elements
    dashboard = widgets.VBox(
        header_widgets + [metrics_grid],
        layout=widgets.Layout(
            margin='10px 0',
            width='100%'
        )
    )
    
    return dashboard