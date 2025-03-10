"""
File: smartcash/ui_handlers/training.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk komponen UI eksekusi training model SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys, time, random, threading
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from smartcash.utils.ui_utils import (
    create_info_alert, create_status_indicator, styled_html,
    create_metric_display
)

def setup_training_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI eksekusi training model."""
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'framework': 'YOLOv5',
                'pretrained': True,
            },
            'training': {
                'epochs': 50,
                'batch_size': 16,
                'optimizer': 'Adam',
                'lr0': 0.01,
            }
        }
    
    # Extract UI components
    components = ui_components
    
    # Data untuk handler
    data = {
        'config': config,
        'training': {
            'running': False,
            'paused': False,
            'current_epoch': 0,
            'total_epochs': config['training'].get('epochs', 50),
            'metrics': {
                'train_loss': [],
                'val_loss': [],
                'precision': [],
                'recall': [],
                'mAP': []
            },
            'best_metrics': {
                'mAP': 0,
                'epoch': 0
            },
            'checkpoints': []
        }
    }
    
    # Thread untuk simulasi training
    training_thread = None
    lock = threading.Lock()
    
    # Handler untuk start training button
    def on_start_training(b):
        if data['training']['running']:
            return
        
        with components['training_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai proses training..."))
            
            try:
                # Ambil options training
                training_mode = components['training_options'].children[0].value
                use_gpu = components['training_options'].children[1].value
                enable_logging = components['training_options'].children[2].value
                enable_checkpointing = components['training_options'].children[3].value
                
                # Reset progress jika from scratch
                if training_mode == "From Scratch":
                    data['training']['current_epoch'] = 0
                    data['training']['metrics'] = {
                        'train_loss': [],
                        'val_loss': [],
                        'precision': [],
                        'recall': [],
                        'mAP': []
                    }
                    data['training']['best_metrics'] = {
                        'mAP': 0,
                        'epoch': 0
                    }
                
                # Reset dan update UI
                components['training_progress'].value = 0
                components['epoch_progress'].value = 0
                
                # Update total epochs dari config
                data['training']['total_epochs'] = data['config']['training'].get('epochs', 50)
                
                # Aktifkan/nonaktifkan buttons
                components['start_training_button'].disabled = True
                components['pause_training_button'].disabled = False
                components['stop_training_button'].disabled = False
                
                # Set flag training
                data['training']['running'] = True
                data['training']['paused'] = False
                
                # Informasi training
                display(create_info_alert(
                    f"Training dimulai dengan backbone {data['config']['model']['backbone']}, "
                    f"{data['training']['total_epochs']} epochs, batch size "
                    f"{data['config']['training']['batch_size']}, optimizer "
                    f"{data['config']['training']['optimizer']}, learning rate "
                    f"{data['config']['training']['lr0']}",
                    "info", "🏋️"
                ))
                
                # Informasi device
                if use_gpu:
                    display(create_status_indicator("success", "✅ GPU terdeteksi dan akan digunakan"))
                else:
                    display(create_status_indicator("warning", "⚠️ Training akan menggunakan CPU (lebih lambat)"))
                
                # Mulai thread training
                global training_thread
                training_thread = threading.Thread(target=run_training_simulation)
                training_thread.daemon = True
                training_thread.start()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                components['start_training_button'].disabled = False
                components['pause_training_button'].disabled = True
                components['stop_training_button'].disabled = True
                data['training']['running'] = False
    
    # Handler untuk pause training button
    def on_pause_training(b):
        with lock:
            data['training']['paused'] = not data['training']['paused']
            
            with components['training_status']:
                if data['training']['paused']:
                    display(create_status_indicator("warning", "⏸ Training dijeda"))
                    b.description = "Resume Training"
                    b.icon = "play"
                else:
                    display(create_status_indicator("info", "▶️ Training dilanjutkan"))
                    b.description = "Pause Training"
                    b.icon = "pause"
    
    # Handler untuk stop training button
    def on_stop_training(b):
        with lock:
            data['training']['running'] = False
            
            with components['training_status']:
                display(create_status_indicator("warning", "🛑 Training dihentikan"))
            
            # Reset button state
            components['start_training_button'].disabled = False
            components['pause_training_button'].disabled = True
            components['stop_training_button'].disabled = True
    
    # Simulasi proses training (berjalan di thread terpisah)
    def run_training_simulation():
        try:
            total_epochs = data['training']['total_epochs']
            start_epoch = data['training']['current_epoch']
            
            # Update progress bar maksimum
            components['training_progress'].max = total_epochs
            
            # Loop untuk setiap epoch
            for epoch in range(start_epoch, total_epochs):
                # Cek jika training dihentikan
                if not data['training']['running']:
                    break
                
                # Update current epoch
                with lock:
                    data['training']['current_epoch'] = epoch
                
                # Simulasi steps dalam satu epoch
                steps_per_epoch = 10  # Simulasi 10 batch per epoch
                
                for step in range(steps_per_epoch):
                    # Cek jika training dihentikan
                    if not data['training']['running']:
                        break
                    
                    # Pause jika diminta
                    while data['training']['paused']:
                        if not data['training']['running']:
                            break
                        time.sleep(0.1)
                    
                    # Update epoch progress
                    update_epoch_progress(step, steps_per_epoch)
                    
                    # Tambahkan delay untuk simulasi
                    time.sleep(0.2)
                
                # Generate metrics untuk epoch ini
                generate_epoch_metrics(epoch)
                
                # Update UI untuk metrics
                update_metrics_display()
                
                # Update visualisasi
                update_visualization()
                
                # Simulasi checkpoint saving
                if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
                    save_checkpoint(epoch)
                
                # Update progress
                components['training_progress'].value = epoch + 1
                components['training_progress'].description = f"Epoch {epoch+1}/{total_epochs}"
            
            # Training selesai
            if data['training']['running']:
                with components['training_status']:
                    display(create_status_indicator("success", "✅ Training selesai!"))
                    
                    # Tampilkan final metrics
                    display(create_info_alert(
                        f"Best mAP: {data['training']['best_metrics']['mAP']:.4f} pada epoch "
                        f"{data['training']['best_metrics']['epoch']}",
                        "success", "🏆"
                    ))
                
                # Update checkpoint list di UI
                update_checkpoint_list()
                
                # Reset UI state
                components['start_training_button'].disabled = False
                components['pause_training_button'].disabled = True
                components['stop_training_button'].disabled = True
                
                data['training']['running'] = False
        
        except Exception as e:
            with components['training_status']:
                display(create_status_indicator("error", f"❌ Error dalam training: {str(e)}"))
            
            # Reset UI state
            components['start_training_button'].disabled = False
            components['pause_training_button'].disabled = True
            components['stop_training_button'].disabled = True
            
            data['training']['running'] = False
    
    # Fungsi untuk update epoch progress bar
    def update_epoch_progress(step, total_steps):
        components['epoch_progress'].value = step + 1
        components['epoch_progress'].max = total_steps
        components['epoch_progress'].description = f"Batch {step+1}/{total_steps}"
    
    # Generate metrics untuk satu epoch
    def generate_epoch_metrics(epoch):
        # Simulasi metrics dengan trend membaik seiring waktu dengan sedikit random noise
        progress_factor = epoch / data['training']['total_epochs']
        random_factor = 0.05 * (random.random() - 0.5)  # +/- 2.5% random noise
        
        # Train loss: dari 1.0 turun ke 0.2
        train_loss = 1.0 - (0.8 * progress_factor) + (random_factor * 0.5)
        
        # Val loss: dari 1.2 turun ke 0.3, sedikit lebih tinggi dari train loss
        val_loss = 1.2 - (0.9 * progress_factor) + random_factor
        
        # Precision: dari 0.6 naik ke 0.9
        precision = 0.6 + (0.3 * progress_factor) + random_factor
        
        # Recall: dari 0.5 naik ke 0.85
        recall = 0.5 + (0.35 * progress_factor) + random_factor
        
        # mAP: dari 0.4 naik ke 0.85
        map_score = 0.4 + (0.45 * progress_factor) + random_factor
        
        # Batasi nilai ke range yang valid
        train_loss = max(0.1, min(1.5, train_loss))
        val_loss = max(0.15, min(1.5, val_loss))
        precision = max(0.3, min(0.99, precision))
        recall = max(0.3, min(0.99, recall))
        map_score = max(0.2, min(0.99, map_score))
        
        # Simpan metrics
        with lock:
            data['training']['metrics']['train_loss'].append(train_loss)
            data['training']['metrics']['val_loss'].append(val_loss)
            data['training']['metrics']['precision'].append(precision)
            data['training']['metrics']['recall'].append(recall)
            data['training']['metrics']['mAP'].append(map_score)
            
            # Update best metrics jika perlu
            if map_score > data['training']['best_metrics']['mAP']:
                data['training']['best_metrics']['mAP'] = map_score
                data['training']['best_metrics']['epoch'] = epoch
    
    # Update metrics display
    def update_metrics_display():
        epoch = data['training']['current_epoch']
        metrics = data['training']['metrics']
        
        with components['metrics_output']:
            clear_output(wait=True)
            
            if epoch >= 0 and len(metrics['train_loss']) > 0:
                # Create row of metrics
                display(widgets.HBox([
                    create_metric_display("Epoch", epoch + 1),
                    create_metric_display("Train Loss", f"{metrics['train_loss'][-1]:.4f}"),
                    create_metric_display("Val Loss", f"{metrics['val_loss'][-1]:.4f}"),
                    create_metric_display("Precision", f"{metrics['precision'][-1]:.4f}"),
                    create_metric_display("Recall", f"{metrics['recall'][-1]:.4f}"),
                    create_metric_display("mAP", f"{metrics['mAP'][-1]:.4f}", 
                                         is_good=metrics['mAP'][-1] > 0.7)
                ]))
                
                # Show best mAP
                display(create_info_alert(
                    f"Best mAP: {data['training']['best_metrics']['mAP']:.4f} (epoch {data['training']['best_metrics']['epoch'] + 1})",
                    "info", "🏆"
                ))
    
    # Fungsi untuk update visualisasi
    def update_visualization():
        metrics = data['training']['metrics']
        
        if len(metrics['train_loss']) == 0:
            return
        
        # Plot loss curves
        with components['visualization_tabs'].children[0]:
            clear_output(wait=True)
            
            plt.figure(figsize=(10, 5))
            epochs = range(1, len(metrics['train_loss']) + 1)
            plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
            plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(linestyle='--', alpha=0.6)
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
        
        # Plot metrics history
        with components['visualization_tabs'].children[1]:
            clear_output(wait=True)
            
            plt.figure(figsize=(10, 5))
            epochs = range(1, len(metrics['precision']) + 1)
            plt.plot(epochs, metrics['precision'], 'g-', label='Precision')
            plt.plot(epochs, metrics['recall'], 'b-', label='Recall')
            plt.plot(epochs, metrics['mAP'], 'r-', label='mAP')
            plt.title('Metrics History')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(linestyle='--', alpha=0.6)
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
        
        # Plot learning rate
        with components['visualization_tabs'].children[2]:
            clear_output(wait=True)
            
            # Simulasi learning rate schedule (cosine decay)
            plt.figure(figsize=(10, 5))
            epochs = range(1, data['training']['total_epochs'] + 1)
            initial_lr = data['config']['training'].get('lr0', 0.01)
            
            # Buat simulasi cosine decay
            lrs = []
            for i in range(len(epochs)):
                progress = i / (len(epochs) - 1)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                lr = initial_lr * cosine_decay
                lrs.append(lr)
            
            plt.plot(epochs, lrs, 'b-')
            # Tambahkan marker untuk posisi epoch saat ini
            if data['training']['current_epoch'] < len(epochs):
                current_epoch = data['training']['current_epoch'] + 1
                current_lr = lrs[data['training']['current_epoch']]
                plt.scatter([current_epoch], [current_lr], color='red', s=100, zorder=5)
                plt.annotate(f'Current: {current_lr:.6f}', 
                            (current_epoch, current_lr),
                            xytext=(10, -20),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(linestyle='--', alpha=0.6)
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
        
        # Plot confusion matrix (dummy)
        with components['visualization_tabs'].children[3]:
            clear_output(wait=True)
            
            # Generate dummy confusion matrix
            plt.figure(figsize=(10, 8))
            classes = ['001', '002', '005', '010', '020', '050', '100']
            num_classes = len(classes)
            
            # Simulasi confusion matrix yang semakin baik seiring waktu
            progress_factor = len(metrics['train_loss']) / data['training']['total_epochs']
            
            # Generate matrix dengan high accuracy di diagonal dan errors menurun over time
            cm = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                # Diagonal - true positives
                cm[i, i] = 100 * (0.5 + 0.4 * progress_factor + 0.1 * random.random())
                
                # Errors - false positives/negatives
                for j in range(num_classes):
                    if i != j:
                        cm[i, j] = 10 * (1.0 - progress_factor) * random.random()
            
            # Plot
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = range(num_classes)
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text
            thresh = cm.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, f"{cm[i, j]:.0f}",
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            display(plt.gcf())
            plt.close()
    
    # Simulasi save checkpoint
    def save_checkpoint(epoch):
        with components['checkpoint_info']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"💾 Menyimpan checkpoint untuk epoch {epoch+1}..."))
            
            # Simulasi delay
            time.sleep(0.5)
            
            # Generate checkpoint filename
            checkpoint_type = "best" if epoch == data['training']['best_metrics']['epoch'] else "epoch"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"smartcash_{data['config']['model']['backbone']}_{checkpoint_type}_{epoch+1}_{timestamp}.pth"
            
            # Simpan info checkpoint