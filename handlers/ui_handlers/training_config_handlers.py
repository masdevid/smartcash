"""
File: smartcash/handlers/ui_handlers/training_config_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk komponen UI konfigurasi training, menangani perubahan konfigurasi 
          dan visualisasi learning rate schedule.
"""

import os
import torch
import yaml
import pickle
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
from datetime import datetime
from pathlib import Path

# Import common utilities
from smartcash.handlers.ui_handlers.common_utils import memory_manager, save_config, load_config

def on_generate_name_button_clicked(ui_components):
    """
    Handler untuk tombol generate nama eksperimen.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_config_ui()
    """
    adjectives = ['Cepat', 'Akurat', 'Kuat', 'Cerdas', 'Adaptif', 'Efisien', 'Optimal']
    nouns = ['Deteksi', 'Training', 'Model', 'Network', 'Percobaan', 'Iterasi']
    suffix = datetime.now().strftime('%m%d_%H%M')
    
    new_name = f"{random.choice(adjectives)}_{random.choice(nouns)}_{suffix}"
    ui_components['experiment_name_input'].value = new_name

def on_save_config_button_clicked(ui_components, config, logger, components=None):
    """
    Handler untuk tombol simpan konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_config_ui()
        config: Dictionary konfigurasi yang akan diupdate
        logger: Logger untuk mencatat aktivitas
        components: Optional dictionary dengan komponen lain yang diperlukan
    """
    if components is None:
        components = {}
        
    with ui_components['config_output']:
        clear_output()
        
        # Update konfigurasi dari UI
        # Backbone
        if 'model' not in config:
            config['model'] = {}
        config['model']['backbone'] = ui_components['backbone_dropdown'].value
        config['model']['pretrained'] = ui_components['pretrained_checkbox'].value
        
        # Training parameters
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = ui_components['epochs_slider'].value
        config['training']['batch_size'] = ui_components['batch_size_slider'].value
        config['training']['learning_rate'] = ui_components['lr_dropdown'].value
        config['training']['optimizer'] = ui_components['optimizer_dropdown'].value
        config['training']['scheduler'] = ui_components['scheduler_dropdown'].value
        config['training']['early_stopping_patience'] = ui_components['early_stopping_slider'].value
        config['training']['weight_decay'] = ui_components['weight_decay_dropdown'].value
        config['training']['save_every'] = ui_components['save_every_slider'].value
        
        # Update active layers
        config['layers'] = list(ui_components['layer_selection'].value)
            
        # Simpan ke file yaml
        try:
            training_config_path = 'configs/training_config.yaml'
            os.makedirs('configs', exist_ok=True)
            
            with open(training_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            # Simpan juga ke pickle untuk digunakan di module lain
            with open('config.pkl', 'wb') as f:
                pickle.dump(config, f)
                
            logger.success(f"✅ Konfigurasi training berhasil disimpan ke {training_config_path}")
            
            # Tampilkan informasi konfigurasi training
            print("💾 Konfigurasi training berhasil disimpan")
            print("\n📋 Parameter Training:")
            print(f"• Epochs: {ui_components['epochs_slider'].value}")
            print(f"• Batch Size: {ui_components['batch_size_slider'].value}")
            print(f"• Learning Rate: {ui_components['lr_dropdown'].value}")
            print(f"• Optimizer: {ui_components['optimizer_dropdown'].value}")
            print(f"• Scheduler: {ui_components['scheduler_dropdown'].value}")
            print(f"• Early Stopping Patience: {ui_components['early_stopping_slider'].value}")
            print(f"• Weight Decay: {ui_components['weight_decay_dropdown'].value}")
            print(f"• Save Checkpoint Every: {ui_components['save_every_slider'].value} epoch")
            print(f"• Layers Aktif: {', '.join(ui_components['layer_selection'].value)}")
            print(f"• Nama Eksperimen: {ui_components['experiment_name_input'].value}")
            
            # Update experiment tracker name jika tersedia
            if 'experiment_tracker' in components:
                components['experiment_tracker'].experiment_name = ui_components['experiment_name_input'].value
                print(f"\n🧪 Experiment tracker diupdate: {ui_components['experiment_name_input'].value}")
                
            # Perbarui dataloaders jika batch size berubah
            if 'dataloaders' in components and 'original_batch_size' in components and components['original_batch_size'] != ui_components['batch_size_slider'].value:
                print(f"\n⚠️ Batch size berubah dari {components['original_batch_size']} menjadi {ui_components['batch_size_slider'].value}")
                print("ℹ️ Jalankan inisialisasi pipeline kembali untuk memperbarui dataloader")
        except Exception as e:
            print(f"❌ Gagal menyimpan konfigurasi: {str(e)}")
            logger.error(f"Gagal menyimpan konfigurasi: {str(e)}")

def simulate_lr_schedule(epochs, lr, scheduler_type):
    """
    Simulasikan learning rate schedule.
    
    Args:
        epochs: Jumlah epoch untuk simulasi
        lr: Learning rate awal
        scheduler_type: Tipe scheduler ('plateau', 'step', 'cosine', 'onecycle')
        
    Returns:
        List learning rates untuk setiap epoch
    """
    lrs = []
    dummy_optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=lr)
    
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dummy_optimizer, mode='min', factor=0.5, patience=5)
        
        # Simulasikan validasi loss yang plateau setelah beberapa epoch
        for epoch in range(epochs):
            val_loss = max(1.0 - 0.05 * epoch, 0.5)
            if epoch > epochs // 3:
                val_loss = 0.5  # Plateau setelah 1/3 total epochs
            
            scheduler.step(val_loss)
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
    elif scheduler_type == 'step':
        step_size = epochs // 5  # 5 steps total
        gamma = 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=step_size, gamma=gamma)
        
        for epoch in range(epochs):
            scheduler.step()
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            scheduler.step()
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
    elif scheduler_type == 'onecycle':
        # OneCycleLR requires max_lr and total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            dummy_optimizer, 
            max_lr=lr*10, 
            total_steps=epochs,
            pct_start=0.3,  # 30% untuk warmup
            div_factor=25,   # initial_lr = max_lr/25
            final_div_factor=10000  # final_lr = initial_lr/final_div_factor
        )
        
        for epoch in range(epochs):
            scheduler.step()
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
    else:
        # Constant LR (no scheduler)
        lrs = [lr] * epochs
    
    return lrs

def on_show_lr_schedule_button_clicked(ui_components, logger):
    """
    Handler untuk tombol visualisasi learning rate schedule.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_config_ui()
        logger: Logger untuk mencatat aktivitas
    """
    with ui_components['lr_schedule_output']:
        clear_output()
        
        # Extract parameters
        epochs = ui_components['epochs_slider'].value
        lr = ui_components['lr_dropdown'].value
        scheduler_type = ui_components['scheduler_dropdown'].value
        
        logger.info(f"📈 Visualisasi learning rate schedule untuk {epochs} epochs...")
        logger.info(f"• Learning rate awal: {lr}")
        logger.info(f"• Tipe scheduler: {scheduler_type}")
        
        try:
            # Simulasikan learning rate schedule
            lrs = simulate_lr_schedule(epochs, lr, scheduler_type)
            
            # Plot learning rate schedule
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, epochs + 1), lrs, 'o-', linewidth=2, markersize=4)
            plt.title(f'Learning Rate Schedule ({scheduler_type})')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Tambahkan keterangan scheduler
            if scheduler_type == 'plateau':
                plateau_start = epochs // 3
                plt.axvline(x=plateau_start, color='r', linestyle='--', alpha=0.5)
                plt.annotate('Plateau terdeteksi', xy=(plateau_start, lrs[plateau_start-1]), 
                            xytext=(plateau_start+5, lrs[0]),
                            arrowprops=dict(facecolor='red', shrink=0.05), color='red')
            elif scheduler_type == 'step':
                step_size = epochs // 5
                for i in range(step_size, epochs, step_size):
                    if i < epochs:
                        plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
                        plt.text(i+0.5, lrs[i-1]/2, 'Step', rotation=90, color='r')
            elif scheduler_type == 'onecycle':
                warmup_end = int(epochs * 0.3)
                plt.axvline(x=warmup_end, color='g', linestyle='--', alpha=0.5)
                plt.text(warmup_end+0.5, lrs[warmup_end]/2, 'Warmup End', rotation=90, color='g')
            
            plt.tight_layout()
            plt.show()
            
            # Tambahkan tips berdasarkan konfigurasi
            print("\n💡 Tips:")
            if scheduler_type == 'plateau':
                print("• ReduceLROnPlateau akan mengurangi learning rate ketika validasi loss tidak membaik")
                print("• Tingkatkan 'Early Stopping Patience' untuk memberi kesempatan pada scheduler bekerja")
            elif scheduler_type == 'onecycle':
                print("• OneCycleLR memiliki fase warmup dan cooldown untuk mencapai konvergensi yang lebih baik")
                print("• Cocok untuk training dari awal dengan dataset besar")
            elif scheduler_type == 'cosine':
                print("• CosineAnnealing menurunkan learning rate dengan halus, baik untuk fine-tuning")
                print("• Optimalkan epoch agar cukup untuk pembelajaran model")
        except Exception as e:
            print(f"❌ Gagal memvisualisasikan: {str(e)}")
            import traceback
            traceback.print_exc()
            
def on_backbone_change(change, ui_components):
    """
    Handler untuk perubahan backbone.
    
    Args:
        change: Nilai perubahan dari observe
        ui_components: Dictionary berisi komponen UI dari create_training_config_ui()
    """
    if change['type'] == 'change' and change['name'] == 'value':
        # Update experiment name berdasarkan backbone
        current_name = ui_components['experiment_name_input'].value
        new_backbone = change['new']
        
        # Buat nama eksperimen baru dengan backbone yang diupdate
        new_name = f"{new_backbone}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ui_components['experiment_name_input'].value = new_name

def setup_training_config_handlers(ui_components, config, logger, components=None):
    """
    Setup event handlers untuk komponen UI konfigurasi training.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_config_ui()
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        components: Dictionary berisi komponen lain yang diperlukan (optional)
        
    Returns:
        Dictionary berisi komponen yang diupdate
    """
    if components is None:
        components = {}
    
    # Simpan batch size awal untuk tracking perubahan
    components['original_batch_size'] = config.get('training', {}).get('batch_size', 16)
    
    # Bind event handlers
    ui_components['generate_name_button'].on_click(
        lambda b: on_generate_name_button_clicked(ui_components)
    )
    
    ui_components['save_config_button'].on_click(
        lambda b: on_save_config_button_clicked(ui_components, config, logger, components)
    )
    
    ui_components['show_lr_schedule_button'].on_click(
        lambda b: on_show_lr_schedule_button_clicked(ui_components, logger)
    )
    
    # Observe perubahan backbone
    ui_components['backbone_dropdown'].observe(
        lambda change: on_backbone_change(change, ui_components), 
        names='value'
    )
    
    return components