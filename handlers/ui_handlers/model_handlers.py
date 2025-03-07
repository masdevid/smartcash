"""
File: smartcash/handlers/ui_handlers/model_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk interaksi UI komponen model, menangani logika untuk inisialisasi, visualisasi, manajemen checkpoint, dan optimasi model.
"""

import os
import time
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output, HTML
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable

@contextmanager
def memory_manager():
    """Context manager untuk mengoptimalkan penggunaan memori."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def setup_gpu():
    """Setup dan dapatkan informasi GPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_info = {'device': device}
    
    if torch.cuda.is_available():
        gpu_info.update({
            'name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0) / (1024**2),
            'memory_reserved': torch.cuda.memory_reserved(0) / (1024**2),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**2)
        })
        
        # Optimize CUDA settings
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
    return gpu_info

def on_initialize_model_clicked(ui_components, model_handler, config, logger):
    """
    Handler untuk tombol inisialisasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_initialization_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    required_components = ['initialize_button', 'output_area', 'backbone_dropdown', 
                         'detection_mode_dropdown', 'img_size_slider', 'pretrained_checkbox']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    ui_components['initialize_button'].disabled = True
    ui_components['initialize_button'].description = "Menginisialisasi..."
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Update konfigurasi berdasarkan pilihan UI
            config['model']['backbone'] = ui_components['backbone_dropdown'].value
            config['model']['pretrained'] = ui_components['pretrained_checkbox'].value
            config['model']['img_size'] = [ui_components['img_size_slider'].value, ui_components['img_size_slider'].value]
            
            # Update layer konfigurasi
            if ui_components['detection_mode_dropdown'].value == 'multi':
                config['layers'] = ['banknote', 'nominal', 'security']
            else:
                config['layers'] = ['banknote']
            
            # Setup environment
            logger.info("🔄 Mempersiapkan lingkungan...")
            gpu_info = setup_gpu()
            
            # Buat model
            with memory_manager():
                logger.info("🧠 Membuat model...")
                start_time = time.time()
                model = model_handler.create_model(backbone_type=config['model']['backbone'])
                creation_time = time.time() - start_time
                
                # Pindahkan model ke device
                model = model.to(gpu_info['device'])
                
                # Hitung parameter
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Log informasi model
                logger.success(f"✅ Model berhasil dibuat dalam {creation_time:.2f} detik!")
                logger.info(f"📊 Total Parameter: {total_params:,}")
                logger.info(f"📊 Parameter Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
                
                # Log GPU info jika tersedia
                if torch.cuda.is_available():
                    logger.info(f"🖥️ Model berada di {gpu_info['device']} ({gpu_info['name']})")
                    logger.info(f"🖥️ VRAM Terpakai: {gpu_info['memory_allocated']:.1f} MB")
                    logger.info(f"🖥️ VRAM Total: {gpu_info['memory_total']:.1f} MB")
                else:
                    logger.info(f"🖥️ Model berada di CPU (GPU tidak tersedia)")
                
                # Tampilkan ringkasan konfigurasi
                print("📋 Ringkasan Konfigurasi Model:")
                print(f"• Backbone: {config['model']['backbone']}")
                print(f"• Mode Deteksi: {'Multi-layer' if len(config['layers']) > 1 else 'Single-layer'}")
                print(f"• Layers: {', '.join(config['layers'])}")
                print(f"• Ukuran Gambar: {config['model']['img_size'][0]}x{config['model']['img_size'][1]}")
                print(f"• Pretrained: {'Ya' if config['model']['pretrained'] else 'Tidak'}")
                
                # Kosongkan model setelah inisialisasi
                del model
                
        except Exception as e:
            logger.error(f"❌ Error saat inisialisasi model: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Bersihkan memori
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Aktifkan kembali tombol
            ui_components['initialize_button'].disabled = False
            ui_components['initialize_button'].description = "Inisialisasi Model"

def on_visualize_model_clicked(ui_components, model_handler, config, logger):
    """
    Handler untuk tombol visualisasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_visualizer_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    required_components = ['create_model_button', 'visualization_output', 'backbone_select',
                         'mode_select', 'viz_module_select']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    ui_components['create_model_button'].disabled = True
    ui_components['create_model_button'].description = "Memproses..."
    
    with ui_components['visualization_output']:
        clear_output()
        
        try:
            # Update konfigurasi berdasarkan pilihan
            config['model']['backbone'] = ui_components['backbone_select'].value
            
            if ui_components['mode_select'].value == 'multi':
                config['layers'] = ['banknote', 'nominal', 'security']
            else:
                config['layers'] = ['banknote']
            
            # Gunakan ModelHandler untuk membuat model
            model = model_handler.create_model(backbone_type=ui_components['backbone_select'].value)
            
            # Initialize layer config if needed
            try:
                from smartcash.utils.layer_config_manager import get_layer_config
                layer_config = get_layer_config()
            except ImportError:
                logger.warning("⚠️ Import layer_config_manager tidak tersedia")
            
            # Visualisasikan model
            try:
                from smartcash.utils.model_visualizer import ModelVisualizer
                
                visualizer = ModelVisualizer(
                    model=model, 
                    logger=logger, 
                    output_dir=config.get('output_dir', 'runs/visualization')
                )
                
                # Tampilkan sesuai opsi visualisasi yang dipilih
                if ui_components['viz_module_select'].value == 'full' or ui_components['viz_module_select'].value == 'parameters':
                    # Tampilkan informasi parameter
                    visualizer.count_parameters()
                    
                    # Tampilkan diagram parameter per modul jika dipilih parameters
                    if ui_components['viz_module_select'].value == 'parameters':
                        module_params = visualizer.get_module_parameters()
                        
                        # Plot diagram batang parameter per modul
                        plt.figure(figsize=(10, 6))
                        modules = list(module_params.keys())
                        values = [module_params[m] / 1e6 for m in modules]  # Convert to millions
                        
                        # Sort by parameter count
                        sorted_indices = np.argsort(values)[::-1]
                        modules = [modules[i] for i in sorted_indices]
                        values = [values[i] for i in sorted_indices]
                        
                        plt.barh(modules, values, color='#3498db')
                        plt.xlabel('Jumlah Parameter (juta)')
                        plt.title('Distribusi Parameter dalam Model')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.show()
                
                if ui_components['viz_module_select'].value == 'full' or ui_components['viz_module_select'].value == 'backbone':
                    # Visualisasikan backbone
                    visualizer.visualize_backbone()
                    
                    # Visualisasikan output layer jika tersedia
                    if ui_components['viz_module_select'].value == 'full':
                        print("\n🔍 Visualisasi Output Layer:")
                        visualizer.visualize_layer_outputs()
                
                # Coba tampilkan summary jika torchsummary tersedia
                if ui_components['viz_module_select'].value == 'full':
                    try:
                        import torchsummary
                        print("\n🔍 Detail struktur model:")
                        visualizer.print_model_summary()
                    except ImportError:
                        print("\n⚠️ Untuk melihat detail struktur model, install torchsummary:")
                        print("!pip install torchsummary")
                
            except ImportError:
                logger.warning("⚠️ ModelVisualizer tidak tersedia, menggunakan visualisasi sederhana")
                
                # Visualisasi sederhana
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"📊 Total Parameter: {total_params:,}")
                print(f"📊 Parameter Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
                print(f"🔍 Backbone: {ui_components['backbone_select'].value}")
                print(f"🔍 Mode: {ui_components['mode_select'].value}")
            
            # Bersihkan model
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ Gagal membuat atau memvisualisasikan model: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Reset button state
            ui_components['create_model_button'].disabled = False
            ui_components['create_model_button'].description = "Buat Model & Visualisasikan"

def on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger):
    """
    Handler untuk tombol melihat checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'checkpoints_output' not in ui_components:
        logger.error("❌ Required UI component 'checkpoints_output' not found")
        return
    
    with ui_components['checkpoints_output']:
        clear_output()
        
        try:
            checkpoint_handler.display_checkpoints()
            
            # Tambahkan visualisasi
            checkpoints = checkpoint_handler.list_checkpoints()
            
            # Count checkpoints by type
            counts = {
                'Best': len(checkpoints.get('best', [])),
                'Latest': len(checkpoints.get('latest', [])),
                'Epoch': len(checkpoints.get('epoch', [])),
                'Emergency': len(checkpoints.get('emergency', []))
            }
            
            # Plot chart if there are any checkpoints
            if sum(counts.values()) > 0:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                bars = plt.bar(
                    counts.keys(),
                    counts.values(),
                    color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                )
                
                # Add values above bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.05,
                            f'{int(height)}',
                            ha='center', 
                            va='bottom'
                        )
                        
                plt.title('Jumlah Checkpoint per Tipe')
                plt.ylabel('Jumlah File')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Create second subplot - size on disk
                plt.subplot(1, 2, 2)
                
                # Calculate size of checkpoints by type
                sizes = {}
                for checkpoint_type, checkpoint_list in checkpoints.items():
                    total_size = sum([ckpt.stat().st_size for ckpt in checkpoint_list]) / (1024 * 1024) # MB
                    sizes[checkpoint_type] = total_size
                    
                if sum(sizes.values()) > 0:
                    plt.pie(
                        list(sizes.values()),
                        labels=list(sizes.keys()),
                        autopct='%1.1f%%',
                        colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                    )
                    plt.title(f'Ukuran Disk ({sum(sizes.values()):.1f} MB)')
                    plt.axis('equal')
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error saat melihat checkpoint: {str(e)}")

def on_cleanup_checkpoints_clicked(ui_components, checkpoint_handler, logger):
    """
    Handler untuk tombol membersihkan checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'checkpoints_output' not in ui_components:
        logger.error("❌ Required UI component 'checkpoints_output' not found")
        return
    
    with ui_components['checkpoints_output']:
        clear_output()
        
        try:
            # Tampilkan konfirmasi
            logger.warning("⚠️ Ini akan menghapus checkpoint yang tidak diperlukan. Hanya checkpoint 'best' dan 'latest' yang dipertahankan.")
            
            # Buat tombol konfirmasi
            confirm_button = widgets.Button(
                description='Konfirmasi Pembersihan',
                button_style='danger',
                icon='trash'
            )
            
            cancel_button = widgets.Button(
                description='Batal',
                button_style='secondary',
                icon='times'
            )
            
            def on_confirm(b):
                clear_output()
                
                try:
                    # Panggil metode clean_checkpoints
                    results = checkpoint_handler.clean_checkpoints(keep_best=True, keep_latest=True)
                    
                    if results and 'removed' in results:
                        logger.success(f"✅ Berhasil membersihkan {results['removed']} checkpoint")
                        logger.info(f"📁 Ukuran yang dibebaskan: {results.get('freed_space_mb', 0):.2f} MB")
                    else:
                        logger.info("ℹ️ Tidak ada checkpoint yang perlu dibersihkan")
                        
                except Exception as e:
                    logger.error(f"❌ Error saat membersihkan checkpoint: {str(e)}")
                    
                # Tampilkan kembali checkpoint yang tersisa
                on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger)
            
            def on_cancel(b):
                clear_output()
                logger.info("ℹ️ Pembersihan checkpoint dibatalkan")
                
                # Tampilkan kembali list checkpoint
                on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger)
            
            confirm_button.on_click(on_confirm)
            cancel_button.on_click(on_cancel)
            
            display(widgets.HBox([confirm_button, cancel_button]))
            
        except Exception as e:
            logger.error(f"❌ Error saat mempersiapkan pembersihan checkpoint: {str(e)}")

def on_compare_checkpoints_clicked(ui_components, checkpoint_handler, model_handler, logger):
    """
    Handler untuk tombol membandingkan checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        model_handler: Instance dari ModelHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'checkpoints_output' not in ui_components:
        logger.error("❌ Required UI component 'checkpoints_output' not found")
        return
    
    with ui_components['checkpoints_output']:
        clear_output()
        
        try:
            # Dapatkan checkpoint yang tersedia
            checkpoints = checkpoint_handler.list_checkpoints()
            
            # Gabungkan semua checkpoint
            all_checkpoints = []
            for checkpoint_type, checkpoint_list in checkpoints.items():
                if checkpoint_list:
                    all_checkpoints.extend(checkpoint_list)
            
            if not all_checkpoints:
                logger.warning("⚠️ Tidak ada checkpoint yang tersedia untuk dibandingkan")
                return
            
            # Ambil max 5 checkpoint terbaru
            all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            recent_checkpoints = all_checkpoints[:5]
            
            # Buat dropdown untuk pemilihan checkpoint
            checkpoint_options = [(ckpt.name, str(ckpt)) for ckpt in recent_checkpoints]
            
            checkpoint1_dropdown = widgets.Dropdown(
                options=checkpoint_options,
                description='Checkpoint 1:',
                style={'description_width': 'initial'}
            )
            
            checkpoint2_dropdown = widgets.Dropdown(
                options=checkpoint_options,
                description='Checkpoint 2:',
                style={'description_width': 'initial'}
            )
            
            # Set default ke nilai berbeda jika ada lebih dari 1 checkpoint
            if len(checkpoint_options) > 1:
                checkpoint2_dropdown.value = checkpoint_options[1][1]
            
            compare_button = widgets.Button(
                description='Bandingkan',
                button_style='primary',
                icon='search'
            )
            
            compare_output = widgets.Output()
            
            def on_compare(b):
                with compare_output:
                    clear_output()
                    
                    try:
                        # Get selected checkpoints
                        ckpt1_path = checkpoint1_dropdown.value
                        ckpt2_path = checkpoint2_dropdown.value
                        
                        if ckpt1_path == ckpt2_path:
                            logger.warning("⚠️ Pilih dua checkpoint yang berbeda untuk dibandingkan")
                            return
                        
                        logger.info(f"🔄 Membandingkan checkpoint:")
                        logger.info(f"• Checkpoint 1: {os.path.basename(ckpt1_path)}")
                        logger.info(f"• Checkpoint 2: {os.path.basename(ckpt2_path)}")
                        
                        # Load checkpoints dan ekstrak informasi
                        ckpt1_info = checkpoint_handler.get_checkpoint_info(ckpt1_path)
                        ckpt2_info = checkpoint_handler.get_checkpoint_info(ckpt2_path)
                        
                        # Tampilkan perbandingan dalam bentuk tabel
                        comparison_data = []
                        
                        # Epoch
                        ckpt1_epoch = ckpt1_info.get('epoch', 'N/A')
                        ckpt2_epoch = ckpt2_info.get('epoch', 'N/A')
                        comparison_data.append({
                            'Metrik': 'Epoch',
                            'Checkpoint 1': ckpt1_epoch,
                            'Checkpoint 2': ckpt2_epoch,
                            'Selisih': f"{ckpt2_epoch - ckpt1_epoch}" if isinstance(ckpt1_epoch, (int, float)) and isinstance(ckpt2_epoch, (int, float)) else "N/A"
                        })
                        
                        # Loss
                        ckpt1_loss = ckpt1_info.get('best_val_loss', 'N/A')
                        ckpt2_loss = ckpt2_info.get('best_val_loss', 'N/A')
                        loss_diff = ckpt2_loss - ckpt1_loss if isinstance(ckpt1_loss, (int, float)) and isinstance(ckpt2_loss, (int, float)) else "N/A"
                        comparison_data.append({
                            'Metrik': 'Validation Loss',
                            'Checkpoint 1': f"{ckpt1_loss:.6f}" if isinstance(ckpt1_loss, (int, float)) else ckpt1_loss,
                            'Checkpoint 2': f"{ckpt2_loss:.6f}" if isinstance(ckpt2_loss, (int, float)) else ckpt2_loss,
                            'Selisih': f"{loss_diff:.6f}" if isinstance(loss_diff, (int, float)) else loss_diff
                        })
                        
                        # Ukuran file
                        ckpt1_size = os.path.getsize(ckpt1_path) / (1024 * 1024)
                        ckpt2_size = os.path.getsize(ckpt2_path) / (1024 * 1024)
                        size_diff = ckpt2_size - ckpt1_size
                        comparison_data.append({
                            'Metrik': 'Ukuran File (MB)',
                            'Checkpoint 1': f"{ckpt1_size:.2f}",
                            'Checkpoint 2': f"{ckpt2_size:.2f}",
                            'Selisih': f"{size_diff:+.2f}"
                        })
                        
                        # Waktu modifikasi
                        ckpt1_mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ckpt1_path)))
                        ckpt2_mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ckpt2_path)))
                        comparison_data.append({
                            'Metrik': 'Waktu Modifikasi',
                            'Checkpoint 1': ckpt1_mtime,
                            'Checkpoint 2': ckpt2_mtime,
                            'Selisih': "N/A"
                        })
                        
                        # Konversi ke DataFrame dan tampilkan
                        comparison_df = pd.DataFrame(comparison_data)
                        display(comparison_df)
                        
                        # Plot validation loss
                        plt.figure(figsize=(10, 5))
                        bars = plt.bar([os.path.basename(ckpt1_path), os.path.basename(ckpt2_path)], 
                                    [ckpt1_loss, ckpt2_loss] if isinstance(ckpt1_loss, (int, float)) and isinstance(ckpt2_loss, (int, float)) else [0, 0])
                        plt.title('Perbandingan Validation Loss')
                        plt.ylabel('Loss')
                        
                        # Tambahkan nilai di atas bar
                        for i, bar in enumerate(bars):
                            loss_val = [ckpt1_loss, ckpt2_loss][i]
                            if isinstance(loss_val, (int, float)):
                                plt.text(
                                    bar.get_x() + bar.get_width()/2.,
                                    bar.get_height() + 0.001,
                                    f'{loss_val:.6f}',
                                    ha='center', 
                                    va='bottom',
                                    fontweight='bold'
                                )
                        
                        plt.tight_layout()
                        plt.show()
                        
                    except Exception as e:
                        logger.error(f"❌ Error saat membandingkan checkpoint: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            compare_button.on_click(on_compare)
            
            # Tampilkan UI
            display(widgets.VBox([
                widgets.HBox([checkpoint1_dropdown, checkpoint2_dropdown]),
                compare_button,
                compare_output
            ]))
            
        except Exception as e:
            logger.error(f"❌ Error saat mempersiapkan perbandingan checkpoint: {str(e)}")

def on_mount_drive_clicked(ui_components, logger):
    """
    Handler untuk tombol mount Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'mount_drive_button' not in ui_components:
        logger.error("❌ Required UI component 'mount_drive_button' not found")
        return
    
    ui_components['mount_drive_button'].disabled = True
    
    try:
        from google.colab import drive
        logger.info("🔄 Melakukan mount Google Drive...")
        drive.mount('/content/drive')
        logger.success("✅ Google Drive berhasil di-mount")
        
        # Update tampilan tombol
        ui_components['mount_drive_button'].description = 'Google Drive Terpasang'
        ui_components['mount_drive_button'].button_style = ''
        ui_components['mount_drive_button'].icon = 'check'
    except ImportError:
        logger.warning("⚠️ Tidak dapat melakukan mount Google Drive (bukan di Colab)")
        ui_components['mount_drive_button'].description = 'Tidak di Colab'
        ui_components['mount_drive_button'].button_style = 'warning'
    except Exception as e:
        logger.error(f"❌ Error saat mount Google Drive: {str(e)}")
        ui_components['mount_drive_button'].disabled = False
        ui_components['mount_drive_button'].description = 'Mount Google Drive (Gagal)'
        ui_components['mount_drive_button'].button_style = 'danger'

def on_check_memory_clicked(ui_components, memory_optimizer, logger):
    """
    Handler untuk tombol cek status memori.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        memory_optimizer: Instance dari MemoryOptimizer
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'memory_output' not in ui_components:
        logger.error("❌ Required UI component 'memory_output' not found")
        return
    
    with ui_components['memory_output']:
        clear_output()
        
        try:
            # Periksa status memori
            memory_optimizer.check_gpu_status()
            
            # Tambahkan informasi CPU dan RAM
            try:
                import psutil
                cpu_usage = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                
                print(f"\n🖥️ CPU Usage: {cpu_usage}%")
                print(f"💾 RAM: {ram.used / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB ({ram.percent}%)")
            except ImportError:
                print("ℹ️ psutil tidak tersedia untuk monitor CPU/RAM")
            
            # Visualisasi jika GPU tersedia
            stats = memory_optimizer.get_optimization_stats()
            
            plt.figure(figsize=(15, 5))
            
            # Plot GPU Memory (if available)
            if 'gpu_available' in stats and stats['gpu_available'] and 'total_memory' in stats:
                plt.subplot(1, 3, 1)
                
                # Data untuk pie chart GPU
                gpu_sizes = [stats['used_memory'], stats['free_memory']]
                gpu_labels = [f'Terpakai\n{gpu_sizes[0]:.1f} MB', f'Bebas\n{gpu_sizes[1]:.1f} MB']
                plt.pie(gpu_sizes, labels=gpu_labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                plt.title(f'GPU Memory ({stats["gpu_name"]})')
            
            # Plot RAM Usage if psutil available
            try:
                import psutil
                ram = psutil.virtual_memory()
                
                plt.subplot(1, 3, 2)
                ram_sizes = [ram.used / (1024**3), (ram.total - ram.used) / (1024**3)]
                ram_labels = [f'Terpakai\n{ram_sizes[0]:.1f} GB', f'Bebas\n{ram_sizes[1]:.1f} GB']
                plt.pie(ram_sizes, labels=ram_labels, autopct='%1.1f%%', colors=['#99ff99','#ffcc99'])
                plt.title('RAM')
            except ImportError:
                pass
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logger.error(f"❌ Error saat memeriksa status memori: {str(e)}")

def on_clear_memory_clicked(ui_components, memory_optimizer, logger):
    """
    Handler untuk tombol membersihkan memori GPU.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        memory_optimizer: Instance dari MemoryOptimizer
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'memory_output' not in ui_components:
        logger.error("❌ Required UI component 'memory_output' not found")
        return
    
    with ui_components['memory_output']:
        clear_output()
        
        try:
            # Bersihkan memori
            before_stats = memory_optimizer.get_optimization_stats()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            after_stats = memory_optimizer.get_optimization_stats()
            
            # Tampilkan hasil pembersihan
            if 'gpu_available' in before_stats and before_stats['gpu_available']:
                freed_memory = before_stats['used_memory'] - after_stats['used_memory']
                logger.success(f"✅ Memori berhasil dibersihkan")
                logger.info(f"📊 Memory sebelum: {before_stats['used_memory']:.1f} MB")
                logger.info(f"📊 Memory sesudah: {after_stats['used_memory']:.1f} MB")
                logger.info(f"📊 Memory dibebaskan: {freed_memory:.1f} MB")
                
                # Plot perbandingan
                plt.figure(figsize=(10, 5))
                bars = plt.bar(['Sebelum', 'Sesudah'], 
                           [before_stats['used_memory'], after_stats['used_memory']])
                
                # Tambahkan label di atas bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{height:.1f} MB',
                        ha='center', 
                        va='bottom'
                    )
                
                plt.title('Penggunaan Memori GPU')
                plt.ylabel('Memori Terpakai (MB)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            else:
                logger.info("ℹ️ Memori berhasil dibersihkan")
                logger.info("ℹ️ GPU tidak tersedia, membersihkan memori CPU")
        
        except Exception as e:
            logger.error(f"❌ Error saat membersihkan memori: {str(e)}")

def on_optimize_batch_size_clicked(ui_components, memory_optimizer, logger):
    """
    Handler untuk tombol optimasi batch size.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        memory_optimizer: Instance dari MemoryOptimizer
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'memory_output' not in ui_components:
        logger.error("❌ Required UI component 'memory_output' not found")
        return
    
    with ui_components['memory_output']:
        clear_output()
        
        try:
            # Jalankan optimasi batch size
            logger.info("🔄 Menjalankan optimasi batch size...")
            
            if hasattr(memory_optimizer, 'find_optimal_batch_size'):
                results = memory_optimizer.find_optimal_batch_size()
                
                if results:
                    logger.success(f"✅ Optimasi batch size selesai")
                    logger.info(f"📊 Batch size optimal: {results.get('optimal_batch_size', 'N/A')}")
                    logger.info(f"📊 Memory usage: {results.get('memory_usage_mb', 0):.1f} MB")
                    logger.info(f"📊 Safety margin: {results.get('safety_margin', 0)*100:.0f}%")
                    
                    # Rekomendasi berdasarkan hasil
                    if 'safe_recommendations' in results:
                        print("\n💡 Rekomendasi Batch Size:")
                        for model_type, batch_size in results['safe_recommendations'].items():
                            print(f"• {model_type}: {batch_size}")
                else:
                    logger.warning("⚠️ Tidak dapat menentukan batch size optimal")
            else:
                # Fallback jika metode tidak tersedia
                batch_sizes = [8, 16, 32, 64]
                memory_usage = []
                
                if torch.cuda.is_available():
                    # Cek memori awal
                    before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    available_mem = total_mem - before_mem
                    
                    # Estimasi batch size berdasarkan memori tersedia
                    recommended_bs = int(((available_mem * 0.7) // 512) * 8)  # Heuristic formula
                    recommended_bs = max(8, recommended_bs)  # Minimal 8
                    recommended_bs = (recommended_bs // 8) * 8  # Round to nearest multiple of 8
                    
                    logger.info(f"📊 Memori GPU tersedia: {available_mem:.1f} MB")
                    logger.info(f"📊 Perkiraan batch size optimal: {recommended_bs}")
                    logger.info("💡 Ini hanya estimasi. Batch size yang lebih kecil biasanya lebih aman.")
                else:
                    logger.warning("⚠️ GPU tidak tersedia, menggunakan estimasi default")
                    logger.info("📊 Rekomendasi batch size untuk CPU: 8-16")
        
        except Exception as e:
            logger.error(f"❌ Error saat optimasi batch size: {str(e)}")

def on_clear_cache_clicked(ui_components, cache, logger):
    """
    Handler untuk tombol clear cache.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        cache: Instance dari EnhancedCache
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'memory_output' not in ui_components:
        logger.error("❌ Required UI component 'memory_output' not found")
        return
    
    with ui_components['memory_output']:
        clear_output()
        
        try:
            # Periksa apakah cache tersedia
            if not cache:
                logger.warning("⚠️ Cache tidak tersedia")
                return
            
            # Get cache stats before cleaning
            before_stats = cache.get_stats()
            
            # Bersihkan cache
            if hasattr(cache, 'clear'):
                cache.clear()
                logger.success("✅ Cache berhasil dibersihkan")
            else:
                logger.warning("⚠️ Metode clear tidak tersedia pada cache")
                return
            
            # Get cache stats after cleaning
            after_stats = cache.get_stats()
            
            # Tampilkan hasil pembersihan
            logger.info(f"📊 Cache sebelum: {before_stats['cache_size_mb']:.1f} MB ({before_stats['file_count']} file)")
            logger.info(f"📊 Cache sesudah: {after_stats['cache_size_mb']:.1f} MB ({after_stats['file_count']} file)")
            logger.info(f"📊 Ruang dibebaskan: {before_stats['cache_size_mb'] - after_stats['cache_size_mb']:.1f} MB")
            
            # Visualisasi hasil
            if before_stats['cache_size_mb'] > 0:
                plt.figure(figsize=(10, 5))
                plt.bar(['Sebelum', 'Sesudah'], 
                      [before_stats['cache_size_mb'], after_stats['cache_size_mb']])
                plt.title('Ukuran Cache')
                plt.ylabel('Ukuran (MB)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            
        except Exception as e:
            logger.error(f"❌ Error saat membersihkan cache: {str(e)}")

def on_verify_cache_clicked(ui_components, cache, logger):
    """
    Handler untuk tombol verify cache.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        cache: Instance dari EnhancedCache
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'memory_output' not in ui_components:
        logger.error("❌ Required UI component 'memory_output' not found")
        return
    
    with ui_components['memory_output']:
        clear_output()
        
        try:
            # Periksa apakah cache tersedia
            if not cache:
                logger.warning("⚠️ Cache tidak tersedia")
                return
            
            # Verifikasi cache
            if hasattr(cache, 'verify'):
                results = cache.verify()
                
                if results:
                    logger.success("✅ Verifikasi cache selesai")
                    logger.info(f"📊 File terverifikasi: {results.get('verified_files', 0)}")
                    logger.info(f"📊 File rusak: {results.get('corrupt_files', 0)}")
                    logger.info(f"📊 File terhapus: {results.get('missing_files', 0)}")
                    
                    # Jika ada file yang rusak atau hilang, beri opsi untuk fix
                    if results.get('corrupt_files', 0) > 0 or results.get('missing_files', 0) > 0:
                        fix_button = widgets.Button(
                            description='Perbaiki Cache',
                            button_style='warning',
                            icon='wrench'
                        )
                        
                        def on_fix(b):
                            with ui_components['memory_output']:
                                clear_output()
                                
                                if hasattr(cache, 'repair'):
                                    repair_results = cache.repair()
                                    
                                    if repair_results:
                                        logger.success("✅ Perbaikan cache selesai")
                                        logger.info(f"📊 File diperbaiki: {repair_results.get('fixed_files', 0)}")
                                        logger.info(f"📊 File dibersihkan: {repair_results.get('cleaned_files', 0)}")
                                    else:
                                        logger.warning("⚠️ Tidak ada perbaikan yang dilakukan")
                                else:
                                    logger.warning("⚠️ Metode repair tidak tersedia pada cache")
                        
                        fix_button.on_click(on_fix)
                        display(fix_button)
                else:
                    logger.warning("⚠️ Verifikasi cache gagal")
            else:
                # Fallback jika verify tidak tersedia
                stats = cache.get_stats()
                logger.info("ℹ️ Informasi Cache:")
                logger.info(f"📊 Ukuran cache: {stats['cache_size_mb']:.1f} MB")
                logger.info(f"📊 Jumlah file: {stats['file_count']}")
                logger.info(f"📊 Batas maksimum: {stats['max_size_mb']:.1f} MB")
                
                # Visualisasi penggunaan cache
                plt.figure(figsize=(10, 5))
                plt.pie(
                    [stats['cache_size_mb'], stats['max_size_mb'] - stats['cache_size_mb']],
                    labels=['Terpakai', 'Tersisa'],
                    autopct='%1.1f%%',
                    colors=['#3498db', '#e5e5e5']
                )
                plt.title('Penggunaan Cache')
                plt.axis('equal')
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            logger.error(f"❌ Error saat verifikasi cache: {str(e)}")

def on_export_format_change(change, ui_components):
    """
    Handler untuk perubahan format ekspor.
    
    Args:
        change: Change event dari dropdown
        ui_components: Dictionary komponen UI dari create_model_exporter_ui()
    """
    # Validate UI components
    required_components = ['onnx_opset_selector', 'optimize_checkbox']
    for component in required_components:
        if component not in ui_components:
            return
    
    if change['new'] == 'onnx':
        ui_components['onnx_opset_selector'].disabled = False
        ui_components['optimize_checkbox'].disabled = True
    else:
        ui_components['onnx_opset_selector'].disabled = True
        ui_components['optimize_checkbox'].disabled = False

def on_export_button_clicked(ui_components, model_exporter, logger):
    """
    Handler untuk tombol ekspor model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_exporter_ui()
        model_exporter: Instance dari ModelExporter
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    required_components = ['export_button', 'export_output', 'export_format_selector',
                         'optimize_checkbox', 'copy_to_drive_checkbox']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    # Update tombol
    ui_components['export_button'].description = "Memproses..."
    ui_components['export_button'].disabled = True
    
    with ui_components['export_output']:
        clear_output()
        
        try:
            # Jalankan ekspor berdasarkan format yang dipilih
            export_format = ui_components['export_format_selector'].value
            
            if export_format == 'torchscript':
                export_path = model_exporter.export_to_torchscript(
                    optimize=ui_components['optimize_checkbox'].value
                )
            elif export_format == 'onnx':
                # Check if onnx_opset_selector exists and is enabled
                if 'onnx_opset_selector' in ui_components and not ui_components['onnx_opset_selector'].disabled:
                    opset_version = ui_components['onnx_opset_selector'].value
                else:
                    opset_version = 12  # Default
                
                export_path = model_exporter.export_to_onnx(
                    opset_version=opset_version
                )
            else:
                print(f"❌ Format ekspor tidak valid: {export_format}")
                return
            
            # Cek hasil ekspor
            if export_path:
                export_file = Path(export_path)
                file_size = export_file.stat().st_size / (1024 * 1024)  # size in MB
                
                print(f"\n✅ Model berhasil diekspor:")
                print(f"📁 Path: {export_path}")
                print(f"📊 Ukuran: {file_size:.2f} MB")
                print(f"🔄 Format: {export_format}")
                print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Salin ke Drive jika diminta
                if ui_components['copy_to_drive_checkbox'].value and os.path.exists("/content/drive"):
                    drive_path = model_exporter.copy_to_drive(export_path)
                    if drive_path:
                        print(f"\n✅ Model berhasil disalin ke Google Drive:")
                        print(f"📁 Path: {drive_path}")
            else:
                print("❌ Ekspor model gagal")
                
        except Exception as e:
            print(f"❌ Error saat mengekspor model: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Reset tombol
            ui_components['export_button'].description = "Ekspor Model"
            ui_components['export_button'].disabled = False

def setup_model_initialization_handlers(ui_components, model_handler, config, logger):
    """
    Setup handler untuk komponen UI inisialisasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_initialization_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Validate required components
    required_components = ['initialize_button']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Bind event handler untuk tombol inisialisasi
    ui_components['initialize_button'].on_click(
        lambda b: on_initialize_model_clicked(ui_components, model_handler, config, logger)
    )
    
    return ui_components

def setup_model_visualizer_handlers(ui_components, model_handler, config, logger):
    """
    Setup handler untuk komponen UI visualisasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_visualizer_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Validate required components
    required_components = ['create_model_button']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Bind event handler untuk tombol visualisasi
    ui_components['create_model_button'].on_click(
        lambda b: on_visualize_model_clicked(ui_components, model_handler, config, logger)
    )
    
    return ui_components

def setup_checkpoint_manager_handlers(ui_components, checkpoint_handler, model_handler, logger):
    """
    Setup handler untuk komponen UI manajemen checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        model_handler: Instance dari ModelHandler
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Validate required components
    required_components = ['list_checkpoints_button', 'cleanup_checkpoints_button', 'compare_button', 'mount_drive_button']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Bind event handler untuk tombol daftar checkpoint
    ui_components['list_checkpoints_button'].on_click(
        lambda b: on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger)
    )
    
    # Bind event handler untuk tombol pembersihan checkpoint
    ui_components['cleanup_checkpoints_button'].on_click(
        lambda b: on_cleanup_checkpoints_clicked(ui_components, checkpoint_handler, logger)
    )
    
    # Bind event handler untuk tombol perbandingan checkpoint
    ui_components['compare_button'].on_click(
        lambda b: on_compare_checkpoints_clicked(ui_components, checkpoint_handler, model_handler, logger)
    )
    
    # Bind event handler untuk tombol mount Google Drive
    ui_components['mount_drive_button'].on_click(
        lambda b: on_mount_drive_clicked(ui_components, logger)
    )
    
    # Periksa apakah berjalan di Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
        
    # Disable tombol mount jika tidak di Colab
    if not is_colab:
        ui_components['mount_drive_button'].disabled = True
        ui_components['mount_drive_button'].description = 'Tidak di Colab'
        ui_components['mount_drive_button'].button_style = 'warning'
    
    return ui_components

def setup_model_optimization_handlers(ui_components, memory_optimizer, cache, logger):
    """
    Setup handler untuk komponen UI optimasi model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_optimization_ui()
        memory_optimizer: Instance dari MemoryOptimizer
        cache: Instance dari EnhancedCache
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Validate required components
    required_components = ['check_memory_button', 'clear_memory_button', 'optimize_button',
                         'clear_cache_button', 'verify_cache_button']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Bind event handler untuk tombol cek memori
    ui_components['check_memory_button'].on_click(
        lambda b: on_check_memory_clicked(ui_components, memory_optimizer, logger)
    )
    
    # Bind event handler untuk tombol bersihkan memori
    ui_components['clear_memory_button'].on_click(
        lambda b: on_clear_memory_clicked(ui_components, memory_optimizer, logger)
    )
    
    # Bind event handler untuk tombol optimasi batch size
    ui_components['optimize_button'].on_click(
        lambda b: on_optimize_batch_size_clicked(ui_components, memory_optimizer, logger)
    )
    
    # Bind event handler untuk tombol bersihkan cache
    ui_components['clear_cache_button'].on_click(
        lambda b: on_clear_cache_clicked(ui_components, cache, logger)
    )
    
    # Bind event handler untuk tombol verifikasi cache
    ui_components['verify_cache_button'].on_click(
        lambda b: on_verify_cache_clicked(ui_components, cache, logger)
    )
    
    return ui_components

def setup_model_exporter_handlers(ui_components, model_exporter, logger):
    """
    Setup handler untuk komponen UI ekspor model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_exporter_ui()
        model_exporter: Instance dari ModelExporter
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi updated UI components dengan handler yang telah ditambahkan
    """
    # Validate required components
    required_components = ['export_format_selector', 'export_button']
    
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return ui_components
    
    # Bind event handler untuk perubahan format
    ui_components['export_format_selector'].observe(
        lambda change: on_export_format_change(change, ui_components), 
        names='value'
    )
    
    # Bind event handler untuk tombol ekspor
    ui_components['export_button'].on_click(
        lambda b: on_export_button_clicked(ui_components, model_exporter, logger)
    )
    
    # Set status awal copy_to_drive_checkbox berdasarkan ketersediaan Google Drive
    if 'copy_to_drive_checkbox' in ui_components:
        ui_components['copy_to_drive_checkbox'].disabled = not os.path.exists("/content/drive")
    
    return ui_components