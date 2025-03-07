"""
File: smartcash/handlers/ui_handlers/model_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk interaksi UI komponen model, menangani logika untuk inisialisasi, visualisasi, 
           manajemen checkpoint, dan optimasi model. Memperbaiki missing imports dan menyeragamkan device handling.
"""
import os
import time
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime  # Added missing import
import ipywidgets as widgets  # Added missing import
from IPython.display import display, clear_output, HTML
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable, Union, Tuple

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
    """
    Setup dan dapatkan informasi GPU.
    
    Returns:
        Dict berisi informasi GPU atau CPU device
    """
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

def on_initialize_model_clicked(ui_components: Dict[str, Any], model_handler: Any, 
                              config: Dict[str, Any], logger: Optional[Any] = None) -> None:
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
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
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
            if logger:
                logger.info("🔄 Mempersiapkan lingkungan...")
            gpu_info = setup_gpu()
            
            # Buat model
            with memory_manager():
                if logger:
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
                if logger:
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
            if logger:
                logger.error(f"❌ Error saat inisialisasi model: {str(e)}")
            print(f"❌ Error saat inisialisasi model: {str(e)}")
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
def on_visualize_model_clicked(ui_components: Dict[str, Any], model_handler: Any,
                             config: Dict[str, Any], logger: Optional[Any] = None) -> None:
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
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
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
            
            # Determine device consistently
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Gunakan ModelHandler untuk membuat model
            model = model_handler.create_model(backbone_type=ui_components['backbone_select'].value)
            
            # Initialize layer config if needed
            try:
                from smartcash.utils.layer_config_manager import get_layer_config
                layer_config = get_layer_config()
            except ImportError:
                if logger:
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
                if logger:
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
            if logger:
                logger.error(f"❌ Error memvisualisasikan model: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Reset button state
            ui_components['create_model_button'].disabled = False
            ui_components['create_model_button'].description = "Buat Model & Visualisasikan"

def on_list_checkpoints_clicked(ui_components: Dict[str, Any], checkpoint_handler: Any, 
                              logger: Optional[Any] = None) -> None:
    """
    Handler untuk tombol melihat checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'checkpoints_output' not in ui_components:
        if logger:
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
            if logger:
                logger.error(f"❌ Error saat melihat checkpoint: {str(e)}")
            print(f"❌ Error saat melihat checkpoint: {str(e)}")

def on_cleanup_checkpoints_clicked(ui_components: Dict[str, Any], checkpoint_handler: Any, 
                                 logger: Optional[Any] = None) -> None:
    """
    Handler untuk tombol membersihkan checkpoint.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        checkpoint_handler: Instance dari CheckpointHandler
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'checkpoints_output' not in ui_components:
        if logger:
            logger.error("❌ Required UI component 'checkpoints_output' not found")
        return
    
    with ui_components['checkpoints_output']:
        clear_output()
        
        try:
            # Tampilkan konfirmasi
            if logger:
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
                        if logger:
                            logger.success(f"✅ Berhasil membersihkan {results['removed']} checkpoint")
                            logger.info(f"📁 Ukuran yang dibebaskan: {results.get('freed_space_mb', 0):.2f} MB")
                        print(f"✅ Berhasil membersihkan {results['removed']} checkpoint")
                        print(f"📁 Ukuran yang dibebaskan: {results.get('freed_space_mb', 0):.2f} MB")
                    else:
                        if logger:
                            logger.info("ℹ️ Tidak ada checkpoint yang perlu dibersihkan")
                        print("ℹ️ Tidak ada checkpoint yang perlu dibersihkan")
                        
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error saat membersihkan checkpoint: {str(e)}")
                    print(f"❌ Error saat membersihkan checkpoint: {str(e)}")
                    
                # Tampilkan kembali checkpoint yang tersisa
                on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger)
            
            def on_cancel(b):
                clear_output()
                if logger:
                    logger.info("ℹ️ Pembersihan checkpoint dibatalkan")
                print("ℹ️ Pembersihan checkpoint dibatalkan")
                
                # Tampilkan kembali list checkpoint
                on_list_checkpoints_clicked(ui_components, checkpoint_handler, logger)
            
            confirm_button.on_click(on_confirm)
            cancel_button.on_click(on_cancel)
            
            display(widgets.HBox([confirm_button, cancel_button]))
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat mempersiapkan pembersihan checkpoint: {str(e)}")
            print(f"❌ Error saat mempersiapkan pembersihan checkpoint: {str(e)}")
def on_compare_checkpoints_clicked(ui_components: Dict[str, Any], checkpoint_handler: Any, 
                                model_handler: Any, logger: Optional[Any] = None) -> None:
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
        if logger:
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
                if logger:
                    logger.warning("⚠️ Tidak ada checkpoint yang tersedia untuk dibandingkan")
                print("⚠️ Tidak ada checkpoint yang tersedia untuk dibandingkan")
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
                            if logger:
                                logger.warning("⚠️ Pilih dua checkpoint yang berbeda untuk dibandingkan")
                            print("⚠️ Pilih dua checkpoint yang berbeda untuk dibandingkan")
                            return
                        
                        if logger:
                            logger.info(f"🔄 Membandingkan checkpoint:")
                            logger.info(f"• Checkpoint 1: {os.path.basename(ckpt1_path)}")
                            logger.info(f"• Checkpoint 2: {os.path.basename(ckpt2_path)}")
                        
                        print(f"🔄 Membandingkan checkpoint:")
                        print(f"• Checkpoint 1: {os.path.basename(ckpt1_path)}")
                        print(f"• Checkpoint 2: {os.path.basename(ckpt2_path)}")
                        
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
                        if logger:
                            logger.error(f"❌ Error saat membandingkan checkpoint: {str(e)}")
                        print(f"❌ Error saat membandingkan checkpoint: {str(e)}")
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
            if logger:
                logger.error(f"❌ Error saat mempersiapkan perbandingan checkpoint: {str(e)}")
            print(f"❌ Error saat mempersiapkan perbandingan checkpoint: {str(e)}")

def on_mount_drive_clicked(ui_components: Dict[str, Any], logger: Optional[Any] = None) -> None:
    """
    Handler untuk tombol mount Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI dari create_checkpoint_manager_ui()
        logger: Logger untuk mencatat aktivitas
    """
    # Validate UI components
    if 'mount_drive_button' not in ui_components:
        if logger:
            logger.error("❌ Required UI component 'mount_drive_button' not found")
        return
    
    ui_components['mount_drive_button'].disabled = True
    
    try:
        from google.colab import drive
        if logger:
            logger.info("🔄 Melakukan mount Google Drive...")
        print("🔄 Melakukan mount Google Drive...")
        drive.mount('/content/drive')
        if logger:
            logger.success("✅ Google Drive berhasil di-mount")
        print("✅ Google Drive berhasil di-mount")
        
        # Update tampilan tombol
        ui_components['mount_drive_button'].description = 'Google Drive Terpasang'
        ui_components['mount_drive_button'].button_style = ''
        ui_components['mount_drive_button'].icon = 'check'
    except ImportError:
        if logger:
            logger.warning("⚠️ Tidak dapat melakukan mount Google Drive (bukan di Colab)")
        print("⚠️ Tidak dapat melakukan mount Google Drive (bukan di Colab)")
        ui_components['mount_drive_button'].description = 'Tidak di Colab'
        ui_components['mount_drive_button'].button_style = 'warning'
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat mount Google Drive: {str(e)}")
        print(f"❌ Error saat mount Google Drive: {str(e)}")
        ui_components['mount_drive_button'].disabled = False
        ui_components['mount_drive_button'].description = 'Mount Google Drive (Gagal)'
        ui_components['mount_drive_button'].button_style = 'danger'

def setup_model_initialization_handlers(ui_components: Dict[str, Any], model_handler: Any, 
                                      config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
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
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
        return ui_components
    
    # Bind event handler untuk tombol inisialisasi
    ui_components['initialize_button'].on_click(
        lambda b: on_initialize_model_clicked(ui_components, model_handler, config, logger)
    )
    
    return ui_components

def setup_model_visualizer_handlers(ui_components: Dict[str, Any], model_handler: Any, 
                                  config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
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
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
        return ui_components
    
    # Bind event handler untuk tombol visualisasi
    ui_components['create_model_button'].on_click(
        lambda b: on_visualize_model_clicked(ui_components, model_handler, config, logger)
    )
    
    return ui_components

def setup_checkpoint_manager_handlers(ui_components: Dict[str, Any], checkpoint_handler: Any, 
                                    model_handler: Any, logger: Optional[Any] = None) -> Dict[str, Any]:
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
    required_components = ['list_checkpoints_button', 'cleanup_checkpoints_button', 
                         'compare_button', 'mount_drive_button']
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
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