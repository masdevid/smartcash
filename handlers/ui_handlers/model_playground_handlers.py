"""
File: handlers/ui_handlers/model_playground_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen model playground, menangani pengujian model dan visualisasi performa.
"""

import torch
import gc
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import traceback

def on_test_model_button_clicked(ui_components, model_handler, config, logger, device=None):
    """
    Handler untuk tombol test model.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_playground_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi aplikasi
        logger: Logger untuk mencatat aktivitas
        device: Device untuk menjalankan model (cuda atau cpu)
    """
    # Update tombol ke status loading
    ui_components['test_model_button'].description = "Memproses..."
    ui_components['test_model_button'].disabled = True
    
    with ui_components['output']:
        clear_output()
        
        # Update konfigurasi berdasarkan pilihan
        config['model']['backbone'] = ui_components['backbone_selector'].value
        config['model']['img_size'] = [ui_components['img_size_slider'].value, ui_components['img_size_slider'].value]
        
        if ui_components['detection_mode_selector'].value == 'multi':
            config['layers'] = ['banknote', 'nominal', 'security']
        else:
            config['layers'] = ['banknote']
        
        config['model']['pretrained'] = ui_components['pretrained_checkbox'].value
        
        # Gunakan GPU jika tersedia
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Bersihkan memori sebelum membuat model baru
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Gunakan ModelHandler untuk membuat model
            test_manager = model_handler.__class__(
                config=config, 
                logger=logger
            )
            
            # Initializes layer config if needed
            try:
                from smartcash.utils.layer_config_manager import get_layer_config
                layer_config = get_layer_config()
            except ImportError:
                logger.warning("⚠️ Tidak dapat mengimpor layer_config_manager")
            
            # Ukur waktu pembuatan model
            start_time = time.time()
            model = test_manager.get_model()
            model_creation_time = (time.time() - start_time) * 1000  # convert to ms
            
            # Pindahkan model ke device yang sesuai
            model = model.to(device)
            
            logger.success(f"✅ Model berhasil dibuat dalam {model_creation_time:.2f}ms")
            
            # Buat input dummy untuk test
            batch_size = 1
            input_dummy = torch.randn(batch_size, 3, ui_components['img_size_slider'].value, ui_components['img_size_slider'].value, device=device)
            
            # Ukur waktu inferensi
            model.eval()
            with torch.no_grad():
                # Warming up
                for _ in range(5):
                    _ = model(input_dummy)
                
                # Ukur waktu
                inference_times = []
                for _ in range(10):  # Jalankan 10 kali untuk mendapatkan rata-rata yang stabil
                    start_time = time.time()
                    output = model(input_dummy)
                    inference_times.append((time.time() - start_time) * 1000)  # convert to ms
                
                avg_inference_time = np.mean(inference_times)
                std_inference_time = np.std(inference_times)
                
                logger.success(f"✅ Inferensi selesai dalam {avg_inference_time:.2f}ms±{std_inference_time:.2f}ms")
                
                # Gunakan ModelVisualizer untuk menampilkan informasi model
                try:
                    from smartcash.utils.model_visualizer import ModelVisualizer
                    visualizer = ModelVisualizer(model, logger)
                    param_count, trainable_count = visualizer.count_parameters()
                except ImportError:
                    logger.warning("⚠️ ModelVisualizer tidak tersedia, menggunakan perhitungan parameter manual")
                    # Fallback untuk menghitung parameter jika visualizer tidak tersedia
                    param_count = sum(p.numel() for p in model.parameters())
                    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Tampilkan info output
                if isinstance(output, dict):
                    logger.info("📊 Output Model (Multi-layer):")
                    for layer_name, layer_outputs in output.items():
                        logger.info(f"  Layer '{layer_name}':")
                        for i, out in enumerate(layer_outputs):
                            logger.info(f"    P{i+3}: shape={tuple(out.shape)}")
                else:
                    for i, out in enumerate(output):
                        logger.info(f"  P{i+3}: shape={tuple(out.shape)}")
                
                # Tampilkan ringkasan visual
                print("\n📈 Ringkasan Performance Model:")
                print(f"  • Backbone: {ui_components['backbone_selector'].value}")
                print(f"  • Mode Deteksi: {ui_components['detection_mode_selector'].value}")
                print(f"  • Pretrained: {'Ya' if ui_components['pretrained_checkbox'].value else 'Tidak'}")
                print(f"  • Ukuran Gambar: {ui_components['img_size_slider'].value}x{ui_components['img_size_slider'].value}")
                print(f"  • Total Parameter: {param_count:,}")
                print(f"  • Parameter Trainable: {trainable_count:,} ({trainable_count/param_count*100:.1f}%)")
                print(f"  • Waktu Pembuatan Model: {model_creation_time:.2f}ms")
                print(f"  • Waktu Inferensi: {avg_inference_time:.2f}ms±{std_inference_time:.2f}ms")
                print(f"  • FPS Estimasi: {1000/avg_inference_time:.1f}")
                
                # Plot barchart waktu
                plt.figure(figsize=(10, 5))
                
                # Plot model creation time
                plt.subplot(1, 2, 1)
                bars = plt.bar(
                    ['Pembuatan Model', 'Inferensi'], 
                    [model_creation_time, avg_inference_time],
                    color=['#3498db', '#2ecc71']
                )
                
                # Tambahkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{height:.1f}ms',
                        ha='center', 
                        va='bottom',
                        fontweight='bold'
                    )
                
                plt.title('Perbandingan Waktu (ms)')
                plt.ylabel('Waktu (ms)')
                plt.yscale('log')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Plot parameter distribution
                plt.subplot(1, 2, 2)
                
                try:
                    module_params = visualizer.get_module_parameters()
                    
                    # Ambil 5 modul dengan parameter terbanyak
                    top_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)[:5]
                    labels = [m[0] for m in top_modules]
                    sizes = [m[1] / param_count * 100 for m in top_modules]
                    
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                           colors=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'])
                    plt.axis('equal')
                    plt.title('Distribusi Parameter')
                except Exception as viz_error:
                    logger.warning(f"⚠️ Tidak dapat mem-plot distribusi parameter: {str(viz_error)}")
                    plt.text(0.5, 0.5, 'Visualisasi parameter tidak tersedia', 
                             ha='center', va='center', fontsize=12)
                    plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Tampilkan visualisasi backbone jika tersedia
                try:
                    print("\n🧠 Visualisasi Arsitektur Backbone:")
                    visualizer.visualize_backbone()
                except Exception as backbone_viz_error:
                    logger.warning(f"⚠️ Tidak dapat memvisualisasikan backbone: {str(backbone_viz_error)}")
                    print("⚠️ Visualisasi backbone tidak tersedia.")
                
                # Bersihkan memori
                del model, input_dummy, output
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
        except Exception as e:
            logger.error(f"❌ Terjadi kesalahan: {str(e)}")
            traceback.print_exc()
        
        # Reset tombol
        ui_components['test_model_button'].description = "Buat & Test Model"
        ui_components['test_model_button'].disabled = False

def setup_model_playground_handlers(ui_components, model_handler, config, logger):
    """
    Setup semua event handlers untuk UI model playground.
    
    Args:
        ui_components: Dictionary komponen UI dari create_model_playground_ui()
        model_handler: Instance dari ModelHandler
        config: Dictionary konfigurasi aplikasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Gunakan GPU jika tersedia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup handler untuk tombol test
    ui_components['test_model_button'].on_click(
        lambda b: on_test_model_button_clicked(ui_components, model_handler, config, logger, device)
    )
    
    return ui_components