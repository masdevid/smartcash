"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/evaluation/handlers/inference_time_handler.py
Deskripsi: Handler untuk menangani metrik waktu inferensi dalam evaluasi model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from smartcash.ui.utils.logger_bridge import log_to_service

def setup_inference_time_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handler untuk checkbox inference time"""
    logger = ui_components.get('logger')
    
    # Dapatkan checkbox inference time
    inference_time_checkbox = ui_components.get('inference_time_checkbox')
    
    if inference_time_checkbox is not None:
        # Tambahkan handler untuk perubahan nilai checkbox
        inference_time_checkbox.observe(
            lambda change: on_inference_time_checkbox_change(change, ui_components, config, logger),
            names='value'
        )
        
        log_to_service(logger, "✅ Handler inference time berhasil diinisialisasi", "info")
    else:
        log_to_service(logger, "⚠️ Checkbox inference time tidak ditemukan", "warning")

def on_inference_time_checkbox_change(change: Dict[str, Any], ui_components: Dict[str, Any], 
                                     config: Dict[str, Any], logger) -> None:
    """Handler untuk perubahan nilai checkbox inference time"""
    new_value = change.get('new', False)
    
    # Update config dengan nilai baru
    if 'evaluation' not in config:
        config['evaluation'] = {}
    
    config['evaluation']['show_inference_time'] = new_value
    
    log_to_service(logger, f"ℹ️ Tampilkan metrik waktu inferensi: {'Ya' if new_value else 'Tidak'}", "info")

def display_inference_time_metrics(inference_metrics: Dict[str, Any], ui_components: Dict[str, Any], 
                                  config: Dict[str, Any], logger) -> None:
    """Tampilkan metrik waktu inferensi dalam UI hasil evaluasi"""
    # Cek apakah metrik inference time harus ditampilkan
    show_inference_time = config.get('evaluation', {}).get('show_inference_time', True)
    
    if not show_inference_time:
        log_to_service(logger, "ℹ️ Metrik waktu inferensi tidak ditampilkan (dinonaktifkan)", "info")
        return
    
    # Cek apakah metrics_tabs tersedia di UI
    metrics_tabs = ui_components.get('metrics_tabs')
    if metrics_tabs is None:
        log_to_service(logger, "⚠️ Metrics tabs tidak ditemukan untuk menampilkan waktu inferensi", "warning")
        return
    
    # Cek apakah inference_metrics valid
    if not inference_metrics or not isinstance(inference_metrics, dict):
        log_to_service(logger, "⚠️ Metrik waktu inferensi tidak valid", "warning")
        return
    
    try:
        # Buat tab baru untuk inference time jika belum ada
        if len(metrics_tabs.children) < 4:  # Asumsi 3 tab yang sudah ada
            # Buat output widget untuk inference time
            inference_time_output = widgets.Output()
            
            # Tambahkan tab baru
            metrics_tabs.children = list(metrics_tabs.children) + [inference_time_output]
            metrics_tabs.set_title(3, "⏱️ Waktu Inferensi")
            
            # Tampilkan metrik waktu inferensi
            with inference_time_output:
                clear_output()
                
                # Ekstrak metrik
                avg_time = inference_metrics.get('avg_inference_time', 0) * 1000  # Convert ke ms
                min_time = inference_metrics.get('min_inference_time', 0) * 1000
                max_time = inference_metrics.get('max_inference_time', 0) * 1000
                fps = inference_metrics.get('fps', 0)
                
                # Buat DataFrame untuk tampilan tabel
                data = {
                    'Metrik': ['Rata-rata (ms)', 'Minimum (ms)', 'Maksimum (ms)', 'FPS'],
                    'Nilai': [f"{avg_time:.2f}", f"{min_time:.2f}", f"{max_time:.2f}", f"{fps:.2f}"]
                }
                df = pd.DataFrame(data)
                
                # Tampilkan tabel
                display(df.style.set_properties(**{'text-align': 'left'}))
                
                # Tampilkan grafik distribusi waktu inferensi jika tersedia
                inference_times = inference_metrics.get('inference_times', [])
                if inference_times:
                    plt.figure(figsize=(10, 5))
                    plt.hist([t * 1000 for t in inference_times], bins=20, alpha=0.7, color='skyblue')
                    plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=2, label=f'Rata-rata: {avg_time:.2f} ms')
                    plt.xlabel('Waktu Inferensi (ms)')
                    plt.ylabel('Frekuensi')
                    plt.title('Distribusi Waktu Inferensi')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.show()
                    
                    # Tampilkan tren waktu inferensi
                    plt.figure(figsize=(10, 5))
                    plt.plot([t * 1000 for t in inference_times], marker='o', linestyle='-', alpha=0.5)
                    plt.axhline(avg_time, color='red', linestyle='dashed', linewidth=2, label=f'Rata-rata: {avg_time:.2f} ms')
                    plt.xlabel('Urutan Gambar')
                    plt.ylabel('Waktu Inferensi (ms)')
                    plt.title('Tren Waktu Inferensi')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.show()
        
        log_to_service(logger, f"✅ Metrik waktu inferensi berhasil ditampilkan: {avg_time:.2f} ms, {fps:.2f} FPS", "info")
        
    except Exception as e:
        log_to_service(logger, f"❌ Error menampilkan metrik waktu inferensi: {str(e)}", "error")
