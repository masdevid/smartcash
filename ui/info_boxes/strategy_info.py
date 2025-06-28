"""
File: smartcash/ui/info_boxes/strategy_info.py
Deskripsi: Info box content for strategy configuration

Modul ini berisi konten HTML untuk menampilkan panduan dan dokumentasi
pada strategy configuration.
"""
import ipywidgets as widgets

def get_strategy_info_content() -> widgets.HTML:
    """Get strategy configuration info box content
    
    Returns:
        IPython HTML widget with formatted help content
    """
    info_html = """
    <div style='padding: 10px; font-size: 14px;'>
        <h4 style='color: #2c3e50; margin-bottom: 10px;'>📖 Panduan Strategy Configuration</h4>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Validation Strategy:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Validation Frequency:</strong> Validasi setiap N epochs</li>
                <li><strong>IoU Threshold:</strong> Threshold untuk Non-Max Suppression (0.4-0.7)</li>
                <li><strong>Confidence Threshold:</strong> Minimum confidence untuk deteksi (0.001-0.1)</li>
                <li><strong>Max Detections:</strong> Maksimum deteksi per gambar (100-1000)</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Training Utilities:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Experiment Name:</strong> Nama unik untuk tracking eksperimen</li>
                <li><strong>TensorBoard:</strong> Enable logging untuk visualisasi training</li>
                <li><strong>Log Metrics:</strong> Frekuensi logging metrics (steps)</li>
                <li><strong>Layer Mode:</strong> Single layer atau multi-layer detection</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Multi-Scale Training:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Enable:</strong> Random resize images untuk robustness</li>
                <li><strong>Min/Max Size:</strong> Range ukuran image (320-640)</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Early Stopping:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Enable:</strong> Stop training jika tidak ada peningkatan</li>
                <li><strong>Patience:</strong> Jumlah epochs tanpa peningkatan sebelum stop</li>
                <li><strong>Min Epochs:</strong> Minimum epochs sebelum early stopping</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Checkpointing:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Enable:</strong> Save model weights pada setiap epoch</li>
                <li><strong>Save Frequency:</strong> Frekuensi save model (epochs)</li>
                <li><strong>Save Best:</strong> Save model dengan performance terbaik</li>
            </ul>
        </div>
        
        <div style='background: #e8f4fd; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <strong>💡 Tips:</strong> Mulai dengan default values, monitor metrics, 
            dan adjust gradual berdasarkan validation performance.
        </div>
        
        <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <strong>🔄 Auto-Sync:</strong> Perubahan pada strategy akan otomatis
            tersinkronisasi dengan Hyperparameters dan Backbone configuration.
        </div>
    </div>
    """
    return widgets.HTML(info_html)
