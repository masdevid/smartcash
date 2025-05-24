"""
File: smartcash/ui/info_boxes/preprocessing_info.py
Deskripsi: Info box untuk preprocessing dengan comprehensive fallback handling
"""

import ipywidgets as widgets
from typing import Dict, Any

def get_preprocessing_info() -> widgets.HTML:
    """
    Create preprocessing info box dengan comprehensive content dan fallback.
    
    Returns:
        widgets.HTML: Info box widget
    """
    
    info_content = """
    <div style="padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 8px; margin: 15px 0; border: 1px solid #dee2e6;">
        
        <h5 style="color: #495057; margin-top: 0; display: flex; align-items: center;">
            <span style="margin-right: 8px;">💡</span>
            Panduan Dataset Preprocessing
        </h5>
        
        <div style="margin: 15px 0;">
            <h6 style="color: #6c757d; margin: 10px 0 5px 0;">🖼️ Parameter Preprocessing:</h6>
            <ul style="margin: 5px 0; padding-left: 20px; color: #495057;">
                <li><strong>Resolusi:</strong> Ukuran output gambar (320x320 hingga 640x640)</li>
                <li><strong>Normalisasi:</strong> Metode normalisasi pixel untuk training optimal</li>
                <li><strong>Workers:</strong> Jumlah thread paralel untuk mempercepat proses</li>
                <li><strong>Target Split:</strong> Bagian dataset yang akan diproses</li>
            </ul>
        </div>

        <div style="margin: 15px 0;">
            <h6 style="color: #6c757d; margin: 10px 0 5px 0;">⚙️ Metode Normalisasi:</h6>
            <div style="background: #ffffff; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <div style="margin: 5px 0;"><strong>MinMax:</strong> Normalisasi ke range [0,1] - Direkomendasikan untuk YOLO</div>
                <div style="margin: 5px 0;"><strong>Standard:</strong> Z-score normalization (mean=0, std=1)</div>
                <div style="margin: 5px 0;"><strong>None:</strong> Tanpa normalisasi (pixel [0,255])</div>
            </div>
        </div>

        <div style="margin: 15px 0;">
            <h6 style="color: #6c757d; margin: 10px 0 5px 0;">🚀 Tips Preprocessing:</h6>
            <div style="background: #e7f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #007bff;">
                <div style="margin: 3px 0;">• Gunakan resolusi 640x640 untuk hasil terbaik</div>
                <div style="margin: 3px 0;">• Sesuaikan jumlah workers dengan kapasitas RAM</div>
                <div style="margin: 3px 0;">• Preprocessing akan mempertahankan aspect ratio gambar</div>
                <div style="margin: 3px 0;">• Hasil preprocessing disimpan terpisah dari dataset asli</div>
            </div>
        </div>

        <div style="margin: 15px 0;">
            <h6 style="color: #6c757d; margin: 10px 0 5px 0;">⚡ Performance Guidelines:</h6>
            <div style="background: #fff8e1; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;">
                <div style="margin: 3px 0;">• <strong>Dataset kecil (&lt;1000 gambar):</strong> 2-4 workers</div>
                <div style="margin: 3px 0;">• <strong>Dataset sedang (1000-5000 gambar):</strong> 4-6 workers</div>
                <div style="margin: 3px 0;">• <strong>Dataset besar (&gt;5000 gambar):</strong> 6-8 workers</div>
                <div style="margin: 3px 0;">• Monitor penggunaan RAM selama proses</div>
            </div>
        </div>

        <div style="margin: 15px 0;">
            <h6 style="color: #6c757d; margin: 10px 0 5px 0;">🔍 Pre-Processing Checklist:</h6>
            <div style="background: #f0f8f0; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                <div style="margin: 3px 0;">✅ Dataset sudah di-download dan tersedia</div>
                <div style="margin: 3px 0;">✅ Struktur folder dataset valid (train/valid/test)</div>
                <div style="margin: 3px 0;">✅ Setiap folder berisi subdirektori images/ dan labels/</div>
                <div style="margin: 3px 0;">✅ Ruang penyimpanan mencukupi untuk hasil preprocessing</div>
            </div>
        </div>

        <div style="margin-top: 15px; padding: 10px; background: #ffe6e6; border-radius: 5px; border-left: 4px solid #dc3545;">
            <strong style="color: #721c24;">⚠️ Penting:</strong>
            <div style="color: #721c24; margin-top: 5px;">
                Proses preprocessing akan membuat salinan dataset dengan format yang dioptimalkan. 
                Pastikan ruang penyimpanan mencukupi sebelum memulai.
            </div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=info_content)

def create_simple_preprocessing_info() -> widgets.HTML:
    """
    Create simple fallback preprocessing info.
    
    Returns:
        widgets.HTML: Simple info widget
    """
    
    simple_content = """
    <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
        <h5>💡 Info Preprocessing</h5>
        <p>Preprocessing akan mengubah ukuran gambar dan menormalisasi pixel untuk training yang optimal.</p>
        <ul>
            <li><strong>Resolusi:</strong> Ukuran output gambar (direkomendasikan 640x640)</li>
            <li><strong>Normalisasi:</strong> Metode normalisasi pixel (minmax direkomendasikan)</li>
            <li><strong>Workers:</strong> Jumlah thread paralel (4-8 untuk performa optimal)</li>
            <li><strong>Split:</strong> Bagian dataset yang diproses (all untuk semua split)</li>
        </ul>
        <div style="margin-top: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px;">
            <strong>💡 Tips:</strong> Gunakan "Check Dataset" untuk memvalidasi struktur dataset sebelum preprocessing.
        </div>
    </div>
    """
    
    return widgets.HTML(value=simple_content)