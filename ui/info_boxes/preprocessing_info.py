"""
File: smartcash/ui/info_boxes/preprocessing_info.py
Deskripsi: Info box untuk preprocessing dengan comprehensive fallback handling
"""

import ipywidgets as widgets
from typing import Dict, Any

"""
File: smartcash/ui/info_boxes/preprocessing_info.py
Deskripsi: Info box untuk preprocessing dalam bentuk accordion yang tertutup by default
"""

import ipywidgets as widgets
from typing import Dict, Any

def get_preprocessing_info() -> widgets.Accordion:
    """
    Create preprocessing info dalam bentuk accordion yang tertutup by default.
    
    Returns:
        widgets.Accordion: Info accordion widget
    """
    
    info_content = """
    <div style="padding: 10px; background: #ffffff; border-radius: 5px;">
        
        <div style="margin: 10px 0;">
            <h6 style="color: #6c757d; margin: 8px 0 5px 0;">🖼️ Parameter Preprocessing:</h6>
            <ul style="margin: 5px 0; padding-left: 20px; color: #495057; font-size: 13px;">
                <li><strong>Resolusi:</strong> Ukuran output gambar (320x320 hingga 640x640)</li>
                <li><strong>Normalisasi:</strong> Metode normalisasi pixel untuk training optimal</li>
                <li><strong>Workers:</strong> Jumlah thread paralel untuk mempercepat proses</li>
                <li><strong>Target Split:</strong> Bagian dataset yang akan diproses</li>
            </ul>
        </div>

        <div style="margin: 10px 0;">
            <h6 style="color: #6c757d; margin: 8px 0 5px 0;">⚙️ Metode Normalisasi:</h6>
            <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 5px 0; font-size: 13px;">
                <div style="margin: 3px 0;"><strong>MinMax:</strong> Normalisasi ke range [0,1] - Direkomendasikan untuk YOLO</div>
                <div style="margin: 3px 0;"><strong>Standard:</strong> Z-score normalization (mean=0, std=1)</div>
                <div style="margin: 3px 0;"><strong>None:</strong> Tanpa normalisasi (pixel [0,255])</div>
            </div>
        </div>

        <div style="margin: 10px 0;">
            <h6 style="color: #6c757d; margin: 8px 0 5px 0;">🚀 Tips Preprocessing:</h6>
            <div style="background: #e7f3ff; padding: 8px; border-radius: 4px; border-left: 3px solid #007bff; font-size: 13px;">
                <div style="margin: 2px 0;">• Gunakan resolusi 640x640 untuk hasil terbaik</div>
                <div style="margin: 2px 0;">• Sesuaikan jumlah workers dengan kapasitas RAM</div>
                <div style="margin: 2px 0;">• Preprocessing akan mempertahankan aspect ratio gambar</div>
                <div style="margin: 2px 0;">• Hasil preprocessing disimpan terpisah dari dataset asli</div>
            </div>
        </div>

        <div style="margin: 10px 0;">
            <h6 style="color: #6c757d; margin: 8px 0 5px 0;">⚡ Performance Guidelines:</h6>
            <div style="background: #fff8e1; padding: 8px; border-radius: 4px; border-left: 3px solid #ffc107; font-size: 13px;">
                <div style="margin: 2px 0;">• <strong>Dataset kecil (&lt;1000 gambar):</strong> 2-4 workers</div>
                <div style="margin: 2px 0;">• <strong>Dataset sedang (1000-5000 gambar):</strong> 4-6 workers</div>
                <div style="margin: 2px 0;">• <strong>Dataset besar (&gt;5000 gambar):</strong> 6-8 workers</div>
                <div style="margin: 2px 0;">• Monitor penggunaan RAM selama proses</div>
            </div>
        </div>

        <div style="margin: 10px 0;">
            <h6 style="color: #6c757d; margin: 8px 0 5px 0;">🔍 Pre-Processing Checklist:</h6>
            <div style="background: #f0f8f0; padding: 8px; border-radius: 4px; border-left: 3px solid #28a745; font-size: 13px;">
                <div style="margin: 2px 0;">✅ Dataset sudah di-download dan tersedia</div>
                <div style="margin: 2px 0;">✅ Struktur folder dataset valid (train/valid/test)</div>
                <div style="margin: 2px 0;">✅ Setiap folder berisi subdirektori images/ dan labels/</div>
                <div style="margin: 2px 0;">✅ Ruang penyimpanan mencukupi untuk hasil preprocessing</div>
            </div>
        </div>

        <div style="margin-top: 10px; padding: 8px; background: #ffe6e6; border-radius: 4px; border-left: 3px solid #dc3545; font-size: 13px;">
            <strong style="color: #721c24;">⚠️ Penting:</strong>
            <div style="color: #721c24; margin-top: 3px;">
                Proses preprocessing akan membuat salinan dataset dengan format yang dioptimalkan. 
                Pastikan ruang penyimpanan mencukupi sebelum memulai.
            </div>
        </div>
    </div>
    """
    
    # Create accordion dengan content
    help_accordion = widgets.Accordion([
        widgets.HTML(value=info_content)
    ])
    
    # Set title dan tutup by default
    help_accordion.set_title(0, "💡 Panduan Dataset Preprocessing")
    help_accordion.selected_index = None  # Tertutup by default
    
    return help_accordion

def create_simple_preprocessing_info() -> widgets.Accordion:
    """
    Create simple fallback preprocessing info dalam accordion.
    
    Returns:
        widgets.Accordion: Simple info accordion widget
    """
    
    simple_content = """
    <div style="padding: 10px; background: #ffffff;">
        <p style="margin: 8px 0; font-size: 14px;">Preprocessing akan mengubah ukuran gambar dan menormalisasi pixel untuk training yang optimal.</p>
        
        <div style="margin: 10px 0;">
            <strong style="color: #495057;">Parameter Utama:</strong>
            <ul style="margin: 5px 0; padding-left: 20px; color: #495057; font-size: 13px;">
                <li><strong>Resolusi:</strong> Ukuran output gambar (direkomendasikan 640x640)</li>
                <li><strong>Normalisasi:</strong> Metode normalisasi pixel (minmax direkomendasikan)</li>
                <li><strong>Workers:</strong> Jumlah thread paralel (4-8 untuk performa optimal)</li>
                <li><strong>Split:</strong> Bagian dataset yang diproses (all untuk semua split)</li>
            </ul>
        </div>
        
        <div style="margin-top: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 13px;">
            <strong>💡 Tips:</strong> Gunakan "Check Dataset" untuk memvalidasi struktur dataset sebelum preprocessing.
        </div>
    </div>
    """
    
    # Create simple accordion
    simple_accordion = widgets.Accordion([
        widgets.HTML(value=simple_content)
    ])
    
    simple_accordion.set_title(0, "💡 Info Preprocessing")
    simple_accordion.selected_index = None  # Tertutup by default
    
    return simple_accordion