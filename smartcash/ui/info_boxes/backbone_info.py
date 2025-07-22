"""
File: smartcash/ui/info_boxes/backbone_info.py
Deskripsi: Info box content for backbone configuration

Modul ini berisi konten HTML untuk menampilkan panduan dan dokumentasi
pada backbone configuration.
"""
import ipywidgets as widgets

def get_backbone_info_content() -> widgets.HTML:
    """Get backbone configuration info box content
    
    Returns:
        IPython HTML widget with formatted help content
    """
    info_html = """
    <div style='padding: 10px; font-size: 14px;'>
        <h4 style='color: #2c3e50; margin-bottom: 10px;'>📖 Panduan Backbone Configuration</h4>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Backbone Architectures:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>EfficientNet-B4:</strong> Modern architecture dengan compound scaling. 
                    Efisien untuk resource terbatas dengan akurasi tinggi</li>
                <li><strong>EfficientNet-B3:</strong> Lebih ringan dari B4, cocok untuk device terbatas</li>
                <li><strong>EfficientNet-B5:</strong> Lebih dalam dari B4, akurasi lebih tinggi</li>
                <li><strong>ResNet50:</strong> Arsitektur klasik dengan residual blocks</li>
                <li><strong>MobileNetV3:</strong> Sangat ringan untuk mobile/edge devices</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Model Types:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>YOLOv5s:</strong> Small - cepat, ringan, akurasi cukup</li>
                <li><strong>YOLOv5m:</strong> Medium - keseimbangan kecepatan-akurasi</li>
                <li><strong>YOLOv5l:</strong> Large - akurasi tinggi, lebih lambat</li>
                <li><strong>YOLOv5x:</strong> Extra large - akurasi tertinggi, paling lambat</li>
                <li><strong>Custom:</strong> Konfigurasi custom (untuk advanced users)</li>
            </ul>
        </div>
        
        <div style='margin-bottom: 15px;'>
            <strong style='color: #3498db;'>Advanced Options:</strong>
            <ul style='margin: 5px 0;'>
                <li><strong>Pretrained Weights:</strong> Gunakan weight yang sudah dilatih di dataset besar</li>
                <li><strong>Attention Mechanism:</strong> Self-attention untuk fokus ke region penting</li>
                <li><strong>ResidualAdapter:</strong> Residual connections untuk gradient flow</li>
                <li><strong>CIoU Loss:</strong> Complete IoU untuk better bounding box regression</li>
            </ul>
        </div>
        
        <div style='background: #e8f4fd; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <strong>💡 Rekomendasi:</strong> Mulai dengan EfficientNet-B4 untuk keseimbangan
            antara performance dan akurasi. Gunakan YOLOv5m sebagai baseline.
        </div>
        
        <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <strong>🔄 Auto-Sync:</strong> Perubahan pada backbone akan otomatis
            mempengaruhi Hyperparameters dan Strategy configuration.
        </div>
    </div>
    """
    return widgets.HTML(info_html)
