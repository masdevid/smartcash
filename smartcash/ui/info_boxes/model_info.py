"""
File: smartcash/ui/info_boxes/model_info.py
Deskripsi: Informasi dan panduan untuk konfigurasi model
"""

def get_model_info_content() -> str:
    """
    Get content untuk info accordion konfigurasi model.
    
    Returns:
        HTML string dengan informasi konfigurasi model
    """
    return """
    <div style="padding: 10px;">
        <h4>🤖 Model Configuration Guide</h4>
        
        <h5>Backbone Options:</h5>
        <ul>
            <li><b>EfficientNet-B4</b>: Recommended untuk accuracy tinggi dengan efficient computation</li>
            <li><b>CSPDarknet</b>: YOLOv5 baseline backbone, lebih ringan tapi kurang akurat</li>
        </ul>
        
        <h5>Detection Layers:</h5>
        <ul>
            <li><b>Banknote</b>: Deteksi keberadaan uang kertas</li>
            <li><b>Nominal</b>: Identifikasi nilai nominal (Rp 1000 - Rp 100.000)</li>
            <li><b>Security</b>: Deteksi fitur keamanan uang</li>
        </ul>
        
        <h5>Layer Modes:</h5>
        <ul>
            <li><b>Single Layer</b>: Cepat, cocok untuk real-time detection</li>
            <li><b>Multi-layer</b>: Lebih akurat, proses semua layer bersamaan</li>
        </ul>
        
        <h5>Optimization:</h5>
        <ul>
            <li><b>Feature Optimization</b>: Channel attention untuk improved accuracy</li>
            <li><b>Mixed Precision</b>: FP16 training untuk faster computation</li>
        </ul>
        
        <p style="margin-top: 15px;">
            <em>Configuration akan disimpan ke <code>model_config.yaml</code></em>
        </p>
    </div>
    """
