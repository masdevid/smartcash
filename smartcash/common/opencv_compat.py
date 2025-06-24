"""
File: smartcash/common/opencv_compat.py
Deskripsi: Kompatibilitas untuk berbagai versi OpenCV
"""

import cv2
import numpy as np

# Kompatibilitas untuk CV_8U yang dihapus di OpenCV versi terbaru
# Gunakan CV_8UC1 sebagai pengganti CV_8U
CV_8U = cv2.CV_8UC1

# Fungsi bantuan untuk thresholding yang kompatibel
def threshold(src, thresh, maxval, type_flags):
    """
    Fungsi threshold yang kompatibel dengan berbagai versi OpenCV
    
    Args:
        src: Sumber gambar
        thresh: Nilai threshold
        maxval: Nilai maksimum untuk thresholding
        type_flags: Tipe thresholding (THRESH_BINARY, dll)
        
    Returns:
        Tuple (retval, dst) seperti cv2.threshold
    """
    # Pastikan input adalah tipe yang benar
    if src.dtype != np.uint8:
        src = src.astype(np.uint8)
        
    # Pastikan type_flags tidak mengandung CV_8U
    type_flags = type_flags & ~cv2.CV_8U if hasattr(cv2, 'CV_8U') else type_flags
    
    # Eksekusi thresholding
    return cv2.threshold(src, thresh, maxval, type_flags)

# Fungsi bantuan untuk konversi tipe yang kompatibel
def convert_to_uint8(image):
    """Konversi gambar ke tipe uint8 dengan rentang 0-255"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image
