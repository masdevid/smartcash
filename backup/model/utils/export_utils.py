"""
File: smartcash/model/utils/export_utils.py
Deskripsi: Utilitas untuk export model ke format ONNX
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import ModelCheckpointError
from smartcash.model.utils.progress_utils import update_progress_safe, ProgressTracker

def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: List[int] = [1, 3, 640, 640],
    opset_version: int = 12,
    dynamic_axes: Optional[Dict] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    logger = None
) -> str:
    """Export PyTorch model ke format ONNX dengan progress tracking"""
    logger = logger or get_logger()
    
    try:
        # Inisialisasi progress dan siapkan input
        update_progress_safe(progress_tracker, 0, 3, "🔄 Memulai export ke ONNX...")
        dummy_input = torch.randn(input_shape, requires_grad=True)
        
        # Default dynamic axes jika tidak disediakan
        dynamic_axes = dynamic_axes or {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
        
        # Siapkan model untuk export
        update_progress_safe(progress_tracker, 1, 3, "📦 Menyiapkan model untuk export...")
        model.eval()
        
        # Export ke ONNX dengan konfigurasi standar
        torch.onnx.export(
            model, dummy_input, output_path,
            export_params=True, opset_version=opset_version, do_constant_folding=True,
            input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes
        )
        
        # Verifikasi hasil export
        update_progress_safe(progress_tracker, 2, 3, "💾 Menyimpan model ONNX...")
        file_size = output_path.stat().st_size
        formatted_size = format_file_size(file_size)
        
        # Selesaikan proses dan log hasil
        success_msg = f"✅ Model berhasil diexport ke ONNX ({formatted_size})"
        update_progress_safe(progress_tracker, 3, 3, success_msg)
        logger.info(f"✅ Model berhasil diexport ke ONNX: {output_path} ({formatted_size})")
        
        return str(output_path)
    
    except Exception as e:
        # Tangani error dan update progress
        error_msg = f"❌ Export ONNX error: {str(e)}"
        logger.error(error_msg)
        update_progress_safe(progress_tracker, 3, 3, error_msg)
        raise ModelCheckpointError(error_msg)

# Format file size dengan one-liner
format_file_size = lambda size_bytes: next((f"{size_bytes/1024**i:.1f} {unit}" for i, unit in enumerate(['B', 'KB', 'MB', 'GB', 'TB']) if size_bytes < 1024**(i+1)), f"{size_bytes/1024**4:.1f} TB")
