#!/bin/bash

# Script untuk membuat struktur direktori service evaluation

# Variabel untuk warna output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Membuat struktur direktori evaluation service...${NC}"

# Buat direktori utama jika belum ada
mkdir -p smartcash/model/services/evaluation

# Buat file __init__.py
cat > smartcash/model/services/evaluation/__init__.py << 'EOF'
"""
File: smartcash/model/services/evaluation/__init__.py
Deskripsi: Modul inisialisasi untuk layanan evaluasi model yang terintegrasi
"""

from smartcash.model.services.evaluation.core import EvaluationService
from smartcash.model.services.evaluation.metrics import MetricsComputation
from smartcash.model.services.evaluation.visualization import EvaluationVisualizer

__all__ = [
    'EvaluationService',
    'MetricsComputation',
    'EvaluationVisualizer'
]
EOF

echo -e "${GREEN}✅ File __init__.py berhasil dibuat${NC}"

# Buat file core.py
cat > smartcash/model/services/evaluation/core.py << 'EOF'
"""
File: smartcash/model/services/evaluation/core.py
Deskripsi: Layanan inti untuk evaluasi model deteksi mata uang
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import time
from tqdm.auto import tqdm

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.model.services.evaluation.metrics import MetricsComputation
from smartcash.model.services.evaluation.visualization import EvaluationVisualizer
from smartcash.model.components.losses import compute_loss

class EvaluationService:
    """
    Layanan untuk evaluasi model deteksi mata uang dengan dukungan lengkap.
    
    Fitur:
    - Evaluasi performa model pada dataset validasi
    - Perhitungan berbagai metrik evaluasi
    - Visualisasi hasil prediksi dan metrik
    - Integrasi dengan eksperimen tracking
    """
    
    def __init__(
        self, 
        config: Dict,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi layanan evaluasi.
        
        Args:
            config: Konfigurasi model dan evaluasi
            output_dir: Direktori output untuk hasil evaluasi
            logger: Logger instance
        """