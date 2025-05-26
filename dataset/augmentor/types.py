"""
File: smartcash/dataset/augmentor/types.py
Deskripsi: Type definitions dan dataclasses untuk augmentasi SmartCash yang lebih clean
"""

from typing import Dict, List, Callable, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

# Type aliases untuk readability
ProgressCallback = Callable[[str, int, int, str], None]
ResultDict = Dict[str, Any]
FileList = List[str]
ClassDistribution = Dict[str, int]
AugmentationTypes = List[str]
PathStr = Union[str, Path]

@dataclass
class AugConfig:
    """Konfigurasi augmentasi dengan default values yang sensible."""
    raw_dir: str = "data"
    aug_dir: str = "data/augmented" 
    prep_dir: str = "data/preprocessed"
    num_variations: int = 2
    target_count: int = 500
    output_prefix: str = "aug"
    process_bboxes: bool = True
    validate_results: bool = False  # Disable untuk memastikan semua gambar dihasilkan
    
    def __post_init__(self):
        """Validasi konfigurasi setelah inisialisasi."""
        if self.num_variations <= 0: self.num_variations = 2
        if self.target_count <= 0: self.target_count = 500

@dataclass 
class ProcessingStats:
    """Statistik hasil pemrosesan augmentasi."""
    total_files: int = 0
    processed_files: int = 0
    generated_images: int = 0
    success_rate: float = 0.0
    duration_seconds: float = 0.0
    
    def calculate_success_rate(self) -> float:
        """Hitung success rate berdasarkan file yang diproses."""
        return (self.processed_files / max(1, self.total_files)) * 100

@dataclass
class ClassBalanceInfo:
    """Informasi balancing kelas untuk augmentasi."""
    class_counts: ClassDistribution
    augmentation_needs: ClassDistribution
    selected_files: FileList
    target_classes: List[int]
    balancing_enabled: bool = False
    
    def get_classes_needing_augmentation(self) -> List[str]:
        """Dapatkan daftar kelas yang perlu augmentasi."""
        return [cls for cls, need in self.augmentation_needs.items() if need > 0]

@dataclass
class AugmentationResult:
    """Hasil augmentasi dengan metadata lengkap."""
    status: str = "pending"
    message: str = ""
    stats: ProcessingStats = None
    class_info: ClassBalanceInfo = None
    output_paths: Dict[str, str] = None
    
    def __post_init__(self):
        """Inisialisasi default values."""
        if self.stats is None: self.stats = ProcessingStats()
        if self.output_paths is None: self.output_paths = {}
    
    def is_success(self) -> bool:
        """Check apakah augmentasi berhasil."""
        return self.status == "success"
    
    def get_summary(self) -> str:
        """Dapatkan ringkasan hasil augmentasi."""
        if self.is_success():
            return f"✅ {self.stats.generated_images} gambar dihasilkan dari {self.stats.processed_files} file"
        return f"❌ {self.message}"

# Constants untuk layer classification
LAYER_1_CLASSES = list(range(0, 7))    # Banknote detection layer
LAYER_2_CLASSES = list(range(7, 14))   # Nominal detection layer  
LAYER_3_CLASSES = list(range(14, 17))  # Security features (diabaikan)
TARGET_CLASSES = LAYER_1_CLASSES + LAYER_2_CLASSES

# File extensions yang didukung
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
SUPPORTED_LABEL_EXTENSIONS = ['.txt']

# Default augmentation types
DEFAULT_AUGMENTATION_TYPES = ['flip', 'rotate', 'brightness', 'contrast', 'hsv']
COMBINED_AUGMENTATION_TYPES = ['flip', 'rotate', 'lighting', 'noise', 'hsv', 'weather']