"""
File: smartcash/ui/charts/__init__.py
Deskripsi: Package untuk visualisasi dan chart (deprecated)

IMPORTANT: Implementasi di modul ini telah dipindahkan ke handler terpisah di:
- smartcash.ui.handlers.visualization_sample_handler
- smartcash.ui.handlers.visualization_compare_handler
- smartcash.ui.handlers.visualization_handler

Menggunakan modul ini tidak direkomendasikan lagi (deprecated).
Silakan gunakan handler baru untuk visualisasi dan komparasi dataset.
"""

# Peringatan deprecation
import warnings
warnings.warn(
    "Modul smartcash.ui.charts telah dipindahkan ke handler terpisah. "
    "Silakan gunakan smartcash.ui.handlers.visualization_* sebagai gantinya.",
    DeprecationWarning,
    stacklevel=2
)
