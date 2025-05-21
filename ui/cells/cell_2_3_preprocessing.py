"""
File: smartcash/ui/cells/cell_2_3_preprocessing.py
Deskripsi: Entry point minimalis untuk preprocessing dataset
"""
from smartcash.ui.dataset.preprocessing import get_preprocessing_ui_components, initialize_preprocessing_ui

# Dapatkan komponen UI yang sudah diinisialisasi atau buat baru jika belum ada
ui = get_preprocessing_ui_components()
if not ui:
    ui = initialize_preprocessing_ui()