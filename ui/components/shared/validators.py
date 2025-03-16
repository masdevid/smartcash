"""
File: smartcash/ui/components/shared/validators.py
Deskripsi: Utilitas validasi untuk input UI dan form handling
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import re
import os
from pathlib import Path
from typing import Callable, Any, Optional, Dict, List, Union, Tuple

from smartcash.ui.utils.constants import COLORS, ICONS

# Validation result type
ValidationResult = Tuple[bool, Optional[str]]

def create_validation_message(message: str, is_error: bool = True) -> widgets.HTML:
    """
    Buat pesan validasi.
    
    Args:
        message: Pesan validasi
        is_error: Apakah pesan error (True) atau success (False)
        
    Returns:
        Widget HTML berisi pesan validasi
    """
    color = COLORS['danger'] if is_error else COLORS['success']
    icon = ICONS['error'] if is_error else ICONS['success']
    
    return widgets.HTML(
        f"""<div style="color: {color}; margin-top: 5px; font-size: 0.9em;">
            {icon} {message}
        </div>"""
    )

def show_validation_message(container: widgets.Box, message: str, is_error: bool = True) -> None:
    """
    Tampilkan pesan validasi di dalam container.
    
    Args:
        container: Container untuk menampilkan pesan
        message: Pesan validasi
        is_error: Apakah pesan error atau success
    """
    with container:
        display(create_validation_message(message, is_error))

def clear_validation_messages(container: widgets.Box) -> None:
    """
    Hapus semua pesan validasi dari container.
    
    Args:
        container: Container yang berisi pesan validasi
    """
    container.children = ()
    
# Basic validators
def validate_required(value: Any) -> ValidationResult:
    """Validasi field tidak boleh kosong."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return False, "Field ini wajib diisi"
    return True, None

def validate_numeric(value: Any) -> ValidationResult:
    """Validasi nilai numerik."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    try:
        float(value)
        return True, None
    except (ValueError, TypeError):
        return False, "Nilai harus berupa angka"

def validate_integer(value: Any) -> ValidationResult:
    """Validasi nilai integer."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    try:
        int(value)
        return True, None
    except (ValueError, TypeError):
        return False, "Nilai harus berupa bilangan bulat"

def validate_min_value(value: Any, min_value: float) -> ValidationResult:
    """Validasi nilai minimum."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    try:
        if float(value) < min_value:
            return False, f"Nilai harus lebih besar atau sama dengan {min_value}"
        return True, None
    except (ValueError, TypeError):
        return False, "Nilai tidak valid"

def validate_max_value(value: Any, max_value: float) -> ValidationResult:
    """Validasi nilai maksimum."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    try:
        if float(value) > max_value:
            return False, f"Nilai harus lebih kecil atau sama dengan {max_value}"
        return True, None
    except (ValueError, TypeError):
        return False, "Nilai tidak valid"

def validate_range(value: Any, min_value: float, max_value: float) -> ValidationResult:
    """Validasi nilai berada dalam range tertentu."""
    min_result = validate_min_value(value, min_value)
    if not min_result[0]:
        return min_result
        
    return validate_max_value(value, max_value)

def validate_min_length(value: str, min_length: int) -> ValidationResult:
    """Validasi panjang minimum string."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    if len(str(value)) < min_length:
        return False, f"Panjang minimal {min_length} karakter"
    return True, None

def validate_max_length(value: str, max_length: int) -> ValidationResult:
    """Validasi panjang maksimum string."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    if len(str(value)) > max_length:
        return False, f"Panjang maksimal {max_length} karakter"
    return True, None

def validate_regex(value: str, pattern: str, message: str = "Format tidak valid") -> ValidationResult:
    """Validasi string dengan regex pattern."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    if not re.match(pattern, str(value)):
        return False, message
    return True, None

def validate_email(value: str) -> ValidationResult:
    """Validasi format email."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return validate_regex(value, pattern, "Format email tidak valid")

def validate_url(value: str) -> ValidationResult:
    """Validasi format URL."""
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    return validate_regex(value, pattern, "Format URL tidak valid")

def validate_file_exists(value: str) -> ValidationResult:
    """Validasi file ada di filesystem."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    if not os.path.isfile(value):
        return False, f"File tidak ditemukan: {value}"
    return True, None

def validate_directory_exists(value: str) -> ValidationResult:
    """Validasi direktori ada di filesystem."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    if not os.path.isdir(value):
        return False, f"Direktori tidak ditemukan: {value}"
    return True, None

def validate_file_extension(value: str, allowed_extensions: List[str]) -> ValidationResult:
    """Validasi ekstensi file."""
    if not value:
        return True, None  # Allow empty for non-required fields
        
    _, ext = os.path.splitext(value)
    ext = ext.lower()
    
    if not ext:
        return False, "File tidak memiliki ekstensi"
        
    if ext[1:] not in [e.lower().lstrip('.') for e in allowed_extensions]:
        return False, f"Ekstensi file harus salah satu dari: {', '.join(allowed_extensions)}"
    return True, None

def validate_api_key(value: str, min_length: int = 10) -> ValidationResult:
    """Validasi API key (panjang minimum dan karakter yang valid)."""
    if not value:
        return False, "API key wajib diisi"
        
    min_result = validate_min_length(value, min_length)
    if not min_result[0]:
        return min_result
    
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', value):
        return False, "API key hanya boleh mengandung huruf, angka, underscore, titik, dan dash"
    
    return True, None

# Form validation
def validate_form(form_data: Dict[str, Any], validation_rules: Dict[str, List[Callable]]) -> Dict[str, str]:
    """
    Validasi form dengan berbagai aturan validasi.
    
    Args:
        form_data: Dictionary data form
        validation_rules: Dictionary aturan validasi (field: [validator1, validator2, ...])
        
    Returns:
        Dictionary berisi pesan error (field: error message)
    """
    errors = {}
    
    for field, validators in validation_rules.items():
        if field not in form_data:
            continue
            
        value = form_data[field]
        
        for validator in validators:
            is_valid, error_message = validator(value)
            if not is_valid:
                errors[field] = error_message
                break
    
    return errors

# Custom validation functions
def create_validator(validation_func: Callable, error_message: str) -> Callable:
    """
    Buat fungsi validator kustom.
    
    Args:
        validation_func: Fungsi yang mengembalikan boolean
        error_message: Pesan error jika validasi gagal
        
    Returns:
        Fungsi validator yang mengembalikan ValidationResult
    """
    def validator(value: Any) -> ValidationResult:
        if validation_func(value):
            return True, None
        return False, error_message
    
    return validator

# Combine validators
def combine_validators(*validators: Callable) -> Callable:
    """
    Gabungkan beberapa validator menjadi satu.
    
    Args:
        *validators: Fungsi-fungsi validator
        
    Returns:
        Fungsi validator gabungan
    """
    def combined_validator(value: Any) -> ValidationResult:
        for validator in validators:
            is_valid, error_message = validator(value)
            if not is_valid:
                return False, error_message
        return True, None
    
    return combined_validator