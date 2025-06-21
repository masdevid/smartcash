"""
File: smartcash/ui/utils/silent_wrapper.py
Deskripsi: Wrapper untuk mencegah dictionary components muncul saat diprint di notebook
"""

class SilentDictWrapper(dict):
    """Wrapper untuk dictionary yang tidak menampilkan output saat diprint
    
    Berguna untuk mencegah output otomatis dari dictionary components
    saat dikembalikan dari fungsi inisialisasi di notebook.
    """
    def __repr__(self) -> str:
        """Override __repr__ untuk mengembalikan string kosong"""
        return ""
    
    def __str__(self) -> str:
        """Override __str__ untuk mengembalikan string kosong"""
        return ""
