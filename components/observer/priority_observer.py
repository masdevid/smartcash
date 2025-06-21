"""
File: smartcash/components/observer/priority_observer.py
Deskripsi: Prioritas observer di SmartCash
"""
# Definisi prioritas observer (untuk urutan eksekusi)
class ObserverPriority:
    """Definisi prioritas untuk observer."""
    CRITICAL = 100  # Observer yang harus dijalankan pertama
    HIGH = 75       # Observer dengan prioritas tinggi
    NORMAL = 50     # Prioritas default
    LOW = 25        # Observer dengan prioritas rendah
    LOWEST = 0      # Observer yang harus dijalankan terakhir