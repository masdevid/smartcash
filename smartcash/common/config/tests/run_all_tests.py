"""
File: smartcash/common/config/tests/run_all_tests.py
Deskripsi: Script untuk menjalankan semua test konfigurasi
"""

import os
import sys
from pathlib import Path
import importlib

def run_all_config_tests():
    """
    Jalankan semua test konfigurasi
    """
    print("🧪 Menjalankan semua test konfigurasi...")
    
    # Dapatkan direktori saat ini
    current_dir = Path(__file__).parent
    
    # Daftar file test yang akan dijalankan
    test_files = [
        "test_config_manager.py",
        "test_colab_config.py",
        "test_simple_config.py"
    ]
    
    # Jalankan setiap file test
    for test_file in test_files:
        test_path = current_dir / test_file
        if test_path.exists():
            print(f"\n📂 Menjalankan test: {test_file}")
            print("=" * 50)
            
            # Import modul dan jalankan test
            module_name = test_file.replace(".py", "")
            try:
                # Import dan jalankan test
                full_module_name = f"smartcash.common.config.tests.{module_name}"
                if module_name in sys.modules:
                    # Reload module jika sudah di-import
                    module = importlib.reload(sys.modules[full_module_name])
                else:
                    # Import module jika belum di-import
                    module = importlib.import_module(full_module_name)
                
                # Jalankan test dengan fungsi run_tests
                if hasattr(module, "run_tests"):
                    module.run_tests()
                # Jalankan test dengan fungsi run_test
                elif hasattr(module, "run_test"):
                    module.run_test()
                else:
                    print(f"⚠️ Tidak dapat menemukan metode pengujian di {module_name}")
            except Exception as e:
                print(f"❌ Error saat menjalankan {test_file}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print("=" * 50)
        else:
            print(f"⚠️ File test tidak ditemukan: {test_path}")
    
    print("\n🎉 Semua test konfigurasi selesai!")

if __name__ == "__main__":
    run_all_config_tests() 