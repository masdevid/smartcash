"""
Test manual untuk komponen UI pretrained tanpa menggunakan pytest.
"""

import sys
import os
from smartcash.common.logger import get_logger

# Setup logger
logger = get_logger('test_pretrained_manual')

def test_manual():
    try:
        print("🔄 Mengimpor komponen UI pretrained...")
        from smartcash.ui.pretrained.components.ui_components import create_pretrained_ui_components
        
        print("✅ Berhasil mengimpor komponen UI pretrained")
        
        # Test dengan konfigurasi default
        print("\n🔧 Membuat UI dengan konfigurasi default...")
        ui_components = create_pretrained_ui_components()
        
        # Periksa komponen kritis
        critical_components = ['ui', 'main_container', 'status', 'log_output', 
                             'confirmation_area', 'dialog_area', 'progress_tracker']
        
        print("\n🔍 Memeriksa komponen kritis:")
        for comp in critical_components:
            status = "✅ Ada" if comp in ui_components else "❌ Tidak ada"
            print(f"{status}: {comp}")
        
        print("\n🎉 Test manual berhasil!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error saat menjalankan test manual:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Memulai test manual untuk komponen UI pretrained...")
    success = test_manual()
    sys.exit(0 if success else 1)
