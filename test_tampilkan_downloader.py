"""
File: test_tampilkan_downloader.py
Deskripsi: File test untuk menampilkan downloader UI dan melihat error yang muncul
"""

import sys
import traceback
from IPython.display import display

def test_downloader():
    """Test fungsi untuk menampilkan downloader UI dengan detail error"""
    try:
        from smartcash.ui.dataset.downloader.downloader_init import initialize_downloader_ui
        
        print("🚀 Memulai inisialisasi downloader...")
        result = initialize_downloader_ui()
        
        if 'ui' in result:
            print(f"✅ UI berhasil dibuat! Keys dalam result: {list(result.keys())}")
            display(result['ui'])
            return True
        else:
            print(f"❌ UI gagal dibuat! Keys dalam result: {list(result.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Error saat inisialisasi downloader: {str(e)}")
        print("📋 Stack trace:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_downloader()
    print(f"\n{'✅ Test berhasil' if success else '❌ Test gagal'}")
