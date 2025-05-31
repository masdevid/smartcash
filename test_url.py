"""
File: test_url.py
Deskripsi: Skrip untuk menguji URL download model EfficientNet-B4
"""

import requests
import sys

def test_url(url):
    print(f"Menguji URL: {url}")
    try:
        response = requests.head(url)
        if response.status_code == 200:
            print(f"✅ URL valid! Status code: {response.status_code}")
            return True
        else:
            print(f"❌ URL tidak valid! Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error saat menguji URL: {str(e)}")
        return False

if __name__ == "__main__":
    # URL yang benar
    correct_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_288-7934f29e.pth"
    
    # URL yang salah (menggunakan huggingface)
    wrong_url = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_288-7934f29e.pth"
    
    print("=== Pengujian URL EfficientNet-B4 ===")
    print("\nMenguji URL yang benar:")
    correct_result = test_url(correct_url)
    
    print("\nMenguji URL yang salah:")
    wrong_result = test_url(wrong_url)
    
    print("\n=== Hasil Pengujian ===")
    print(f"URL rwightman: {'✅ Valid' if correct_result else '❌ Tidak Valid'}")
    print(f"URL huggingface: {'✅ Valid' if wrong_result else '❌ Tidak Valid'}")
    
    if not correct_result:
        print("\n⚠️ Perhatian: URL yang seharusnya benar ternyata tidak valid!")
        print("Kemungkinan ada masalah dengan koneksi atau URL telah berubah.")
        sys.exit(1)
