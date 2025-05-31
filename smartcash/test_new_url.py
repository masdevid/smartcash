#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file_path: /Users/masdevid/Projects/smartcash/smartcash/test_new_url.py
# Skrip pengujian untuk memverifikasi URL download model EfficientNet-B4 yang baru

import requests
from pathlib import Path
import time
import sys
from tqdm import tqdm

def test_url(url, description):
    """
    Menguji URL dengan melakukan HTTP request dan menampilkan status code dan pesan.
    
    Args:
        url (str): URL yang akan diuji
        description (str): Deskripsi URL untuk ditampilkan
    
    Returns:
        tuple: (status_code, success_flag)
    """
    print(f"\n🔍 Menguji {description}: {url}")
    
    try:
        # Gunakan stream=True untuk tidak mengunduh seluruh konten
        response = requests.get(url, stream=True, timeout=10)
        status_code = response.status_code
        
        if 200 <= status_code < 300:
            print(f"✅ Sukses! Status code: {status_code}")
            
            # Dapatkan ukuran file
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                print(f"📦 Ukuran file: {total_size / (1024*1024):.2f} MB")
            
            # Cek jika redirect terjadi
            if response.history:
                print(f"⚠️ Redirect terjadi dari {url} ke {response.url}")
            
            return status_code, True
        else:
            print(f"❌ Gagal! Status code: {status_code}")
            if status_code == 301 or status_code == 302:
                print(f"⚠️ Redirect ke: {response.headers.get('Location', 'Tidak diketahui')}")
            return status_code, False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 0, False

def download_sample(url, output_path, chunk_size=8192):
    """
    Mengunduh sebagian kecil dari file untuk memastikan konten dapat diunduh.
    
    Args:
        url (str): URL untuk diunduh
        output_path (str): Path untuk menyimpan file
        chunk_size (int): Ukuran chunk untuk download
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            # Hanya unduh maksimal 1MB untuk pengujian
            max_size = min(1024*1024, total_size)
            
            # Buat progress bar
            progress_bar = tqdm(total=max_size, unit='B', unit_scale=True, 
                               desc=f"Mengunduh sampel ke {output_path}")
            
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        if downloaded < max_size:
                            size_to_write = min(len(chunk), max_size - downloaded)
                            f.write(chunk[:size_to_write])
                            downloaded += size_to_write
                            progress_bar.update(size_to_write)
                        else:
                            break
            
            progress_bar.close()
            print(f"✅ Berhasil mengunduh sampel ({downloaded / (1024*1024):.2f} MB) ke {output_path}")
            return True
        else:
            print(f"❌ Gagal mengunduh: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error saat mengunduh: {str(e)}")
        return False

if __name__ == "__main__":
    # URL model yang diuji
    old_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_288-7934f29e.pth"
    new_url = "https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin"
    
    print("=" * 80)
    print("🧪 PENGUJIAN URL MODEL EFFICIENTNET-B4")
    print("=" * 80)
    
    # Uji URL lama
    old_status, old_success = test_url(old_url, "URL lama (rwightman)")
    
    # Uji URL baru
    new_status, new_success = test_url(new_url, "URL baru (Hugging Face)")
    
    # Jika URL baru berhasil, coba unduh sampel
    if new_success:
        print("\n🔽 Mencoba mengunduh sampel dari URL baru...")
        output_dir = Path("./test_downloads")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "efficientnet_b4_sample.bin"
        
        download_success = download_sample(new_url, output_path)
        
        if download_success:
            print(f"\n✅ Pengujian berhasil! URL baru dapat diakses dan diunduh.")
            print(f"📁 Sampel tersimpan di: {output_path}")
        else:
            print(f"\n❌ Pengujian gagal! URL baru dapat diakses tetapi tidak dapat diunduh.")
    else:
        print("\n❌ Pengujian gagal! URL baru tidak dapat diakses.")
    
    print("\n" + "=" * 80)
    print("📊 HASIL PENGUJIAN:")
    print(f"URL lama: {'✅ Berhasil' if old_success else '❌ Gagal'} (Status: {old_status})")
    print(f"URL baru: {'✅ Berhasil' if new_success else '❌ Gagal'} (Status: {new_status})")
    print("=" * 80)
