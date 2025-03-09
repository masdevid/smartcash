#!/usr/bin/env python
"""
File: run_app.py
Author: Alfrida Sabar
Deskripsi: Script untuk menjalankan aplikasi SmartCash dengan Streamlit.
           Menyediakan environment setup dan error handling.
"""

import os
import sys
import subprocess
import argparse

def setup_environment():
    """Setup environment untuk menjalankan aplikasi."""
    # Buat direktori yang diperlukan
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Cek jika requirements terpenuhi
    try:
        import streamlit
        import torch
        import numpy
        import pandas
        import yaml
        print("✅ Semua dependensi utama tersedia")
    except ImportError as e:
        print(f"❌ Dependensi tidak lengkap: {e}")
        print("🔄 Menginstall dependensi...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependensi berhasil diinstall")
        except subprocess.CalledProcessError:
            print("❌ Gagal menginstall dependensi. Silahkan jalankan:")
            print("   pip install -r requirements.txt")
            sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SmartCash App")
    parser.add_argument("--port", type=int, default=8501, help="Port untuk Streamlit")
    parser.add_argument("--share", action="store_true", help="Share aplikasi secara publik")
    return parser.parse_args()

def run_app(args):
    """Jalankan aplikasi Streamlit."""
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(args.port)
    ]
    
    if args.share:
        cmd.append("--server.headless=true")
        cmd.append("--server.enableCORS=false")
        cmd.append("--server.enableXsrfProtection=false")
    
    try:
        print(f"🚀 Memulai SmartCash App pada port {args.port}...")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Aplikasi dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error saat menjalankan aplikasi: {e}")
        sys.exit(1)

def main():
    """Entry point utama."""
    print("🔍 SmartCash: Deteksi Nilai Mata Uang Rupiah")
    setup_environment()
    args = parse_args()
    run_app(args)

if __name__ == "__main__":
    main()