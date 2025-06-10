"""
File: update_yolov5_deps.py
Deskripsi: Script untuk update dependencies YOLOv5
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, cwd: str = None):
    """Jalankan command dan print output"""
    print(f"🚀 Menjalankan: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def main():
    print("🔄 Memperbarui dependencies YOLOv5...")
    
    # Update pip
    if not run_command("pip install --upgrade pip"):
        print("❌ Gagal mengupdate pip")
        return False
    
    # Uninstall package lama yang bermasalah
    run_command("pip uninstall -y yolov5")
    
    # Install package yang diperlukan
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.18.5",
        "matplotlib>=3.2.2",
        "pillow>=7.1.2",
        "pyyaml>=5.3.1",
        "seaborn>=0.11.0",
        "pandas>=1.1.4",
        "tqdm>=4.64.0",
        "tensorboard>=2.4.1"
    ]
    
    for pkg in packages:
        if not run_command(f"pip install {pkg}"):
            print(f"❌ Gagal menginstall {pkg}")
            return False
    
    print("✅ Update dependencies selesai!")
    return True

if __name__ == "__main__":
    if main():
        print("\n🎉 Proses update berhasil!")
        print("Silakan restart kernel/notebook Anda dan coba import YOLOv5 lagi.")
    else:
        print("\n❌ Terjadi kesalahan saat update dependencies.")
        sys.exit(1)
