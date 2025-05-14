"""
File: smartcash/model/services/pretrained_setup.py
Deskripsi: Fungsi setup untuk model pre-trained
"""

from smartcash.model.services.pretrained_downloader import PretrainedModelDownloader

def setup_pretrained_models(models_dir='/content/models'):
    """
    Download dan setup model pre-trained YOLOv5 dan EfficientNet-B4.
    
    Args:
        models_dir: Direktori untuk menyimpan model
        
    Returns:
        Dict berisi informasi model
    """
    # Inisialisasi downloader
    downloader = PretrainedModelDownloader(models_dir=models_dir)
    
    # Download semua model yang diperlukan
    print("🚀 Memulai download model pre-trained...")
    
    # Download YOLOv5
    try:
        print("\n📥 Memeriksa YOLOv5s...")
        yolo_info = downloader.download_yolov5()
        print(f"✅ Model YOLOv5s tersedia di {yolo_info['path']}")
    except Exception as e:
        print(f"❌ Gagal memproses YOLOv5s: {str(e)}")

    
    # Download EfficientNet-B4
    try:
        print("\n📥 Memeriksa EfficientNet-B4...")
        efficientnet_info = downloader.download_efficientnet()
        print(f"✅ Model EfficientNet-B4 tersedia di {efficientnet_info['path']}")
    except Exception as e:
        print(f"❌ Gagal memproses EfficientNet-B4: {str(e)}")

    
    # Tampilkan ringkasan informasi model
    model_info = downloader.get_model_info()
    print("\n✨ Proses download selesai!\n")
    
    if 'yolov5s' in model_info['models'] and 'efficientnet_b4' in model_info['models']:
        print("Ringkasan model yang tersedia:")
        print(f"- YOLOv5s: {model_info['models']['yolov5s']['size_mb']} MB")
        print(f"- EfficientNet-B4: {model_info['models']['efficientnet_b4']['size_mb']} MB")
    
    # Kembalikan informasi model
    return model_info
