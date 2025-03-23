"""
File: smartcash/ui/visualization/visualize_preprocessed_samples.py
Deskripsi: Utilitas untuk menampilkan sampel dataset yang telah dipreprocessing
"""

def visualize_preprocessed_samples(ui_components: Dict[str, Any], preprocessed_dir: str, original_dir: str, num_samples: int = 5):
    """
    Visualisasi sampel dataset yang telah dipreprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        original_dir: Direktori dataset mentah
        num_samples: Jumlah sampel yang akan divisualisasikan
    """
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
    output_widget = ui_components.get('status')
    if not output_widget:
        return
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mengambil sampel dari dataset..."))
        
        # Cari sampel dari train split
        train_dir = Path(preprocessed_dir) / 'train'
        if not train_dir.exists():
            # Coba split lain jika train tidak tersedia
            for split in ['valid', 'test']:
                split_dir = Path(preprocessed_dir) / split
                if split_dir.exists():
                    train_dir = split_dir
                    break
            else:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada split yang tersedia di {preprocessed_dir}"))
                return
        
        # Dapatkan sampel gambar
        images_dir = train_dir / 'images'
        if not images_dir.exists():
            display(create_status_indicator('warning', f"{ICONS['warning']} Direktori gambar tidak ditemukan di {train_dir}"))
            return
            
        # Ambil semua gambar
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.npy'))
        if not image_files:
            display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada file gambar ditemukan di {images_dir}"))
            return
            
        # Batasi jumlah sampel
        image_files = image_files[:min(num_samples, len(image_files))]
        
        # Tampilkan deskripsi dengan create_info_alert standar
        display(create_info_alert(
            f"Menampilkan {len(image_files)} sampel dataset yang telah dipreprocessing dari {train_dir.name}",
            "info"
        ))
        
        # Visualisasi sampel
        fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
        if len(image_files) == 1:
            axes = [axes]
            
        for i, img_path in enumerate(image_files):
            # Load gambar
            try:
                if img_path.suffix == '.npy':
                    # Handle numpy array preprocessed
                    img = np.load(str(img_path))
                    # Denormalisasi jika perlu
                    if img.dtype == np.float32 and img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                else:
                    # Handle gambar biasa
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Tampilkan gambar
                axes[i].imshow(img)
                # Tampilkan nama file yang pendek
                img_name = img_path.name
                if len(img_name) > 10:
                    img_name = f"...{img_name[-10:]}"
                axes[i].set_title(f"{img_name}")
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan informasi ukuran gambar
        for img_path in image_files:
            try:
                if img_path.suffix == '.npy':
                    img = np.load(str(img_path))
                    h, w = img.shape[:2]
                else:
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                
                # Tampilkan nama file yang pendek
                img_name = img_path.name
                if len(img_name) > 15:
                    img_name = f"...{img_name[-10:]}"
                
                display(HTML(f"<p style='color:{COLORS['dark']}'><strong>{img_name}</strong>: {w}x{h} piksel</p>"))
            except Exception:
                pass