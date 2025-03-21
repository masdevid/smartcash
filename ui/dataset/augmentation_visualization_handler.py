"""
File: smartcash/ui/dataset/augmentation_visualization_handler.py
Deskripsi: Update handler visualisasi untuk mendukung lokasi augmentasi baru
"""

def setup_visualization_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk visualisasi dataset augmentasi dengan dukungan lokasi baru."""
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi dengan fungsi konsolidasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='single')

    # Handler untuk tombol komparasi dengan fungsi konsolidasi
    def on_compare_click(b):
        """Handler untuk komparasi sampel dataset asli dengan yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='compare')
    
    # Setup handlers untuk tombol visualisasi
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'children') and len(visualization_buttons.children) >= 2:
        visualization_buttons.children[0].on_click(on_visualize_click)
        visualization_buttons.children[1].on_click(on_compare_click)
    
    # Tambahkan handlers ke komponen UI
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'visualize_dataset': visualize_dataset
    })
    
    return ui_components

def visualize_dataset(ui_components: Dict[str, Any], mode: str = 'single', num_samples: int = 5):
    """
    Fungsi visualisasi dataset yang mendukung lokasi baru augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        mode: Mode visualisasi ('single' untuk augmented saja, 'compare' untuk komparasi)
        num_samples: Jumlah sampel yang akan ditampilkan
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
    # Use the visualization container if available
    output_widget = ui_components.get('visualization_container', ui_components.get('status'))
    if not output_widget: return
    
    # Dapatkan lokasi dari konfigurasi
    config = ui_components.get('config', {})
    
    # Dapatkan lokasi preprocessed (untuk mencari hasil augmentasi)
    preprocessed_dir = config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
    
    # Dapatkan lokasi temp augmented (untuk backup)
    augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
    
    # Siapkan lokasi yang akan dicari (prioritaskan preprocessed/train)
    primary_path = Path(preprocessed_dir) / 'train' / 'images'
    secondary_path = Path(augmented_dir) / 'images'
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan {'komparasi' if mode == 'compare' else 'visualisasi'}..."))
        
        # Periksa lokasi mana yang memiliki file hasil augmentasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Cek apakah ada file augmentasi di primary path
        primary_aug_files = list(primary_path.glob(f"{aug_prefix}_*.jpg")) if primary_path.exists() else []
        
        # Jika tidak ada di primary path, cek di secondary path
        active_path = primary_path
        if not primary_aug_files and secondary_path.exists():
            secondary_aug_files = list(secondary_path.glob(f"{aug_prefix}_*.jpg"))
            if secondary_aug_files:
                active_path = secondary_path
                display(create_info_alert(f"Menggunakan lokasi augmentasi temporary: {active_path}", "info"))
        
        # Jika mode compare, dapatkan lokasi data asli
        if mode == 'compare':
            # Cek apakah ada file original di lokasi preprocessed
            original_path = Path(preprocessed_dir) / 'train' / 'images'
            if not original_path.exists():
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset untuk komparasi tidak ditemukan"))
                return
            
            # Gunakan visualisasi komparasi
            compare_original_vs_augmented(original_path, active_path, output_widget, ui_components, num_samples)
        else:
            # Visualisasi sampel augmented
            visualize_augmented_samples(active_path, output_widget, ui_components, num_samples)
        
        # Show visualization container
        if 'visualization_container' in ui_components:
            ui_components['visualization_container'].layout.display = 'block'

def find_label_path(img_path: Path) -> Optional[Path]:
    """
    Fungsi helper untuk mencari label path dari image path.
    
    Args:
        img_path: Path gambar
        
    Returns:
        Path file label atau None jika tidak ditemukan
    """
    # Cek apakah ada file label di folder paralel
    parent_dir = img_path.parent.parent
    label_path = parent_dir / 'labels' / f"{img_path.stem}.txt"
    
    if label_path.exists():
        return label_path
    
    # Cek apakah ada file label di folder sibling
    sibling_label_dir = img_path.parent.parent / 'labels'
    if sibling_label_dir.exists():
        sibling_label_path = sibling_label_dir / f"{img_path.stem}.txt"
        if sibling_label_path.exists():
            return sibling_label_path
    
    return None

def visualize_augmented_samples(images_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 5):
    """Visualisasi sampel dataset yang telah diaugmentasi dengan peningkatan tampilan nama file."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Get augmentation prefix
    aug_prefix = "aug"
    if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
        aug_prefix = ui_components['aug_options'].children[2].value
    
    # Ambil gambar augmentasi
    image_files = list(images_dir.glob(f'{aug_prefix}_*.jpg'))
    if not image_files:
        display(create_info_alert(f"Tidak ada file gambar augmentasi ditemukan di {images_dir}", "warning"))
        return
    
    # Batasi jumlah sampel
    image_files = image_files[:min(num_samples, len(image_files))]
    
    # Tampilkan deskripsi
    display(create_info_alert(f"Menampilkan {len(image_files)} sampel dataset yang telah diaugmentasi", "info"))
    
    # Visualisasi sampel
    fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
    if len(image_files) == 1: axes = [axes]
        
    for i, img_path in enumerate(image_files):
        try:
            img = load_image(img_path)
            axes[i].imshow(img)
            # Tampilkan nama file yang dipersingkat
            shortened_name = shorten_filename(img_path.name)
            axes[i].set_title(f"{shortened_name}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan informasi label jika tersedia
    labels = []
    for img_path in image_files:
        label_path = find_label_path(img_path)
        if label_path:
            labels.append((img_path, label_path))
    
    if labels:
        display_label_info(labels)

def compare_original_vs_augmented(original_dir: Path, augmented_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 3):
    """Komparasi sampel dataset asli dengan yang telah diaugmentasi."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Get augmentation prefix
    aug_prefix = "aug"
    if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
        aug_prefix = ui_components['aug_options'].children[2].value
    
    # Cari gambar augmentasi
    augmented_images = list(augmented_dir.glob(f'{aug_prefix}_*.jpg'))
    
    if not augmented_images:
        display(create_info_alert(f"Tidak ada file gambar augmentasi ditemukan di {augmented_dir}", "warning"))
        return
    
    # Ekstrak stem original dari nama file augmentasi (tanpa prefix)
    aug_to_orig = {}
    for aug_img in augmented_images:
        parts = aug_img.stem.split('_')
        if len(parts) > 2 and parts[0] == aug_prefix:
            orig_name_parts = []
            for i in range(1, len(parts)-1):  # Skip prefix dan random ID di akhir
                orig_name_parts.append(parts[i])
            orig_stem = '_'.join(orig_name_parts)
            aug_to_orig[aug_img] = orig_stem
    
    # Cari file original dengan prefix rp
    original_images = list(original_dir.glob('rp_*.jpg'))
    
    # Cari pasangan gambar
    matched_pairs = []
    for aug_img, orig_stem in aug_to_orig.items():
        for orig_img in original_images:
            if orig_img.stem.startswith('rp_' + orig_stem) or orig_stem in orig_img.stem:
                matched_pairs.append((orig_img, aug_img))
                if len(matched_pairs) >= num_samples:
                    break
        if len(matched_pairs) >= num_samples:
            break
    
    # Fallback jika tidak menemukan pasangan yang cocok
    if not matched_pairs:
        display(create_info_alert("Tidak menemukan pasangan gambar yang cocok. Menggunakan sampel acak...", "warning"))
        import random
        for i in range(min(num_samples, len(augmented_images))):
            if original_images and i < len(original_images):
                matched_pairs.append((original_images[i], augmented_images[i]))
            elif original_images:
                matched_pairs.append((random.choice(original_images), augmented_images[i]))
    
    if not matched_pairs:
        display(create_info_alert("Tidak dapat menemukan pasangan gambar untuk komparasi", "error"))
        return
    
    # Tampilkan deskripsi
    display(create_info_alert(f"Komparasi {len(matched_pairs)} sampel: asli vs augmentasi", "info"))
    
    # Visualisasi komparasi
    fig, axes = plt.subplots(len(matched_pairs), 2, figsize=(10, 4*len(matched_pairs)))
    if len(matched_pairs) == 1: axes = axes.reshape(1, 2)
        
    for i, (orig_path, aug_path) in enumerate(matched_pairs):
        try:
            # Load gambar
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            # Tampilkan gambar dengan nama file yang dipersingkat
            orig_name = shorten_filename(orig_path.name)
            aug_name = shorten_filename(aug_path.name)
            
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {orig_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_img)
            axes[i, 1].set_title(f"Augmented: {aug_name}")
            axes[i, 1].axis('off')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan info perbandingan
    for orig_path, aug_path in matched_pairs:
        try:
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            orig_h, orig_w = orig_img.shape[:2]
            aug_h, aug_w = aug_img.shape[:2]
            
            orig_size = os.path.getsize(orig_path)
            aug_size = os.path.getsize(aug_path)
            
            # Tampilkan nama file yang dipersingkat
            orig_name = shorten_filename(orig_path.stem, 20)
            aug_name = shorten_filename(aug_path.stem, 20)
            
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['dark']};"><strong>Original:</strong> {orig_name} | <strong>Augmented:</strong> {aug_name}</p>
                <p style="color:{COLORS['dark']};">Original: {orig_w}×{orig_h} px | Augmented: {aug_w}×{aug_h} px</p>
                <p style="color:{COLORS['dark']};">Size ratio: {aug_size/orig_size:.2f}× ({format_size(orig_size)} → {format_size(aug_size)})</p>
            </div>
            """))
        except Exception:
            pass