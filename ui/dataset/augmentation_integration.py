"""
File: smartcash/ui/dataset/augmentation_integration.py
Deskripsi: File utama untuk integrasi perbaikan class balancer dengan modul augmentasi
"""

def setup_augmentation_integration():
    """Setup integrasi perbaikan class balancer dan visualisasi kelas ke dalam modul augmentasi."""
    try:
        # Import UI components augmentasi
        from smartcash.ui.dataset.augmentation import setup_augmentation
        
        # Simpan referensi ke setup_augmentation original
        original_setup_augmentation = setup_augmentation
        
        # Definisikan fungsi setup_augmentation yang baru dengan integrasi class balancer dan visualisasi
        def enhanced_setup_augmentation():
            """Versi yang ditingkatkan dari setup_augmentation dengan class balancer dan visualisasi yang diperbaiki."""
            # Panggil setup_augmentation original untuk mendapatkan ui_components dasar
            ui_components = original_setup_augmentation()
            
            try:
                # Import modul integrasi
                from smartcash.ui.dataset.augmentation_visualization_integration import update_augmentation_handler
                
                # Update ui_components dengan integrasi class balancer dan visualisasi
                ui_components = update_augmentation_handler(ui_components)
                
                # Log success jika berhasil
                if 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].info("✅ Integrasi class balancer dan visualisasi berhasil")
            except Exception as e:
                # Log error jika gagal
                if 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].warning(f"⚠️ Error integrasi: {str(e)}")
            
            # Kembalikan ui_components yang sudah diupdate
            return ui_components
        
        # Ganti fungsi original dengan versi yang ditingkatkan
        import sys
        sys.modules['smartcash.ui.dataset.augmentation'].setup_augmentation = enhanced_setup_augmentation
        
        return True
    except Exception as e:
        print(f"⚠️ Gagal setup integrasi augmentasi: {str(e)}")
        return False

# Jalankan setup integrasi
success = setup_augmentation_integration()
if success:
    print("✅ Integrasi berhasil")
else:
    print("❌ Integrasi gagal")