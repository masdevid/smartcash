# Preprocessing Refactor Progress

## 📋 Overview
Buat ulang preprocesing init, dan handlernya untuk mengintegrasikan dengan perubahan pada `logger.py`, `logger_bridge.py`, `ui_logger.py`, `environment.py` dan `config/manager.py`. Jangan rubah komponen yang UI yang sudah ada. Folder `smartcash/ui/dataset/preprocessing/components` hanya berisi UI components saja.

## ⚠️ Perhatian Khusus

1. **Concurrent.futures**: Semua async operations menggunakan ThreadPoolExecutor untuk Colab compatibility
2. **Confirmation Dialog**: Lakukan pengecekan data terlebih sebelum memulai proses preprocessing maupun cleanup.
3. **Symlink Safety**: Cleanup harus cek symlink augmentasi dan tidak menghapusnya
4. **Drive Persistence**: Data preprocessing harus disimpan di Drive untuk persistence
5. **2-Level Progress**: Overall progress + step progress untuk UX yang lebih baik
6. **Error Handling**: Graceful error handling dengan UI reset
7. **Resource Cleanup**: Proper cleanup untuk futures dan threads
8. **Context Separation**Pisahkan alur untuk logika UI dengan backend preprocessing `smartcash.dataset.preprocessing.*`. Adapun logika jembatan integrasi antar UI dan Backend bisa ditaruh di modul UI. 

## Code Generation
1. Pecah handlers-nya menjadi unit SRP file lebih kecil (micro) untuk menerapkan prinsip DRY supaya code footprint tetap minimal 
2. Good code coverage, reusable dan mudah dipelihara.
3. Jangan gunakan threading yang tidak disupport dilingkungan colab, gunakan concurrent.future sebagai alternatifnya.
4. Buat summary singkat tentang apa yang ditambahkan, diubah dan dihapus dalam bentuk `.md` file artefact sebagai referensi thread sebelumnya. Jangan ulangi penjelasan setelah dokumentasi dibuat.
5. Tidak perlu unit testing code maupun examples.