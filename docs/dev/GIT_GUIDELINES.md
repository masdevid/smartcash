# Git Hooks & Panduan Pesan Commit untuk Detector SmartCash 🚦

## 🔧 Penyiapan Git Hooks

### Prasyarat
- Python 3.8+
- Pasang dependensi yang diperlukan:
  ```bash
  pip install flake8 mypy pytest
  ```
  atau
  ```bash
  pip install -r dev_requirements.txt
  ```

### Instalasi
1. Salin skrip `pre-commit` ke `.git/hooks/pre-commit`
   ```bash
   cp pre-commit.sh .git/hooks/pre-commit
   ```
2. Jadikan dapat dieksekusi:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

## 📝 Konvensi Pesan Commit

### Format
```
<jenis>: (<lingkup opsional>)<pesan deskriptif>
```

### Jenis
- `fitur` 🚀: Fitur baru atau peningkatan signifikan
- `perbaikan` 🐛: Perbaikan bug
- `dok` 📄: Pembaruan dokumentasi
- `gaya` 🎨: Pemformatan kode, tanpa perubahan kode produksi
- `refaktor` 🔧: Restrukturisasi kode tanpa mengubah fungsionalitas
- `uji` 🧪: Menambah atau mengubah pengujian
- `tugas` 🔨: Tugas pemeliharaan, pembaruan proses build

### Lingkup (Opsional)
Tentukan modul atau komponen yang terpengaruh, contoh:
- `fitur`: (backbone) Fitur baru di modul backbone
- `perbaikan`: (dataset) Perbaikan bug di penanganan dataset

### Contoh
```
git commit -m "fitur: (backbone) Tambahkan dukungan penskalaan kompound EfficientNet-B4"
git commit -m "perbaikan: (dataset) Selesaikan kebocoran memori di data loader"
git commit -m "dok: Perbarui petunjuk instalasi"
```

## 🚨 Pemeriksaan Pra-Commit
Hook pra-commit melakukan pemeriksaan berikut:
1. Linting Python (flake8)
2. Pengecekan Tipe (mypy)
3. Pengujian Unit (pytest)
4. Validasi Pesan Commit

### Mengabaikan Pemeriksaan (Gunakan Dengan Hati-Hati)
Jika Anda benar-benar perlu melewati pemeriksaan pra-commit:
```bash
git commit --no-verify -m "Pesan commit Anda"
```

## 💡 Praktik Terbaik
- Buat commit kecil dan terfokus
- Tulis pesan commit yang jelas dan deskriptif
- Jalankan pemeriksaan secara lokal sebelum push
- Gunakan awalan tipe yang bermakna

## 🤝 Pedoman Kontribusi
1. Buat cabang baru untuk perubahan Anda
2. Tulis kode yang bersih dan terdokumentasi
3. Tambah/perbarui pengujian untuk fungsionalitas baru
4. Pastikan semua pemeriksaan lolos sebelum membuat permintaan tarik