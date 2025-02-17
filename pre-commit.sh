#!/bin/bash

# Warna untuk keluaran
MERAH='\033[0;31m'
HIJAU='\033[0;32m'
KUNING='\033[1;33m'
NC='\033[0m' # Tidak berwarna

# Periksa gaya penulisan Python dengan flake8
echo -e "${KUNING}🕵️ Menjalankan Pemeriksa Gaya Kode (flake8)...${NC}"
flake8 ./src --max-line-length=120 --exclude=.git,__pycache__,venv
if [ $? -ne 0 ]; then
    echo -e "${MERAH}❌ Pemeriksaan gaya kode gagal. Silakan perbaiki masalah gaya penulisan.${NC}"
    exit 1
fi

# Jalankan pemeriksaan tipe dengan mypy
echo -e "${KUNING}🔍 Menjalankan Pemeriksa Tipe (mypy)...${NC}"
mypy ./src --ignore-missing-imports
if [ $? -ne 0 ]; then
    echo -e "${MERAH}❌ Pemeriksaan tipe gagal. Silakan perbaiki anotasi tipe.${NC}"
    exit 1
fi

# Jalankan pengujian unit
echo -e "${KUNING}🧪 Menjalankan Pengujian Unit...${NC}"
python -m pytest tests/
if [ $? -ne 0 ]; then
    echo -e "${MERAH}❌ Pengujian gagal. Silakan perbaiki kegagalan pengujian.${NC}"
    exit 1
fi

# Periksa format pesan commit
BERKAS_PESAN_COMMIT=$1
PESAN_COMMIT=$(cat "$BERKAS_PESAN_COMMIT")

# Validasi format pesan commit
if ! [[ "$PESAN_COMMIT" =~ ^(fitur|perbaikan|dok|gaya|refaktor|uji|tugas)(\([a-z-]+\))?:\ .{10,} ]]; then
    echo -e "${MERAH}❌ Format pesan commit tidak valid!${NC}"
    echo -e "Pesan commit harus mengikuti format:"
    echo -e "  jenis(lingkup): pesan deskriptif"
    echo -e "\nJenis:"
    echo -e "  - fitur     : Fitur baru"
    echo -e "  - perbaikan : Perbaikan bug"
    echo -e "  - dok       : Perubahan dokumentasi"
    echo -e "  - gaya      : Pemformatan kode"
    echo -e "  - refaktor  : Refaktor kode"
    echo -e "  - uji       : Menambah atau mengubah pengujian"
    echo -e "  - tugas     : Tugas pemeliharaan"
    exit 1
fi

echo -e "${HIJAU}✅ Pemeriksaan pra-commit berhasil!${NC}"
exit 0