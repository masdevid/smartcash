#!/bin/bash

# File: clear-training-data.sh
# Author: Alfrida Sabar
# Deskripsi: Script untuk menghapus file training kecuali .gitkeep

# Direktori yang akan dibersihkan
DIRS=(
    "data/rupiah/train/images"
    "data/rupiah/train/labels"
    "data/rupiah/val/images"
    "data/rupiah/val/labels"
    "data/rupiah/test/images"
    "data/rupiah/test/labels"
)

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fungsi untuk membersihkan direktori
clear_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        # Hapus semua file kecuali .gitkeep
        find "$dir" -type f ! -name '.gitkeep' -delete
        echo -e "${GREEN}[✓]${NC} Direktori ${YELLOW}$dir${NC} dibersihkan"
    else
        echo -e "${RED}[✗]${NC} Direktori ${YELLOW}$dir${NC} tidak ditemukan"
    fi
}

# Konfirmasi dari pengguna
read -p "$(echo -e "${YELLOW}Apakah Anda yakin ingin menghapus semua file training? (y/n):${NC} ")" confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    # Proses pembersihan untuk setiap direktori
    for dir in "${DIRS[@]}"; do
        clear_directory "$dir"
    done
    
    echo -e "${GREEN}✨ Pembersihan data training selesai!${NC}"
else
    echo -e "${RED}❌ Operasi dibatalkan.${NC}"
fi

# Berikan izin eksekusi
chmod +x "$0"