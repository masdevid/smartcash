#!/bin/bash

# File: project-tree.sh
# Author: Alfrida Sabar
# Deskripsi: Script untuk menyalin dan mengatur ulang dataset Rupiah

# Warna untuk output console
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fungsi untuk mencetak pesan dengan warna
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Lokasi direktori sumber dan tujuan
SOURCE_DIR="data/rupiah_baru"
DESTINATION_DIR="data/rupiah"

# Parse argumen
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--source)
            SOURCE_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            print_error "Argumen tidak dikenal: $1"
            exit 1
            ;;
    esac
done

# Periksa apakah direktori sumber ditentukan
if [ -z "$SOURCE_DIR" ]; then
    print_error "Harap tentukan direktori sumber dengan -s atau --source"
    exit 1
fi

# Buat direktori tujuan jika belum ada
mkdir -p "$DESTINATION_DIR/train/images"
mkdir -p "$DESTINATION_DIR/train/labels"
mkdir -p "$DESTINATION_DIR/val/images"
mkdir -p "$DESTINATION_DIR/val/labels"
mkdir -p "$DESTINATION_DIR/test/images"
mkdir -p "$DESTINATION_DIR/test/labels"

# Salin gambar dan label untuk training
print_status "Menyalin gambar dan label training..."
cp "$SOURCE_DIR/train/images/"* "$DESTINATION_DIR/train/images/" 2>/dev/null
cp "$SOURCE_DIR/train/labels/"* "$DESTINATION_DIR/train/labels/" 2>/dev/null

# Salin gambar dan label validasi (konversi dari "valid" ke "val")
print_status "Menyalin gambar dan label validasi..."
cp "$SOURCE_DIR/valid/images/"* "$DESTINATION_DIR/val/images/" 2>/dev/null
cp "$SOURCE_DIR/valid/labels/"* "$DESTINATION_DIR/val/labels/" 2>/dev/null

# Salin gambar dan label testing
print_status "Menyalin gambar dan label testing..."
cp "$SOURCE_DIR/test/images/"* "$DESTINATION_DIR/test/images/" 2>/dev/null
cp "$SOURCE_DIR/test/labels/"* "$DESTINATION_DIR/test/labels/" 2>/dev/null

# Buat file konfigurasi YAML
print_status "Membuat file konfigurasi rupiah.yaml..."
cat > "$DESTINATION_DIR/rupiah.yaml" << EOL
# Dataset configuration for Rupiah Banknote Detection

# Train and validation paths
train: $DESTINATION_DIR/train/images
val: $DESTINATION_DIR/val/images
test: $DESTINATION_DIR/test/images

# Number of classes
nc: 7

# Class names
names: ['1000', '2000', '5000', '10000', '20000', '50000', '100000']

# Optional: Additional dataset information
description: 'SmartCash Rupiah Banknote Detection Dataset'
version: '1.0'
EOL

# Verifikasi jumlah file
train_images=$(ls "$DESTINATION_DIR/train/images" | wc -l)
train_labels=$(ls "$DESTINATION_DIR/train/labels" | wc -l)
val_images=$(ls "$DESTINATION_DIR/val/images" | wc -l)
val_labels=$(ls "$DESTINATION_DIR/val/labels" | wc -l)
test_images=$(ls "$DESTINATION_DIR/test/images" | wc -l)
test_labels=$(ls "$DESTINATION_DIR/test/labels" | wc -l)

echo ""
print_status "Ringkasan Dataset:"
echo -e "Training:   ${YELLOW}$train_images gambar${NC}, ${YELLOW}$train_labels label${NC}"
echo -e "Validasi:   ${YELLOW}$val_images gambar${NC}, ${YELLOW}$val_labels label${NC}"
echo -e "Testing:    ${YELLOW}$test_images gambar${NC}, ${YELLOW}$test_labels label${NC}"

# Berikan izin eksekusi
chmod +x "$0"

print_status "Penyalinan dan pengaturan ulang dataset selesai!"