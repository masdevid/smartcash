#!/bin/bash

# Fungsi untuk membuat file dengan isi header
create_file_with_header() {
    local filename="$1.txt"
    local header="#$1"
    
    # Coba buat file dan tulis header
    if echo "$header" > "$filename"; then
        echo "Berhasil membuat: $filename"
    else
        echo "Gagal membuat: $filename"
        exit 1
    fi
}

# Pastikan direktori saat ini bisa ditulis
if [ ! -w . ]; then
    echo "Error: Direktori saat ini tidak bisa ditulis!"
    exit 1
fi

# Daftar cell dan sub-cell
create_file_with_header "Cell 10 - Project Setup"
create_file_with_header "Cell 11 - Repository Clone"
create_file_with_header "Cell 12 - Environment Configuration"
create_file_with_header "Cell 13 - Dependency Installation"

create_file_with_header "Cell 20 - Dataset Preparation"
create_file_with_header "Cell 21 - Dataset Download"
create_file_with_header "Cell 22 - Preprocessing"
create_file_with_header "Cell 23 - Split Configuration"
create_file_with_header "Cell 24 - Data Augmentation"

create_file_with_header "Cell 30 - Training Configuration"
create_file_with_header "Cell 31 - Backbone Selection"
create_file_with_header "Cell 32 - Model Hyperparameters"
create_file_with_header "Cell 33 - Layer Configuration"
create_file_with_header "Cell 34 - Training Strategy"

create_file_with_header "Cell 40 - Training Execution"
create_file_with_header "Cell 41 - Model Training"
create_file_with_header "Cell 42 - Performance Tracking"
create_file_with_header "Cell 43 - Checkpoint Management"
create_file_with_header "Cell 44 - Live Metrics Visualization"

create_file_with_header "Cell 50 - Model Evaluation"
create_file_with_header "Cell 51 - Performance Metrics"
create_file_with_header "Cell 52 - Comparative Analysis"
create_file_with_header "Cell 53 - Statistical Insights"
create_file_with_header "Cell 54 - Visualization"

create_file_with_header "Cell 60 - Model Prediction Interface"
create_file_with_header "Cell 61 - Image Upload Mode"
create_file_with_header "Cell 62 - Webcam Capture Mode"
create_file_with_header "Cell 63 - Model Selection"
create_file_with_header "Cell 64 - Real-time Detection"
create_file_with_header "Cell 65 - Results Visualization"
create_file_with_header "Cell 66 - Export Detection Results"

echo "Selesai membuat semua file."