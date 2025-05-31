#!/bin/bash
# File: /Users/masdevid/Projects/smartcash/run_preprocessing_tests.sh
# Script untuk menjalankan test suite preprocessing di conda environment smartcash_test

# Aktifkan conda environment
echo "🚀 Mengaktifkan conda environment smartcash_test..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate smartcash_test || { echo "❌ Gagal mengaktifkan environment smartcash_test"; exit 1; }

# Jalankan test
echo "🧪 Menjalankan test suite preprocessing..."
cd /Users/masdevid/Projects/smartcash
python -m unittest smartcash.ui.evaluation.tests.test_preprocessing

# Tampilkan hasil
if [ $? -eq 0 ]; then
    echo "✅ Semua test berhasil!"
else
    echo "❌ Ada test yang gagal."
fi

# Deaktifkan conda environment
conda deactivate
