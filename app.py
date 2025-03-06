#!/usr/bin/env python
"""
SmartCash: Deteksi Nilai Mata Uang Rupiah

Entry point utama untuk aplikasi SmartCash dengan antarmuka pengguna grafis.
"""

import os
import sys
import argparse
import logging
import gradio as gr
from datetime import datetime

# Import UI components
from ui_components import (
    create_training_ui, create_evaluation_ui, create_model_playground_ui,
    create_dataset_ui, create_model_manager_ui, create_research_ui,
    create_directory_ui, create_repository_ui, create_augmentation_ui,
    create_data_handling_ui, create_global_config_ui
)

# Import UI handlers
from handlers.ui_handlers import (
    setup_training_handlers, setup_evaluation_handlers, setup_model_playground_handlers,
    setup_dataset_handlers, setup_model_initialization_handlers, setup_model_visualizer_handlers,
    setup_checkpoint_manager_handlers, setup_model_optimization_handlers, setup_model_exporter_handlers,
    setup_directory_handlers, setup_repository_handlers, setup_research_handlers,
    setup_augmentation_handlers, setup_dataset_info_handlers, setup_split_dataset_handlers,
    setup_global_config_handlers
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", f"smartcash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger(__name__)

def setup_dirs():
    """Buat direktori yang diperlukan jika belum ada."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)

def parse_args():
    """Parse argumen baris perintah."""
    parser = argparse.ArgumentParser(description="SmartCash: Deteksi Nilai Mata Uang Rupiah")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host untuk aplikasi web")
    parser.add_argument("--port", type=int, default=7860, help="Port untuk aplikasi web")
    parser.add_argument("--share", action="store_true", help="Berbagi aplikasi secara publik")
    parser.add_argument("--debug", action="store_true", help="Tampilkan pesan debug")
    parser.add_argument("--theme", type=str, default="default", help="Tema UI (default, dark)")
    return parser.parse_args()

def main():
    """Entry point utama aplikasi."""
    args = parse_args()
    setup_dirs()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Inisialisasi SmartCash App...")
    
    # Buat aplikasi dengan tab
    with gr.Blocks(theme=args.theme, title="SmartCash: Deteksi Nilai Mata Uang Rupiah") as app:
        gr.Markdown("# 🔍 SmartCash: Deteksi Nilai Mata Uang Rupiah")
        gr.Markdown("Sistem deteksi nilai mata uang Rupiah menggunakan YOLOv5 dengan berbagai backbone")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("📁 Setup & Dataset"):
                setup_tab = gr.Group()
                with setup_tab:
                    dir_ui = create_directory_ui()
                    dataset_ui = create_dataset_ui()
                    data_handling_ui = create_data_handling_ui()
                    
                    # Setup handlers
                    setup_directory_handlers(dir_ui)
                    setup_dataset_handlers(dataset_ui)
                    setup_dataset_info_handlers(data_handling_ui)
                    setup_split_dataset_handlers(data_handling_ui)
            
            with gr.TabItem("🔧 Augmentasi Data"):
                augmentation_tab = gr.Group()
                with augmentation_tab:
                    augmentation_ui = create_augmentation_ui()
                    
                    # Setup handlers
                    setup_augmentation_handlers(augmentation_ui)
            
            with gr.TabItem("⚙️ Konfigurasi"):
                config_tab = gr.Group()
                with config_tab:
                    config_ui = create_global_config_ui()
                    
                    # Setup handlers
                    setup_global_config_handlers(config_ui)
            
            with gr.TabItem("🛠️ Model Manager"):
                model_tab = gr.Group()
                with model_tab:
                    model_ui = create_model_manager_ui()
                    
                    # Setup handlers
                    setup_model_initialization_handlers(model_ui)
                    setup_model_visualizer_handlers(model_ui)
                    setup_checkpoint_manager_handlers(model_ui)
                    setup_model_optimization_handlers(model_ui)
                    setup_model_exporter_handlers(model_ui)
            
            with gr.TabItem("🏋️ Training"):
                training_tab = gr.Group()
                with training_tab:
                    training_ui = create_training_ui()
                    
                    # Setup handlers
                    setup_training_handlers(training_ui)
            
            with gr.TabItem("📊 Evaluation"):
                evaluation_tab = gr.Group()
                with evaluation_tab:
                    evaluation_ui = create_evaluation_ui()
                    
                    # Setup handlers
                    setup_evaluation_handlers(evaluation_ui)
            
            with gr.TabItem("🔬 Research"):
                research_tab = gr.Group()
                with research_tab:
                    research_ui = create_research_ui()
                    
                    # Setup handlers
                    setup_research_handlers(research_ui)
            
            with gr.TabItem("🎮 Playground"):
                playground_tab = gr.Group()
                with playground_tab:
                    playground_ui = create_model_playground_ui()
                    
                    # Setup handlers
                    setup_model_playground_handlers(playground_ui)
            
            with gr.TabItem("📚 Repository"):
                repo_tab = gr.Group()
                with repo_tab:
                    repo_ui = create_repository_ui()
                    
                    # Setup handlers
                    setup_repository_handlers(repo_ui)
            
            with gr.TabItem("ℹ️ About"):
                about_tab = gr.Group()
                with about_tab:
                    gr.Markdown("""
                    # SmartCash: Deteksi Nilai Mata Uang Rupiah
                    
                    SmartCash adalah sistem deteksi nilai mata uang Rupiah yang menggunakan YOLOv5 dengan berbagai backbone, termasuk EfficientNet. 
                    Sistem ini dioptimasi untuk akurasi tinggi dalam berbagai kondisi pengambilan gambar.
                    
                    ## Fitur Utama
                    - Deteksi nilai mata uang Rupiah
                    - Dukungan berbagai backbone model
                    - Manajemen dan augmentasi dataset
                    - Pelatihan dan evaluasi model
                    - Antarmuka penelitian untuk eksperimen
                    - Playground untuk pengujian model
                    
                    ## Lisensi / License
                    Project ini dilisensikan di bawah [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
                    """)
    
    # Launch aplikasi
    logger.info(f"Launching SmartCash App at {args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()