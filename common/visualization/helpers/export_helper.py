"""
File: smartcash/dataset/visualization/helpers/export_helper.py
Deskripsi: Utilitas untuk ekspor visualisasi ke berbagai format
"""

import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from matplotlib.figure import Figure

from smartcash.common.logger import get_logger


class ExportHelper:
    """Helper untuk ekspor visualisasi ke berbagai format."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi ExportHelper.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("export_helper")
        self.logger.info("💾 ExportHelper diinisialisasi")
    
    def save_figure(
        self, 
        fig: Figure, 
        output_path: Union[str, Path], 
        dpi: int = 300,
        format: Optional[str] = None,
        transparent: bool = False,
        close_fig: bool = True,
        create_dir: bool = True
    ) -> str:
        """
        Simpan figure matplotlib ke file.
        
        Args:
            fig: Figure matplotlib
            output_path: Path untuk menyimpan file
            dpi: Dots per inch untuk output
            format: Format file (png, pdf, svg, etc.)
            transparent: Apakah background transparan
            close_fig: Apakah menutup figure setelah disimpan
            create_dir: Apakah membuat direktori jika belum ada
            
        Returns:
            Path lengkap ke file yang disimpan
        """
        output_path = Path(output_path)
        
        # Buat direktori jika perlu
        if create_dir:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tentukan format dari ekstensi jika tidak disebutkan
        if format is None:
            format = output_path.suffix.lstrip('.')
            
            # Default ke PNG jika tidak ada ekstensi
            if not format:
                format = 'png'
                output_path = output_path.with_suffix(f'.{format}')
        
        # Simpan figure
        fig.savefig(
            output_path,
            dpi=dpi,
            format=format,
            bbox_inches='tight',
            transparent=transparent
        )
        
        # Tutup figure jika diminta
        if close_fig:
            plt.close(fig)
            
        self.logger.info(f"💾 Figure disimpan ke: {output_path}")
        return str(output_path)
    
    def figure_to_base64(
        self, 
        fig: Figure, 
        format: str = 'png', 
        dpi: int = 100,
        close_fig: bool = True
    ) -> str:
        """
        Konversi figure matplotlib ke string base64.
        
        Args:
            fig: Figure matplotlib
            format: Format gambar (png, jpg, svg)
            dpi: Dots per inch
            close_fig: Apakah menutup figure setelah konversi
            
        Returns:
            String base64 dari gambar
        """
        # Simpan gambar ke buffer in-memory
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        
        # Reset posisi buffer ke awal
        buf.seek(0)
        
        # Encode sebagai base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Tutup figure jika diminta
        if close_fig:
            plt.close(fig)
            
        # Format untuk embedding di HTML
        mime_type = f'image/{format}'
        if format == 'svg':
            mime_type = 'image/svg+xml'
            
        return f'data:{mime_type};base64,{img_str}'
    
    def save_as_html(
        self, 
        fig: Figure, 
        output_path: Union[str, Path],
        title: str = "Visualization",
        include_plotlyjs: bool = True,
        create_dir: bool = True,
        close_fig: bool = True
    ) -> str:
        """
        Simpan figure sebagai file HTML (Plotly jika tersedia, atau embedding gambar).
        
        Args:
            fig: Figure matplotlib
            output_path: Path untuk menyimpan file
            title: Judul untuk HTML
            include_plotlyjs: Apakah menyertakan library Plotly
            create_dir: Apakah membuat direktori jika belum ada
            close_fig: Apakah menutup figure setelah disimpan
            
        Returns:
            Path lengkap ke file yang disimpan
        """
        output_path = Path(output_path)
        
        # Buat direktori jika perlu
        if create_dir:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Tambahkan ekstensi .html jika belum ada
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
            
        # Coba simpan dengan Plotly jika tersedia
        try:
            from plotly.tools import mpl_to_plotly
            import plotly
            
            plotly_fig = mpl_to_plotly(fig)
            
            # Tambahkan judul
            plotly_fig.update_layout(title=title)
            
            # Simpan sebagai HTML
            plotly.offline.plot(
                plotly_fig,
                filename=str(output_path),
                auto_open=False,
                include_plotlyjs=include_plotlyjs
            )
            
            if close_fig:
                plt.close(fig)
                
            self.logger.info(f"💾 Figure disimpan sebagai HTML interaktif ke: {output_path}")
            
        except ImportError:
            # Fallback ke embedding gambar jika Plotly tidak tersedia
            img_base64 = self.figure_to_base64(fig, format='png', close_fig=close_fig)
            
            # Buat HTML sederhana
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="figure">
            <img src="{img_base64}" alt="{title}">
        </div>
    </div>
</body>
</html>"""
            
            # Simpan HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"💾 Figure disimpan sebagai HTML statis ke: {output_path}")
        
        return str(output_path)
    
    def save_as_dashboard(
        self, 
        figures: Dict[str, Figure], 
        output_path: Union[str, Path],
        title: str = "Dashboard",
        subtitle: str = "",
        layout: Optional[List[List[str]]] = None,
        create_dir: bool = True,
        close_figs: bool = True
    ) -> str:
        """
        Simpan beberapa figure sebagai dashboard HTML.
        
        Args:
            figures: Dictionary {id: Figure}
            output_path: Path untuk menyimpan file
            title: Judul dashboard
            subtitle: Subjudul dashboard
            layout: List of lists mendefinisikan layout grid
            create_dir: Apakah membuat direktori jika belum ada
            close_figs: Apakah menutup figure setelah disimpan
            
        Returns:
            Path lengkap ke file yang disimpan
        """
        output_path = Path(output_path)
        
        # Buat direktori jika perlu
        if create_dir:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Tambahkan ekstensi .html jika belum ada
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        
        # Konversi semua figure ke base64
        encoded_figures = {}
        for fig_id, fig in figures.items():
            encoded_figures[fig_id] = self.figure_to_base64(fig, close_fig=close_figs)
        
        # Gunakan layout default jika tidak ditentukan
        if layout is None:
            # Buat layout 2-column grid sederhana
            fig_ids = list(figures.keys())
            n_figures = len(fig_ids)
            
            layout = []
            row = []
            
            for i, fig_id in enumerate(fig_ids):
                row.append(fig_id)
                
                # Tutup row setelah 2 item
                if len(row) == 2 or i == n_figures - 1:
                    layout.append(row)
                    row = []
        
        # Buat konten HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 5px; }}
        h3 {{ color: #666; margin-top: 0; font-weight: normal; }}
        .row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
        .col {{ padding: 10px; box-sizing: border-box; }}
        .figure {{ background-color: white; padding: 15px; border-radius: 5px; text-align: center; height: 100%; }}
        img {{ max-width: 100%; }}
        .col-1 {{ width: 100%; }}
        .col-2 {{ width: 50%; }}
        .col-3 {{ width: 33.33%; }}
        
        @media (max-width: 768px) {{
            .col-2, .col-3 {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{title}</h1>
        <h3>{subtitle}</h3>
"""
        
        # Tambahkan rows dan columns
        for row in layout:
            html_content += '        <div class="row">\n'
            
            # Tentukan ukuran kolom
            col_class = "col-1" if len(row) == 1 else "col-2" if len(row) == 2 else "col-3"
            
            for fig_id in row:
                if fig_id in encoded_figures:
                    html_content += f'            <div class="col {col_class}">\n'
                    html_content += f'                <div class="figure">\n'
                    html_content += f'                    <img src="{encoded_figures[fig_id]}" alt="{fig_id}">\n'
                    html_content += f'                </div>\n'
                    html_content += f'            </div>\n'
                    
            html_content += '        </div>\n'
        
        # Tutup HTML
        html_content += """    </div>
</body>
</html>"""
        
        # Simpan HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"💾 Dashboard disimpan ke: {output_path}")
        return str(output_path)
    
    def create_output_directory(self, output_dir: Union[str, Path]) -> Path:
        """
        Buat direktori output jika belum ada.
        
        Args:
            output_dir: Path direktori yang akan dibuat
            
        Returns:
            Path direktori
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path