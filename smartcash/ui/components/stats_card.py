"""
file_path: smartcash/ui/components/stats_card.py

Komponen untuk menampilkan statistik dalam bentuk kartu dashboard.
"""
from typing import Dict, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML

class StatsCard:
    """Komponen kartu statistik untuk dashboard."""
    
    def __init__(self, title: str, icon: str = "📊"):
        """Inisialisasi kartu statistik.
        
        Args:
            title: Judul kartu
            icon: Ikon untuk judul kartu
        """
        self.title = title
        self.icon = icon
        self._value = 0
        self._subtitle = ""
        self._progress = 0.0
        
        # Buat elemen UI
        self._create_widgets()
    
    def _create_widgets(self):
        """Buat widget untuk kartu statistik."""
        # Style untuk kartu
        card_style = """
        .stats-card {
            border-left: 4px solid #3498db;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-card h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-weight: 600;
        }
        .stats-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 5px 0;
        }
        .stats-subtitle {
            font-size: 12px;
            color: #7f8c8d;
            margin: 5px 0;
        }
        .progress-container {
            height: 4px;
            background: #ecf0f1;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #3498db;
            width: 0%;
            transition: width 0.3s ease;
        }
        """
        
        # Tambahkan style ke notebook
        display(HTML(f'<style>{card_style}</style>'))
        
        # Buat widget HTML untuk kartu
        self.card = widgets.HTML()
        self.update()
    
    def update(self, value: Optional[int] = None, subtitle: str = "", progress: float = 0.0):
        """Perbarui nilai dan tampilan kartu.
        
        Args:
            value: Nilai utama yang akan ditampilkan
            subtitle: Subjudul/keterangan tambahan
            progress: Nilai progress (0.0 - 1.0)
        """
        if value is not None:
            self._value = value
        if subtitle:
            self._subtitle = subtitle
        if 0 <= progress <= 1.0:
            self._progress = progress
        
        # Format nilai dengan pemisah ribuan
        formatted_value = f"{self._value:,}".replace(",", ".")
        
        # Buat HTML untuk kartu
        progress_width = int(self._progress * 100)
        html = f"""
        <div class="stats-card">
            <h4>{self.icon} {self.title}</h4>
            <div class="stats-value">{formatted_value}</div>
            <div class="stats-subtitle">{self._subtitle}</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress_width}%;"></div>
            </div>
        </div>
        """
        
        self.card.value = html
    
    def get_widget(self):
        """Dapatkan widget kartu."""
        return self.card
    
    def __repr__(self):
        return f"<StatsCard: {self.title}>"


def create_dashboard_cards() -> Dict[str, Tuple[widgets.Widget, StatsCard]]:
    """Buat set lengkap kartu dashboard untuk dataset.
    
    Returns:
        Dictionary berisi widget dan instance StatsCard untuk setiap split
    """
    # Buat container untuk baris kartu
    cards_row = widgets.HBox(
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='space-between',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Buat kartu untuk setiap split
    splits = ["Train", "Validation", "Test", "Total"]
    cards = {}
    
    for split in splits:
        card = StatsCard(title=split)
        cards[split.lower()] = (widgets.VBox([card.get_widget()], layout=widgets.Layout(width='24%')), card)
    
    # Tambahkan kartu ke dalam baris
    cards_row.children = [cards[split.lower()][0] for split in splits]
    
    return {"container": cards_row, "cards": {k: v[1] for k, v in cards.items()}}
