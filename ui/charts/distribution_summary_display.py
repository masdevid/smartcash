"""
File: smartcash/ui/charts/distribution_summary_display.py
Deskripsi: Utilitas untuk menampilkan ringkasan distribusi kelas dalam format tabel
"""

from typing import Dict
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS

def display_distribution_summary(
    all_counts: Dict[str, int],
    orig_counts: Dict[str, int], 
    aug_counts: Dict[str, int],
    target_count: int = 1000
) -> None:
    """
    Tampilkan ringkasan distribusi kelas dalam format tabel.
    
    Args:
        all_counts: Dictionary {class_id: jumlah_instance} semua data
        orig_counts: Dictionary {class_id: jumlah_instance} data asli
        aug_counts: Dictionary {class_id: jumlah_instance} data augmentasi
        target_count: Target jumlah instance per kelas
    """
    all_classes = sorted(set(all_counts) | set(orig_counts) | set(aug_counts))
    
    html_parts = [
        f'<div style="margin: 10px 0; padding: 10px; background-color: {COLORS.get("light", "#f8f9fa")}; border-radius: 5px;">',
        f'<h4 style="color: {COLORS.get("dark", "#2F58CD")};">{ICONS.get("stats", "📊")} Ringkasan Distribusi Kelas</h4>',
        '<style>',
        '.table-cell { padding: 8px; text-align: center; border: 1px solid #ddd; }',
        '.header { background-color: #3498db; color: white; }',
        '</style>',
        '<table style="width: 100%; border-collapse: collapse;">',
        '<thead><tr class="header">',
        '<th class="table-cell">Kelas</th>',
        '<th class="table-cell">Asli</th>',
        '<th class="table-cell">Augmentasi</th>',
        '<th class="table-cell">Total</th>',
        '<th class="table-cell">Target</th>',
        '<th class="table-cell">Status</th>',
        '</tr></thead><tbody>'
    ]
    
    success_color = COLORS.get("success", "#28a745")
    danger_color = COLORS.get("danger", "#dc3545")
    
    for cls in all_classes:
        orig = orig_counts.get(cls, 0)
        aug = aug_counts.get(cls, 0)
        total = all_counts.get(cls, 0)
        status = (
            f'<span style="color: {success_color};">✅ Tercapai</span>' 
            if total >= target_count 
            else f'<span style="color: {danger_color};">⚠️ Kurang {target_count - total}</span>'
        )
        
        html_parts.append(
            f'<tr>'
            f'<td class="table-cell">{cls}</td>'
            f'<td class="table-cell">{orig}</td>'
            f'<td class="table-cell">{aug}</td>'
            f'<td class="table-cell">{total}</td>'
            f'<td class="table-cell">{target_count}</td>'
            f'<td class="table-cell">{status}</td>'
            f'</tr>'
        )
    
    html_parts.extend(['</tbody></table></div>'])
    display(HTML(''.join(html_parts)))

def display_file_summary(prefix_counts: Dict[str, int], total_files: int = None) -> None:
    """
    Tampilkan ringkasan jumlah file berdasarkan prefix.
    
    Args:
        prefix_counts: Dictionary {prefix: jumlah_file}
        total_files: Total jumlah file (akan dihitung jika None)
    """
    total_files = total_files if total_files is not None else sum(prefix_counts.values())
    if not total_files:  # Handle kasus total_files = 0
        display(HTML('<p>Tidak ada data untuk ditampilkan</p>'))
        return
        
    html_parts = [
        f'<div style="margin: 10px 0; padding: 10px; background-color: {COLORS.get("light", "#f8f9fa")}; border-radius: 5px;">',
        f'<h4 style="color: {COLORS.get("dark", "#2F58CD")};">{ICONS.get("file", "📄")} Ringkasan Jumlah File</h4>',
        '<style>',
        '.table-cell { padding: 8px; text-align: center; border: 1px solid #ddd; }',
        '.header { background-color: #3498db; color: white; }',
        '.total-row { background-color: #f8f9fa; font-weight: bold; }',
        '</style>',
        '<table style="width: 100%; border-collapse: collapse;">',
        '<thead><tr class="header">',
        '<th class="table-cell">Prefix</th>',
        '<th class="table-cell">Jumlah File</th>',
        '<th class="table-cell">Persentase</th>',
        '</tr></thead><tbody>'
    ]
    
    for prefix in sorted(prefix_counts):
        count = prefix_counts[prefix]
        percentage = count / total_files * 100
        
        html_parts.append(
            f'<tr>'
            f'<td class="table-cell">{prefix}</td>'
            f'<td class="table-cell">{count}</td>'
            f'<td class="table-cell">{percentage:.1f}%</td>'
            f'</tr>'
        )
    
    html_parts.extend([
        '<tr class="total-row">',
        '<td class="table-cell">Total</td>',
        f'<td class="table-cell">{total_files}</td>',
        '<td class="table-cell">100%</td>',
        '</tr></tbody></table></div>'
    ])
    
    display(HTML(''.join(html_parts)))