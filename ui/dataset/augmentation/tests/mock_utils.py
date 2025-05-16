"""
File: smartcash/ui/dataset/augmentation/tests/mock_utils.py
Deskripsi: Utilitas untuk mocking modul eksternal dalam pengujian
"""

import sys
from unittest.mock import MagicMock

def mock_all_dependencies():
    """
    Mock semua dependensi eksternal yang dibutuhkan untuk pengujian.
    Harus dipanggil di awal file pengujian sebelum mengimpor modul lain.
    """
    mock_opencv()
    mock_matplotlib()
    mock_ipywidgets()
    mock_albumentations()
    mock_pandas()
    mock_numpy()

def mock_opencv():
    """
    Mock modul OpenCV (cv2) untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor cv2 diimpor.
    """
    # Buat mock untuk cv2
    mock_cv2 = MagicMock()
    
    # Tambahkan fungsi-fungsi yang sering digunakan
    mock_cv2.imread = MagicMock(return_value=MagicMock())
    mock_cv2.imwrite = MagicMock(return_value=True)
    mock_cv2.resize = MagicMock(return_value=MagicMock())
    mock_cv2.cvtColor = MagicMock(return_value=MagicMock())
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_RGB2BGR = 4
    
    # Tambahkan ke sys.modules
    sys.modules['cv2'] = mock_cv2
    
    return mock_cv2

def mock_matplotlib():
    """
    Mock modul matplotlib untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor matplotlib diimpor.
    """
    # Buat mock untuk matplotlib
    mock_mpl = MagicMock()
    mock_plt = MagicMock()
    
    # Tambahkan fungsi-fungsi yang sering digunakan
    mock_plt.figure = MagicMock(return_value=MagicMock())
    mock_plt.subplot = MagicMock(return_value=MagicMock())
    mock_plt.imshow = MagicMock()
    mock_plt.title = MagicMock()
    mock_plt.axis = MagicMock()
    mock_plt.show = MagicMock()
    mock_plt.savefig = MagicMock()
    
    # Tambahkan ke sys.modules
    sys.modules['matplotlib'] = mock_mpl
    sys.modules['matplotlib.pyplot'] = mock_plt
    
    return mock_plt

def mock_ipywidgets():
    """
    Mock modul ipywidgets untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor ipywidgets diimpor.
    """
    # Buat mock untuk ipywidgets
    mock_widgets = MagicMock()
    
    # Tambahkan widget yang sering digunakan
    mock_widgets.HBox = MagicMock(return_value=MagicMock())
    mock_widgets.VBox = MagicMock(return_value=MagicMock())
    mock_widgets.Button = MagicMock(return_value=MagicMock())
    mock_widgets.Checkbox = MagicMock(return_value=MagicMock())
    mock_widgets.Dropdown = MagicMock(return_value=MagicMock())
    mock_widgets.RadioButtons = MagicMock(return_value=MagicMock())
    mock_widgets.Output = MagicMock(return_value=MagicMock())
    mock_widgets.Text = MagicMock(return_value=MagicMock())
    mock_widgets.IntSlider = MagicMock(return_value=MagicMock())
    mock_widgets.FloatSlider = MagicMock(return_value=MagicMock())
    mock_widgets.Layout = MagicMock(return_value=MagicMock())
    
    # Tambahkan ke sys.modules
    sys.modules['ipywidgets'] = mock_widgets
    
    return mock_widgets

def mock_albumentations():
    """
    Mock modul albumentations untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor albumentations diimpor.
    """
    # Buat mock untuk albumentations
    mock_albu = MagicMock()
    
    # Tambahkan fungsi-fungsi yang sering digunakan
    mock_albu.Compose = MagicMock(return_value=MagicMock())
    mock_albu.HorizontalFlip = MagicMock(return_value=MagicMock())
    mock_albu.VerticalFlip = MagicMock(return_value=MagicMock())
    mock_albu.Rotate = MagicMock(return_value=MagicMock())
    mock_albu.RandomBrightnessContrast = MagicMock(return_value=MagicMock())
    
    # Tambahkan ke sys.modules
    sys.modules['albumentations'] = mock_albu
    
    return mock_albu

def mock_pandas():
    """
    Mock modul pandas untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor pandas diimpor.
    """
    # Buat mock untuk pandas
    mock_pd = MagicMock()
    
    # Tambahkan fungsi-fungsi yang sering digunakan
    mock_pd.DataFrame = MagicMock(return_value=MagicMock())
    mock_pd.Series = MagicMock(return_value=MagicMock())
    mock_pd.read_csv = MagicMock(return_value=MagicMock())
    
    # Tambahkan ke sys.modules
    sys.modules['pandas'] = mock_pd
    
    return mock_pd

def mock_numpy():
    """
    Mock modul numpy untuk pengujian.
    Harus dipanggil sebelum modul yang mengimpor numpy diimpor.
    """
    # Buat mock untuk numpy
    mock_np = MagicMock()
    
    # Tambahkan fungsi-fungsi yang sering digunakan
    mock_np.array = MagicMock(return_value=MagicMock())
    mock_np.zeros = MagicMock(return_value=MagicMock())
    mock_np.ones = MagicMock(return_value=MagicMock())
    mock_np.random = MagicMock()
    mock_np.random.rand = MagicMock(return_value=MagicMock())
    
    # Tambahkan ke sys.modules
    sys.modules['numpy'] = mock_np
    
    return mock_np
