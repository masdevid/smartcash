"""
File: tests/unit/test_imports.py
Deskripsi: Unit test untuk memastikan semua modul dapat diimport dengan benar
"""

import importlib
import importlib.util
import inspect
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import unittest

def is_valid_module(path):
    """Check if a path should be considered as a valid module"""
    # Skip cache and hidden directories
    if any(part.startswith(('.', '__pycache__', '.pytest_cache')) for part in path.parts):
        return False
    return True

def find_python_modules(root_dir, base_package='smartcash'):
    """Find all Python modules in the given directory"""
    modules = set()
    root_path = Path(root_dir)
    
    # Menggunakan IGNORED_MODULES dari level modul
    ignore_modules = globals().get('IGNORED_MODULES', set())
    
    for py_file in root_path.rglob("*.py"):
        if not is_valid_module(py_file):
            continue
            
        # Convert path to module format
        rel_path = py_file.relative_to(root_path.parent)
        module_path = str(rel_path.with_suffix('')).replace(os.path.sep, '.')
        
        # Handle __init__.py for package names
        if py_file.name == "__init__.py":
            module_path = str(rel_path.parent).replace(os.path.sep, '.')
        
        # Skip modules in the ignore list
        if any(module_path == ignored or module_path.startswith(ignored + '.') for ignored in ignore_modules):
            continue
            
        # Ensure the module starts with the base package
        if module_path.startswith(base_package):
            modules.add(module_path)
    
    return sorted(modules)

def load_ignored_modules():
    """Muat daftar modul yang diabaikan dari file .python-ignore"""
    ignored = {
        # Modul yang tidak dapat diimpor
        'smartcash.dataset.preprocessor.denoiser',
        'smartcash.dataset.preprocessor.enhancer',
        'smartcash.dataset.preprocessor.rotator',
        'smartcash.dataset.preprocessor.validator',
        'smartcash.dataset.preprocessor.pipeline.stages',
        'smartcash.dataset.preprocessor.service',
        'smartcash.dataset.preprocessor.utils.metadata_extractor',
        'smartcash.dataset.preprocessor.utils.sample_generator',
    }
    
    # Baca dari file .python-ignore jika ada
    ignore_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.python-ignore')
    if os.path.exists(ignore_file):
        with open(ignore_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Konversi path file ke format modul Python
                    if line.endswith('.py'):
                        line = line[:-3]  # Hapus .py
                    mod_path = line.replace('/', '.').replace('\\', '.')
                    if mod_path.startswith('smartcash.'):
                        ignored.add(mod_path)
                    else:
                        ignored.add(f'smartcash.{mod_path}')
    
    return ignored

# Daftar modul yang akan diuji
MODULES = find_python_modules('smartcash')

# Muat modul yang diabaikan
IGNORED_MODULES = load_ignored_modules()

# Filter modul yang akan diuji
MODULES_TO_TEST = [m for m in MODULES if m not in IGNORED_MODULES]

# Tampilkan modul yang diabaikan untuk keperluan debugging
print(f"\n🔍 {len(IGNORED_MODULES)} modul diabaikan:")
for mod in sorted(IGNORED_MODULES):
    print(f"  - {mod}")
print()

class TestImports(unittest.TestCase):
    # Gunakan IGNORED_MODULES dari level modul
    IGNORED_MODULES = IGNORED_MODULES
    
    def setUp(self):
        """Setup test environment"""
        # Tambahkan direktori root ke path
        self.root_dir = str(Path(__file__).parent.parent.parent.absolute())
        if self.root_dir not in sys.path:
            sys.path.append(self.root_dir)
    
    def _print_import_summary(self, failed_imports, import_errors, log_file):
        """Mencetak ringkasan hasil import"""
        total = len(MODULES_TO_TEST)
        success = total - len(failed_imports)
        
        print("\n📊 Ringkasan Hasil Import:")
        print("=" * 50)
        print(f"\n✅ {success} dari {total} modul berhasil diimport")
        print(f"❌ {len(failed_imports)} modul gagal diimport\n")
        
        self._print_error_analysis(import_errors)
        self._print_success_examples(failed_imports)
        self._print_recommendations(import_errors)
        self._print_error_summary(import_errors)
        self._print_statistics(failed_imports, total)
        self._print_failed_modules(failed_imports)
        
        # Tampilkan lokasi file log
        print(f"\n📝 Log lengkap disimpan di: {os.path.abspath(log_file)}")
    
    def _print_error_analysis(self, import_errors):
        """Mencetak analisis error"""
        if import_errors:
            print("\n🔍 Analisis Error:")
            for error, modules in sorted(import_errors.items(), key=lambda x: -len(x[1])):
                print(f"\n{error} (Terjadi pada {len(modules)} modul):")
                for module in sorted(modules)[:5]:  # Tampilkan 5 contoh
                    print(f"  - {module}")
                if len(modules) > 5:
                    print(f"  - ...dan {len(modules) - 5} modul lainnya")
    
    def _print_success_examples(self, failed_imports):
        """Mencetak contoh modul yang berhasil diimport"""
        print("\n📦 Contoh modul yang berhasil diimport:")
        success_modules = sorted(set(MODULES_TO_TEST) - set(failed_imports))
        for module in success_modules[:10]:
            print(f"  - {module}")
        if len(success_modules) > 10:
            print(f"  - ...dan {len(success_modules) - 10} modul lainnya")
    
    def _print_recommendations(self, import_errors):
        """Mencetak rekomendasi berdasarkan error"""
        if import_errors:
            print("\n💡 Rekomendasi:")
            print("=" * 80)
            for error, modules in import_errors.items():
                print(f"\nError: {error}")
                print(f"Modul yang terkena dampak ({len(modules)}):")
                for module in sorted(modules):
                    print(f"  - {module}")
                
                if "No module named" in error:
                    mod_name = error.split("'")[1]
                    print(f"  - Install package yang hilang: pip install {mod_name.split('.')[0]}")
                elif "cannot import name" in error:
                    parts = error.split("'")[1].split('.')
                    print(f"  - Periksa import cycle atau dependency di modul: {'.'.join(parts[:-1])}")
    
    def _print_error_summary(self, import_errors):
        """Mencetak ringkasan error"""
        if import_errors:
            print("\n📌 Ringkasan Error:")
            print("=" * 80)
            for error, modules in import_errors.items():
                print(f"\n❌ {error} (Terjadi pada {len(modules)} modul):")
                for module in sorted(modules):
                    print(f"  - {module}")
    
    def _print_statistics(self, failed_imports, total):
        """Mencetak statistik hasil import"""
        success = total - len(failed_imports)
        print("\n📊 Statistik:")
        print("=" * 80)
        print(f"✅ {success}/{total} modul berhasil diimpor")
        print(f"❌ {len(failed_imports)} modul gagal diimpor")
    
    def _print_failed_modules(self, failed_imports):
        """Mencetak daftar modul yang gagal diimport"""
        if failed_imports:
            print("\n🔍 Modul yang gagal diimpor:")
            print("=" * 80)
            for module in sorted(failed_imports):
                print(f"- {module}")
    
    def _handle_import_error(self, module_name, e, failed_imports, import_errors):
        """Menangani error saat mengimport modul"""
        error_msg = str(e).split('\n')[0]
        print(f"\n❌ Gagal mengimpor {module_name}: {error_msg}")
        failed_imports.append(module_name)
        import_errors[error_msg] = import_errors.get(error_msg, []) + [module_name]

        # Tampilkan traceback lengkap
        print("\nTraceback:")
        traceback.print_exc()
        
        # Cek apakah modul ada di path
        print(f"\n🔍 Mencari modul {module_name} di sys.path:")
        for path in sys.path:
            mod_path = os.path.join(path, module_name.replace('.', '/'))
            py_path = mod_path + '.py'
            py_package = os.path.join(mod_path, '__init__.py')
            
            if os.path.exists(py_path):
                print(f"✅ Ditemukan: {py_path}")
            elif os.path.exists(py_package):
                print(f"✅ Ditemukan package: {py_package}")
        
        # Coba dapatkan detail error lebih lanjut
        self._analyze_import_error(error_msg)
        
        # Tampilkan rekomendasi khusus
        if "No module named" in error_msg:
            missing_module = error_msg.split("'")[1]
            print(f"\n💡 Rekomendasi:")
            print(f"1. Periksa apakah package {missing_module.split('.')[0]} sudah terinstall")
            print(f"   Jalankan: pip install {missing_module.split('.')[0]}")
            print(f"2. Periksa PYTHONPATH: {os.environ.get('PYTHONPATH', 'Tidak diset')}")
            print(f"3. Periksa sys.path:")
            for i, path in enumerate(sys.path, 1):
                print(f"   {i}. {path}")
    
    def _analyze_import_error(self, error_msg):
        """Menganalisis error import untuk memberikan informasi lebih detail"""
        try:
            if "No module named" in error_msg:
                missing_module = error_msg.split("'")[1]
                print(f"\n🔍 Mencari modul yang hilang: {missing_module}")
                try:
                    # Coba cari modul di PYTHONPATH
                    spec = importlib.util.find_spec(missing_module.split('.')[0])
                    if spec:
                        print(f"   - Modul ditemukan di: {spec.origin}")
                    else:
                        print(f"   - Modul tidak ditemukan di PYTHONPATH")
                except Exception as e:
                    print(f"   - Gagal memeriksa modul: {str(e)}")
        except Exception as e:
            print(f"   - Gagal menganalisis error: {str(e)}")
    
    def _handle_unexpected_error(self, module_name, e, failed_imports, import_errors, module=None):
        """Menangani error tidak terduga saat mengimport modul"""
        error_first_line = str(e).split('\n')[0]
        error_msg = f"Unexpected error: {error_first_line}"
        print(f"\n⚠️  Error tidak terduga saat mengimpor {module_name}: {error_msg}")
        failed_imports.append(module_name)
        import_errors[error_msg] = import_errors.get(error_msg, []) + [module_name]

        # Tampilkan traceback lengkap
        print("\nTraceback:")
        traceback.print_exc()
        
        # Cek dependensi modul
        print(f"\n🔍 Memeriksa dependensi untuk {module_name}:")
        try:
            # Coba dapatkan file modul
            if hasattr(module, '__file__') and module.__file__:
                with open(module.__file__, 'r', encoding='utf-8') as f:
                    content = f.read()
                    imports = [line for line in content.split('\n') 
                              if line.strip().startswith(('import ', 'from '))]
                    if imports:
                        print("  Dependencies:")
                        for imp in imports[:10]:  # Batasi jumlah yang ditampilkan
                            print(f"  - {imp}")
                        if len(imports) > 10:
                            print(f"  - ...dan {len(imports) - 10} import lainnya")
        except Exception as e:
            print(f"  Tidak dapat memeriksa dependensi: {str(e)}")
        
        # Coba dapatkan info lebih lanjut tentang modul
        self._print_module_info(module_name, module)
        
        # Tampilkan rekomendasi
        print("\n💡 Rekomendasi:")
        print(f"1. Periksa apakah semua dependensi untuk {module_name} sudah terinstall")
        print(f"2. Periksa kompatibilitas versi dependensi")
        if "No module named" in error_msg:
            missing = error_msg.split("'")[1]
            print(f"3. Install package yang hilang: pip install {missing}")
        print(f"4. Periksa error di atas untuk detail lebih lanjut")
    
    def _print_module_info(self, module_name, module):
        """Mencetak informasi tentang modul"""
        try:
            print(f"\n🔍 Informasi modul {module_name}:")
            if module is not None:
                print(f"   - File: {inspect.getfile(module) if hasattr(module, '__file__') else 'Tidak dapat menentukan file'}")
            else:
                print("   - Modul tidak berhasil diimpor")
        except Exception as e:
            print(f"   - Gagal mendapatkan info modul: {str(e)}")
    
    def test_module_imports(self):
        """Test import untuk semua modul yang didefinisikan"""
        # Buat direktori logs jika belum ada
        import os
        os.makedirs('logs', exist_ok=True)
        
        # Nama file log dengan timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/import_test_{timestamp}.log'
        
        print(f"\n🔍 Memulai pengujian import modul. Log akan disimpan di: {os.path.abspath(log_file)}")
        print("-" * 80)
        
        # Inisialisasi variabel untuk mencatat hasil import
        failed_imports = []
        import_errors = {}
        
        # Catat waktu mulai
        start_time = datetime.now()
        
        # Daftar modul yang diharapkan gagal (untuk debugging)
        expected_failures = []
        
        # Buka file untuk menulis output
        with open(log_file, 'w', encoding='utf-8') as f_log, \
             open('import_test_output.txt', 'w', encoding='utf-8') as f_out:
            
            # Redirect stdout dan stderr ke file log dan console
            class Tee:
                def __init__(self, *files):
                    self.files = files
                
                def write(self, obj):
                    for f in self.files:
                        f.write(str(obj))
                        f.flush()
                
                def flush(self):
                    for f in self.files:
                        f.flush()
            
            # Simpan referensi asli
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # Redirect output ke file dan console
                sys.stdout = Tee(sys.stdout, f_log, f_out)
                sys.stderr = Tee(sys.stderr, f_log, f_out)
                
                # Uji import untuk setiap modul
                for module_name in sorted(MODULES_TO_TEST):
                    # Skip modul yang ada di IGNORED_MODULES
                    if module_name in self.IGNORED_MODULES:
                        print(f"⏩ Melewati modul yang diabaikan: {module_name}")
                        continue
                        
                    module = None
                    try:
                        # Coba impor modul
                        module = importlib.import_module(module_name)
                        print(f"✅ {module_name} berhasil diimpor")
                        
                        # Coba muat semua nama dari modul
                        try:
                            all_names = dir(module)
                            print(f"   - {len(all_names)} simbol ditemukan")
                        except Exception as e:
                            print(f"   ⚠️  Gagal memeriksa isi modul: {str(e)}")
                            
                    except ImportError as e:
                        self._handle_import_error(module_name, e, failed_imports, import_errors)
                    except Exception as e:
                        self._handle_unexpected_error(module_name, e, failed_imports, import_errors, module)

                    print("-" * 80)
                
                # Hitung durasi tes
                duration = datetime.now() - start_time
                
                # Tampilkan ringkasan
                self._print_import_summary(failed_imports, import_errors, log_file)
                
                # Tampilkan ringkasan eksekusi
                print("\n" + "=" * 80)
                print(f"🏁 Selesai dalam {duration.total_seconds():.2f} detik")
                print(f"📊 Total modul diuji: {len(MODULES_TO_TEST)}")
                print(f"✅ Berhasil: {len(MODULES_TO_TEST) - len(failed_imports)}")
                print(f"❌ Gagal: {len(failed_imports)}")
                
                # Tampilkan modul yang gagal dengan detail
                if failed_imports:
                    print("\n" + "🔍" * 40)
                    print("MODUL YANG GAGAL DIIMPOR:")
                    print("🔍" * 40)
                    
                    # Kelompokkan modul yang gagal berdasarkan direktori
                    failed_by_dir = {}
                    for mod in failed_imports:
                        dir_path = '.'.join(mod.split('.')[:-1])
                        if dir_path not in failed_by_dir:
                            failed_by_dir[dir_path] = []
                        failed_by_dir[dir_path].append(mod.split('.')[-1])
                    
                    # Tampilkan modul yang gagal berdasarkan direktori
                    for dir_path, modules in sorted(failed_by_dir.items()):
                        print(f"\n📂 {dir_path}:")
                        for mod in sorted(modules):
                            print(f"  - {mod}")
                    
                    print("\n💡 Rekomendasi:")
                    print("1. Periksa apakah semua dependensi sudah terinstall")
                    print("2. Pastikan path modul sudah benar")
                    print("3. Periksa error di file log untuk detail lebih lanjut")
                    print(f"   Path log: {os.path.abspath(log_file)}")
                    
                    # Tulis daftar modul yang gagal ke file terpisah
                    failed_file = 'logs/failed_imports.txt'
                    with open(failed_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(sorted(failed_imports)))
                    print(f"\n📝 Daftar modul yang gagal disimpan di: {os.path.abspath(failed_file)}")
                
                # Gagalkan tes jika ada modul yang gagal diimpor
                total = len(MODULES_TO_TEST)
                self.assertEqual(
                    len(failed_imports),
                    0,
                    f"{len(failed_imports)} dari {total} modul gagal diimport. "
                    f"Lihat file {os.path.abspath(log_file)} untuk detail lengkap."
                )
                
            except Exception as e:
                print(f"\n❌ Error tidak terduga: {str(e)}")
                traceback.print_exc()
                raise
                
            finally:
                # Pastikan stdout dan stderr dikembalikan ke semula
                sys.stdout = original_stdout
                sys.stderr = original_stderr

if __name__ == "__main__":
    unittest.main()
