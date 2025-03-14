"""
File: tests/test_header_fixer.py
Deskripsi: Modul untuk file test_header_fixer.py
"""

import os
import unittest
import tempfile
from smartcash.common.header_fixer import (
    generate_file_header, 
    extract_existing_header, 
    update_file_header, 
    update_headers_recursively
)

class TestHeaderFixer(unittest.TestCase):
    def setUp(self):
        """Siapkan direktori tes sementara"""
        self.test_dir = tempfile.mkdtemp()
    
    def _create_test_file(self, filename, content):
        """Buat file tes"""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    
    def test_generate_file_header(self):
        """Tes pembuatan header file"""
        test_file_path = os.path.join(self.test_dir, 'test_module.py')
        header = generate_file_header(
            test_file_path, 
            "Modul tes untuk header fixer"
        )
        
        # Periksa komponen header
        self.assertIn('File:', header)
        self.assertIn('Deskripsi: Modul tes untuk header fixer', header)
        self.assertNotIn('Author:', header)
    
    def test_extract_existing_header(self):
        """Tes ekstraksi header yang sudah ada"""
        content = '''"""
Existing header
"""

def sample_function():
    pass
'''
        header, cleaned_content = extract_existing_header(content)
        
        self.assertIn('Existing header', header)
        self.assertIn('def sample_function', cleaned_content)
    
    def test_update_file_header(self):
        """Tes update header file"""
        # Buat file dengan header lama yang kompleks
        test_content = '''"""
File: old/path/test_module.py
Author: Old Author
Deskripsi: Deskripsi lama
"""

def old_function():
    return "Hello"
'''
        test_file_path = self._create_test_file('test_module.py', test_content)
        
        # Update header
        result = update_file_header(test_file_path)
        self.assertTrue(result)
        
        # Baca konten file yang sudah diupdate
        with open(test_file_path, 'r', encoding='utf-8') as f:
            updated_content = f.read()
        
        # Periksa header baru
        self.assertIn('File:', updated_content)
        self.assertIn('Deskripsi:', updated_content)
        self.assertNotIn('Author:', updated_content)
        self.assertIn('def old_function', updated_content)
    
    def test_update_headers_recursively(self):
        """Tes update header secara rekursif"""
        # Buat struktur direktori dan file palsu
        os.makedirs(os.path.join(self.test_dir, 'subdir'), exist_ok=True)
        
        # Buat beberapa file tes
        test_files = [
            os.path.join(self.test_dir, 'file1.py'),
            os.path.join(self.test_dir, 'subdir', 'file2.py'),
            os.path.join(self.test_dir, 'file3.txt')  # File yang tidak akan diupdate
        ]
        
        # Buat file dengan konten berbeda
        for filepath in test_files:
            content = 'def sample_function():\n    pass\n' if filepath.endswith('.py') else 'Bukan file Python'
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Update header
        updated = update_headers_recursively(
            self.test_dir, 
            exclude_dirs=['non_existent_dir', '.git', 'venv', 'tests']
        )
        
        # Periksa hanya file Python yang diupdate
        self.assertEqual(len(updated), 2)  # file1.py dan file2.py
        
        # Periksa isi file yang diupdate
        for filepath in updated:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('File:', content)
            self.assertIn('Deskripsi:', content)
            self.assertNotIn('Author:', content)
    
    def test_file_with_no_header(self):
        """Tes file tanpa header"""
        test_file_path = self._create_test_file('no_header.py', 'def test_function():\n    pass\n')
        
        result = update_file_header(test_file_path)
        self.assertTrue(result)
        
        # Baca konten file yang sudah diupdate
        with open(test_file_path, 'r', encoding='utf-8') as f:
            updated_content = f.read()
        
        # Periksa header baru
        self.assertIn('File:', updated_content)
        self.assertIn('Deskripsi:', updated_content)
        self.assertNotIn('Author:', updated_content)
        self.assertIn('def test_function', updated_content)

if __name__ == '__main__':
    unittest.main()