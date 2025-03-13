# Struktur direktori baru sesuai dengan model_migration.md

mkdir -p smartcash/model/architectures/backbones
mkdir -p smartcash/model/architectures/necks
mkdir -p smartcash/model/architectures/heads
mkdir -p smartcash/model/services/checkpoint
mkdir -p smartcash/model/services/training
mkdir -p smartcash/model/services/evaluation
mkdir -p smartcash/model/services/prediction
mkdir -p smartcash/model/services/experiment
mkdir -p smartcash/model/services/research
mkdir -p smartcash/model/config
mkdir -p smartcash/model/utils
mkdir -p smartcash/model/components

# File-file inisialisasi yang diperlukan
touch smartcash/model/__init__.py
touch smartcash/model/manager.py
touch smartcash/model/exceptions.py

touch smartcash/model/architectures/__init__.py
touch smartcash/model/architectures/backbones/__init__.py
touch smartcash/model/architectures/necks/__init__.py
touch smartcash/model/architectures/heads/__init__.py

touch smartcash/model/services/__init__.py
touch smartcash/model/services/checkpoint/__init__.py
touch smartcash/model/services/training/__init__.py
touch smartcash/model/services/evaluation/__init__.py
touch smartcash/model/services/prediction/__init__.py
touch smartcash/model/services/experiment/__init__.py
touch smartcash/model/services/research/__init__.py

touch smartcash/model/config/__init__.py
touch smartcash/model/utils/__init__.py
touch smartcash/model/components/__init__.py