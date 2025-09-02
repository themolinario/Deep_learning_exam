#!/usr/bin/env python3
"""
Script di avvio per l'applicazione CLIP Scene Search.
"""

import os
import sys

# Aggiungi la directory root del progetto al PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Importa e avvia l'applicazione
from src.ui.app import main

if __name__ == "__main__":
    print("🚀 Avvio CLIP Scene Search...")
    print(f"📁 Directory di lavoro: {project_root}")
    main()
