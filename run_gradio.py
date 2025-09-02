#!/usr/bin/env python3
"""
Script di avvio principale per l'applicazione CLIP Scene Search.
Ora utilizza esclusivamente l'interfaccia Gradio.
"""

import os
import sys

# Aggiungi la directory root del progetto al PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Importa e avvia l'applicazione Gradio
from src.ui.gradio_app import main

if __name__ == "__main__":
    print("🚀 Avvio CLIP Scene Search...")
    print(f"📁 Directory di lavoro: {project_root}")
    print("🌐 Interfaccia: Gradio")
    main()
