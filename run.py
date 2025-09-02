#!/usr/bin/env python3
"""
Script di avvio principale per CLIP Scene Search.
"""

import os
import sys

# Aggiungi il percorso del progetto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ui.gradio_app import main

if __name__ == "__main__":
    main()
