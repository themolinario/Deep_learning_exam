#!/usr/bin/env python3
"""
Script per ricostruire il vector database con il parsing corretto del CSV.
"""

import os
import sys
import yaml
from pathlib import Path

# Aggiungi il percorso src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.index_dataset import DatasetIndexer


def rebuild_index():
    """
    Ricostruisce l'indice vettoriale con il parsing corretto del CSV.
    """
    print("🔧 Ricostruzione dell'indice vettoriale...")

    # Carica configurazione
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Crea indicizzatore
    indexer = DatasetIndexer(config)

    # Dataset path
    dataset_path = "data/raw/Anime-Naruto"

    # Controlla che il dataset esista
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trovato: {dataset_path}")
        return False

    try:
        # Ricostruisci l'indice
        indexer.build_full_index(dataset_path, use_faiss=True)
        print("✅ Indice ricostruito con successo!")
        return True

    except Exception as e:
        print(f"❌ Errore durante la ricostruzione: {e}")
        return False


if __name__ == "__main__":
    success = rebuild_index()
    sys.exit(0 if success else 1)
