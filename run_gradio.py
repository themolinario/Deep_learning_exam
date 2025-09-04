#!/usr/bin/env python3
"""
Script principale per avviare l'interfaccia Gradio del sistema di Scene Analysis.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Aggiungi il percorso src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.gradio_app import GradioApp


def main():
    """
    Funzione principale per avviare l'applicazione Gradio.
    """
    parser = argparse.ArgumentParser(description="Sistema di Scene Analysis - Interfaccia Gradio")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Percorso del file di configurazione (default: config.yaml)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Porta per l'interfaccia Gradio (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Condividi l'interfaccia pubblicamente"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Abilita modalità debug"
    )

    args = parser.parse_args()

    # Verifica che il file di configurazione esista
    if not os.path.exists(args.config):
        print(f"❌ File di configurazione non trovato: {args.config}")
        print("Assicurati che config.yaml sia presente nella directory principale.")
        return 1

    try:
        # Carica e aggiorna configurazione
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Override configurazione UI con argomenti da linea di comando
        if 'ui' not in config:
            config['ui'] = {}

        config['ui']['port'] = args.port
        config['ui']['share'] = args.share
        config['ui']['debug'] = args.debug

        print("🎭 Inizializzazione Sistema di Scene Analysis...")
        print(f"📁 Configurazione: {args.config}")
        print(f"🌐 Porta: {args.port}")
        print(f"🔗 Condivisione: {args.share}")

        # Crea e avvia l'applicazione Gradio
        app = GradioApp(args.config)
        app.launch()

    except KeyboardInterrupt:
        print("\n👋 Applicazione interrotta dall'utente")
        return 0
    except Exception as e:
        print(f"❌ Errore durante l'avvio: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
