#!/usr/bin/env python3
"""
Script principale per eseguire l'intero pipeline del sistema di Scene Analysis.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Aggiungi il percorso src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.index_dataset import DatasetIndexer
from src.pipelines.fine_tune_clip import CLIPFineTuner, split_dataset
from src.pipelines.segment_and_search import SceneAnalyzer
from src.pipelines.performance_evaluation import PerformanceEvaluator
from src.models.backbones import ModelFactory, AlternativeModelEvaluator


def build_index(config: dict, dataset_path: str):
    """
    Costruisce l'indice vettoriale del dataset.

    Args:
        config: Configurazione del progetto
        dataset_path: Percorso del dataset
    """
    print("🏗️ === COSTRUZIONE INDICE DATASET ===")

    indexer = DatasetIndexer(config)
    indexer.build_full_index(dataset_path, use_faiss=True)

    print("✅ Indice costruito con successo!")


def finetune_model(config: dict, dataset_path: str):
    """
    Esegue il fine-tuning del modello CLIP.

    Args:
        config: Configurazione del progetto
        dataset_path: Percorso del dataset
    """
    print("🎯 === FINE-TUNING CLIP ===")

    # Carica dataset
    indexer = DatasetIndexer(config)
    dataset = indexer.load_dataset(dataset_path)

    if len(dataset) == 0:
        print("❌ Dataset vuoto o non valido")
        return

    # Dividi dataset
    train_set, val_set, test_set = split_dataset(dataset)

    # Esegui fine-tuning
    fine_tuner = CLIPFineTuner(config)
    fine_tuner.train(train_set, val_set)

    print("✅ Fine-tuning completato!")


def evaluate_system(config: dict, test_path: str):
    """
    Valuta le performance del sistema.

    Args:
        config: Configurazione del progetto
        test_path: Percorso del dataset di test
    """
    print("📊 === VALUTAZIONE PERFORMANCE ===")

    # Inizializza analizzatore
    scene_analyzer = SceneAnalyzer(config)

    # Carica vector database
    vector_db_config = config.get('vector_db', {})
    index_path = vector_db_config.get('index_path', 'data/vector_db/image_index.faiss')
    metadata_path = vector_db_config.get('metadata_path', 'data/vector_db/image_metadata.json')

    if os.path.exists(metadata_path):
        scene_analyzer.load_vector_database(index_path, metadata_path)
    else:
        print("⚠️ Vector database non trovato. Esegui prima la costruzione dell'indice.")
        return

    # Valuta performance
    evaluator = PerformanceEvaluator(config, scene_analyzer)
    test_dataset = evaluator.load_test_dataset(test_path)

    if not test_dataset:
        print("⚠️ Dataset di test non trovato o vuoto")
        return

    results = evaluator.evaluate_scene_analysis(test_dataset)
    report_path = evaluator.generate_performance_report()

    print(f"✅ Valutazione completata! Report salvato: {report_path}")


def compare_models(config: dict):
    """
    Confronta modelli alternativi con CLIP.

    Args:
        config: Configurazione del progetto
    """
    print("🔍 === CONFRONTO MODELLI ALTERNATIVI ===")

    # Configura modelli da confrontare
    model_configs = [
        {"type": "dinov2", "variant": "dinov2-base"},
        {"type": "blip2", "variant": "blip2-opt-2.7b"}
    ]

    # Inizializza valutatore
    evaluator = AlternativeModelEvaluator(config)
    evaluator.load_models(model_configs)

    # Carica immagini di test (usa alcune immagini dal dataset)
    dataset_path = config.get('dataset', {}).get('raw_path', 'data/raw/Anime-Naruto')
    test_images = []

    if os.path.exists(dataset_path):
        from PIL import Image
        import glob

        # Carica prime 10 immagini per il test
        image_files = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)[:10]

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                test_images.append(img)
            except Exception as e:
                print(f"Errore caricando {img_path}: {e}")

    if not test_images:
        print("⚠️ Nessuna immagine di test trovata")
        return

    # Confronta embeddings
    embedding_results = evaluator.compare_embeddings(test_images)

    # Benchmark velocità
    speed_results = evaluator.benchmark_speed(test_images)

    print("📊 Risultati confronto modelli:")
    for model_name, stats in embedding_results['embedding_stats'].items():
        print(f"  {model_name}:")
        print(f"    - Dimensione embedding: {stats['embedding_dim']}")
        print(f"    - Norma media: {stats['mean_norm']:.3f}")
        print(f"    - Tempo medio: {speed_results.get(model_name, 'N/A'):.3f}s")

    print("✅ Confronto modelli completato!")


def main():
    """
    Funzione principale del sistema.
    """
    parser = argparse.ArgumentParser(description="Sistema di Scene Analysis - Pipeline Completa")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Percorso del file di configurazione"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/Anime-Naruto",
        help="Percorso del dataset"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="data/test_scenes",
        help="Percorso del dataset di test"
    )

    # Azioni disponibili
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Costruisci l'indice vettoriale del dataset"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Esegui fine-tuning del modello CLIP"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Valuta le performance del sistema"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Confronta modelli alternativi"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Esegui l'intero pipeline (equivale a --build-index --finetune --evaluate)"
    )

    args = parser.parse_args()

    # Verifica file di configurazione
    if not os.path.exists(args.config):
        print(f"❌ File di configurazione non trovato: {args.config}")
        return 1

    # Carica configurazione
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("🎭 Sistema di Scene Analysis - Pipeline Completa")
    print(f"📁 Configurazione: {args.config}")
    print(f"📂 Dataset: {args.dataset}")
    print()

    try:
        # Esegui azioni richieste
        if args.all or args.build_index:
            build_index(config, args.dataset)
            print()

        if args.all or args.finetune:
            finetune_model(config, args.dataset)
            print()

        if args.all or args.evaluate:
            evaluate_system(config, args.test_dataset)
            print()

        if args.compare_models:
            compare_models(config)
            print()

        # Se nessuna azione specificata, mostra help
        if not any([args.build_index, args.finetune, args.evaluate, args.compare_models, args.all]):
            print("❓ Nessuna azione specificata. Usa --help per vedere le opzioni disponibili.")
            parser.print_help()

        print("🎉 Pipeline completata con successo!")

    except KeyboardInterrupt:
        print("\n👋 Pipeline interrotta dall'utente")
        return 0
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
