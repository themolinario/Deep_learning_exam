"""
Pipeline per l'indicizzazione del dataset e creazione del vector database.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("⚠️ FAISS non installato. Installa con: pip install faiss-cpu")
    faiss = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("⚠️ Transformers non installato. Installa con: pip install transformers")
    CLIPProcessor = None
    CLIPModel = None


class DatasetIndexer:
    """
    Classe per indicizzare il dataset e creare il vector database.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza l'indicizzatore del dataset.

        Args:
            config: Configurazione del progetto
        """
        self.config = config
        self.device = config.get('models', {}).get('clip', {}).get('device', 'cpu')

        # Inizializza il modello CLIP
        model_name = config.get('models', {}).get('clip', {}).get('model_name', 'openai/clip-vit-base-patch32')

        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("Transformers non installato. Installa con: pip install transformers")

        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Configurazione vector database
        self.embedding_dim = config.get('vector_db', {}).get('embedding_dim', 512)
        self.vector_db_path = config.get('vector_db', {}).get('index_path', 'data/vector_db/image_index.faiss')
        self.metadata_path = config.get('vector_db', {}).get('metadata_path', 'data/vector_db/image_metadata.json')

        # Database vettoriale
        self.index = None
        self.metadata = []

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Carica il dataset organizzato per personaggi.

        Args:
            dataset_path: Percorso della directory del dataset

        Returns:
            Lista di dizionari con informazioni delle immagini
        """
        dataset = []
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

        # Estensioni immagine supportate
        image_extensions = self.config.get('dataset', {}).get('image_extensions', ['.jpg', '.jpeg', '.png'])

        # Attraversa le directory del dataset
        for split_dir in ['train', 'valid', 'test']:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                continue

            print(f"Caricamento split: {split_dir}")

            # Cerca file di classe (_classes.csv) se presente
            classes_file = split_path / '_classes.csv'
            if classes_file.exists():
                print(f"Caricamento CSV: {classes_file}")
                # Carica le classi dal file CSV multi-label
                with open(classes_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        continue

                    # Leggi header per ottenere i nomi dei personaggi
                    header = lines[0].strip().split(',')
                    character_columns = header[1:]  # Escludi la colonna filename
                    print(f"Personaggi trovati: {character_columns}")

                    # Processa ogni riga di dati
                    for line in lines[1:]:  # Salta header
                        parts = line.strip().split(',')
                        if len(parts) < len(header):
                            continue

                        filename = parts[0]
                        labels = [int(x) for x in parts[1:]]

                        # Trova quale personaggio è presente (valore 1)
                        characters_present = []
                        for i, label in enumerate(labels):
                            if label == 1:
                                characters_present.append(character_columns[i])

                        # Se nessun personaggio è etichettato o è "Unlabeled", salta
                        if not characters_present or (len(characters_present) == 1 and characters_present[0] == 'Unlabeled'):
                            continue

                        # Usa il primo personaggio se ce ne sono multipli
                        character = characters_present[0]

                        # Costruisci il percorso completo
                        image_path = split_path / filename
                        if image_path.exists() and image_path.suffix.lower() in image_extensions:
                            dataset.append({
                                'path': str(image_path),
                                'character': character,
                                'split': split_dir
                            })
                            print(f"Aggiunto: {filename} -> {character}")
                        else:
                            print(f"⚠️ File non trovato: {image_path}")
            else:
                # Se non c'è file CSV, usa la struttura delle directory
                for character_dir in split_path.iterdir():
                    if character_dir.is_dir():
                        character_name = character_dir.name

                        for image_file in character_dir.iterdir():
                            if image_file.suffix.lower() in image_extensions:
                                dataset.append({
                                    'path': str(image_file),
                                    'character': character_name,
                                    'split': split_dir
                                })

                # Se non ci sono sottodirectory, tutte le immagini sono nella directory split
                if not any(child.is_dir() for child in split_path.iterdir()):
                    for image_file in split_path.iterdir():
                        if image_file.suffix.lower() in image_extensions:
                            # Estrai il nome del personaggio dal nome del file
                            character_name = self._extract_character_from_filename(image_file.name)
                            dataset.append({
                                'path': str(image_file),
                                'character': character_name,
                                'split': split_dir
                            })

        print(f"Caricato dataset con {len(dataset)} immagini")
        return dataset

    def _extract_character_from_filename(self, filename: str) -> str:
        """
        Estrae il nome del personaggio dal nome del file.

        Args:
            filename: Nome del file

        Returns:
            Nome del personaggio estratto
        """
        # Rimuovi estensione
        name = Path(filename).stem

        # Logica per estrarre il personaggio dal nome del file
        # Questo dipende dalla convenzione di nomenclatura del dataset

        # Se il nome contiene underscore, prendi la prima parte
        if '_' in name:
            return name.split('_')[0]

        # Se contiene trattini, prendi la prima parte
        if '-' in name:
            return name.split('-')[0]

        # Altrimenti usa tutto il nome
        return name

    def compute_embeddings(self, dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Calcola gli embedding per tutte le immagini del dataset.

        Args:
            dataset: Lista di informazioni delle immagini

        Returns:
            Tupla con array degli embedding e metadati aggiornati
        """
        embeddings = []
        metadata = []

        print("Calcolo embedding delle immagini...")

        with torch.no_grad():
            for item in tqdm(dataset, desc="Embedding"):
                try:
                    # Carica l'immagine
                    image = Image.open(item['path']).convert('RGB')

                    # Preprocessa l'immagine
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Calcola l'embedding
                    image_features = self.model.get_image_features(**inputs)

                    # Normalizza l'embedding
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                    # Converti in numpy
                    embedding = image_features.cpu().numpy().flatten()

                    embeddings.append(embedding)
                    metadata.append({
                        'path': item['path'],
                        'character': item['character'],
                        'split': item['split'],
                        'embedding_id': len(metadata)
                    })

                except Exception as e:
                    print(f"Errore processando {item['path']}: {e}")
                    continue

        embeddings_array = np.vstack(embeddings).astype(np.float32)
        print(f"Calcolati {len(embeddings_array)} embedding di dimensione {embeddings_array.shape[1]}")

        return embeddings_array, metadata

    def create_vector_index(self, embeddings: np.ndarray, use_faiss: bool = True) -> Any:
        """
        Crea l'indice vettoriale per la ricerca veloce.

        Args:
            embeddings: Array degli embedding
            use_faiss: Se usare FAISS per l'indicizzazione

        Returns:
            Indice vettoriale creato
        """
        if use_faiss and faiss is not None:
            print("Creazione indice FAISS...")

            # Crea indice FAISS per similarità coseno
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product per cosine similarity

            # Normalizza gli embedding per la similarità coseno
            faiss.normalize_L2(embeddings)

            # Aggiungi embedding all'indice
            index.add(embeddings)

            print(f"Indice FAISS creato con {index.ntotal} vettori")
            return index
        else:
            print("Usando ricerca lineare semplice...")
            # Fallback a ricerca lineare
            return embeddings

    def save_index(self, index: Any, metadata: List[Dict[str, Any]], use_faiss: bool = True):
        """
        Salva l'indice vettoriale e i metadati su disco.

        Args:
            index: Indice vettoriale
            metadata: Metadati delle immagini
            use_faiss: Se l'indice è FAISS
        """
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)

        if use_faiss and faiss is not None:
            # Salva indice FAISS
            faiss.write_index(index, self.vector_db_path)
            print(f"Indice FAISS salvato in: {self.vector_db_path}")
        else:
            # Salva array numpy
            np.save(self.vector_db_path.replace('.faiss', '.npy'), index)
            print(f"Array embedding salvato in: {self.vector_db_path.replace('.faiss', '.npy')}")

        # Salva metadati
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadati salvati in: {self.metadata_path}")

    def load_index(self, use_faiss: bool = True) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Carica l'indice vettoriale e i metadati da disco.

        Args:
            use_faiss: Se l'indice è FAISS

        Returns:
            Tupla con indice e metadati
        """
        # Carica metadati
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"File metadati non trovato: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Carica indice
        if use_faiss and faiss is not None and os.path.exists(self.vector_db_path):
            index = faiss.read_index(self.vector_db_path)
            print(f"Indice FAISS caricato: {index.ntotal} vettori")
        else:
            npy_path = self.vector_db_path.replace('.faiss', '.npy')
            if os.path.exists(npy_path):
                index = np.load(npy_path)
                print(f"Array embedding caricato: {index.shape}")
            else:
                raise FileNotFoundError(f"Indice non trovato: {self.vector_db_path} o {npy_path}")

        return index, metadata

    def build_full_index(self, dataset_path: str, use_faiss: bool = True):
        """
        Costruisce l'indice completo dal dataset.

        Args:
            dataset_path: Percorso del dataset
            use_faiss: Se usare FAISS
        """
        print("=== Inizio indicizzazione dataset ===")

        # 1. Carica dataset
        dataset = self.load_dataset(dataset_path)

        # 2. Calcola embedding
        embeddings, metadata = self.compute_embeddings(dataset)

        # 3. Crea indice
        index = self.create_vector_index(embeddings, use_faiss)

        # 4. Salva indice e metadati
        self.save_index(index, metadata, use_faiss)

        # 5. Aggiorna attributi di classe
        self.index = index
        self.metadata = metadata

        print("=== Indicizzazione completata ===")

        # Statistiche finali
        characters = set(item['character'] for item in metadata)
        print(f"Personaggi nel database: {len(characters)}")
        print(f"Immagini totali: {len(metadata)}")
        print(f"Personaggi trovati: {sorted(characters)}")


class SimpleVectorDB:
    """
    Implementazione semplice di vector database per il fallback.
    """

    def __init__(self):
        self.embeddings = None
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Aggiunge embedding al database."""
        self.embeddings = embeddings
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cerca i k embedding più simili.

        Args:
            query_embedding: Embedding di query
            k: Numero di risultati da restituire

        Returns:
            Tupla con (scores, indices)
        """
        if self.embeddings is None:
            return np.array([]), np.array([])

        # Normalizza query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calcola similarità coseno
        similarities = np.dot(self.embeddings, query_embedding)

        # Trova i top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]

        return top_scores, top_indices

