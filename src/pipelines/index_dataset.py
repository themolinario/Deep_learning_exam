"""
Pipeline per l'indicizzazione del dataset e la creazione del database vettoriale.
"""

import os
import numpy as np
import torch
from PIL import Image
import yaml
import pickle
import faiss
from tqdm import tqdm
import json

from src.models.backbones import CLIPBackbone


class DatasetIndexer:
    """
    Classe per l'indicizzazione del dataset e la creazione del database vettoriale.
    """

    def __init__(self, config_path="config.yaml"):
        # Carica la configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['clip']['device'] if torch.cuda.is_available() else 'cpu')

        # Inizializza il modello CLIP
        self.model = CLIPBackbone(
            model_name=self.config['clip']['model_name'],
            device=self.device
        )

        # Parametri del database vettoriale
        self.vector_db_path = self.config['vector_db']['path']
        self.embedding_dim = self.config['vector_db']['embedding_dim']

        # Inizializza l'indice FAISS
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product per similarità coseno

        # Metadata delle immagini
        self.image_metadata = []

    def load_images_from_directory(self, directory_path):
        """
        Carica tutte le immagini da una directory.

        Args:
            directory_path: Percorso della directory contenente le immagini

        Returns:
            Lista di percorsi delle immagini
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        return image_paths

    def extract_image_features(self, image_paths, batch_size=32):
        """
        Estrae le feature dalle immagini usando CLIP.

        Args:
            image_paths: Lista dei percorsi delle immagini
            batch_size: Dimensione del batch per l'elaborazione

        Returns:
            Array numpy delle feature estratte
        """
        features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Estraendo feature"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            # Preprocessa le immagini del batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.model.preprocess(image).unsqueeze(0)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Errore nel caricare {path}: {e}")
                    continue

            if batch_images:
                # Combina le immagini in un batch
                batch_tensor = torch.cat(batch_images).to(self.device)

                # Estrai le feature
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    features.append(batch_features.cpu().numpy())

        return np.vstack(features) if features else np.array([])

    def index_dataset(self, data_directory=None):
        """
        Indicizza tutto il dataset e crea il database vettoriale.

        Args:
            data_directory: Directory contenente le immagini (se None, usa la config)
        """
        if data_directory is None:
            data_directory = self.config['dataset']['raw_data_path']

        print(f"Indicizzazione del dataset da: {data_directory}")

        # Carica i percorsi delle immagini
        image_paths = self.load_images_from_directory(data_directory)
        print(f"Trovate {len(image_paths)} immagini")

        if not image_paths:
            print("Nessuna immagine trovata!")
            return

        # Estrai le feature
        features = self.extract_image_features(image_paths)

        if features.size == 0:
            print("Nessuna feature estratta!")
            return

        # Normalizza le feature per la similarità coseno
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Aggiungi le feature all'indice FAISS
        self.index.add(features.astype('float32'))

        # Salva i metadata delle immagini
        for i, path in enumerate(image_paths):
            metadata = {
                'id': i,
                'path': path,
                'filename': os.path.basename(path),
                'directory': os.path.dirname(path)
            }
            self.image_metadata.append(metadata)

        # Salva l'indice e i metadata
        self.save_index()

        print(f"Indicizzazione completata! {len(image_paths)} immagini indicizzate.")

    def save_index(self):
        """
        Salva l'indice FAISS e i metadata su disco.
        """
        os.makedirs(self.vector_db_path, exist_ok=True)

        # Salva l'indice FAISS
        index_path = os.path.join(self.vector_db_path, "image_index.faiss")
        faiss.write_index(self.index, index_path)

        # Salva i metadata
        metadata_path = os.path.join(self.vector_db_path, "image_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.image_metadata, f, indent=2)

        print(f"Indice salvato in: {index_path}")
        print(f"Metadata salvati in: {metadata_path}")

    def load_index(self):
        """
        Carica l'indice FAISS e i metadata da disco.
        """
        index_path = os.path.join(self.vector_db_path, "image_index.faiss")
        metadata_path = os.path.join(self.vector_db_path, "image_metadata.json")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Carica l'indice FAISS
            self.index = faiss.read_index(index_path)

            # Carica i metadata
            with open(metadata_path, 'r') as f:
                self.image_metadata = json.load(f)

            print(f"Indice caricato: {self.index.ntotal} immagini")
            return True
        else:
            print("Indice non trovato!")
            return False

    def search_similar_images(self, query_text, top_k=10):
        """
        Cerca immagini simili basandosi su una query testuale.

        Args:
            query_text: Testo della query
            top_k: Numero di risultati da restituire

        Returns:
            Lista di risultati con score e metadata
        """
        if self.index.ntotal == 0:
            print("Indice vuoto! Esegui prima l'indicizzazione.")
            return []

        # Estrai le feature del testo
        text_features = self.model.encode_text([query_text])
        text_features = text_features.cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        # Cerca nell'indice
        similarities, indices = self.index.search(text_features.astype('float32'), top_k)

        # Prepara i risultati
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.image_metadata):
                result = {
                    'rank': i + 1,
                    'similarity': float(similarity),
                    'metadata': self.image_metadata[idx]
                }
                results.append(result)

        return results


def main():
    """
    Funzione principale per eseguire l'indicizzazione.
    """
    indexer = DatasetIndexer()

    # Indicizza il dataset
    indexer.index_dataset()

    # Test di ricerca
    print("\nTest di ricerca:")
    results = indexer.search_similar_images("a beautiful landscape", top_k=5)

    for result in results:
        print(f"Rank {result['rank']}: {result['metadata']['filename']} "
              f"(similarity: {result['similarity']:.3f})")


if __name__ == "__main__":
    main()
