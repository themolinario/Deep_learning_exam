"""
Pipeline per il fine-tuning del modello CLIP.
"""

import os
import yaml
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.models.clip_finetune import CLIPFineTuner, CLIPDataset


class FineTunePipeline:
    """
    Pipeline completa per il fine-tuning di CLIP.
    """

    def __init__(self, config_path="config.yaml"):
        # Carica la configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Inizializza il fine-tuner
        self.fine_tuner = CLIPFineTuner(config_path)

    def load_data_from_json(self, json_path):
        """
        Carica i dati di training da un file JSON.

        Formato JSON atteso:
        [
            {
                "image_path": "path/to/image.jpg",
                "text": "descrizione dell'immagine",
                "category": "categoria_opzionale"
            },
            ...
        ]

        Args:
            json_path: Percorso del file JSON

        Returns:
            Tuple (image_paths, texts)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_paths = [item['image_path'] for item in data]
        texts = [item['text'] for item in data]

        return image_paths, texts

    def load_data_from_directory(self, data_dir, text_file=None):
        """
        Carica i dati da una directory di immagini.

        Args:
            data_dir: Directory contenente le immagini
            text_file: File opzionale con le descrizioni (formato: filename\tdescription)

        Returns:
            Tuple (image_paths, texts)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        # Trova tutte le immagini
        for filename in os.listdir(data_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(data_dir, filename))

        # Carica le descrizioni se fornite
        texts = []
        if text_file and os.path.exists(text_file):
            text_dict = {}
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text_dict[parts[0]] = parts[1]

            texts = [text_dict.get(os.path.basename(path), f"Image {os.path.basename(path)}")
                    for path in image_paths]
        else:
            # Genera descrizioni di default basate sul nome del file
            texts = [f"Image showing {os.path.splitext(os.path.basename(path))[0].replace('_', ' ')}"
                    for path in image_paths]

        return image_paths, texts

    def validate_data(self, image_paths, texts):
        """
        Valida che tutti i file di immagine esistano e siano leggibili.

        Args:
            image_paths: Lista dei percorsi delle immagini
            texts: Lista dei testi corrispondenti

        Returns:
            Tuple (valid_image_paths, valid_texts)
        """
        valid_image_paths = []
        valid_texts = []

        for img_path, text in zip(image_paths, texts):
            try:
                # Prova ad aprire l'immagine
                with Image.open(img_path) as img:
                    img.verify()
                valid_image_paths.append(img_path)
                valid_texts.append(text)
            except Exception as e:
                print(f"Errore con l'immagine {img_path}: {e}")
                continue

        print(f"Dati validati: {len(valid_image_paths)}/{len(image_paths)} immagini valide")
        return valid_image_paths, valid_texts

    def prepare_splits(self, image_paths, texts):
        """
        Divide i dati in training e validation set.

        Args:
            image_paths: Lista dei percorsi delle immagini
            texts: Lista dei testi

        Returns:
            Tuple (train_images, val_images, train_texts, val_texts)
        """
        train_split = self.config['dataset']['train_split']

        train_images, val_images, train_texts, val_texts = train_test_split(
            image_paths, texts,
            train_size=train_split,
            random_state=42,
            shuffle=True
        )

        print(f"Split dei dati:")
        print(f"  Training: {len(train_images)} campioni")
        print(f"  Validation: {len(val_images)} campioni")

        return train_images, val_images, train_texts, val_texts

    def run_fine_tuning(self, data_source, data_type="json"):
        """
        Esegue la pipeline completa di fine-tuning.

        Args:
            data_source: Percorso del file JSON o directory
            data_type: Tipo di dato ("json" o "directory")
        """
        print("Iniziando la pipeline di fine-tuning...")

        # 1. Carica i dati
        print("1. Caricamento dati...")
        if data_type == "json":
            image_paths, texts = self.load_data_from_json(data_source)
        elif data_type == "directory":
            image_paths, texts = self.load_data_from_directory(data_source)
        else:
            raise ValueError("data_type deve essere 'json' o 'directory'")

        print(f"Caricati {len(image_paths)} campioni")

        # 2. Valida i dati
        print("2. Validazione dati...")
        image_paths, texts = self.validate_data(image_paths, texts)

        if len(image_paths) == 0:
            print("Nessun dato valido trovato!")
            return

        # 3. Divide i dati
        print("3. Splitting dei dati...")
        train_images, val_images, train_texts, val_texts = self.prepare_splits(image_paths, texts)

        # 4. Avvia il fine-tuning
        print("4. Avvio fine-tuning...")
        self.fine_tuner.train(
            image_paths=train_images,
            texts=train_texts,
            val_image_paths=val_images,
            val_texts=val_texts
        )

        print("Fine-tuning completato!")

    def create_sample_data(self, output_path="data/sample_training_data.json"):
        """
        Crea un file di esempio per i dati di training.

        Args:
            output_path: Percorso dove salvare il file di esempio
        """
        sample_data = [
            {
                "image_path": "data/scene_examples/landscape1.jpg",
                "text": "A beautiful mountain landscape with snow-capped peaks",
                "category": "landscape"
            },
            {
                "image_path": "data/scene_examples/city1.jpg",
                "text": "A bustling city street with tall buildings",
                "category": "urban"
            },
            {
                "image_path": "data/scene_examples/beach1.jpg",
                "text": "A tropical beach with crystal clear water",
                "category": "nature"
            },
            {
                "image_path": "data/scene_examples/forest1.jpg",
                "text": "A dense forest with tall green trees",
                "category": "nature"
            }
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        print(f"File di esempio creato: {output_path}")


def main():
    """
    Funzione principale per eseguire la pipeline di fine-tuning.
    """
    pipeline = FineTunePipeline()

    # Crea dati di esempio
    pipeline.create_sample_data()

    # Esempio di utilizzo
    print("Pipeline di fine-tuning pronta!")
    print("Per avviare il fine-tuning:")
    print("1. Prepara i tuoi dati nel formato JSON o in una directory")
    print("2. Esegui: pipeline.run_fine_tuning('path/to/data', 'json')")


if __name__ == "__main__":
    main()
