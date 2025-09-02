"""
Pipeline per la segmentazione delle immagini e la ricerca semantica.
"""

import os
import numpy as np
import cv2
import torch
import yaml
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.models.backbones import CLIPBackbone
from src.pipelines.index_dataset import DatasetIndexer


class ImageSegmenter:
    """
    Classe per la segmentazione delle immagini.
    """

    def __init__(self, method="superpixel"):
        self.method = method

    def segment_superpixel(self, image, n_segments=100):
        """
        Segmentazione usando algoritmo SLIC superpixel.

        Args:
            image: Immagine PIL o numpy array
            n_segments: Numero di superpixel

        Returns:
            Maschera di segmentazione
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Converte in formato OpenCV
        if len(image.shape) == 3:
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            image_cv = image

        # Applica SLIC
        slic = cv2.ximgproc.createSuperpixelSLIC(image_cv,
                                                 cv2.ximgproc.SLIC,
                                                 n_segments)
        slic.iterate(10)
        mask = slic.getLabels()

        return mask

    def segment_kmeans_color(self, image, n_clusters=8):
        """
        Segmentazione basata su clustering dei colori.

        Args:
            image: Immagine PIL o numpy array
            n_clusters: Numero di cluster

        Returns:
            Maschera di segmentazione
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Reshape per clustering
        h, w, c = image.shape
        image_flat = image.reshape(-1, c)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(image_flat)

        # Reshape back
        mask = labels.reshape(h, w)

        return mask

    def segment_grid(self, image, grid_size=(4, 4)):
        """
        Segmentazione a griglia uniforme.

        Args:
            image: Immagine PIL o numpy array
            grid_size: Dimensioni della griglia (rows, cols)

        Returns:
            Maschera di segmentazione
        """
        if isinstance(image, Image.Image):
            h, w = image.size[::-1]
        else:
            h, w = image.shape[:2]

        rows, cols = grid_size
        mask = np.zeros((h, w), dtype=int)

        row_size = h // rows
        col_size = w // cols

        segment_id = 0
        for i in range(rows):
            for j in range(cols):
                start_row = i * row_size
                end_row = (i + 1) * row_size if i < rows - 1 else h
                start_col = j * col_size
                end_col = (j + 1) * col_size if j < cols - 1 else w

                mask[start_row:end_row, start_col:end_col] = segment_id
                segment_id += 1

        return mask

    def segment_image(self, image, **kwargs):
        """
        Segmenta un'immagine usando il metodo specificato.

        Args:
            image: Immagine da segmentare
            **kwargs: Parametri specifici per il metodo

        Returns:
            Maschera di segmentazione
        """
        if self.method == "superpixel":
            return self.segment_superpixel(image, **kwargs)
        elif self.method == "kmeans":
            return self.segment_kmeans_color(image, **kwargs)
        elif self.method == "grid":
            return self.segment_grid(image, **kwargs)
        else:
            raise ValueError(f"Metodo di segmentazione non supportato: {self.method}")


class SemanticSearchPipeline:
    """
    Pipeline completa per la segmentazione e ricerca semantica.
    """

    def __init__(self, config_path="config.yaml"):
        # Carica la configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Inizializza i componenti
        self.clip_model = CLIPBackbone(
            model_name=self.config['clip']['model_name'],
            device=self.config['clip']['device']
        )

        self.indexer = DatasetIndexer(config_path)
        self.segmenter = ImageSegmenter()

        # Carica l'indice se esiste
        self.indexer.load_index()

    def extract_segment_features(self, image, mask):
        """
        Estrae le feature di ogni segmento dell'immagine.

        Args:
            image: Immagine originale
            mask: Maschera di segmentazione

        Returns:
            Lista di feature per ogni segmento
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        unique_segments = np.unique(mask)
        segment_features = []
        segment_info = []

        for segment_id in unique_segments:
            # Crea maschera per questo segmento
            segment_mask = (mask == segment_id)

            # Calcola bounding box
            rows, cols = np.where(segment_mask)
            if len(rows) == 0:
                continue

            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Estrai il segmento
            segment_image = image_array[min_row:max_row+1, min_col:max_col+1]

            # Applica la maschera
            segment_mask_crop = segment_mask[min_row:max_row+1, min_col:max_col+1]
            segment_image = segment_image * segment_mask_crop[:, :, np.newaxis]

            # Converte in PIL e preprocessa
            segment_pil = Image.fromarray(segment_image.astype(np.uint8))

            try:
                # Estrai feature con CLIP
                preprocessed = self.clip_model.preprocess(segment_pil).unsqueeze(0)
                with torch.no_grad():
                    features = self.clip_model.encode_image(preprocessed.to(self.clip_model.device))

                segment_features.append(features.cpu().numpy())
                segment_info.append({
                    'segment_id': int(segment_id),
                    'bbox': (min_col, min_row, max_col, max_row),
                    'area': int(np.sum(segment_mask))
                })

            except Exception as e:
                print(f"Errore nell'estrazione delle feature per il segmento {segment_id}: {e}")
                continue

        return segment_features, segment_info

    def search_in_segments(self, image_path, query_text, top_k=5,
                          segmentation_method="grid", **seg_kwargs):
        """
        Cerca segmenti di un'immagine che corrispondono alla query.

        Args:
            image_path: Percorso dell'immagine
            query_text: Testo della query
            top_k: Numero di segmenti da restituire
            segmentation_method: Metodo di segmentazione
            **seg_kwargs: Parametri per la segmentazione

        Returns:
            Lista di risultati con score e informazioni sui segmenti
        """
        # Carica l'immagine
        image = Image.open(image_path).convert('RGB')

        # Segmenta l'immagine
        self.segmenter.method = segmentation_method
        mask = self.segmenter.segment_image(image, **seg_kwargs)

        # Estrai feature dei segmenti
        segment_features, segment_info = self.extract_segment_features(image, mask)

        if not segment_features:
            return []

        # Estrai feature della query
        query_features = self.clip_model.encode_text([query_text])
        query_features = query_features.cpu().numpy()

        # Calcola similarità
        similarities = []
        for features in segment_features:
            similarity = np.dot(features.flatten(), query_features.flatten())
            similarities.append(similarity)

        # Ordina per similarità
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        # Prepara i risultati
        results = []
        for i, idx in enumerate(sorted_indices):
            result = {
                'rank': i + 1,
                'similarity': float(similarities[idx]),
                'segment_info': segment_info[idx],
                'features': segment_features[idx]
            }
            results.append(result)

        return results, mask

    def visualize_search_results(self, image_path, results, mask,
                                output_path=None, show_plot=True):
        """
        Visualizza i risultati della ricerca sui segmenti.

        Args:
            image_path: Percorso dell'immagine originale
            results: Risultati della ricerca
            mask: Maschera di segmentazione
            output_path: Percorso per salvare l'immagine (opzionale)
            show_plot: Se mostrare il plot
        """
        # Carica l'immagine
        image = Image.open(image_path)

        # Crea il plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Immagine originale
        axes[0].imshow(image)
        axes[0].set_title("Immagine Originale")
        axes[0].axis('off')

        # Immagine con risultati
        axes[1].imshow(image)
        axes[1].set_title("Segmenti Trovati")
        axes[1].axis('off')

        # Disegna i bounding box dei risultati
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

        for i, (result, color) in enumerate(zip(results, colors)):
            bbox = result['segment_info']['bbox']
            x, y, x2, y2 = bbox

            # Disegna il rettangolo
            rect = patches.Rectangle((x, y), x2-x, y2-y,
                                   linewidth=3, edgecolor=color,
                                   facecolor='none', alpha=0.8)
            axes[1].add_patch(rect)

            # Aggiungi etichetta
            axes[1].text(x, y-5, f"#{i+1}\n{result['similarity']:.3f}",
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Risultati salvati in: {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def batch_search_scenes(self, query_text, image_directory=None, top_k=10):
        """
        Cerca scene simili in tutto il dataset.

        Args:
            query_text: Testo della query
            image_directory: Directory delle immagini (opzionale)
            top_k: Numero di risultati

        Returns:
            Lista di risultati
        """
        if self.indexer.index.ntotal == 0:
            print("Database vettoriale vuoto! Esegui prima l'indicizzazione.")
            return []

        # Cerca nel database vettoriale
        results = self.indexer.search_similar_images(query_text, top_k)

        return results


def main():
    """
    Funzione principale per testare la pipeline.
    """
    pipeline = SemanticSearchPipeline()

    print("Pipeline di segmentazione e ricerca semantica pronta!")
    print("\nEsempi di utilizzo:")
    print("1. Ricerca in segmenti:")
    print("   results, mask = pipeline.search_in_segments('image.jpg', 'blue sky')")
    print("2. Ricerca nel dataset:")
    print("   results = pipeline.batch_search_scenes('mountain landscape')")


if __name__ == "__main__":
    main()
