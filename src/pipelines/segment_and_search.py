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

# Import SAM se disponibile
try:
    from src.pipelines.sam_integration import SAMSegmenter
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM non disponibile. Installa con: pip install git+https://github.com/facebookresearch/segment-anything.git")


class ImageSegmenter:
    """
    Classe per la segmentazione delle immagini con supporto per SAM.
    """

    def __init__(self, method="superpixel"):
        self.method = method
        self.sam_segmenter = None

        # Inizializza SAM se disponibile e richiesto
        if method == "sam" and SAM_AVAILABLE:
            try:
                self.sam_segmenter = SAMSegmenter(device="cpu")
            except Exception as e:
                print(f"⚠️ Impossibile inizializzare SAM: {e}")
                print("  Fallback a metodo superpixel")
                self.method = "superpixel"

    def segment(self, image, **kwargs):
        """
        Segmenta un'immagine usando il metodo specificato.

        Args:
            image: Immagine PIL o numpy array
            **kwargs: Parametri specifici per ogni metodo

        Returns:
            Lista di segmenti con informazioni geometriche
        """
        if self.method == "sam" and self.sam_segmenter is not None:
            return self.segment_sam(image, **kwargs)
        elif self.method == "superpixel":
            return self.segment_superpixel(image, **kwargs)
        elif self.method == "kmeans":
            return self.segment_kmeans_color(image, **kwargs)
        elif self.method == "grid":
            return self.segment_grid(image, **kwargs)
        else:
            raise ValueError(f"Metodo di segmentazione non supportato: {self.method}")

    def segment_sam(self, image, **kwargs):
        """
        Segmentazione usando SAM (Segment Anything Model).

        Args:
            image: Immagine PIL o numpy array
            **kwargs: Parametri SAM (ignorati per ora)

        Returns:
            Lista di segmenti
        """
        if self.sam_segmenter is None:
            raise RuntimeError("SAM non inizializzato")

        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Genera maschere SAM
        masks = self.sam_segmenter.segment_image(image_np)

        # Converti in formato standard
        segments = self.sam_segmenter.masks_to_segments(masks, image_np.shape[:2])

        return segments

    def segment_superpixel(self, image, n_segments=100):
        """
        Segmentazione usando algoritmo SLIC superpixel.

        Args:
            image: Immagine PIL o numpy array
            n_segments: Numero di superpixel

        Returns:
            Lista di segmenti
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

        # Converti in lista di segmenti
        segments = self._mask_to_segments(mask, image.shape[:2])
        return segments

    def segment_kmeans_color(self, image, n_clusters=8):
        """
        Segmentazione basata su clustering dei colori.

        Args:
            image: Immagine PIL o numpy array
            n_clusters: Numero di cluster

        Returns:
            Lista di segmenti
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

        # Converti in lista di segmenti
        segments = self._mask_to_segments(mask, image.shape[:2])
        return segments

    def segment_grid(self, image, grid_size=(4, 4)):
        """
        Segmentazione a griglia uniforme.

        Args:
            image: Immagine PIL o numpy array
            grid_size: Dimensioni della griglia (rows, cols)

        Returns:
            Lista di segmenti
        """
        if isinstance(image, Image.Image):
            h, w = image.size[::-1]
        else:
            h, w = image.shape[:2]

        rows, cols = grid_size

        # Crea griglia
        segments = []
        segment_id = 0

        for i in range(rows):
            for j in range(cols):
                # Calcola coordinate del segmento
                y1 = i * h // rows
                y2 = (i + 1) * h // rows
                x1 = j * w // cols
                x2 = (j + 1) * w // cols

                # Crea maschera per questo segmento
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True

                segment_info = {
                    'id': segment_id,
                    'mask': mask,
                    'bbox': [x1, y1, x2, y2],
                    'area': (x2 - x1) * (y2 - y1)
                }

                segments.append(segment_info)
                segment_id += 1

        return segments

    def _mask_to_segments(self, mask, image_shape):
        """
        Converte una maschera di segmentazione in lista di segmenti.

        Args:
            mask: Maschera con ID dei segmenti
            image_shape: Dimensioni dell'immagine (H, W)

        Returns:
            Lista di segmenti con informazioni geometriche
        """
        segments = []
        unique_labels = np.unique(mask)

        for label in unique_labels:
            if label == -1:  # Skip background/noise
                continue

            # Crea maschera binaria per questo segmento
            segment_mask = (mask == label)

            # Trova bounding box
            coords = np.where(segment_mask)
            if len(coords[0]) == 0:
                continue

            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Calcola area
            area = np.sum(segment_mask)

            segment_info = {
                'id': int(label),
                'mask': segment_mask,
                'bbox': [x_min, y_min, x_max, y_max],
                'area': int(area)
            }

            segments.append(segment_info)

        return segments

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
        segments = self.segmenter.segment(image, **seg_kwargs)

        # Se il metodo non restituisce segmenti nel formato atteso, converti
        if segments and isinstance(segments[0], dict) and 'bbox' in segments[0]:
            # Formato già corretto con segmenti
            segment_info = segments
            # Crea una maschera dummy per compatibilità
            mask = np.zeros((image.size[1], image.size[0]), dtype=int)
        else:
            # Formato legacy con maschera numerica - converti in segmenti
            mask = segments if segments is not None else np.zeros((image.size[1], image.size[0]), dtype=int)
            segment_info = self._convert_mask_to_segments(mask, image.size)

        # Estrai feature dei segmenti usando il nuovo formato
        segment_features = []
        results = []

        for i, seg_info in enumerate(segment_info):
            try:
                bbox = seg_info['bbox']
                x1, y1, x2, y2 = bbox

                # Estrai il segmento dell'immagine
                segment_image = image.crop((x1, y1, x2, y2))

                # Estrai feature con CLIP
                preprocessed = self.clip_model.preprocess(segment_image).unsqueeze(0)
                with torch.no_grad():
                    features = self.clip_model.encode_image(preprocessed.to(self.clip_model.device))

                segment_features.append(features.cpu().numpy())

            except Exception as e:
                print(f"Errore nell'estrazione delle feature per il segmento {i}: {e}")
                continue

        if not segment_features:
            return [], mask

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
            if idx < len(segment_info):
                result = {
                    'rank': i + 1,
                    'similarity': float(similarities[idx]),
                    'segment_info': segment_info[idx],
                    'features': segment_features[idx]
                }
                results.append(result)

        return results, mask

    def _convert_mask_to_segments(self, mask, image_size):
        """
        Converte una maschera numerica in lista di segmenti.

        Args:
            mask: Maschera con ID dei segmenti
            image_size: Dimensioni dell'immagine (W, H)

        Returns:
            Lista di segmenti con informazioni geometriche
        """
        segments = []
        unique_labels = np.unique(mask)

        for label in unique_labels:
            if label == -1 or label == 0:  # Skip background/noise
                continue

            # Crea maschera binaria per questo segmento
            segment_mask = (mask == label)

            # Trova bounding box
            coords = np.where(segment_mask)
            if len(coords[0]) == 0:
                continue

            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Calcola area
            area = np.sum(segment_mask)

            segment_info = {
                'id': int(label),
                'mask': segment_mask,
                'bbox': [x_min, y_min, x_max, y_max],
                'area': int(area)
            }

            segments.append(segment_info)

        return segments

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
