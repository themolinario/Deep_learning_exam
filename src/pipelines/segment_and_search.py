"""
Pipeline per l'analisi delle scene che combina SAM per la segmentazione
e CLIP per l'identificazione dei personaggi.
"""

import os
import numpy as np
from PIL import Image
import cv2
import torch
from typing import List, Dict, Any, Tuple, Optional
import json

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("⚠️ Transformers non installato. Installa con: pip install transformers")
    CLIPProcessor = None
    CLIPModel = None

try:
    import faiss
except ImportError:
    print("⚠️ FAISS non installato. Installa con: pip install faiss-cpu")
    faiss = None

from .sam_integration import SAMSegmenter
from .index_dataset import SimpleVectorDB


class SceneAnalyzer:
    """
    Classe principale per l'analisi delle scene.
    Combina SAM per la segmentazione e CLIP per l'identificazione.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza l'analizzatore di scene.

        Args:
            config: Configurazione del progetto
        """
        self.config = config
        self.device = config.get('models', {}).get('clip', {}).get('device', 'cpu')

        # Inizializza SAM
        sam_config = config.get('models', {}).get('sam', {})
        self.sam_segmenter = SAMSegmenter(
            model_type=sam_config.get('model_type', 'vit_b'),
            checkpoint_path=sam_config.get('checkpoint_path'),
            device=self.device
        )

        # Inizializza CLIP
        clip_config = config.get('models', {}).get('clip', {})
        model_name = clip_config.get('model_name', 'openai/clip-vit-base-patch32')

        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("Transformers non installato. Installa con: pip install transformers")

        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Carica modello fine-tuned se disponibile
        checkpoint_path = clip_config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_finetuned_model(checkpoint_path)

        # Inizializza vector database
        self.vector_index = None
        self.metadata = []
        self.use_faiss = faiss is not None

        # Configurazione ricerca
        search_config = config.get('search', {})
        self.top_k = search_config.get('top_k', 5)
        self.similarity_threshold = search_config.get('similarity_threshold', 0.7)

        # Configurazione segmentazione
        seg_config = config.get('segmentation', {})
        self.min_mask_area = seg_config.get('min_mask_area', 1000)

    def load_finetuned_model(self, checkpoint_path: str):
        """
        Carica un modello CLIP fine-tuned.

        Args:
            checkpoint_path: Percorso del checkpoint
        """
        try:
            # Fix per PyTorch 2.6+ con TorchScript archives
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modello CLIP fine-tuned caricato da: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Errore caricando modello fine-tuned: {e}")
            print("Usando modello CLIP pre-addestrato")

    def load_vector_database(self, index_path: str, metadata_path: str):
        """
        Carica il vector database.

        Args:
            index_path: Percorso dell'indice vettoriale
            metadata_path: Percorso dei metadati
        """
        try:
            # Carica metadati
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Carica indice
            if self.use_faiss and index_path.endswith('.faiss') and os.path.exists(index_path):
                self.vector_index = faiss.read_index(index_path)
                print(f"✅ Indice FAISS caricato: {self.vector_index.ntotal} vettori")
            else:
                # Fallback a array numpy
                npy_path = index_path.replace('.faiss', '.npy')
                if os.path.exists(npy_path):
                    embeddings = np.load(npy_path)
                    self.vector_index = SimpleVectorDB()
                    self.vector_index.add(embeddings, self.metadata)
                    print(f"✅ Vector database caricato: {embeddings.shape[0]} vettori")
                else:
                    raise FileNotFoundError(f"Indice non trovato: {index_path} o {npy_path}")

        except Exception as e:
            print(f"❌ Errore caricando vector database: {e}")
            self.vector_index = None
            self.metadata = []

    def compute_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Calcola l'embedding di un'immagine usando CLIP.

        Args:
            image: Immagine come array numpy

        Returns:
            Embedding normalizzato
        """
        try:
            if isinstance(image, np.ndarray):
                # Controlla le dimensioni dell'immagine
                if len(image.shape) == 1:
                    # Se l'immagine è 1D, non è valida
                    print(f"⚠️ Immagine 1D non valida, shape: {image.shape}")
                    return np.zeros(512, dtype=np.float32)  # Restituisci embedding zero

                elif len(image.shape) == 2:
                    # Immagine grayscale, convertila a RGB
                    image = np.stack([image, image, image], axis=-1)

                elif len(image.shape) == 3:
                    # Controlla che abbia canali validi
                    if image.shape[-1] not in [3, 4]:
                        print(f"⚠️ Numero di canali non valido: {image.shape[-1]}")
                        return np.zeros(512, dtype=np.float32)

                    # Converti da array numpy a PIL Image
                    if image.shape[-1] == 4:  # RGBA
                        image = Image.fromarray(image.astype(np.uint8), mode='RGBA').convert('RGB')
                    else:  # RGB
                        image = Image.fromarray(image.astype(np.uint8), mode='RGB')
                else:
                    print(f"⚠️ Dimensioni immagine non supportate: {image.shape}")
                    return np.zeros(512, dtype=np.float32)

            elif isinstance(image, Image.Image):
                # Se è già una PIL Image, assicurati che sia RGB
                image = image.convert('RGB')
            else:
                print(f"⚠️ Tipo di immagine non supportato: {type(image)}")
                return np.zeros(512, dtype=np.float32)

            # Verifica che l'immagine abbia dimensioni minime
            if hasattr(image, 'size') and (image.size[0] < 10 or image.size[1] < 10):
                print(f"⚠️ Immagine troppo piccola: {image.size}")
                return np.zeros(512, dtype=np.float32)

            # Preprocessa l'immagine
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Calcola embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

                # Controlla che l'output abbia la forma corretta
                if len(image_features.shape) == 1:
                    # Se è 1D, aggiungi dimensione batch
                    image_features = image_features.unsqueeze(0)

                # Normalizza
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()

        except Exception as e:
            print(f"❌ Errore nel calcolo embedding: {e}")
            print(f"Tipo immagine: {type(image)}")
            if isinstance(image, np.ndarray):
                print(f"Shape immagine: {image.shape}")
            # Restituisci embedding zero come fallback
            return np.zeros(512, dtype=np.float32)

    def search_similar_characters(self, embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Cerca personaggi simili nel vector database.

        Args:
            embedding: Embedding dell'immagine query

        Returns:
            Lista di risultati con similarità e metadati
        """
        if self.vector_index is None:
            return []

        try:
            if self.use_faiss and hasattr(self.vector_index, 'search'):
                # Ricerca FAISS
                embedding = embedding.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(embedding)

                scores, indices = self.vector_index.search(embedding, self.top_k)
                scores = scores[0]
                indices = indices[0]

                # Filtra risultati validi
                valid_indices = indices >= 0
                scores = scores[valid_indices]
                indices = indices[valid_indices]

            else:
                # Ricerca lineare
                scores, indices = self.vector_index.search(embedding, self.top_k)

            # Costruisci risultati
            results = []
            for score, idx in zip(scores, indices):
                if score >= self.similarity_threshold and idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)

            return results

        except Exception as e:
            print(f"Errore nella ricerca: {e}")
            return []

    def analyze_scene(self, image: np.ndarray, use_prompts: bool = False,
                     prompt_points: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Analizza una scena completa identificando tutti i personaggi.

        Args:
            image: Immagine della scena
            use_prompts: Se usare punti prompt per la segmentazione
            prompt_points: Punti per la segmentazione guidata

        Returns:
            Dizionario con risultati dell'analisi
        """
        results = {
            'original_image': image,
            'detected_characters': [],
            'segmentation_masks': [],
            'annotated_image': None,
            'analysis_summary': {}
        }

        try:
            # 1. Segmentazione con SAM
            if use_prompts and prompt_points:
                # Segmentazione guidata
                mask_data = self.sam_segmenter.segment_with_prompts(image, prompt_points)
                masks = [mask_data]
            else:
                # Segmentazione automatica
                masks = self.sam_segmenter.segment_automatic(image)

            print(f"Trovate {len(masks)} maschere")

            # 2. Estrazione oggetti
            extracted_objects = self.sam_segmenter.extract_objects(
                image, masks, min_area=self.min_mask_area
            )

            print(f"Estratti {len(extracted_objects)} oggetti validi")

            # 3. Identificazione personaggi
            for obj in extracted_objects:
                try:
                    # Calcola embedding dell'oggetto
                    obj_embedding = self.compute_image_embedding(obj['image'])

                    # Cerca personaggi simili
                    similar_chars = self.search_similar_characters(obj_embedding)

                    # Determina il personaggio più probabile
                    if similar_chars:
                        best_match = similar_chars[0]
                        character_name = best_match['character']
                        confidence = best_match['similarity_score']
                    else:
                        character_name = "Sconosciuto"
                        confidence = 0.0

                    character_info = {
                        'object_id': obj['object_id'],
                        'character_name': character_name,
                        'confidence': confidence,
                        'bbox': obj['bbox'],
                        'area': obj['area'],
                        'segmentation_confidence': obj['confidence'],
                        'similar_matches': similar_chars[:3]  # Top 3 matches
                    }

                    results['detected_characters'].append(character_info)

                except Exception as e:
                    print(f"Errore identificando oggetto {obj['object_id']}: {e}")
                    continue

            # 4. Salva maschere per visualizzazione
            results['segmentation_masks'] = masks

            # 5. Crea immagine annotata
            results['annotated_image'] = self.create_annotated_image(
                image, results['detected_characters']
            )

            # 6. Crea riassunto
            characters_found = [char['character_name'] for char in results['detected_characters']
                              if char['character_name'] != "Sconosciuto"]
            unique_characters = list(set(characters_found))

            results['analysis_summary'] = {
                'total_objects_detected': len(extracted_objects),
                'characters_identified': len([c for c in results['detected_characters']
                                            if c['character_name'] != "Sconosciuto"]),
                'unique_characters': unique_characters,
                'unknown_objects': len([c for c in results['detected_characters']
                                      if c['character_name'] == "Sconosciuto"]),
                'average_confidence': np.mean([c['confidence'] for c in results['detected_characters']])
                                    if results['detected_characters'] else 0.0
            }

            print(f"✅ Analisi completata: {len(unique_characters)} personaggi unici identificati")

        except Exception as e:
            print(f"❌ Errore nell'analisi della scena: {e}")
            results['error'] = str(e)

        return results

    def create_annotated_image(self, image: np.ndarray,
                             characters: List[Dict[str, Any]]) -> np.ndarray:
        """
        Crea un'immagine annotata con i personaggi identificati.

        Args:
            image: Immagine originale
            characters: Lista di personaggi identificati

        Returns:
            Immagine annotata
        """
        annotated = image.copy()

        # Colori per le annotazioni
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]

        for i, char in enumerate(characters):
            color = colors[i % len(colors)]
            bbox = char['bbox']
            name = char['character_name']
            confidence = char['confidence']

            # Disegna bounding box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Prepara testo
            if confidence > 0:
                label = f"{name} ({confidence:.2f})"
            else:
                label = name

            # Calcola posizione testo
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Sfondo per il testo
            cv2.rectangle(annotated,
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)

            # Testo
            cv2.putText(annotated, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def query_database(self, query_image: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Cerca immagini simili nel database usando un'immagine query.

        Args:
            query_image: Immagine di query
            top_k: Numero di risultati da restituire

        Returns:
            Lista di immagini simili con metadati
        """
        if self.vector_index is None:
            return []

        try:
            # Calcola embedding della query
            query_embedding = self.compute_image_embedding(query_image)

            # Cerca risultati simili
            if self.use_faiss and hasattr(self.vector_index, 'search'):
                # FAISS search
                query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(query_embedding)

                scores, indices = self.vector_index.search(query_embedding, top_k)
                scores = scores[0]
                indices = indices[0]

                # Filtra risultati validi
                valid_indices = indices >= 0
                scores = scores[valid_indices]
                indices = indices[valid_indices]
            else:
                # Linear search
                scores, indices = self.vector_index.search(query_embedding, top_k)

            # Costruisci risultati
            results = []
            for score, idx in zip(scores, indices):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)

                    # Carica immagine se esiste
                    if os.path.exists(result['path']):
                        result['image'] = np.array(Image.open(result['path']).convert('RGB'))

                    results.append(result)

            return results

        except Exception as e:
            print(f"Errore nella query del database: {e}")
            return []
