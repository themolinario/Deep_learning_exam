"""
Implementazione di modelli di embedding alternativi a CLIP.
Include DINOv2 e BLIP-2 come richiesto dal progetto.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import warnings

try:
    from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
    import timm
except ImportError:
    print("⚠️ Alcune librerie non installate. Installa con: pip install transformers timm")
    AutoModel = None
    AutoProcessor = None
    BlipProcessor = None
    BlipForConditionalGeneration = None
    timm = None


class AlternativeEmbeddingModel:
    """
    Classe base per modelli di embedding alternativi.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def compute_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Calcola l'embedding di un'immagine.

        Args:
            image: Immagine PIL

        Returns:
            Embedding normalizzato
        """
        raise NotImplementedError("Deve essere implementato dalle sottoclassi")

    def get_embedding_dim(self) -> int:
        """
        Restituisce la dimensione dell'embedding.

        Returns:
            Dimensione dell'embedding
        """
        raise NotImplementedError("Deve essere implementato dalle sottoclassi")


class DINOv2Model(AlternativeEmbeddingModel):
    """
    Implementazione del modello DINOv2 per embedding di immagini.
    """

    def __init__(self, model_variant: str = "dinov2-base", device: str = "cpu"):
        """
        Inizializza il modello DINOv2.

        Args:
            model_variant: Variante del modello ("dinov2-base", "dinov2-large", "dinov2-giant")
            device: Device per l'inferenza
        """
        model_name = f"facebook/{model_variant}"
        super().__init__(model_name, device)

        if AutoModel is None or AutoProcessor is None:
            raise ImportError("Transformers non installato. Installa con: pip install transformers")

        try:
            # Carica modello e processore
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()

            # Determina dimensione embedding
            self.embedding_dim = self.model.config.hidden_size

            print(f"✅ DINOv2 caricato: {model_name}")

        except Exception as e:
            print(f"❌ Errore caricando DINOv2: {e}")
            # Fallback a implementazione con timm se disponibile
            self._try_timm_fallback(model_variant)

    def _try_timm_fallback(self, model_variant: str):
        """
        Prova a caricare DINOv2 tramite timm come fallback.

        Args:
            model_variant: Variante del modello
        """
        if timm is None:
            raise ImportError("Né transformers né timm disponibili per DINOv2")

        try:
            # Mappa nomi modelli per timm
            timm_model_names = {
                "dinov2-base": "vit_base_patch14_dinov2.lvd142m",
                "dinov2-large": "vit_large_patch14_dinov2.lvd142m",
                "dinov2-giant": "vit_giant_patch14_dinov2.lvd142m"
            }

            timm_name = timm_model_names.get(model_variant, "vit_base_patch14_dinov2.lvd142m")

            self.model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            self.model.to(self.device)
            self.model.eval()

            # Per timm, usiamo trasformazioni standard
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            self.embedding_dim = self.model.num_features
            self.processor = None  # Useremo transform personalizzato

            print(f"✅ DINOv2 caricato tramite timm: {timm_name}")

        except Exception as e:
            raise RuntimeError(f"Impossibile caricare DINOv2: {e}")

    def compute_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Calcola l'embedding usando DINOv2.

        Args:
            image: Immagine PIL

        Returns:
            Embedding normalizzato
        """
        try:
            with torch.no_grad():
                if self.processor is not None:
                    # Usa processor di transformers
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs)
                    # Prendi l'embedding del token CLS
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                else:
                    # Usa transform di timm
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    embedding = self.model(input_tensor).squeeze()

                # Normalizza l'embedding
                embedding = embedding / embedding.norm(p=2)

                return embedding.cpu().numpy()

        except Exception as e:
            print(f"Errore calcolando embedding DINOv2: {e}")
            # Restituisci embedding zero come fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_embedding_dim(self) -> int:
        """Restituisce la dimensione dell'embedding DINOv2."""
        return self.embedding_dim


class BLIP2Model(AlternativeEmbeddingModel):
    """
    Implementazione del modello BLIP-2 per embedding di immagini.
    """

    def __init__(self, model_variant: str = "blip2-opt-2.7b", device: str = "cpu"):
        """
        Inizializza il modello BLIP-2.

        Args:
            model_variant: Variante del modello BLIP-2
            device: Device per l'inferenza
        """
        model_name = f"Salesforce/{model_variant}"
        super().__init__(model_name, device)

        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise ImportError("Transformers non installato. Installa con: pip install transformers")

        try:
            # Per BLIP-2, usiamo una versione più leggera per gli embedding
            fallback_model = "Salesforce/blip2-opt-2.7b"

            self.processor = BlipProcessor.from_pretrained(fallback_model)
            self.model = BlipForConditionalGeneration.from_pretrained(fallback_model)
            self.model.to(device)
            self.model.eval()

            # BLIP-2 usa il vision encoder per gli embedding
            self.embedding_dim = self.model.config.vision_config.hidden_size

            print(f"✅ BLIP-2 caricato: {fallback_model}")

        except Exception as e:
            print(f"❌ Errore caricando BLIP-2: {e}")
            # Prova con modello più semplice
            self._try_simple_blip()

    def _try_simple_blip(self):
        """
        Prova a caricare un modello BLIP più semplice come fallback.
        """
        try:
            from transformers import BlipModel

            simple_model = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(simple_model)
            self.model = BlipModel.from_pretrained(simple_model)
            self.model.to(self.device)
            self.model.eval()

            self.embedding_dim = self.model.config.vision_config.hidden_size

            print(f"✅ BLIP semplice caricato: {simple_model}")

        except Exception as e:
            raise RuntimeError(f"Impossibile caricare BLIP: {e}")

    def compute_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Calcola l'embedding usando BLIP-2.

        Args:
            image: Immagine PIL

        Returns:
            Embedding normalizzato
        """
        try:
            with torch.no_grad():
                # Preprocessa l'immagine
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Estrai features dal vision encoder
                if hasattr(self.model, 'vision_model'):
                    # BLIP-2
                    vision_outputs = self.model.vision_model(**inputs)
                    # Usa l'embedding pooled
                    embedding = vision_outputs.pooler_output.squeeze()
                else:
                    # BLIP semplice
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs.squeeze()

                # Normalizza l'embedding
                embedding = embedding / embedding.norm(p=2)

                return embedding.cpu().numpy()

        except Exception as e:
            print(f"Errore calcolando embedding BLIP-2: {e}")
            # Restituisci embedding zero come fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_embedding_dim(self) -> int:
        """Restituisce la dimensione dell'embedding BLIP-2."""
        return self.embedding_dim


class ModelFactory:
    """
    Factory per creare modelli di embedding alternativi.
    """

    @staticmethod
    def create_model(model_type: str, model_variant: str = None, device: str = "cpu") -> AlternativeEmbeddingModel:
        """
        Crea un modello di embedding alternativo.

        Args:
            model_type: Tipo di modello ("dinov2", "blip2")
            model_variant: Variante specifica del modello
            device: Device per l'inferenza

        Returns:
            Istanza del modello
        """
        model_type = model_type.lower()

        if model_type == "dinov2":
            variant = model_variant or "dinov2-base"
            return DINOv2Model(variant, device)

        elif model_type == "blip2":
            variant = model_variant or "blip2-opt-2.7b"
            return BLIP2Model(variant, device)

        else:
            raise ValueError(f"Tipo di modello non supportato: {model_type}")

    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """
        Restituisce i modelli disponibili.

        Returns:
            Dizionario con tipi e varianti disponibili
        """
        return {
            "dinov2": ["dinov2-base", "dinov2-large", "dinov2-giant"],
            "blip2": ["blip2-opt-2.7b", "blip2-flan-t5-xl"]
        }


class AlternativeModelEvaluator:
    """
    Classe per valutare e confrontare modelli alternativi.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il valutatore di modelli alternativi.

        Args:
            config: Configurazione del progetto
        """
        self.config = config
        self.device = config.get('models', {}).get('clip', {}).get('device', 'cpu')

        # Modelli da confrontare
        self.models = {}

    def load_models(self, model_configs: list):
        """
        Carica i modelli da valutare.

        Args:
            model_configs: Lista di configurazioni modelli
                          [{"type": "dinov2", "variant": "dinov2-base"}, ...]
        """
        for config in model_configs:
            model_type = config['type']
            variant = config.get('variant')

            try:
                model = ModelFactory.create_model(model_type, variant, self.device)
                model_name = f"{model_type}_{variant or 'default'}"
                self.models[model_name] = model
                print(f"✅ Modello caricato: {model_name}")

            except Exception as e:
                print(f"❌ Errore caricando {model_type}: {e}")

    def compare_embeddings(self, test_images: list) -> Dict[str, Any]:
        """
        Confronta gli embedding generati dai diversi modelli.

        Args:
            test_images: Lista di immagini PIL per il test

        Returns:
            Dizionario con risultati del confronto
        """
        if not self.models:
            print("❌ Nessun modello caricato")
            return {}

        results = {
            'model_embeddings': {},
            'similarity_matrices': {},
            'embedding_stats': {}
        }

        print(f"Confronto embeddings su {len(test_images)} immagini...")

        # Calcola embeddings per ogni modello
        for model_name, model in self.models.items():
            print(f"Calcolando embeddings per {model_name}...")

            embeddings = []
            for i, image in enumerate(test_images):
                try:
                    embedding = model.compute_embedding(image)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Errore con immagine {i} per {model_name}: {e}")
                    # Aggiungi embedding zero come fallback
                    embeddings.append(np.zeros(model.get_embedding_dim()))

            embeddings_array = np.vstack(embeddings)
            results['model_embeddings'][model_name] = embeddings_array

            # Calcola statistiche
            results['embedding_stats'][model_name] = {
                'mean_norm': np.mean(np.linalg.norm(embeddings_array, axis=1)),
                'std_norm': np.std(np.linalg.norm(embeddings_array, axis=1)),
                'embedding_dim': model.get_embedding_dim()
            }

            # Calcola matrice di similarità
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            results['similarity_matrices'][model_name] = similarity_matrix

        return results

    def benchmark_speed(self, test_images: list) -> Dict[str, float]:
        """
        Confronta la velocità dei diversi modelli.

        Args:
            test_images: Lista di immagini per il benchmark

        Returns:
            Dizionario con tempi medi per modello
        """
        import time

        speed_results = {}

        for model_name, model in self.models.items():
            times = []

            print(f"Benchmark velocità per {model_name}...")

            for image in test_images:
                start_time = time.time()
                try:
                    _ = model.compute_embedding(image)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"Errore durante benchmark {model_name}: {e}")
                    times.append(float('inf'))

            avg_time = np.mean(times)
            speed_results[model_name] = avg_time

            print(f"{model_name}: {avg_time:.3f}s per immagine")

        return speed_results
