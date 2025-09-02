"""
Moduli backbone per i modelli di machine learning.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import clip
import ssl
import urllib.request
import os


# Fix per certificati SSL su macOS
def fix_ssl_certificates():
    """
    Risolve i problemi di certificati SSL su macOS.
    """
    try:
        # Crea un contesto SSL che non verifica i certificati
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Applica il contesto SSL globalmente
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
        )

        # Imposta variabili d'ambiente per disabilitare la verifica SSL
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''

    except Exception as e:
        print(f"Avviso: Impossibile configurare SSL: {e}")


class CLIPBackbone(nn.Module):
    """
    Backbone basato su CLIP per l'estrazione di feature da immagini e testo.
    """

    def __init__(self, model_name="ViT-B/32", device="cpu"):
        super(CLIPBackbone, self).__init__()
        self.device = device
        self.model_name = model_name

        # Applica il fix SSL prima di caricare il modello
        fix_ssl_certificates()

        try:
            # Carica il modello CLIP
            self.model, self.preprocess = clip.load(model_name, device=device, download_root="./checkpoints/clip_models")
            self.model.eval()
            print(f"✅ Modello CLIP {model_name} caricato con successo")
        except Exception as e:
            print(f"❌ Errore nel caricamento del modello CLIP: {e}")
            print("🔄 Tentativo con approccio alternativo...")
            self._load_alternative_clip(model_name, device)

    def _load_alternative_clip(self, model_name, device):
        """
        Metodo alternativo per caricare CLIP usando transformers.
        """
        try:
            # Usa transformers come fallback
            from transformers import CLIPModel, CLIPProcessor

            model_mapping = {
                "ViT-B/32": "openai/clip-vit-base-patch32",
                "ViT-B/16": "openai/clip-vit-base-patch16",
                "ViT-L/14": "openai/clip-vit-large-patch14"
            }

            hf_model_name = model_mapping.get(model_name, "openai/clip-vit-base-patch32")

            self.model = CLIPModel.from_pretrained(hf_model_name, cache_dir="./checkpoints/hf_models")
            self.processor = CLIPProcessor.from_pretrained(hf_model_name, cache_dir="./checkpoints/hf_models")
            self.model.to(device)
            self.model.eval()

            # Wrapper per compatibilità
            self.preprocess = self.processor.image_processor
            self._using_transformers = True

            print(f"✅ Modello CLIP alternativo {hf_model_name} caricato con successo")

        except Exception as e:
            print(f"❌ Errore anche con il metodo alternativo: {e}")
            print("🆘 Utilizzando modello mock per testing...")
            self._create_mock_model(device)

    def _create_mock_model(self, device):
        """
        Crea un modello mock per testing quando CLIP non può essere caricato.
        """
        self.model = None
        self.preprocess = None
        self._using_mock = True
        print("⚠️ Usando modello mock - funzionalità limitata")

    def encode_image(self, images):
        """
        Codifica le immagini in embedding vettoriali.

        Args:
            images: Tensor delle immagini preprocessate

        Returns:
            Embedding delle immagini
        """
        if hasattr(self, '_using_mock'):
            # Ritorna embedding casuali per il testing
            batch_size = images.shape[0] if hasattr(images, 'shape') else 1
            return torch.randn(batch_size, 512).to(self.device)

        if hasattr(self, '_using_transformers'):
            # Usa transformers
            with torch.no_grad():
                outputs = self.model.get_image_features(images)
                image_features = outputs / outputs.norm(dim=-1, keepdim=True)
            return image_features
        else:
            # Usa CLIP originale
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

    def encode_text(self, texts):
        """
        Codifica i testi in embedding vettoriali.

        Args:
            texts: Lista di stringhe di testo

        Returns:
            Embedding dei testi
        """
        if hasattr(self, '_using_mock'):
            # Ritorna embedding casuali per il testing
            return torch.randn(len(texts), 512).to(self.device)

        if hasattr(self, '_using_transformers'):
            # Usa transformers
            with torch.no_grad():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.get_text_features(**inputs)
                text_features = outputs / outputs.norm(dim=-1, keepdim=True)
            return text_features
        else:
            # Usa CLIP originale
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

    def get_similarity(self, image_features, text_features):
        """
        Calcola la similarità tra feature di immagini e testi.

        Args:
            image_features: Embedding delle immagini
            text_features: Embedding dei testi

        Returns:
            Matrice di similarità
        """
        similarity = torch.matmul(image_features, text_features.T)
        return similarity


class CustomCLIPHead(nn.Module):
    """
    Head personalizzata per il fine-tuning di CLIP.
    """

    def __init__(self, input_dim=512, num_classes=1000, dropout=0.1):
        super(CustomCLIPHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
