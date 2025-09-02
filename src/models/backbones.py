"""
Moduli backbone per i modelli di machine learning.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import clip


class CLIPBackbone(nn.Module):
    """
    Backbone basato su CLIP per l'estrazione di feature da immagini e testo.
    """

    def __init__(self, model_name="ViT-B/32", device="cpu"):
        super(CLIPBackbone, self).__init__()
        self.device = device
        self.model_name = model_name

        # Carica il modello CLIP
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def encode_image(self, images):
        """
        Codifica le immagini in embedding vettoriali.

        Args:
            images: Tensor delle immagini preprocessate

        Returns:
            Embedding delle immagini
        """
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
