"""
Modulo per il fine-tuning del modello CLIP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import clip
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import os

from src.models.backbones import CLIPBackbone


class CLIPDataset(Dataset):
    """
    Dataset personalizzato per il fine-tuning di CLIP.
    """

    def __init__(self, image_paths, texts, preprocess, device="cpu"):
        self.image_paths = image_paths
        self.texts = texts
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.preprocess(image)
        text = self.texts[idx]

        return image, text


class CLIPFineTuner:
    """
    Classe per il fine-tuning del modello CLIP.
    """

    def __init__(self, config_path="config.yaml"):
        # Carica la configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['clip']['device'] if torch.cuda.is_available() else 'cpu')

        # Inizializza il modello
        self.model = CLIPBackbone(
            model_name=self.config['clip']['model_name'],
            device=self.device
        )

        # Parametri di training
        self.batch_size = self.config['clip']['batch_size']
        self.learning_rate = self.config['clip']['learning_rate']
        self.epochs = self.config['clip']['epochs']

        # Optimizer
        self.optimizer = optim.Adam(self.model.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def prepare_data(self, image_paths, texts):
        """
        Prepara i dati per il training.

        Args:
            image_paths: Lista dei percorsi delle immagini
            texts: Lista dei testi corrispondenti
        """
        dataset = CLIPDataset(image_paths, texts, self.model.preprocess, self.device)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def train_epoch(self):
        """
        Esegue un'epoca di training.
        """
        self.model.model.train()
        total_loss = 0

        for batch_idx, (images, texts) in enumerate(tqdm(self.dataloader, desc="Training")):
            images = images.to(self.device)

            # Tokenizza i testi
            text_tokens = clip.tokenize(texts).to(self.device)

            # Forward pass
            logits_per_image, logits_per_text = self.model.model(images, text_tokens)

            # Calcola la loss
            batch_size = images.shape[0]
            labels = torch.arange(batch_size).to(self.device)

            loss_img = self.criterion(logits_per_image, labels)
            loss_txt = self.criterion(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def evaluate(self, val_dataloader):
        """
        Valuta il modello sul set di validazione.
        """
        self.model.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validation"):
                images = images.to(self.device)
                text_tokens = clip.tokenize(texts).to(self.device)

                logits_per_image, logits_per_text = self.model.model(images, text_tokens)

                batch_size = images.shape[0]
                labels = torch.arange(batch_size).to(self.device)

                loss_img = self.criterion(logits_per_image, labels)
                loss_txt = self.criterion(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2

                total_loss += loss.item()

                # Calcola l'accuratezza
                _, predicted = torch.max(logits_per_image.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(val_dataloader)

        return avg_loss, accuracy

    def train(self, image_paths, texts, val_image_paths=None, val_texts=None):
        """
        Esegue il training completo del modello.
        """
        # Prepara i dati di training
        self.prepare_data(image_paths, texts)

        # Prepara i dati di validazione se forniti
        val_dataloader = None
        if val_image_paths and val_texts:
            val_dataset = CLIPDataset(val_image_paths, val_texts, self.model.preprocess, self.device)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")

            # Validation
            if val_dataloader:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

                # Salva il miglior modello
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)

            # Salva checkpoint periodicamente
            if (epoch + 1) % self.config['checkpoints']['save_frequency'] == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch, is_best=False):
        """
        Salva un checkpoint del modello.
        """
        checkpoint_dir = self.config['checkpoints']['save_path']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        filename = f"clip_epoch_{epoch + 1}.pth"
        if is_best:
            filename = "clip_best.pth"

        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
        print(f"Checkpoint salvato: {filename}")

    def load_checkpoint(self, checkpoint_path):
        """
        Carica un checkpoint del modello.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']
