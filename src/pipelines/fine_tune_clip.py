"""
Pipeline per il fine-tuning del modello CLIP sui personaggi di Naruto.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import random

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("⚠️ Transformers non installato. Installa con: pip install transformers")
    CLIPProcessor = None
    CLIPModel = None


class ContrastiveDataset(Dataset):
    """
    Dataset per l'addestramento contrastivo di CLIP.
    """

    def __init__(self, dataset: List[Dict[str, Any]], processor, augment: bool = True):
        """
        Inizializza il dataset contrastivo.

        Args:
            dataset: Lista di informazioni delle immagini
            processor: Processore CLIP
            augment: Se applicare data augmentation
        """
        self.dataset = dataset
        self.processor = processor
        self.augment = augment

        # Organizza per personaggio
        self.character_to_images = {}
        for item in dataset:
            character = item['character']
            if character not in self.character_to_images:
                self.character_to_images[character] = []
            self.character_to_images[character].append(item)

        self.characters = list(self.character_to_images.keys())
        print(f"Dataset contrastivo: {len(self.characters)} personaggi, {len(dataset)} immagini")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Restituisce una tripla (anchor, positive, negative).
        """
        # Anchor: immagine corrente
        anchor_item = self.dataset[idx]
        anchor_character = anchor_item['character']

        # Positive: altra immagine dello stesso personaggio
        positive_candidates = [item for item in self.character_to_images[anchor_character]
                             if item['path'] != anchor_item['path']]

        if len(positive_candidates) == 0:
            # Se non ci sono altre immagini dello stesso personaggio, usa la stessa
            positive_item = anchor_item
        else:
            positive_item = random.choice(positive_candidates)

        # Negative: immagine di un personaggio diverso
        negative_characters = [char for char in self.characters if char != anchor_character]
        if len(negative_characters) == 0:
            # Fallback se c'è solo un personaggio
            negative_item = anchor_item
        else:
            negative_character = random.choice(negative_characters)
            negative_item = random.choice(self.character_to_images[negative_character])

        # Carica e preprocessa le immagini
        anchor_image = self._load_and_process_image(anchor_item['path'])
        positive_image = self._load_and_process_image(positive_item['path'])
        negative_image = self._load_and_process_image(negative_item['path'])

        return {
            'anchor': anchor_image,
            'positive': positive_image,
            'negative': negative_image,
            'anchor_character': anchor_character,
            'positive_character': positive_item['character'],
            'negative_character': negative_item['character']
        }

    def _load_and_process_image(self, image_path: str) -> torch.Tensor:
        """
        Carica e preprocessa un'immagine.

        Args:
            image_path: Percorso dell'immagine

        Returns:
            Tensor preprocessato
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # Applica data augmentation se richiesto
            if self.augment and random.random() > 0.5:
                # Semplice augmentation (flip orizzontale)
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Preprocessa con CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)

        except Exception as e:
            print(f"Errore caricando {image_path}: {e}")
            # Restituisci un'immagine nera come fallback
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(images=dummy_image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)


class CLIPFineTuner:
    """
    Classe per il fine-tuning di CLIP con apprendimento contrastivo.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il fine-tuner.

        Args:
            config: Configurazione del progetto
        """
        self.config = config
        self.device = config.get('models', {}).get('clip', {}).get('device', 'cpu')

        # Configurazione training
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        self.learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
        self.num_epochs = config.get('training', {}).get('num_epochs', 10)
        self.temperature = config.get('training', {}).get('temperature', 0.1)
        self.margin = config.get('training', {}).get('margin', 0.2)
        self.weight_decay = config.get('training', {}).get('weight_decay', 1e-4)
        self.save_every = config.get('training', {}).get('save_every', 2)
        self.checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints/fine_tuned')

        # Inizializza modello e processore
        model_name = config.get('models', {}).get('clip', {}).get('model_name', 'openai/clip-vit-base-patch32')

        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("Transformers non installato. Installa con: pip install transformers")

        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Ottimizzatore
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = None

        # Metriche di training
        self.train_losses = []
        self.val_losses = []

        # Crea directory checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def contrastive_loss(self, anchor_features, positive_features, negative_features):
        """
        Calcola la contrastive loss.

        Args:
            anchor_features: Features dell'immagine anchor
            positive_features: Features dell'immagine positive
            negative_features: Features dell'immagine negative

        Returns:
            Loss calcolata
        """
        # Normalizza le features
        anchor_features = anchor_features / anchor_features.norm(p=2, dim=-1, keepdim=True)
        positive_features = positive_features / positive_features.norm(p=2, dim=-1, keepdim=True)
        negative_features = negative_features / negative_features.norm(p=2, dim=-1, keepdim=True)

        # Calcola similarità
        pos_similarity = torch.sum(anchor_features * positive_features, dim=-1) / self.temperature
        neg_similarity = torch.sum(anchor_features * negative_features, dim=-1) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss

    def triplet_loss(self, anchor_features, positive_features, negative_features):
        """
        Calcola la triplet loss come alternativa.

        Args:
            anchor_features: Features dell'immagine anchor
            positive_features: Features dell'immagine positive
            negative_features: Features dell'immagine negative

        Returns:
            Loss calcolata
        """
        # Normalizza le features
        anchor_features = anchor_features / anchor_features.norm(p=2, dim=-1, keepdim=True)
        positive_features = positive_features / positive_features.norm(p=2, dim=-1, keepdim=True)
        negative_features = negative_features / negative_features.norm(p=2, dim=-1, keepdim=True)

        # Distanze
        pos_distance = torch.norm(anchor_features - positive_features, p=2, dim=1)
        neg_distance = torch.norm(anchor_features - negative_features, p=2, dim=1)

        # Triplet loss
        loss = torch.relu(pos_distance - neg_distance + self.margin).mean()

        return loss

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Addestra per un'epoca.

        Args:
            dataloader: DataLoader del training set
            epoch: Numero dell'epoca corrente

        Returns:
            Loss media dell'epoca
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            self.optimizer.zero_grad()

            # Sposta i dati sul device
            anchor_images = batch['anchor'].to(self.device)
            positive_images = batch['positive'].to(self.device)
            negative_images = batch['negative'].to(self.device)

            # Forward pass
            anchor_features = self.model.get_image_features(anchor_images)
            positive_features = self.model.get_image_features(positive_images)
            negative_features = self.model.get_image_features(negative_images)

            # Calcola loss (usa contrastive loss di default)
            loss = self.contrastive_loss(anchor_features, positive_features, negative_features)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Aggiorna progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        """
        Valuta il modello sul validation set.

        Args:
            dataloader: DataLoader del validation set

        Returns:
            Loss media di validazione
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validazione"):
                # Sposta i dati sul device
                anchor_images = batch['anchor'].to(self.device)
                positive_images = batch['positive'].to(self.device)
                negative_images = batch['negative'].to(self.device)

                # Forward pass
                anchor_features = self.model.get_image_features(anchor_images)
                positive_features = self.model.get_image_features(positive_images)
                negative_features = self.model.get_image_features(negative_images)

                # Calcola loss
                loss = self.contrastive_loss(anchor_features, positive_features, negative_features)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """
        Salva un checkpoint del modello.

        Args:
            epoch: Numero dell'epoca
            train_loss: Loss di training
            val_loss: Loss di validazione
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f'clip_finetuned_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Carica un checkpoint del modello.

        Args:
            checkpoint_path: Percorso del checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint caricato da: {checkpoint_path}")
        print(f"Epoca: {checkpoint['epoch']}, Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

    def train(self, train_dataset: List[Dict[str, Any]], val_dataset: List[Dict[str, Any]] = None):
        """
        Addestra il modello CLIP.

        Args:
            train_dataset: Dataset di training
            val_dataset: Dataset di validazione (opzionale)
        """
        print("=== Inizio Fine-tuning CLIP ===")

        # Crea dataset contrastivi
        train_contrastive = ContrastiveDataset(train_dataset, self.processor, augment=True)
        train_loader = DataLoader(train_contrastive, batch_size=self.batch_size, shuffle=True, num_workers=0)

        val_loader = None
        if val_dataset:
            val_contrastive = ContrastiveDataset(val_dataset, self.processor, augment=False)
            val_loader = DataLoader(val_contrastive, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Setup scheduler
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)

        print(f"Training su {len(train_dataset)} immagini per {self.num_epochs} epoche")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validazione
            val_loss = 0.0
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log risultati
            print(f"Epoca {epoch+1}/{self.num_epochs}: Train Loss = {train_loss:.4f}", end="")
            if val_loader:
                print(f", Val Loss = {val_loss:.4f}")

                # Salva il miglior modello
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': self.config
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"🏆 Nuovo miglior modello salvato! Val Loss: {val_loss:.4f}")
            else:
                print()

            # Salva checkpoint periodicamente
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

        # Salva modello finale
        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        final_checkpoint = {
            'epoch': self.num_epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(final_checkpoint, final_checkpoint_path)

        print("=== Fine-tuning completato ===")
        print(f"Modello finale salvato in: {final_checkpoint_path}")

        # Salva metriche di training
        metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metriche salvate in: {metrics_path}")


def split_dataset(dataset: List[Dict[str, Any]], train_ratio: float = 0.8,
                 val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Divide il dataset in training, validation e test set.

    Args:
        dataset: Dataset completo
        train_ratio: Rapporto per il training set
        val_ratio: Rapporto per il validation set

    Returns:
        Tupla con (train_set, val_set, test_set)
    """
    # Organizza per personaggio per garantire che ogni personaggio sia in tutti i set
    character_to_images = {}
    for item in dataset:
        character = item['character']
        if character not in character_to_images:
            character_to_images[character] = []
        character_to_images[character].append(item)

    train_set = []
    val_set = []
    test_set = []

    for character, images in character_to_images.items():
        # Mescola le immagini per ogni personaggio
        random.shuffle(images)

        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        # Assicurati che ogni set abbia almeno un'immagine se possibile
        if n_train == 0 and n_images > 0:
            n_train = 1
        if n_val == 0 and n_images > 1:
            n_val = 1

        train_set.extend(images[:n_train])
        val_set.extend(images[n_train:n_train + n_val])
        test_set.extend(images[n_train + n_val:])

    print(f"Dataset diviso: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set

