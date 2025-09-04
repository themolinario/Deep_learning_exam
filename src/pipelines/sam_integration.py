"""
Integrazione del modello SAM (Segment Anything Model) per la segmentazione automatica.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Any
import os

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError:
    print("⚠️ SAM non installato. Installa con: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SamAutomaticMaskGenerator = None
    sam_model_registry = None
    SamPredictor = None


class SAMSegmenter:
    """
    Wrapper per il modello SAM (Segment Anything Model).
    """

    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = None, device: str = "cpu"):
        """
        Inizializza il segmentatore SAM.

        Args:
            model_type: Tipo di modello SAM ("vit_b", "vit_l", "vit_h")
            checkpoint_path: Percorso del checkpoint SAM
            device: Device per l'inferenza
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        if sam_model_registry is None:
            raise ImportError("SAM non installato. Installa con: pip install git+https://github.com/facebookresearch/segment-anything.git")

        # Scarica checkpoint se non esiste
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"📥 Scaricamento checkpoint SAM {model_type}...")
            checkpoint_path = self.download_sam_checkpoint(model_type, "checkpoints")
            self.checkpoint_path = checkpoint_path

        # Carica il modello SAM
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            print(f"✅ SAM {model_type} caricato con checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Errore caricando SAM con checkpoint: {e}")
            print("Tentativo di caricamento senza checkpoint...")
            try:
                self.sam = sam_model_registry[model_type]()
                print("⚠️ SAM caricato senza checkpoint (performance ridotte)")
            except Exception as e2:
                raise RuntimeError(f"Impossibile caricare SAM: {e2}")

        self.sam.to(device=device)

        # Inizializza il generatore di maschere automatico
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000
        )

        # Inizializza il predictor per segmentazione guidata
        self.predictor = SamPredictor(self.sam)

    def segment_automatic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmenta automaticamente tutti gli oggetti nell'immagine.

        Args:
            image: Immagine RGB come array numpy

        Returns:
            Lista di maschere con metadati
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Converti in RGB se necessario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        else:
            image_rgb = image

        masks = self.mask_generator.generate(image_rgb)

        # Ordina le maschere per area (dalle più grandi alle più piccole)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        return masks

    def segment_with_prompts(self, image: np.ndarray, points: List[Tuple[int, int]],
                           labels: List[int] = None) -> Dict[str, Any]:
        """
        Segmenta l'immagine usando prompt di punti.

        Args:
            image: Immagine RGB come array numpy
            points: Lista di coordinate (x, y) dei punti
            labels: Lista di etichette (1 per positivo, 0 per negativo)

        Returns:
            Dizionario con maschere e metadati
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if labels is None:
            labels = [1] * len(points)

        self.predictor.set_image(image)

        input_points = np.array(points)
        input_labels = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

        # Seleziona la maschera con il punteggio più alto
        best_mask_idx = np.argmax(scores)

        return {
            'segmentation': masks[best_mask_idx],
            'score': scores[best_mask_idx],
            'logits': logits[best_mask_idx]
        }

    def extract_objects(self, image: np.ndarray, masks: List[Dict[str, Any]],
                       min_area: int = 1000) -> List[Dict[str, Any]]:
        """
        Estrae gli oggetti segmentati dall'immagine originale.

        Args:
            image: Immagine originale
            masks: Lista di maschere da SAM
            min_area: Area minima per considerare una maschera

        Returns:
            Lista di oggetti estratti con metadati
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        extracted_objects = []

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            area = mask_data.get('area', np.sum(mask))

            # Filtra maschere troppo piccole
            if area < min_area:
                continue

            # Trova il bounding box della maschera
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue

            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Estrai la regione dell'immagine
            roi = image[y_min:y_max+1, x_min:x_max+1]
            roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

            # Crea immagine RGBA con sfondo trasparente
            if len(roi.shape) == 3:
                rgba_image = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.uint8)
                rgba_image[:, :, :3] = roi
                rgba_image[:, :, 3] = roi_mask * 255
            else:
                rgba_image = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.uint8)
                rgba_image[:, :, 0] = roi
                rgba_image[:, :, 1] = roi
                rgba_image[:, :, 2] = roi
                rgba_image[:, :, 3] = roi_mask * 255

            extracted_objects.append({
                'object_id': i,
                'image': rgba_image,
                'mask': roi_mask,
                'bbox': (x_min, y_min, x_max, y_max),
                'area': area,
                'confidence': mask_data.get('stability_score', 0.0)
            })

        return extracted_objects

    def visualize_masks(self, image: np.ndarray, masks: List[Dict[str, Any]],
                       show_bbox: bool = True, show_labels: bool = True) -> np.ndarray:
        """
        Visualizza le maschere sull'immagine originale.

        Args:
            image: Immagine originale
            masks: Lista di maschere
            show_bbox: Mostra bounding box
            show_labels: Mostra etichette

        Returns:
            Immagine con maschere visualizzate
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        vis_image = image.copy()

        # Colori per le maschere
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0]
        ]

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = colors[i % len(colors)]

            # Applica la maschera con trasparenza
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask] = color
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)

            if show_bbox:
                # Disegna bounding box
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

            if show_labels:
                # Aggiungi etichetta
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_center = int(coords[0].mean())
                    x_center = int(coords[1].mean())
                    cv2.putText(vis_image, f"Object {i}", (x_center-30, y_center),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_image

    @staticmethod
    def download_sam_checkpoint(model_type: str = "vit_b", save_dir: str = "checkpoints") -> str:
        """
        Scarica il checkpoint SAM se non esiste.

        Args:
            model_type: Tipo di modello SAM
            save_dir: Directory dove salvare il checkpoint

        Returns:
            Percorso del checkpoint scaricato
        """
        import urllib.request

        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

        if model_type not in checkpoint_urls:
            raise ValueError(f"Tipo di modello non supportato: {model_type}")

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"sam_{model_type}_checkpoint.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Scaricando checkpoint SAM {model_type}...")
            url = checkpoint_urls[model_type]
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Checkpoint salvato in: {checkpoint_path}")

        return checkpoint_path
