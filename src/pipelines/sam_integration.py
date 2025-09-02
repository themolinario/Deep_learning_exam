"""
Integrazione del modello SAM (Segment Anything Model) per la segmentazione automatica.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Any

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("⚠️ SAM non installato. Installa con: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


class SAMSegmenter:
    """
    Wrapper per il modello SAM (Segment Anything Model).
    """

    def __init__(self, model_type: str = "vit_b", checkpoint_path: str = None, device: str = "cpu"):
        """
        Inizializza il segmentatore SAM.

        Args:
            model_type: Tipo di modello SAM ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Percorso del checkpoint SAM
            device: Device per l'inferenza
        """
        self.device = device
        self.model_type = model_type

        if sam_model_registry is None:
            raise ImportError("SAM non disponibile. Installa con: pip install git+https://github.com/facebookresearch/segment-anything.git")

        if checkpoint_path is None:
            # Percorsi di default per i checkpoint SAM
            checkpoint_paths = {
                "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
                "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
                "vit_h": "checkpoints/sam_vit_h_4b8939.pth"
            }
            checkpoint_path = checkpoint_paths.get(model_type)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint SAM non trovato: {checkpoint_path}")

        # Carica il modello SAM
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)

        # Inizializza il generatore di maschere automatico
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def segment_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmenta un'immagine usando SAM.

        Args:
            image: Immagine numpy array in formato RGB

        Returns:
            Lista di maschere con metadata
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Converti da RGB a BGR se necessario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Genera le maschere
        masks = self.mask_generator.generate(image_bgr)

        return masks

    def masks_to_segments(self, masks: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Converte le maschere SAM in segmenti con bounding box.

        Args:
            masks: Lista di maschere SAM
            image_shape: Dimensioni dell'immagine (H, W)

        Returns:
            Lista di segmenti con informazioni geometriche
        """
        segments = []

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']

            # Calcola bounding box
            bbox = mask_data['bbox']  # SAM fornisce già la bbox in formato [x, y, w, h]
            x, y, w, h = bbox

            # Converti in formato [x1, y1, x2, y2]
            bbox_xyxy = [x, y, x + w, y + h]

            segment_info = {
                'id': i,
                'mask': mask,
                'bbox': bbox_xyxy,
                'area': mask_data['area'],
                'stability_score': mask_data['stability_score'],
                'predicted_iou': mask_data['predicted_iou']
            }

            segments.append(segment_info)

        return segments


def download_sam_checkpoint(model_type: str = "vit_b", save_dir: str = "checkpoints/"):
    """
    Scarica il checkpoint SAM se non presente.

    Args:
        model_type: Tipo di modello ('vit_b', 'vit_l', 'vit_h')
        save_dir: Directory dove salvare il checkpoint
    """
    import urllib.request
    import os

    # URL dei checkpoint SAM
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }

    if model_type not in checkpoint_urls:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

    url = checkpoint_urls[model_type]
    filename = url.split('/')[-1]
    filepath = os.path.join(save_dir, filename)

    # Crea la directory se non esiste
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(filepath):
        print(f"Checkpoint già presente: {filepath}")
        return filepath

    print(f"Scaricamento checkpoint SAM {model_type}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Checkpoint salvato in: {filepath}")

    return filepath
