import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BokehDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []
        
        # Standard transform for FLUX/OminiControl (Resize + Normalize)
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self._load_dataset()

    def _load_dataset(self):
        # Scan all scene folders
        for scene_path in self.root_dir.iterdir():
            if not scene_path.is_dir(): continue

            image_dir = scene_path / "image"
            config_dir = scene_path / "config"

            # Define the 'Condition' (Sharpest Image: f/8.00)
            condition_path = image_dir / "aperture_8.00.png"
            if not condition_path.exists(): continue

            # Iterate over all target images (different apertures)
            for json_file in config_dir.glob("aperture_*.json"):
                # Map json -> png (e.g., aperture_0.05.json -> aperture_0.05.png)
                target_img_name = json_file.stem + ".png"
                target_path = image_dir / target_img_name

                if not target_path.exists(): continue
                
                # Skip if target is the same as condition (optional, prevents identity mapping)
                if target_path == condition_path: continue

                self.samples.append({
                    "target_path": str(target_path),
                    "condition_path": str(condition_path),
                    "metadata_path": str(json_file)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Load Images
        target_img = Image.open(item['target_path']).convert("RGB")
        condition_img = Image.open(item['condition_path']).convert("RGB")

        # 2. Parse Metadata for Text Prompt
        with open(item['metadata_path'], 'r') as f:
            meta = json.load(f)
            
        focal = meta.get('focal_length', 50)
        aperture = meta.get('aperture', 8.0)
        focus_dist = meta.get('focus_distance', 1.0)
        
        # Create the Description (Prompt)
        # This teaches the model to associate "aperture f/X" with the visual blur
        text = f"A photo of a scene, focal length {focal}mm, aperture f/{aperture}, focus distance {focus_dist}"

        # 3. Apply Transforms
        # Both images must undergo the same transform to stay aligned
        pixel_values = self.transform(target_img)
        condition_pixel_values = self.transform(condition_img)

        # 4. Define Position Delta
        # Your task is Spatially Aligned (pixels don't move, they just blur).
        # Therefore, we use (0, 0) as per documentation.
        position_delta = [0, 0]

        return {
            "image": pixel_values,           # The Target (Blurred Result)
            "text": text,                    # The Description containing aperture info
            "condition": condition_pixel_values, # The Input (Sharp Reference)
            "position_delta": position_delta # (0,0) for aligned control
        }