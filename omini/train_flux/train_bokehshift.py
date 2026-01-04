# import torch
# import os
# import random
# from torch.utils.data import DataLoader, Dataset

# from PIL import Image

# from datasets import load_dataset

# from .trainer import OminiModel, get_config, train
# from ..pipeline.flux_omini import Condition, generate


# class CustomDataset(Dataset):
#     def __getitem__(self, idx):
#         # TODO: Implement the logic to load your custom dataset
#         raise NotImplementedError("Custom dataset loading not implemented")


# @torch.no_grad()
# def test_function(model, save_path, file_name):
#     # TODO: Implement the logic to generate a sample using the model
#     raise NotImplementedError("Sample generation not implemented")


# def main():
#     # Initialize
#     config = get_config()
#     training_config = config["train"]
#     torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

#     # Initialize custom dataset
#     dataset = CustomDataset()

#     # Initialize model
#     trainable_model = OminiModel(
#         flux_pipe_id=config["flux_path"],
#         lora_config=training_config["lora_config"],
#         device=f"cuda",
#         dtype=getattr(torch, config["dtype"]),
#         optimizer_config=training_config["optimizer"],
#         model_config=config.get("model", {}),
#         gradient_checkpointing=training_config.get("gradient_checkpointing", False),
#     )

#     train(dataset, trainable_model, config, test_function)


# if __name__ == "__main__":
#     main()




# BOKEHSHIFT #
import torch
import os
import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Standard OminiControl imports
from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate

# --- 1. Custom Dataset Implementation ---
class CustomDataset(Dataset):
    def __init__(self, root_dir="./scenes", image_size=512):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []

        # Standard transforms for FLUX (Resize, Crop, Normalize to [-1, 1])
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self._load_dataset()
        print(f"Dataset loaded: {len(self.samples)} samples found in {root_dir}")

    def _load_dataset(self):
        # Scan all scene folders (e.g., sceneA, sceneB)
        for scene_path in self.root_dir.iterdir():
            if not scene_path.is_dir(): continue

            image_dir = scene_path / "image"
            config_dir = scene_path / "config"

            # Define the 'Condition' (Sharpest Image: f/8.00)
            # This is the "structure" we want to preserve while changing blur
            condition_path = image_dir / "aperture_8.00.png"
            if not condition_path.exists(): 
                print(f"Warning: Skipping {scene_path.name}, missing aperture_8.00.png")
                continue

            # Iterate over all target images (different apertures)
            for json_file in config_dir.glob("aperture_*.json"):
                # Map json -> png (e.g., aperture_0.05.json -> aperture_0.05.png)
                target_img_name = json_file.stem + ".png"
                target_path = image_dir / target_img_name

                if not target_path.exists(): continue
                
                # Optional: Skip using the sharp image as a target to force learning blur
                # if target_path == condition_path: continue 

                self.samples.append({
                    "target_path": str(target_path),
                    "condition_path": str(condition_path),
                    "metadata_path": str(json_file)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # A. Load Images
        target_img = Image.open(item['target_path']).convert("RGB")
        condition_img = Image.open(item['condition_path']).convert("RGB")

        # B. Parse Metadata for Prompt
        with open(item['metadata_path'], 'r') as f:
            meta = json.load(f)
            
        focal = meta.get('focal_length', 50)
        aperture = meta.get('aperture', 8.0)
        focus_dist = meta.get('focus_distance', 1.0)
        
        # C. Construct Prompt
        # We embed the parameters so the model learns that "f/0.05" = BLUR
        text_prompt = f"A photo of a scene, focal length {focal}mm, aperture f/{aperture}, focus distance {focus_dist}"

        # D. Apply Transforms
        pixel_values = self.transform(target_img)
        condition_pixel_values = self.transform(condition_img)

        # E. Define Position Delta
        # Your task is Spatially Aligned (pixels don't move, just blur).
        # Thus, position_delta is (0,0).
        position_delta = [0, 0]

        return {
            "image": pixel_values,            # Target (Blurred)
            "description": text_prompt,              # Description
            "conditions": condition_pixel_values, # Input (Sharp)
            "position_delta": position_delta  # Alignment
        }


# --- 2. Test Function Implementation ---
@torch.no_grad()
def test_function(model, save_path, file_name):
    """
    Generates validation images to check if aperture control works.
    We take a sharp image and try to generate 3 versions: Blurry, Medium, Sharp.
    """
    print(f"Running test generation for {file_name}...")
    
    # 1. Access the underlying pipeline from OminiModel
    # Note: Depending on OminiModel implementation, this is usually model.pipeline or model.pipe
    pipe = model.pipeline 
    
    # 2. Get a test condition (Use a fixed image from dataset path or generated noise)
    # Here we try to load the first valid condition from the dataset directory for consistency
    dataset_root = Path("./scenes")
    first_scene = next(dataset_root.iterdir())
    condition_path = first_scene / "image" / "aperture_8.00.png"
    
    if not condition_path.exists():
        print("Test condition not found, skipping generation.")
        return

    condition_img = Image.open(condition_path).convert("RGB").resize((512, 512))
    
    # 3. Define Test Prompts (Varying Aperture)
    prompts = [
        "A photo of a scene, focal length 50mm, aperture f/0.05, focus distance 1.0", # Heavy Bokeh
        "A photo of a scene, focal length 50mm, aperture f/2.0, focus distance 1.0",  # Medium
        "A photo of a scene, focal length 50mm, aperture f/8.0, focus distance 1.0"   # Sharp
    ]
    
    generated_images = []
    
    # 4. Generate
    for prompt in prompts:
        # Wrap image in Condition object (required by generate function)
        # Note: 'condition_type' might be 'canny', 'depth', or 'subject'. 
        # For general spatial control, we often treat it as 'depth' or just generic 'spatial'.
        cond = Condition(condition_type="spatial", condition_image=condition_img)
        
        result = generate(
            pipe,
            prompt=prompt,
            conditions=[cond],
            num_inference_steps=20,
            height=512,
            width=512,
            guidance_scale=3.5
        ).images[0]
        generated_images.append(result)
        
    # 5. Save Grid
    # Combine images side-by-side: [Blurry | Medium | Sharp]
    grid_width = 512 * 3
    grid = Image.new("RGB", (grid_width, 512))
    grid.paste(generated_images[0], (0, 0))
    grid.paste(generated_images[1], (512, 0))
    grid.paste(generated_images[2], (1024, 0))
    
    full_save_path = os.path.join(save_path, f"{file_name}.png")
    grid.save(full_save_path)
    print(f"Saved validation grid to {full_save_path}")


def main():
    # Initialize
    config = get_config()
    config["train"]["batch_size"] = 1
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Initialize custom dataset
    # Ensure your "./scenes" directory exists and follows the structure
    dataset = CustomDataset(root_dir="./scenes", image_size=512)

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()