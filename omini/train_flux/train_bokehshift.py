


# # BOKEHSHIFT #
# import torch
# import os
# import json
# import random
# from pathlib import Path
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as T

# # Standard OminiControl imports
# from .trainer import OminiModel, get_config, train
# from ..pipeline.flux_omini import Condition, generate

# # --- 1. Custom Dataset Implementation ---
# class CustomDataset(Dataset):
#     def __init__(self, root_dir="./scenes", image_size=512):
#         self.root_dir = Path(root_dir)
#         self.image_size = image_size
#         self.samples = []

#         # Standard transforms for FLUX (Resize, Crop, Normalize to [-1, 1])
#         self.transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.CenterCrop(image_size),
#             T.ToTensor(),
#             T.Normalize([0.5], [0.5])
#         ])

#         self._load_dataset()
#         print(f"Dataset loaded: {len(self.samples)} samples found in {root_dir}")

#     def _load_dataset(self):
#         # Scan all scene folders (e.g., sceneA, sceneB)
#         for scene_path in self.root_dir.iterdir():
#             if not scene_path.is_dir(): continue

#             image_dir = scene_path / "image"
#             config_dir = scene_path / "config"

#             # Define the 'Condition' (Sharpest Image: f/8.00)
#             # This is the "structure" we want to preserve while changing blur
#             condition_path = image_dir / "aperture_8.00.png"
#             if not condition_path.exists(): 
#                 print(f"Warning: Skipping {scene_path.name}, missing aperture_8.00.png")
#                 continue

#             # Iterate over all target images (different apertures)
#             for json_file in config_dir.glob("aperture_*.json"):
#                 # Map json -> png (e.g., aperture_0.05.json -> aperture_0.05.png)
#                 target_img_name = json_file.stem + ".png"
#                 target_path = image_dir / target_img_name

#                 if not target_path.exists(): continue
                
#                 # Optional: Skip using the sharp image as a target to force learning blur
#                 # if target_path == condition_path: continue 

#                 self.samples.append({
#                     "target_path": str(target_path),
#                     "condition_path": str(condition_path),
#                     "metadata_path": str(json_file)
#                 })

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]

#         # A. Load Images
#         target_img = Image.open(item['target_path']).convert("RGB")
#         condition_img = Image.open(item['condition_path']).convert("RGB")

#         # B. Parse Metadata for Prompt
#         with open(item['metadata_path'], 'r') as f:
#             meta = json.load(f)
            
#         focal = meta.get('focal_length', 50)
#         aperture = meta.get('aperture', 8.0)
#         focus_dist = meta.get('focus_distance', 1.0)
        
#         # C. Construct Prompt
#         # We embed the parameters so the model learns that "f/0.05" = BLUR
#         text_prompt = f"A photo of a scene, focal length {focal}mm, aperture f/{aperture}, focus distance {focus_dist}"

#         # D. Apply Transforms
#         pixel_values = self.transform(target_img)
#         condition_pixel_values = self.transform(condition_img)

#         # E. Define Position Delta
#         # Your task is Spatially Aligned (pixels don't move, just blur).
#         # Thus, position_delta is (0,0).
#         position_delta = [0, 0]

#         return {
#             "image": pixel_values,            # Target (Blurred)
#             "description": text_prompt,              # Description
#             "conditions": condition_pixel_values, # Input (Sharp)
#             "position_delta": position_delta  # Alignment
#         }


# # --- 2. Test Function Implementation ---
# @torch.no_grad()
# def test_function(model, save_path, file_name):
#     """
#     Generates validation images to check if aperture control works.
#     We take a sharp image and try to generate 3 versions: Blurry, Medium, Sharp.
#     """
#     print(f"Running test generation for {file_name}...")
    
#     # 1. Access the underlying pipeline from OminiModel
#     # Note: Depending on OminiModel implementation, this is usually model.pipeline or model.pipe
#     pipe = model.pipeline 
    
#     # 2. Get a test condition (Use a fixed image from dataset path or generated noise)
#     # Here we try to load the first valid condition from the dataset directory for consistency
#     dataset_root = Path("./scenes")
#     first_scene = next(dataset_root.iterdir())
#     condition_path = first_scene / "image" / "aperture_8.00.png"
    
#     if not condition_path.exists():
#         print("Test condition not found, skipping generation.")
#         return

#     condition_img = Image.open(condition_path).convert("RGB").resize((512, 512))
    
#     # 3. Define Test Prompts (Varying Aperture)
#     prompts = [
#         "A photo of a scene, focal length 50mm, aperture f/0.05, focus distance 1.0", # Heavy Bokeh
#         "A photo of a scene, focal length 50mm, aperture f/2.0, focus distance 1.0",  # Medium
#         "A photo of a scene, focal length 50mm, aperture f/8.0, focus distance 1.0"   # Sharp
#     ]
    
#     generated_images = []
    
#     # 4. Generate
#     for prompt in prompts:
#         # Wrap image in Condition object (required by generate function)
#         # Note: 'condition_type' might be 'canny', 'depth', or 'subject'. 
#         # For general spatial control, we often treat it as 'depth' or just generic 'spatial'.
#         cond = Condition(condition_type="spatial", condition_image=condition_img)
        
#         result = generate(
#             pipe,
#             prompt=prompt,
#             conditions=[cond],
#             num_inference_steps=20,
#             height=512,
#             width=512,
#             guidance_scale=3.5
#         ).images[0]
#         generated_images.append(result)
        
#     # 5. Save Grid
#     # Combine images side-by-side: [Blurry | Medium | Sharp]
#     grid_width = 512 * 3
#     grid = Image.new("RGB", (grid_width, 512))
#     grid.paste(generated_images[0], (0, 0))
#     grid.paste(generated_images[1], (512, 0))
#     grid.paste(generated_images[2], (1024, 0))
    
#     full_save_path = os.path.join(save_path, f"{file_name}.png")
#     grid.save(full_save_path)
#     print(f"Saved validation grid to {full_save_path}")


# def main():
#     # Initialize
#     config = get_config()
#     config["train"]["batch_size"] = 1
#     training_config = config["train"]
#     torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

#     # Initialize custom dataset
#     # Ensure your "./scenes" directory exists and follows the structure
#     dataset = CustomDataset(root_dir="./scenes", image_size=512)

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

import torch
import os
import random
import glob
import json
import numpy as np
import time
import gc
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import lightning as L

from diffusers import FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model_state_dict  # <--- Added Import

from .trainer import get_config, get_rank, TrainingCallback, init_wandb
from ..pipeline.flux_omini import Condition, generate, transformer_forward, encode_images

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def get_metadata(img_path):
    try:
        image_dir = os.path.dirname(img_path)
        subfolder_dir = os.path.dirname(image_dir)
        filename = os.path.basename(img_path).rsplit('.', 1)[0]
        json_path = os.path.join(subfolder_dir, "config", f"{filename}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def construct_prompt(metadata):
    fl = metadata.get('focal_length', 'standard')
    aperture = metadata.get('aperture', 'standard')
    dist = metadata.get('focus_distance', 'unknown')
    return f"A clear photo with focal length {fl}, aperture {aperture}, focus distance {dist}."

# ==========================================
# 1. DATASET
# ==========================================
class CustomDataset(Dataset):
    def __init__(self, root_dir, condition_size=(512, 512), target_size=(512, 512)):
        self.root_dir = root_dir
        self.condition_size = condition_size
        self.target_size = target_size
        self.to_tensor = T.ToTensor()
        self.scene_groups = {}
        self._parse_dataset()
        self.scene_keys = [k for k, v in self.scene_groups.items() if len(v) >= 1]
        print(f">> Dataset loaded. Found {len(self.scene_keys)} scene groups in '{root_dir}'.")

    def _parse_dataset(self):
        search_pattern = os.path.join(self.root_dir, "*", "*", "image", "*")
        all_images = glob.glob(search_pattern)
        for img_path in all_images:
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            image_dir = os.path.dirname(img_path)
            group_id = os.path.dirname(image_dir) 
            if group_id not in self.scene_groups:
                self.scene_groups[group_id] = []
            self.scene_groups[group_id].append(img_path)

    def __len__(self):
        return len(self.scene_keys)

    def __getitem__(self, idx):
        group_id = self.scene_keys[idx]
        images_in_group = self.scene_groups[group_id]
        target_path = random.choice(images_in_group)
        if random.random() < 0.5 and len(images_in_group) > 1:
            source_path = random.choice([p for p in images_in_group if p != target_path])
        else:
            source_path = target_path

        try:
            target_img = Image.open(target_path).convert("RGB")
            source_img = Image.open(source_path).convert("RGB")
            target_meta = get_metadata(target_path)
            prompt = construct_prompt(target_meta)
            target_img = target_img.resize(self.target_size)
            source_img = source_img.resize(self.condition_size)
            return {
                "image": self.to_tensor(target_img),
                "condition_0": self.to_tensor(source_img),
                "condition_type_0": "bokeh_edit", 
                "position_delta_0": np.array([0, 0]),
                "description": prompt,
                "debug_path": target_path 
            }
        except Exception as e:
            print(f"Error loading {target_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

# ==========================================
# 2. MODEL (A100 Version)
# ==========================================
class HighPerformanceOminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.adapter_names = [None, None, "default"]
        self.my_device = device
        self.my_dtype = dtype

        print(f">> [Model] Loading Full FLUX.1-dev in {dtype} on {device}...")
        self.flux_pipe = FluxPipeline.from_pretrained(flux_pipe_id, torch_dtype=dtype).to(device)
        self.transformer = self.flux_pipe.transformer
        
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing() 
        
        self.flux_pipe.text_encoder.requires_grad_(False)
        self.flux_pipe.text_encoder_2.requires_grad_(False)
        self.flux_pipe.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.lora_layers = self.init_lora(lora_config)

    def init_lora(self, lora_config: dict):
        self.transformer.add_adapter(LoraConfig(**lora_config), adapter_name="default")
        lora_layers = [p for p in self.transformer.parameters() if p.requires_grad]
        print(f">> [Model] Trainable LoRA parameters: {len(lora_layers)}")
        return lora_layers

    def save_lora(self, path: str):
        # [FIXED METHOD]
        lora_state_dict = get_peft_model_state_dict(self.transformer, adapter_name="default")
        self.flux_pipe.save_lora_weights(
            save_directory=path,
            weight_name="default.safetensors",
            transformer_lora_layers=lora_state_dict,
            safe_serialization=True
        )

    def configure_optimizers(self):
        print(">> [Optimizer] Using Standard AdamW (BF16)...")
        lr = float(self.optimizer_config["params"]["lr"])
        weight_decay = float(self.optimizer_config["params"].get("weight_decay", 1e-2))
        optimizer = torch.optim.AdamW(self.lora_layers, lr=lr, weight_decay=weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        pipe = self.flux_pipe
        
        conditions = []
        if "condition_0" in batch:
            conditions.append(batch["condition_0"])

        with torch.no_grad():
            x_0, img_ids = encode_images(pipe, imgs)
            (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
                prompt=prompts, prompt_2=None, device=self.device, 
                num_images_per_prompt=1, max_sequence_length=512
            )
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.my_dtype)

            condition_latents, condition_ids = [], []
            for cond in conditions:
                c_latents, c_ids = encode_images(pipe, cond)
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            guidance = torch.ones_like(t) if self.transformer.config.guidance_embeds else None

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions), device=self.device))

        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        self.log_loss = step_loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + step_loss.item() * 0.05
        return step_loss

@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = (512, 512)
    target_size = (512, 512)
    prompt = "Default Prompt"
    img_path = "assets/test_in.jpg"
    try:
        dataset_path = model.training_config["dataset"]["path"]
        test_images = glob.glob(os.path.join(dataset_path, "*", "*", "image", "*.png"))
        if test_images:
            img_path = random.choice(test_images)
            meta = get_metadata(img_path)
            prompt = construct_prompt(meta)
            print(f"\n[Test] Image: {img_path} | Prompt: {prompt}\n")
            image = Image.open(img_path).convert("RGB").resize(condition_size)
        else:
            image = Image.new("RGB", condition_size, (128, 128, 128))
    except Exception:
        image = Image.new("RGB", condition_size, (128, 128, 128))

    condition = Condition(image, "default", position_delta=[0,0])
    os.makedirs(save_path, exist_ok=True)
    pipe = model.flux_pipe
    generator = torch.Generator(device=pipe.device).manual_seed(42)

    res = generate(
        pipe, prompt=prompt, conditions=[condition],
        height=target_size[1], width=target_size[0],
        generator=generator, model_config=model.model_config,
    )
    
    concat = Image.new("RGB", (target_size[0] * 2, target_size[1]))
    concat.paste(image.resize(target_size), (0, 0))
    concat.paste(res.images[0], (target_size[0], 0))
    concat.save(os.path.join(save_path, f"{file_name}_test.jpg"))

def train_custom(dataset, trainable_model, config, test_function):
    is_main_process, rank = get_rank() == 0, get_rank()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    if training_config.get("wandb", None) and is_main_process:
        init_wandb(training_config["wandb"], run_name)

    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config.get("dataloader_workers", 2),
    )

    callbacks = [TrainingCallback(run_name, training_config, test_function)] if is_main_process else []
    
    trainer = L.Trainer(
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True, 
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        precision="bf16-true", 
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    trainer.fit(trainable_model, train_loader)

def main():
    config = get_config()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    dataset = CustomDataset(
        root_dir=config["train"]["dataset"]["path"],
        condition_size=config["train"]["dataset"]["condition_size"],
        target_size=config["train"]["dataset"]["target_size"]
    )

    trainable_model = HighPerformanceOminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
    )

    train_custom(dataset, trainable_model, config, test_function)

if __name__ == "__main__":
    main()