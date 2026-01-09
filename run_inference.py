import torch
import os
from PIL import Image
from diffusers import FluxPipeline

# Import OminiControl Utils
# Ensure omini/ folder is in the same directory
from omini.pipeline.flux_omini import Condition, generate

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "runs/20260109-151908/ckpt/3000/default.safetensors"  # <--- UPDATE THIS PATH
INPUT_IMAGE = "assets/ominicontrol_art/oiiai.png"                                # <--- YOUR INPUT IMAGE
OUTPUT_FOLDER = "results/"

# The prompt controls the blur. Change 'aperture' to see the effect.
# Try: f/0.1 (very blurry), f/5.6 (normal), f/22 (very sharp)
PROMPT = "A clear photo with focal length 70mm, aperture f/4, focus distance 1.0m"

# ==========================================
# MAIN INFERENCE
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f">> Loading Base Model: {BASE_MODEL_ID}...")
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype
    ).to(device)

    print(f">> Loading LoRA from: {LORA_PATH}")
    pipe.load_lora_weights(LORA_PATH, adapter_name="default")
    pipe.fuse_lora(lora_scale=1.0) # Merge weights for speed (Scale 1.0 = 100% effect)

    # Prepare Input Image (Resize to 512x512 to match training)
    if os.path.exists(INPUT_IMAGE):
        print(f">> Processing Input Image: {INPUT_IMAGE}")
        raw_img = Image.open(INPUT_IMAGE).convert("RGB")
        
        # Resize logic to match your training (Resize Shortest -> Center Crop)
        short_side = min(raw_img.size)
        scale = 512 / short_side
        new_size = (int(raw_img.size[0] * scale), int(raw_img.size[1] * scale))
        raw_img = raw_img.resize(new_size, Image.Resampling.BILINEAR)
        
        # Center Crop
        left = (raw_img.width - 512) // 2
        top = (raw_img.height - 512) // 2
        input_img = raw_img.crop((left, top, left + 512, top + 512))
    else:
        print("!! Input image not found. Creating dummy noise image.")
        input_img = Image.new("RGB", (512, 512), (128, 128, 128))

    # Create Condition Object
    # "condition_type" isn't strictly used by the logic but good for tracking
    condition = Condition(input_img, "default", position_delta=[0, 0])

    print(f">> Generating with prompt: '{PROMPT}'")
    generator = torch.Generator(device=device).manual_seed(42)

    res = generate(
        pipe,
        prompt=PROMPT,
        conditions=[condition],
        height=512,
        width=512,
        num_inference_steps=28, # Standard FLUX steps
        guidance_scale=3.5,
        generator=generator
    )

    # Save Result
    output_filename = f"result_f4.jpg"
    save_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Save Side-by-Side for easy comparison
    concat = Image.new("RGB", (1024, 512))
    concat.paste(input_img, (0, 0))
    concat.paste(res.images[0], (512, 0))
    concat.save(save_path)
    
    print(f">> Saved result to {save_path}")

if __name__ == "__main__":
    main()