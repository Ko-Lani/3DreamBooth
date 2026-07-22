"""Prepare condition images by removing the background and normalizing framing.

This script is intended for dataset preprocessing before training or validation.
It does the following for a small set of input images:
1. Removes the background with BiRefNet via `rembg`.
2. Composites the foreground onto a white background.
3. Resizes and crops the image so the main object fits within a fixed square.
4. Optionally scales the final object down to leave more white margin.

Update the constants below to match your local dataset layout before running.
"""

from PIL import Image
from rembg import remove, new_session
import os
import glob

# Source directory containing raw object images.
INPUT_DIR = "./data/figure1/full"
# Output directory for processed condition images.
OUTPUT_DIR = "./data/figure1/cond"

# Number of images to pick from the input folder for processing.
NUM_IMAGES = 3

# Final square output size.
CROP_SIZE = 512

# Keep the detected object within this fraction of the crop.
OBJ_MAX_RATIO = 0.7

# Optional final down-scaling for condition images.
# A value of 1.0 preserves more detail and texture, but it can also make the
# condition image too similar to the original training image. In joint training,
# that may encourage the model to copy the 3DreamBooth reconstruction too
# literally, including background fitting artifacts. Using a smaller value helps
# keep the condition image slightly different from the original.
# We recommend trying values between 0.5 and 0.8 to find the best balance for your dataset.
REF_SCALE = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the rembg session once so multiple images can reuse the same model.
session = new_session("birefnet-massive")


# all_images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
# Accept common image extensions from the input folder.
extensions = ["*.jpg", "*.jpeg", "*.png"]
all_images = []
for ext in extensions:
    all_images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

all_images = sorted(all_images)


# By default, choose four images as uniformly as possible across the sequence.
# If your frames are not ordered uniformly enough for this to be meaningful,
# replace `pick_indices` manually with the exact indices you want.
num_picks = min(NUM_IMAGES, len(all_images))
if num_picks <= 1:
    pick_indices = list(range(num_picks))
else:
    pick_indices = [round(i * (len(all_images) - 1) / (num_picks - 1)) for i in range(num_picks)]

## To pick specific frames manually, comment out the above block and set `pick_indices` directly, e.g.:
# pick_indices = [0, 10, 20]  # Example: pick the first, 11th, and 21st images from the sorted list. Adjust as needed.


pick_indices = sorted(set(i for i in pick_indices if 0 <= i < len(all_images)))
selected_images = [all_images[i] for i in pick_indices]
print(f"Total images: {len(all_images)}, Selected {len(selected_images)}: indices {pick_indices}")


# Process each selected image: remove background, crop around the object, and save.
for img_path in selected_images:
    fname = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")

    # Remove the background and keep the alpha channel for object localization.
    result_rgba = remove(img, session=session)
    alpha = result_rgba.split()[3]

    # Composite the cutout onto a white background.
    white_bg = Image.new("RGB", result_rgba.size, (255, 255, 255))
    white_bg.paste(result_rgba, mask=alpha)

    # Find the object bounding box from the alpha mask.
    bbox = alpha.getbbox()  # (left, top, right, bottom) or None

    w, h = white_bg.size

    if bbox:
        obj_left, obj_top, obj_right, obj_bottom = bbox
        obj_w = obj_right - obj_left
        obj_h = obj_bottom - obj_top
        obj_cx = (obj_left + obj_right) / 2.0
        obj_cy = (obj_top + obj_bottom) / 2.0

        # Scale the image so the object fits comfortably inside the target crop.
        target_obj_h = CROP_SIZE * OBJ_MAX_RATIO
        target_obj_w = CROP_SIZE * OBJ_MAX_RATIO
        scale_h = target_obj_h / max(obj_h, 1)
        scale_w = target_obj_w / max(obj_w, 1)
        scale = min(scale_h, scale_w)

        # Also ensure the resized image is still large enough for a full crop.
        min_scale = max(CROP_SIZE / w, CROP_SIZE / h)
        scale = max(scale, min_scale)

        new_w = max(int(w * scale), CROP_SIZE)
        new_h = max(int(h * scale), CROP_SIZE)
        white_bg = white_bg.resize((new_w, new_h), Image.BICUBIC)
        obj_cx *= scale
        obj_cy *= scale
        w, h = new_w, new_h

        # Center the crop around the detected object.
        left = int(obj_cx - CROP_SIZE / 2.0)
        top = int(obj_cy - CROP_SIZE / 2.0)
        left = max(0, min(left, w - CROP_SIZE))
        top = max(0, min(top, h - CROP_SIZE))
    else:
        # Fall back to a simple centered crop if no foreground was detected.
        if w / h > 1:
            new_h = CROP_SIZE
            new_w = int(w * CROP_SIZE / h)
        else:
            new_w = CROP_SIZE
            new_h = int(h * CROP_SIZE / w)
        white_bg = white_bg.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h
        left = (w - CROP_SIZE) // 2
        top = (h - CROP_SIZE) // 2

    white_bg = white_bg.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # Optionally shrink the final image and place it on a clean white canvas.
    if REF_SCALE < 1.0:
        scaled_size = int(CROP_SIZE * REF_SCALE)
        scaled = white_bg.resize((scaled_size, scaled_size), Image.BICUBIC)
        canvas = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
        paste_x = (CROP_SIZE - scaled_size) // 2
        paste_y = (CROP_SIZE - scaled_size) // 2
        canvas.paste(scaled, (paste_x, paste_y))
        white_bg = canvas

    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".png")
    white_bg.save(out_path)
    print(f"Saved: {out_path} (crop offset: top={top}, left={left}, ref_scale={REF_SCALE})")

print(f"\nDone! Images saved to {OUTPUT_DIR}")
