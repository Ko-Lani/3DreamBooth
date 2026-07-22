# Subjects200K Dataset for OminiControl-style training
# Dataset: https://huggingface.co/datasets/Yuanshi/Subjects200K

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from typing import Optional, Dict, Any, Tuple
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize
import numpy as np


import os



class Subjects200KDataset(Dataset):
    """
    Dataset for Subjects200K paired images.

    Each sample contains:
    - A composite image with two images side by side (16-pixel padding)
    - Quality assessment scores
    - Description of the image pair

    This dataset splits the composite into:
    - Reference image (left): The subject reference
    - Target image (right): The target scene with the same subject
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 720,
        collection: str = "collection_2",  # collection_1 or collection_2
        min_quality_score: int = 5,
        use_quality_filter: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        padding: int = 16,  # Padding between images in composite
    ):
        """
        Args:
            height: Target height for output images
            width: Target width for output images
            collection: Which collection to use (collection_1, collection_2, collection_3)
            min_quality_score: Minimum score for quality filtering (0-5)
            use_quality_filter: Whether to filter by quality assessment
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Directory for caching dataset
            padding: Padding pixels between paired images in composite
        """
        super().__init__()

        self.height = height
        self.width = width
        self.collection = collection
        self.min_quality_score = min_quality_score
        self.padding = padding

        print(f"Loading Subjects200K dataset (collection: {collection})...")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            'Yuanshi/Subjects200K',
            cache_dir=cache_dir,
            split='train'
        )

        # Filter by collection
        if collection:
            self.dataset = self.dataset.filter(
                lambda x: x.get("collection") == collection,
                num_proc=4,
            )
            print(f"Filtered to {collection}: {len(self.dataset)} samples")

        # Filter by quality
        # Use standalone function instead of bound method to avoid pickling
        # issues with num_proc > 1 (self contains the dataset object)
        if use_quality_filter:
            min_q = self.min_quality_score
            def quality_filter(item):
                desc = item.get("description")
                if isinstance(desc, dict) and desc.get("description_valid") is False:
                    return False
                qa = item.get("quality_assessment")
                if not qa:
                    return False
                return all(
                    qa.get(key, 0) >= min_q
                    for key in ["compositeStructure", "objectConsistency", "imageQuality"]
                )
            self.dataset = self.dataset.filter(
                quality_filter,
                num_proc=4,
            )
            print(f"After quality filter (>= {min_quality_score}): {len(self.dataset)} samples")

        # Limit samples if specified
        if max_samples is not None and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")

        print(f"Final dataset size: {len(self.dataset)} samples")

    def _quality_filter(self, item: Dict[str, Any]) -> bool:
        """Filter by quality assessment scores and description validity."""
        # Check description validity (only reject if explicitly False, not None)
        description = item.get("description")
        if isinstance(description, dict) and description.get("description_valid") is False:
            return False

        qa = item.get("quality_assessment")
        if not qa:
            return False

        required_keys = ["compositeStructure", "objectConsistency", "imageQuality"]
        return all(
            qa.get(key, 0) >= self.min_quality_score
            for key in required_keys
        )

    def _split_composite_image(self, composite: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Split composite image into reference and target images.

        The composite image has format: [ref_image | padding | target_image]
        """
        width = composite.width
        height = composite.height

        # Calculate single image width (accounting for padding)
        single_width = (width - self.padding) // 2

        # Split images
        reference = composite.crop((0, 0, single_width, height))
        target = composite.crop((single_width + self.padding, 0, width, height))

        return reference, target

    def _get_prompt(self, item: Dict[str, Any]) -> str:
        """
        Extract prompt from description.

        Subjects200K description structure:
            item: str           - object name (e.g. "Eames Lounge Chair")
            description_0: str  - description of left/reference image
            description_1: str  - description of right/target image
            category: str       - object category
            description_valid: bool
        """
        description = item.get("description", {})

        if isinstance(description, dict):
            # Use target image description (right side = description_1)
            if "description_1" in description and description["description_1"]:
                return description["description_1"]
            # Fallback to reference description
            if "description_0" in description and description["description_0"]:
                return description["description_0"]
            # Fallback to item name with category
            item_name = description.get("item", "")
            category = description.get("category", "")
            if item_name:
                if category:
                    return f"A photo of {item_name}, {category}"
                return f"A photo of {item_name}"
        elif isinstance(description, str):
            return description

        return "A photo of an object"

    def _resize_and_crop(self, image: Image.Image) -> torch.Tensor:
        """Resize and center crop image to target size."""
        # Convert to tensor [C, H, W] in range [0, 1]
        tensor = TT.functional.to_tensor(image)

        # Add batch dimension for resize: [1, C, H, W]
        tensor = tensor.unsqueeze(0)

        target_h, target_w = self.height, self.width
        _, _, h, w = tensor.shape

        # Calculate resize dimensions (maintain aspect ratio)
        if w / h > target_w / target_h:
            # Width is wider - resize by height
            new_h = target_h
            new_w = int(w * target_h / h)
        else:
            # Height is taller - resize by width
            new_w = target_w
            new_h = int(h * target_w / w)

        # Resize
        tensor = resize(tensor, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)

        # Center crop
        delta_h = new_h - target_h
        delta_w = new_w - target_w
        top = delta_h // 2
        left = delta_w // 2
        tensor = TT.functional.crop(tensor, top, left, target_h, target_w)

        # Remove batch dimension and normalize to [-1, 1]
        tensor = tensor.squeeze(0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.clamp(-1.0, 1.0)

        return tensor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        # Get composite image
        composite = item["image"]
        if isinstance(composite, str):
            composite = Image.open(composite).convert("RGB")
        elif not isinstance(composite, Image.Image):
            composite = composite.convert("RGB")

        # Split into reference and target
        # reference, target = self._split_composite_image(composite)
        target, reference = self._split_composite_image(composite)


        # Get prompt
        prompt = self._get_prompt(item)

        # Process images
        reference_tensor = self._resize_and_crop(reference)
        target_tensor = self._resize_and_crop(target)

        # For video training, add temporal dimension [C, H, W] -> [C, 1, H, W]
        target_video = target_tensor.unsqueeze(1)
        reference_image = reference_tensor  # Keep as [C, H, W] for reference


        return {
            "pixel_values": target_video,  # [C, 1, H, W] - target as single-frame video
            "reference_image": reference_image,  # [C, H, W] - reference image
            "text": prompt,
            "quality_assessment": item.get("quality_assessment", {}),
        }


class Subjects200KVideoDataset(Subjects200KDataset):
    """
    Extended dataset that generates pseudo-video from paired images.

    Creates a video sequence where:
    - First frame: Reference image
    - Remaining frames: Interpolated transition to target (or repeated target)
    """

    def __init__(
        self,
        num_frames: int = 17,  # 4k+1 format for HunyuanVideo
        interpolation_mode: str = "repeat",  # "repeat" or "interpolate"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.interpolation_mode = interpolation_mode

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        # Get composite image
        composite = item["image"]
        if isinstance(composite, str):
            composite = Image.open(composite).convert("RGB")
        elif not isinstance(composite, Image.Image):
            composite = composite.convert("RGB")

        # Split into reference and target
        reference, target = self._split_composite_image(composite)

        # Get prompt
        prompt = self._get_prompt(item)

        # Process images
        reference_tensor = self._resize_and_crop(reference)  # [C, H, W]
        target_tensor = self._resize_and_crop(target)  # [C, H, W]

        # Create video sequence
        if self.interpolation_mode == "repeat":
            # Simply repeat target for all frames
            video_frames = target_tensor.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        else:
            # Interpolate from reference to target
            video_frames = []
            for i in range(self.num_frames):
                alpha = i / (self.num_frames - 1) if self.num_frames > 1 else 1.0
                frame = (1 - alpha) * reference_tensor + alpha * target_tensor
                video_frames.append(frame)
            video_frames = torch.stack(video_frames, dim=1)  # [C, T, H, W]

        return {
            "pixel_values": video_frames,  # [C, T, H, W]
            "reference_image": reference_tensor,  # [C, H, W]
            "text": prompt,
            "quality_assessment": item.get("quality_assessment", {}),
        }


def create_subjects200k_dataloader(
    batch_size: int = 1,
    height: int = 480,
    width: int = 720,
    collection: str = "collection_2",
    min_quality_score: int = 5,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    use_video_dataset: bool = False,
    num_frames: int = 17,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for Subjects200K dataset.

    Args:
        batch_size: Batch size
        height: Target image height
        width: Target image width
        collection: Dataset collection to use
        min_quality_score: Minimum quality score for filtering
        max_samples: Maximum samples to load
        num_workers: Number of data loading workers
        use_video_dataset: Whether to use video dataset (with multiple frames)
        num_frames: Number of frames for video dataset

    Returns:
        DataLoader instance
    """
    if use_video_dataset:
        dataset = Subjects200KVideoDataset(
            height=height,
            width=width,
            collection=collection,
            min_quality_score=min_quality_score,
            max_samples=max_samples,
            num_frames=num_frames,
            **kwargs
        )
    else:
        dataset = Subjects200KDataset(
            height=height,
            width=width,
            collection=collection,
            min_quality_score=min_quality_score,
            max_samples=max_samples,
            **kwargs
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Subjects200K dataset...")

    dataset = Subjects200KDataset(
        height=480,
        width=720,
        collection="collection_2",
        min_quality_score=5,
        max_samples=100,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Target video shape: {sample['pixel_values'].shape}")
    print(f"Reference image shape: {sample['reference_image'].shape}")
    print(f"Prompt: {sample['text'][:100]}...")
    print(f"Quality: {sample['quality_assessment']}")
