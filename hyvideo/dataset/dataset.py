from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize

import random

from rembg import remove
from rembg import new_session

class ImageToVideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video", # 이름은 유지하지만 실제론 이미지 경로
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 49, # 여기서 설정한 만큼 이미지를 복사함
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""

        # 데이터 로드 (로컬 경로에서 이미지 찾기)
        self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)

        # 4k + 1 규칙에 맞게 프레임 수 조정 (CogVideoX 필수)
        # 예: 49 -> 49 (OK), 50 -> 49 (조정)
        self.actual_num_frames = self.max_num_frames - (self.max_num_frames - 1) % 4

        # 전처리 수행 (이미지 로딩 -> 비디오로 변환)
        self.instance_videos = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "text": self.id_token + self.instance_prompts[index],
            "pixel_values": self.instance_videos[index],
        }

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance root folder does not exist")

        # 1. 이미지 파일들만 싹 긁어오기 (jpg, png, webp 등)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        instance_images = [
            p for p in self.instance_data_root.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if len(instance_images) == 0:
            raise ValueError(f"No images found in {self.instance_data_root}")

        # 2. 프롬프트 처리
        # (간단하게: 별도 텍스트 파일 없으면 파일명을 프롬프트로 쓰거나 빈 문자열 사용)
        # 여기서는 파일명(확장자 제외)을 프롬프트로 쓰는 예시
        instance_prompts = [p.stem for p in instance_images]

        # 만약 'prompt.txt' 같은 걸 쓰고 싶으면 기존 로직을 살리면 됨.
        # 지금은 심플하게 이미지 개수만큼 리스트를 맞춤.

        return instance_prompts, instance_images

    def _resize_for_rectangle_crop(self, arr):
        # 기존 로직 그대로 사용 (Tensor [F, C, H, W] 입력을 기대함)
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode

        # arr: [F, C, H, W] -> 높이(H), 너비(W) 추출
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]

        # Crop 계산
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError

        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_data(self):
        # decord 제거하고 PIL 사용
        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)),
            desc="Loading and repeating images as videos",
        )
        videos = []

        for image_path in self.instance_video_paths:
            # 1. 이미지 열기 & RGB 변환
            pil_image = Image.open(image_path).convert("RGB")

            # 2. Tensor로 변환 [C, H, W] (0~1 사이 값)
            tensor_image = TT.functional.to_tensor(pil_image)

            # 3. Normalization (-1 ~ 1 사이 값으로)
            # to_tensor는 0~1이니까, (x - 0.5) / 0.5 하면 -1~1이 됨
            tensor_image = (tensor_image - 0.5) / 0.5

            # 4. 🔥 [핵심] 비디오처럼 차원 확장 (Repeat)
            # [C, H, W] -> [1, C, H, W] -> [F, C, H, W]
            # video_tensor = tensor_image.unsqueeze(0).repeat(self.actual_num_frames, 1, 1, 1)
            # video_tensor = tensor_image.unsqueeze(0)
            video_tensor = tensor_image.unsqueeze(1)



            # 5. 리사이즈 및 크롭 (기존 함수 재사용)
            # _resize_for_rectangle_crop 함수는 [F, C, H, W] 입력을 처리함
            progress_dataset_bar.set_description(
                f"Resizing image-video from {video_tensor.shape[2]}x{video_tensor.shape[3]} to {self.height}x{self.width}"
            )
            video_tensor = self._resize_for_rectangle_crop(video_tensor)

            video_tensor = video_tensor.clamp(-1.0, 1.0)

            # 6. 저장
            videos.append(video_tensor.contiguous())
            progress_dataset_bar.update(1)

        progress_dataset_bar.close()
        return videos





class ImageToVideoMaskDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video", # 이름은 유지하지만 실제론 이미지 경로
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 49, # 여기서 설정한 만큼 이미지를 복사함
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""

        # 데이터 로드 (로컬 경로에서 이미지 찾기)
        self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)

        # 4k + 1 규칙에 맞게 프레임 수 조정 (CogVideoX 필수)
        # 예: 49 -> 49 (OK), 50 -> 49 (조정)
        self.actual_num_frames = self.max_num_frames - (self.max_num_frames - 1) % 4

        # 전처리 수행 (이미지 로딩 -> 비디오로 변환)
        self.instance_videos, self.instance_masks = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "text": self.id_token + self.instance_prompts[index],
            "pixel_values": self.instance_videos[index],
            "loss_mask": self.instance_masks[index],
        }

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance root folder does not exist")

        # 1. 이미지 파일들만 싹 긁어오기 (jpg, png, webp 등)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        instance_images = [
            p for p in self.instance_data_root.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if len(instance_images) == 0:
            raise ValueError(f"No images found in {self.instance_data_root}")

        # 2. 프롬프트 처리
        # (간단하게: 별도 텍스트 파일 없으면 파일명을 프롬프트로 쓰거나 빈 문자열 사용)
        # 여기서는 파일명(확장자 제외)을 프롬프트로 쓰는 예시
        instance_prompts = [p.stem for p in instance_images]

        # 만약 'prompt.txt' 같은 걸 쓰고 싶으면 기존 로직을 살리면 됨.
        # 지금은 심플하게 이미지 개수만큼 리스트를 맞춤.

        return instance_prompts, instance_images

    def _resize_for_rectangle_crop(self, arr, obj_max_ratio=0.7):
        """
        객체 중심 리사이즈 & crop.

        obj_max_ratio: 객체가 crop 영역에서 차지할 최대 비율 (0.7 = 70%).
                       나머지 30%가 배경 여백으로 확보됨.
        """
        image_size = self.height, self.width
        crop_h, crop_w = image_size
        reshape_mode = self.video_reshape_mode
        orig_h, orig_w = arr.shape[2], arr.shape[3]

        # 4ch (RGB+mask) 이면 객체 기반 리사이즈 + crop
        if arr.shape[0] == 4:
            mask_ch = arr[3, 0]  # [H, W] 원본 해상도의 마스크
            obj_pixels = (mask_ch > 0.1).nonzero(as_tuple=False)

            if obj_pixels.numel() > 0:
                obj_top = obj_pixels[:, 0].min().item()
                obj_bottom = obj_pixels[:, 0].max().item()
                obj_left = obj_pixels[:, 1].min().item()
                obj_right = obj_pixels[:, 1].max().item()
                obj_h = obj_bottom - obj_top
                obj_w = obj_right - obj_left

                # 객체가 crop의 obj_max_ratio 비율에 맞도록 스케일 계산
                # 예: obj_max_ratio=0.7이면 객체가 512*0.7=358px 이내
                target_obj_h = crop_h * obj_max_ratio
                target_obj_w = crop_w * obj_max_ratio
                scale_h = target_obj_h / max(obj_h, 1)
                scale_w = target_obj_w / max(obj_w, 1)
                scale = min(scale_h, scale_w)

                # 리사이즈 후 이미지가 crop보다 작아지면 안 됨
                # → 최소 스케일: 짧은 변이 crop 크기 이상
                min_scale = max(crop_h / orig_h, crop_w / orig_w)
                scale = max(scale, min_scale)

                new_h = max(int(orig_h * scale), crop_h)
                new_w = max(int(orig_w * scale), crop_w)
                arr = resize(arr, size=[new_h, new_w], interpolation=InterpolationMode.BICUBIC)

                # 리사이즈 후 객체 중심 좌표 재계산
                obj_center_h = ((obj_top + obj_bottom) / 2.0) * scale
                obj_center_w = ((obj_left + obj_right) / 2.0) * scale
                h, w = new_h, new_w

                # 객체 중심으로 crop 오프셋
                top = int(obj_center_h - crop_h / 2.0)
                left = int(obj_center_w - crop_w / 2.0)
                top = max(0, min(top, h - crop_h))
                left = max(0, min(left, w - crop_w))
            else:
                # 마스크에 객체 없음 → 기존 비율 리사이즈 + center crop
                arr = self._aspect_ratio_resize(arr, image_size)
                h, w = arr.shape[2], arr.shape[3]
                top = (h - crop_h) // 2
                left = (w - crop_w) // 2
        else:
            # 4ch 아닌 경우 → 기존 로직
            arr = self._aspect_ratio_resize(arr, image_size)
            h, w = arr.shape[2], arr.shape[3]
            delta_h = h - crop_h
            delta_w = w - crop_w
            if reshape_mode == "random" or reshape_mode == "none":
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            elif reshape_mode == "center":
                top, left = delta_h // 2, delta_w // 2
            else:
                raise NotImplementedError

        arr = TT.functional.crop(arr, top=top, left=left, height=crop_h, width=crop_w)
        return arr

    def _aspect_ratio_resize(self, arr, image_size):
        """비율 유지 리사이즈: 짧은 변을 target에 맞춤"""
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )
        return arr

    def _preprocess_data(self):
        # decord 제거하고 PIL 사용
        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)),
            desc="Loading and repeating images as videos",
        )
        videos = []
        masks = []

        session = new_session("u2net", providers=['CPUExecutionProvider'])


        for image_path in self.instance_video_paths:

            pil_image = Image.open(image_path).convert("RGB")
            W_orig, H_orig = pil_image.size


            tensor_image = TT.functional.to_tensor(pil_image)
            tensor_image = (tensor_image - 0.5) / 0.5

            tensor_mask = torch.ones((1, H_orig, W_orig), dtype=tensor_image.dtype)

            combined = torch.cat([tensor_image.unsqueeze(1), tensor_mask.unsqueeze(1)], dim=0) # [4, 1, H_orig, W_orig]


            # video_tensor = tensor_image.unsqueeze(1)
            # mask_tensor = tensor_mask.unsqueeze(1)


            # 5. 리사이즈 및 크롭 (기존 함수 재사용)
            # _resize_for_rectangle_crop 함수는 [F, C, H, W] 입력을 처리함
            # progress_dataset_bar.set_description(
            #     f"Resizing image-video from {video_tensor.shape[2]}x{video_tensor.shape[3]} to {self.height}x{self.width}"
            # )

            progress_dataset_bar.set_description(
                f"Resizing image-video from {combined.shape[2]}x{combined.shape[3]} to {self.height}x{self.width}"
            )
            # video_tensor = self._resize_for_rectangle_crop(video_tensor)
            combined = self._resize_for_rectangle_crop(combined)


            video_tensor = combined[:3] # 앞의 3채널은 이미지
            mask_tensor = combined[3:]  # 마지막 1채널은 마스크

            video_tensor = video_tensor.clamp(-1.0, 1.0)
            videos.append(video_tensor.contiguous())
            masks.append(mask_tensor.contiguous())

            progress_dataset_bar.update(1)

        progress_dataset_bar.close()
        return videos, masks
