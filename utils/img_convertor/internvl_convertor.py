from .base_convertor import BaseConvertor
from transformers import AutoTokenizer, AutoModel
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from typing import Optional


class InternVLConvertor(BaseConvertor):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=True, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(self.device)
        )

    def convert(
        self, img_path: str, query: str, generation_config: dict, max_num: int = 6
    ):
        assert img_path.endswith(
            (".png", ".jpg", ".jpeg")
        ), "Only support png, jpg, jpeg format"
        assert os.path.exists(img_path), f"{img_path} does not exist"

        pixel_values = (
            self.load_image(img_path, max_num=max_num)
            .to(torch.bfloat16)
            .to(self.device)
        )
        response = self.model.chat(
            self.tokenizer, pixel_values, query, generation_config
        )
        return response

    def build_transform(self, input_size: int):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=6):
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def clear_GPU_mem(self):
        del self.model
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
