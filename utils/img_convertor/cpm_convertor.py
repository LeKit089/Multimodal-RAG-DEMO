from .base_convertor import BaseConvertor
from ..system_prompt import TABLES_AND_CHARTS_CONVERT_SYSTEM
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class CPMConvertor(BaseConvertor):
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device},
            torch_dtype=torch.float16,
        ).eval()

    def convert(
        self,
        img_path: str,
        query: str,
        sampling=True,
        temperature=0.1,
        system_prompt: str = TABLES_AND_CHARTS_CONVERT_SYSTEM,
    ):
        assert (
            img_path.endswith(".png")
            or img_path.endswith(".jpg")
            or img_path.endswith(".jpeg")
        ), "Only support png, jpg, jpeg format"
        assert os.path.exists(img_path), f"{img_path} does not exist"

        image = Image.open(img_path).convert("RGB")
        msgs = [{"role": "user", "content": query}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        return res

    def clear_GPU_mem(self):
        del self.model
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
