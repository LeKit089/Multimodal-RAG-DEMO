from .system_prompt import TABLES_AND_CHARTS_CONVERT_SYSTEM
import os

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from typing import Optional



CPM_MODEL_PATH = "/home/project/data/jc/mmRAG/model/MiniCPM-Llama3-V-2_5"

# # 定义最大内存使用量和GPU设备ID
# max_memory_each_gpu = '15GiB'
# gpu_device_ids = [0, 1]  # 假设有两个GPU

# max_memory = {device_id: max_memory_each_gpu for device_id in gpu_device_ids}
# no_split_module_classes = ["LlamaDecoderLayer"]

class CPMConvertorFactory:
    _instance = None

    @staticmethod
    def get_instance(device="cuda:1" if torch.cuda.is_available() else "cpu"):
        """Static method to get the singleton instance of CPMConvertor."""
        
        CPMConvertorFactory._instance = CPMConvertor(device=device)
        return CPMConvertorFactory._instance
        
        # if CPMConvertorFactory._instance is None:
        #     CPMConvertorFactory._instance = CPMConvertor(device=device)
        # return CPMConvertorFactory._instance

class CPMConvertor:
    def __init__(
        self,
        model_name_or_path: Optional[str] = CPM_MODEL_PATH,
        device: Optional[str] = "cuda:1" if torch.cuda.is_available() else "cpu",
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

        # with init_empty_weights():
        #     self.model = AutoModel.from_pretrained(
        #         model_name_or_path,
        #         trust_remote_code=True,
        #         torch_dtype=torch.float16,
        #     )
        
        # device_map = infer_auto_device_map(
        #     self.model,
        #     max_memory=max_memory,
        #     no_split_module_classes=no_split_module_classes
        # )

        # # 确保输入和输出层都在第一个GPU上
        # device_map["llm.model.embed_tokens"] = 0
        # device_map["llm.model.layers.0"] = 0
        # device_map["llm.lm_head"] = 0
        # device_map["vpm"] = 0
        # device_map["resampler"] = 0

        # load_checkpoint_in_model(self.model, model_name_or_path, device_map=device_map)
        # self.model = dispatch_model(self.model, device_map=device_map).eval()
        
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
        del self.tokenizer
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
