from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer
from typing import List
import torch
from collections import ChainMap
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction


class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
#,model_name="/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/models/bge-m3"
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModel.from_pretrained(model_name)
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result


# class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):

#     def __init__(self):

#         # 初始化模型

#         self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

#         # self.move_to_cuda0()


#     # def move_to_cuda0(self):

#     #     # 获取模型的所有参数和缓冲区

#     #     # 使用列表推导式来获取参数和缓冲区

#     #     params_buffers = [param for param in self.model.parameters()] + [buffer for buffer in self.model.buffers()]


#     #     # 移动所有参数和缓冲区到 cuda:0

#     #     for param_buffer in params_buffers:

#     #         param_buffer.data = param_buffer.data.to(torch.device('cuda:0'))


#     def encode_queries(self, queries: List[str]):

#         outputs = self.model.encode(

#             queries,

#             return_dense=False,

#             return_sparse=True,

#             return_colbert_vecs=False,

#         )["lexical_weights"]

#         return [self._to_standard_dict(output) for output in outputs]


#     def encode_documents(self, documents: List[str]):

#         outputs = self.model.encode(

#             documents,

#             return_dense=False,

#             return_sparse=True,

#             return_colbert_vecs=False,

#         )["lexical_weights"]

#         return [self._to_standard_dict(output) for output in outputs]


#     def _to_standard_dict(self, raw_output):

#         result = {}

#         for k in raw_output:

#             result[int(k)] = raw_output[k]

#         return result


#     # 检查所有参数和缓冲区是否在 cuda:0 上

#     def check_cuda0(self):

#         for t in ChainMap(self.model.parameters(), self.model.buffers()):

#             if t.device != torch.device('cuda:0'):

#                 raise RuntimeError("module must have its parameters and buffers "

#                                    "on device cuda:0 (device_ids[0]) but found one of "

#                                    "them on device: {}".format(t.device))