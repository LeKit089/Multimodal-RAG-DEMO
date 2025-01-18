# import os
# from typing import Optional, Sequence
# from tqdm import tqdm
# import pandas as pd
# from pymilvus import MilvusClient, connections, Collection, utility
# from PyPDF2 import PdfReader
# from typing import Sequence
# from llama_index.core.response_synthesizers import Refine
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )
# from src.chatbot_web_demo.pages.qa_demo.system_prompt import EXPERT_Q_AND_A_SYSTEM
# from llama_index.core import Document, Settings, ServiceContext, StorageContext, VectorStoreIndex
# from unstructured.partition.pdf import partition_pdf
# from unstructured.chunking.basic import chunk_elements
# import nltk
# from src.chatbot_web_demo.pages.qa_demo.hybrid_dev import ExampleEmbeddingFunction
# from llama_index.llms.ollama import Ollama
# import logging
# from llama_index.vector_stores.milvus import MilvusVectorStore

# logging.basicConfig(level=logging.INFO)

# ROOT_DIR = "/home/project/data/jc/Chatbot_Web_Demo/qa_data"
# DATA_DIR = os.path.join(ROOT_DIR, "data")

# # 生成摘要
# def summarize_docs(text: Sequence[str]) -> str:
#     """
#     Summarize a list of documents using the Llama Index Refine Response Synthesizers.
#     """
#     summarizer = Refine(verbose=True)
#     response = summarizer.get_response(
#         "请你着重关注文档的数据部分，分点给出该文档的摘要(保留重要数据)，最后给出总结",
#         text
#     )
#     return response

# # 保存摘要
# def save_summary(summary, output_dir):
#     summary_path = os.path.join(output_dir, "summary.txt")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write(summary)
#     print(f"Summary saved to {summary_path}")

# # 创建向量索引
# def create_vector_index(documents, collection_name, llm, embed_model):
#     service_context = ServiceContext.from_defaults(
#         llm=llm,
#         embed_model=embed_model,
#         system_prompt=EXPERT_Q_AND_A_SYSTEM,
#     )

#     vector_store = MilvusVectorStore(
#         uri="http://localhost:19530/",
#         token="root:Milvus",
#         collection_name=f"doc_{collection_name}",
#         dim=1024,
#         overwrite=True,
#         enable_sparse=True,
#         sparse_embedding_function=ExampleEmbeddingFunction(),
#         hybrid_ranker="RRFRanker",
#         hybrid_ranker_params={"k": 60},
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_documents(
#         documents,
#         service_context=service_context,
#         storage_context=storage_context,
#     )

#     # 查看存储在 Milvus 数据库中的数据
#     print("#############################################################")
#     check_milvus_collection(f"doc_{upload_file_name.split('.')[0]}")
#     print("#############################################################")
#     return index

# def get_embed_model(embed_path):
#     logging.info(f"Loading {embed_path}")
#     embed_model = HuggingFaceEmbedding(model_name=embed_path, device="cuda:1")
#     return embed_model

# def load_model():
#     llm = Ollama(model="qwen:14b", request_timeout=60.0)
#     # llm = None
#     embed_model = get_embed_model(embed_path="/home/project/data/jc/mmRAG/model/bge-m3")
#     Settings.llm = llm
#     Settings.embed_model = embed_model
#     # st.session_state["llm"] = llm
#     # st.session_state["embed_model"] = embed_model
#     return FlagEmbeddingReranker(
#         model="BAAI/bge-reranker-large", top_n=5
#     )

# # 提取图像块并保存到指定目录
# def parse_pdf(
#     filepath,
#     extract_image_block_output_dir: Optional[str] = None,
#     extract_images_in_pdf: bool = True,
# ):
#     """
#     Parse a PDF file using the `unstructured` library.
#     """

#     assert os.path.exists(filepath), f"PDF file {filepath} does not exist"
#     if extract_images_in_pdf:
#         return partition_pdf(
#             filepath,
#             strategy="hi_res",
#             extract_images_in_pdf=True,
#             extract_image_block_types=["Image", "Table"],
#             extract_image_block_to_payload=False,
#             extract_image_block_output_dir=extract_image_block_output_dir,
#         )
#     else:
#         return partition_pdf(filepath, strategy="hi_res", extract_images_in_pdf=False)

# # 将提取的图像转换为表格数据
# def convert_img_to_tables(raw_docs, output_dir):
#     """
#     Convert the image blocks to tables using the MiniCPM model.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     from src.chatbot_web_demo.pages.qa_demo.cpm_convertor import CPMConvertorFactory

#     factory = CPMConvertorFactory()
#     cpm_convertor = factory.get_instance()
#     question = "请你将图像转换为markdown形式的表格"
#     documents = []
#     img_captions = []

#     for doc in tqdm(raw_docs, "Converting images to tables"):
#         if doc.to_dict()["type"] in ["Image", "Table"]:
#             img_path = doc.to_dict()["metadata"]["image_path"]
#             converted_text = cpm_convertor.convert(query=question, img_path=img_path)
#             img_captions.append(
#                 {"image_path": img_path.split("/")[-1], "caption": converted_text}
#             )
#             converted_doc = doc
#             converted_doc.text = converted_text
#             documents.append(converted_doc)
#         else:
#             documents.append(doc)

#     img_captions_df = pd.DataFrame(img_captions)
#     img_captions_df.to_csv(f"{output_dir}/img_captions.csv", index=False)

#     cpm_convertor.clear_GPU_mem()

#     return documents

# # 将表格数据转换为文档对象，并提取文本序列。
# def convert_to_documents(documents, max_characters=512, overlap=50):
#     """
#     convert the partitioned documents to llamaindex Document objects.
#     """

#     chunks = chunk_elements(documents, max_characters=max_characters, overlap=overlap)
#     documents = []
#     text_seq = []
#     for chunk in tqdm(chunks, desc="Converting to documents"):
#         text = chunk.to_dict()["text"]
#         text_seq.append(text)
#         document = Document(
#             doc_id=chunk.to_dict()["element_id"],
#             text=text,
#             metadata={
#                 "page_number": chunk.to_dict()["metadata"]["page_number"],
#                 "filename": chunk.to_dict()["metadata"]["filename"],
#             },
#         )
#         documents.append(document)

#     return documents, text_seq

# def read_summary(document_path: str) -> str:
#     """
#     Read the summary from a file.
#     """
#     summary_filepath = os.path.join(document_path, "summary.txt")
#     with open(summary_filepath, 'r', encoding='utf-8') as file:
#         summary = file.read()
#     return summary
# # def create_summary_document(summary):
# #     return Document(
# #         doc_id="summary",
# #         text=summary,
# #         metadata={"type": "summary"}
# #     )

# # 读取本地PDF文件并进行处理
# def process_local_pdf(filepath, output_dir, extract_images_in_pdf=False):
#     # 读取PDF文件
#     with open(filepath, 'rb') as f:
#         pdf = PdfReader(f)
#         # print(f"PDF file '{pdf_path}' has {len(pdf)} pages.")

#     # 加载模型
#     load_model()
#     # 提取图像块并保存到指定目录（如果需要）
#     raw_docs = parse_pdf(filepath, extract_image_block_output_dir=file_data_path, extract_images_in_pdf=extract_images_in_pdf)

#     # 将提取的图像转换为表格数据
#     documents_with_tables = convert_img_to_tables(raw_docs, output_dir)

#     # 将表格数据转换为文档对象，并提取文本序列
#     documents, text_seq = convert_to_documents(documents_with_tables)

#     return documents, text_seq

# # 查看存储在 Milvus 数据库中的数据
# def check_milvus_collection(collection_name):
#     connections.connect(alias="default", host="localhost", port="19530")
#     if utility.has_collection(collection_name):
#         print(f"Collection '{collection_name}' exists in Milvus.")
#         collection = Collection(collection_name)
#         expr = "doc_id != ''"
#         results = collection.query(expr, output_fields=["*"])
#         print(f"Number of vectors in collection '{collection_name}': {len(results)}")
#         for i, result in enumerate(results):
#             print(f"\nResult {i + 1}:")
#             for key, value in result.items():
#                 print(f"{key}: {value}")
#     else:
#         print(f"Collection '{collection_name}' does not exist in Milvus.")
#     connections.disconnect(alias="default")

# if __name__ == "__main__":
#     # print(nltk.data.path)

#     filepath = '/home/project/data/datasets/att/大文件/9489418.pdf'
#     # output_dir = 'home/project/Chatbot_Web_Demo/doc_parse_data/output'
#     upload_file_name = '9489418.pdf'
#     extract_images_in_pdf = True

#     file_id = upload_file_name.split(".")[0]
#     os.makedirs(os.path.join(DATA_DIR, file_id), exist_ok=True)
#     file_data_path = os.path.join(DATA_DIR, file_id)

#     documents, text_seq = process_local_pdf(filepath, file_data_path, extract_images_in_pdf)

#     summary = summarize_docs(text_seq)
#     # documents = create_summary_document(summary)
#     save_summary(summary, file_data_path)
#     read_summary = read_summary(file_data_path)
#     index = create_vector_index(documents, upload_file_name.split(".")[0], Settings.llm, Settings.embed_model)

#     print('#############################################################################')
#     # print(f"text_seq is {text_seq}")
#     print('#############################################################################')
#     print(f"summary is {read_summary}")
#     print('#############################################################################')
#     print(index)

#     # 查看存储在 Milvus 数据库中的数据
#     check_milvus_collection(f"doc_{upload_file_name.split('.')[0]}")



import os
from typing import Optional
from tqdm import tqdm
import pandas as pd
from pymilvus import MilvusClient
from PyPDF2 import PdfReader
from typing import Sequence
from llama_index.core.response_synthesizers import Refine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from src.chatbot_web_demo.pages.qa_demo.system_prompt import EXPERT_Q_AND_A_SYSTEM
from llama_index.core import Document, Settings, ServiceContext, StorageContext, VectorStoreIndex
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
import nltk
from src.chatbot_web_demo.pages.qa_demo.hybrid_dev import ExampleEmbeddingFunction
from llama_index.llms.ollama import Ollama
import logging
from llama_index.vector_stores.milvus import MilvusVectorStore

logging.basicConfig(level=logging.INFO)

ROOT_DIR = "/home/project/data/jc/Chatbot_Web_Demo/qa_data"
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 生成摘要
def summarize_docs(text: Sequence[str]) -> str:
    """
    Summarize a list of documents using the Llama Index Refine Response Synthesizers.
    """
    summarizer = Refine(verbose=True)
    response = summarizer.get_response(
        "请你着重关注文档的数据部分，分点给出该文档的摘要(保留重要数据)，最后给出总结",
        text
    )
    return response

# 保存摘要
def save_summary(summary, output_dir):
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")

# 创建向量索引
def create_vector_index(documents, collection_name, llm, embed_model):
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=EXPERT_Q_AND_A_SYSTEM,
    )

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530/",
        token="root:Milvus",
        collection_name=f"doc_{collection_name}",
        dim=1024,
        overwrite=True,
        enable_sparse=True,
        sparse_embedding_function=ExampleEmbeddingFunction(),
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )
    return index

def get_embed_model(embed_path):
    logging.info(f"Loading {embed_path}")
    embed_model = HuggingFaceEmbedding(model_name=embed_path, device="cuda:1")
    return embed_model

def load_model():
    llm = Ollama(model="qwen:14b", request_timeout=60.0)
    # llm = None
    embed_model = get_embed_model(embed_path="/home/project/data/jc/mmRAG/model/bge-m3")
    Settings.llm = llm
    Settings.embed_model = embed_model
    # st.session_state["llm"] = llm
    # st.session_state["embed_model"] = embed_model
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large", top_n=5
    )

# 提取图像块并保存到指定目录
def parse_pdf(
    filepath,
    extract_image_block_output_dir: Optional[str] = None,
    extract_images_in_pdf: bool = True,
):
    """
    Parse a PDF file using the `unstructured` library.
    """

    assert os.path.exists(filepath), f"PDF file {filepath} does not exist"
    if extract_images_in_pdf:
        return partition_pdf(
            filepath,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=extract_image_block_output_dir,
        )
    else:
        return partition_pdf(filepath, strategy="hi_res", extract_images_in_pdf=False)

# 将提取的图像转换为表格数据
def convert_img_to_tables(raw_docs, output_dir):
    """
    Convert the image blocks to tables using the MiniCPM model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from src.chatbot_web_demo.pages.qa_demo.cpm_convertor import CPMConvertorFactory

    factory = CPMConvertorFactory()
    cpm_convertor = factory.get_instance()
    question = "请你将图像转换为markdown形式的表格"
    documents = []
    img_captions = []

    for doc in tqdm(raw_docs, "Converting images to tables"):
        if doc.to_dict()["type"] in ["Image", "Table"]:
            img_path = doc.to_dict()["metadata"]["image_path"]
            converted_text = cpm_convertor.convert(query=question, img_path=img_path)
            img_captions.append(
                {"image_path": img_path.split("/")[-1], "caption": converted_text}
            )
            converted_doc = doc
            converted_doc.text = converted_text
            documents.append(converted_doc)
        else:
            documents.append(doc)

    img_captions_df = pd.DataFrame(img_captions)
    img_captions_df.to_csv(f"{output_dir}/img_captions.csv", index=False)

    cpm_convertor.clear_GPU_mem()

    return documents

# 将表格数据转换为文档对象，并提取文本序列。
def convert_to_documents(documents, max_characters=512, overlap=50):
    """
    convert the partitioned documents to llamaindex Document objects.
    """

    chunks = chunk_elements(documents, max_characters=max_characters, overlap=overlap)
    documents = []
    text_seq = []
    for chunk in tqdm(chunks, desc="Converting to documents"):
        text = chunk.to_dict()["text"]
        text_seq.append(text)
        document = Document(
            doc_id=chunk.to_dict()["element_id"],
            text=text,
            metadata={
                "page_number": chunk.to_dict()["metadata"]["page_number"],
                "filename": chunk.to_dict()["metadata"]["filename"],
            },
        )
        documents.append(document)

    return documents, text_seq

def read_summary(document_path: str) -> str:
    """
    Read the summary from a file.
    """
    summary_filepath = os.path.join(document_path, "summary.txt")
    with open(summary_filepath, 'r', encoding='utf-8') as file:
        summary = file.read()
    return summary
# def create_summary_document(summary):
#     return Document(
#         doc_id="summary",
#         text=summary,
#         metadata={"type": "summary"}
#     )

# 读取本地PDF文件并进行处理
def process_local_pdf(filepath, output_dir, extract_images_in_pdf=False):
    # 读取PDF文件
    with open(filepath, 'rb') as f:
        pdf = PdfReader(f)
        # print(f"PDF file '{pdf_path}' has {len(pdf)} pages.")

    # 加载模型
    load_model()
    # 提取图像块并保存到指定目录（如果需要）
    raw_docs = parse_pdf(filepath, extract_image_block_output_dir=file_data_path, extract_images_in_pdf=extract_images_in_pdf)

    # 将提取的图像转换为表格数据
    documents_with_tables = convert_img_to_tables(raw_docs, output_dir)

    # 将表格数据转换为文档对象，并提取文本序列
    documents, text_seq = convert_to_documents(documents_with_tables)

    return documents, text_seq


if __name__ == "__main__":
    # print(nltk.data.path)

    directory = '/home/project/data/datasets/att/大文件'
    extract_images_in_pdf = True

    # 加载模型
    load_model()

    for filename in os.listdir(directory):
        # upload_file_name = filename
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            file_id = filename.split(".")[0]
            file_data_path = os.path.join(DATA_DIR, file_id)
            os.makedirs(file_data_path, exist_ok=True)

            documents, text_seq = process_local_pdf(filepath, file_data_path, extract_images_in_pdf)
            summary = summarize_docs(text_seq)
            save_summary(summary, file_data_path)

            index = create_vector_index(documents, filename.split(".")[0], Settings.llm, Settings.embed_model)

            print('#############################################################################')
            print(f"summary for {filename} is {read_summary(file_data_path)}")
            print('#############################################################################')

