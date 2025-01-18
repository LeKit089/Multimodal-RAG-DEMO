import streamlit as st
from .ui_dev import clear_query_history
from .hybrid_dev import ExampleEmbeddingFunction
from .system_prompt import EXPERT_Q_AND_A_SYSTEM
from .data_preprocessing import parse_pdf, convert_to_documents, convert_img_to_tables
from .summary_utils import summarize_docs, save_summary
import torch
import os
from io import StringIO
import PyPDF2
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    Settings,
)
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama

import logging

logging.basicConfig(level=logging.INFO)

ROOT_DIR = "/home/project/data/jc/Chatbot_Web_Demo/qa_data"
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")


def clear_dirs():
    # make sure the directories exist and no files are in them
    # So this is a bit of a hack, but it works for now
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    else:
        for file in os.listdir(INPUT_DIR):
            os.remove(os.path.join(INPUT_DIR, file))


def get_milvus_collections_list():
    # Connect to Milvus server
    milvus_client = MilvusClient(
        uri="http://localhost:19530/", db_name="default", token="root:Milvus"
    )
    st.session_state["milvus_collections"] = sorted(milvus_client.list_collections())


def reset_engine():
    # st.session_state.clear()
    st.session_state["is_ready"] = False
    get_milvus_collections_list()


if "uploaded_files" not in st.session_state.keys():
    st.session_state["uploaded_files"] = []

if "is_ready" not in st.session_state.keys():
    st.session_state["is_ready"] = False

if "milvus_collections" not in st.session_state:
    st.session_state["milvus_collections"] = []

if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

if "selected_docs" not in st.session_state:
    st.session_state["selected_doc"] = None

def get_embed_model(embed_path):
    logging.info(f"Loading {embed_path}")
    embed_model = HuggingFaceEmbedding(model_name=embed_path, device="cuda:1")
    return embed_model


def load_model():
    llm = Ollama(model="qwen:14b", device="cuda:1", request_timeout=60.0)
    embed_model = get_embed_model(embed_path="/home/project/data/jc/mmRAG/model/bge-m3")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    st.session_state["llm"] = llm
    st.session_state["embed_model"] = embed_model
    st.session_state["reranker"] = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large", top_n=5
    )

# @st.cache_data
def choose_docs():
    get_milvus_collections_list()  # 确保从Milvus获取文档集合

    pdf_doc_list = []
    for collection in st.session_state["milvus_collections"]:
        if collection.startswith("doc_"):
            pdf_doc_list.append(collection.split("_")[1] + ".pdf")

    selected_doc = st.selectbox(
        "选择文档",
        pdf_doc_list,
        #index=None,
        placeholder="请选择文档",
    )
    if selected_doc:
        st.session_state["selected_doc"] = "doc_" + selected_doc.split(".")[0]
    else:
        st.session_state["selected_doc"] = ""

'''
def choose_docs():
    get_milvus_collections_list()  # 确保从Milvus获取文档集合

    pdf_doc_list = []
    for collection in st.session_state["milvus_collections"]:
        if collection.startswith("doc_"):
            pdf_doc_list.append(collection.split("_")[1] + ".pdf")
    
    # 使用multiselect以允许选择多个已有文档
    # selected_docs = st.multiselect(
    #     "选择已有文档",
    #     pdf_doc_list,
    #     placeholder="请选择文档"
    # )

    # 使用file_uploader支持多文件上传
    uploaded_files = st.file_uploader("上传文档", accept_multiple_files=True, type=["pdf"])

    # 处理上传的文件并添加到selected_docs中
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["uploaded_files"] = st.session_state.get("uploaded_files", []) + [file_path]
            selected_docs.append(uploaded_file.name)

    # 将选中的和上传的文档处理为内部存储格式
    st.session_state["selected_docs"] = ["doc_" + doc.split(".")[0] for doc in selected_docs]

    # 可以用来显示用户选择的文档（调试用）
    # st.write("您选择的文档是：", st.session_state["selected_docs"])
''' 

# @st.cache_data cannot be used opon st.file_uploader
def upload_data():
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1
    uploaded_files = st.file_uploader(
         "请上传您的文档",
        on_change=reset_engine,
        accept_multiple_files=True,
        type=["pdf", "md", "txt"],
        key=str(st.session_state["uploader_key"]) + "/file_uploader"
    )
    
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        process_data()
        st.session_state["uploader_key"] += 1
        st.rerun()

        # st.session_state["seleted_doc"] = st.session_state["uploaded_file_name"]


def parse_data():
    documents = []
    for uploaded_file in st.session_state["uploaded_files"]:
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                clear_dirs()
                filepath = f"{INPUT_DIR}/{uploaded_file.name}"
                file_id = uploaded_file.name.split(".")[0]
                os.makedirs(os.path.join(DATA_DIR, file_id), exist_ok=True)
                file_data_path = os.path.join(DATA_DIR, file_id)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                print(f"Processing {filepath}")
                raw_docs = parse_pdf(
                    filepath,
                    extract_image_block_output_dir=os.path.join(
                        file_data_path, "images"
                    ),
                    extract_images_in_pdf=True,
                )
                docs = convert_img_to_tables(raw_docs, file_data_path)
                documents, text_seq = convert_to_documents(docs)
                summary = summarize_docs(text_seq)
                save_summary(summary, file_data_path)
            else:
                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # To read file as string:
                string_data = stringio.read()
                documents.append(
                    Document(
                        text=string_data, metadata={"file_name": uploaded_file.name}
                    )
                )
        st.session_state["uploaded_file_name"] = uploaded_file.name

    return documents


def create_vector_index(documents):
    collection_name = st.session_state["uploaded_file_name"].split(".")[0]
    service_context = ServiceContext.from_defaults(
        llm=st.session_state["llm"],
        embed_model=st.session_state["embed_model"],
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
    # st.write(documents)
    return index


def build_query_engine_from_db(collection_name):
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530/",
        token="root:Milvus",
        collection_name=collection_name,
        dim=1024,
        overwrite=False,
        enable_sparse=True,
        sparse_embedding_function=ExampleEmbeddingFunction(),
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[st.session_state["reranker"]]
    )
    return query_engine


def build_query_engine_from_index(index):
    re_ranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)
    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[st.session_state["reranker"]]
    )

    return query_engine


def success_message():
    st.success("文档加载成功！", icon="👉")


def success_doc_processing_message():
    st.success("文档处理成功！现在，您可在上方选择文档后输入问题进行查询！", icon="✅")


def init_engine():
    st.session_state["is_ready"] = True


def build_query_engine():
    if "selected_docs" not in st.session_state:
        st.session_state["selected_docs"] = []
    if st.session_state["selected_docs"]:
        collection_names = st.session_state["selected_docs"]
        all_query_engines = []

        for collection_name in collection_names:
            query_engine = build_query_engine_from_db(collection_name)
            all_query_engines.append(query_engine)

        st.session_state["query_engines"] = all_query_engines
        init_engine()
        success_message()



def process_data():
    with st.sidebar:
        with st.spinner("处理中"):
            documents = parse_data()
            index = create_vector_index(documents)
            # st.session_state["query_engine"] = build_query_engine_from_index(index)

            # query_engine = build_query_engine_from_index(index)
            # st.session_state["query_engine"] = query_engine
            st.session_state["uploaded_files"] = []
            get_milvus_collections_list()
            success_doc_processing_message()


def sidebar():
    with st.sidebar:
        st.image(
            "/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/374920_tech-logo-png.png",
            use_column_width=True,
        )
        st.markdown(
            "## 指引\n"
            "1. 选择知识库中已有的文档📄\n"
            "2. 输入您想问的问题💬\n"
        )

        if "llm" not in st.session_state and "embed_model" not in st.session_state:
            load_model()

        choose_docs()
        upload_data()
        

        st.button("清除聊天记录", on_click=clear_query_history)
        
        del st.session_state["llm"]
        del st.session_state["embed_model"]

        import gc
        gc.collect()

        if 'torch' in globals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

