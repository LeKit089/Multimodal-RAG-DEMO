import streamlit as st
import os
import logging
import torch


from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..detect_demo.table_detection.inference_pdf import detect_pdf
from ..qa_demo.summary_utils import summarize_docs, save_summary, read_summary
from ..qa_demo.data_preprocessing import (
    parse_pdf,
    convert_img_to_tables,
    convert_to_documents,
)
from ..qa_demo.sidebar_dev import create_vector_index


if "uploaded_detect_file" not in st.session_state.keys():
    st.session_state["uploaded_detect_file"] = []

if "output_file_or_folder" not in st.session_state.keys():
    st.session_state["output_file_or_folder"] = [] 

#if "unstructured_output_dir" not in st.session_state.keys():
    #st.session_state["unstructured_output_dir"] = []

#DATA_DIR = "/home/project/Chatbot_Web_Demo/doc_parse_data"
ROOT_DIR="/home/project/data/jc/Chatbot_Web_Demo/qa_data"
DATA_DIR=os.path.join(ROOT_DIR,"data")
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")


def get_embed_model(embed_path):
    logging.info(f"Loading {embed_path}")
    embed_model = HuggingFaceEmbedding(model_name=embed_path, device="cuda:1")
    return embed_model

# def move_model_to_cuda_0(model):
#     for name, param in model.named_parameters():
#         if param.device != torch.device('cuda:0'):
#             param.data = param.data.to('cuda:0')

#     for name, buffer in model.named_buffers():
#         if buffer.device != torch.device('cuda:0'):
#             buffer.data = buffer.data.to('cuda:0')

#     return model

def load_model():
    llm = Ollama(model="qwen2:7b", device="cuda:0", context_window = 2048, request_timeout=60.0)
    embed_model = get_embed_model(embed_path="/home/project/data/jc/mmRAG/model/bge-m3")
    # embed_model = move_model_to_cuda_0(embed_model)

            
    Settings.llm = llm
    Settings.embed_model = embed_model

    st.session_state["llm"] = llm
    st.session_state["embed_model"] = embed_model

    

def clear_dirs():
    # make sure the directories exist and no files are in them
    # So this is a bit of a hack, but it works for now
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    else:
        for file in os.listdir(INPUT_DIR):
            os.remove(os.path.join(INPUT_DIR, file))


def process_data():
    file = st.session_state["uploaded_detect_file"]
    if file:
        clear_dirs()
        file_name = file.name
        filepath = os.path.join(INPUT_DIR, file_name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
        
        output_dir = os.path.join(DATA_DIR, file_name.split(".")[0])
        os.makedirs(output_dir,exist_ok=True)

        detect_pdf(filepath, output_dir)

        raw_docs = parse_pdf(
            filepath,
            extract_image_block_output_dir=os.path.join(
                output_dir, "images"
            ),
            extract_images_in_pdf=True,
        )
        docs = convert_img_to_tables(raw_docs, output_dir)
        documents, text_seq = convert_to_documents(docs)
        summary = summarize_docs(text_seq)
        save_summary(summary, output_dir)
        
        import gc
        gc.collect()

        st.session_state["uploaded_file_name"] = file.name
        index=create_vector_index(documents)

        st.session_state["output_file_or_folder"] = output_dir
        st.success("解析成功！")   


def visulize_result():
    if "output_file_or_folder" in st.session_state.keys():
        output_file_or_folder = st.session_state["output_file_or_folder"]

    #if "unstructured_output_dir" in st.session_state.keys():
        #unstructured_output_dir = st.session_state["unstructured_output_dir"]

    try:
        if output_file_or_folder:
            summary = read_summary(output_file_or_folder)
            st.write(summary)
            
        #if output_file_or_folder and unstructured_output_dir:
            #summary = read_summary(unstructured_output_dir)
            #st.write(f"{output_file_or_folder}")
            #st.write(summary)
            #st.write(f"{output_file_or_folder}")
            
            if os.path.isdir(output_file_or_folder):
                all_files = os.listdir(output_file_or_folder)
                detected_files=[
                    file for file in all_files
                    if file.startswith("detected_") and 
                    file.endswith(".png") and
                    file.count('_')==1
                ]
                detected_files.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
                detected_files_path = [os.path.join(output_file_or_folder, file) for file in detected_files]
                detected_files_captions = [f"{file}识别结果" for file in detected_files]
                st.image(detected_files_path, width=200, caption=detected_files_captions)

            
            
            #all_files = os.listdir(output_file_or_folder)
       #     all_images = [file for file in all_files if file.endswith(".png")]
        #    all_files_path = [
         #       os.path.join(output_file_or_folder, file)
          #      for file in all_images
            #]
           # st.image(all_files_path, width=200, caption=all_images)
    except NameError:
        output_file_or_folder = ""


def upload_data():

    if "uploaded_detect_file" in st.session_state:
        del st.session_state["uploaded_detect_file"]  

    if "output_file_or_folder" in st.session_state:
        del st.session_state["output_file_or_folder"] 
        
    uploaded_detect_file = st.file_uploader(
        "请上传您的文档",
        type=["pdf"],
    )

    st.session_state["uploaded_detect_file"] = uploaded_detect_file
    if uploaded_detect_file:
        with st.spinner("正在解析..."):
            process_data()


def doc_parse_demo():
    st.header("研报解析")
    with st.sidebar:
        st.image("/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/374920_tech-logo-png.png", use_container_width=True)

    if "llm" not in st.session_state and "embed_model" not in st.session_state:
        load_model()

    # print(f'************************************session_state : {st.session_state["llm"]} ************************************')

    upload_data()
    visulize_result()

    del st.session_state["llm"]
    del st.session_state["embed_model"]
    
    # 删除 Settings 类的属性
    # delattr(Settings, 'llm')
    # delattr(Settings, 'embed_model')
    # del Settings.llm
    # del Settings.embed_model

    

    # if torch.cuda.is_available():
    if 'torch' in globals() and torch.cuda.is_available():
        torch.cuda.empty_cache()
