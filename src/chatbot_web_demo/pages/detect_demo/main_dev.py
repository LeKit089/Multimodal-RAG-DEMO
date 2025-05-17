# import streamlit as st
# import os
# import json
# import pandas as pd
# import torch
# from .table_detection.inference import detect_image
# from .table_detection.inference_pdf import detect_pdf 
# # from ..qa_demo.data_preprocessing import parse_pdf,convert_img_to_tables,convert_to_documents
# from ..qa_demo.image import convert_img_to_tables

# if "uploaded_detect_file" not in st.session_state.keys():
#     st.session_state["uploaded_detect_file"] = []

# if "output_file_or_folder" not in st.session_state.keys():
#     st.session_state["output_file_or_folder"] = []

# DATA_DIR = "/home/project/data/jc/Chatbot_Web_Demo/detect_data"
# INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")




# def clear_dirs():
#     # make sure the directories exist and no files are in them
#     # So this is a bit of a hack, but it works for now
#     if not os.path.exists(INPUT_DIR):
#         os.makedirs(INPUT_DIR)
#     else:
#         for file in os.listdir(INPUT_DIR):
#             os.remove(os.path.join(INPUT_DIR, file))
                    
# def process_data():
#     file = st.session_state["uploaded_detect_file"]
#     if file:
#         clear_dirs()
#         file_name = file.name
#         filepath = os.path.join(INPUT_DIR, file_name)
        
#         with open(filepath, "wb") as f:
#             f.write(file.getbuffer())

#         # raw_docs=parse_pdf(filepath,extract_images_in_pdf=True)

#         if file.type == "application/pdf":
#             os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
#             output_dir = os.path.join(DATA_DIR, file_name.split(".")[0])
#             detect_pdf(filepath, output_dir)

#             # converted_docs=convert_img_to_tables(raw_docs,output_dir)
#             # documents,_=convert_to_documents(converted_docs)
#             # table_docs=[doc for doc in documents if doc.text.strip().startswith("|")]

#             st.session_state["output_file_or_folder"] = output_dir
#         else:
#             os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
#             output_file_path = os.path.join(DATA_DIR, file_name.split(".")[0])
#             detect_image(filepath, output_file_path)

#             st.session_state["output_file_or_folder"] = output_file_path
#         st.success("识别成功！")
#         print(f"识别的文件:{filepath}")  
#         # st.session_state["table_docs"]=table_docs    


# def visulize_img():
#     if "output_file_or_folder" in st.session_state.keys():
#         output_file_or_folder = st.session_state["output_file_or_folder"]
#     try:
#         if output_file_or_folder:
#             #st.write(f"{output_file_or_folder}")
#             if os.path.isdir(output_file_or_folder):
#                 all_files = os.listdir(output_file_or_folder)
#                 detected_files=[
#                     file for file in all_files
#                     if file.startswith("detected_") and 
#                     file.endswith(".png") and
#                     file.count('_')==1
#                 ]
#                 detected_files.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
#                 detected_files_path = [os.path.join(output_file_or_folder, file) for file in detected_files]
#                 detected_files_captions = [f"{file}识别结果" for file in detected_files]
#                 st.image(detected_files_path, width=200, caption=detected_files_captions)
                
#                 for file in detected_files:

#                     file_path = os.path.join(output_file_or_folder, file)
                    
#                     # Convert the image to a table
#                     converted_doc=convert_img_to_tables(file_path,output_file_or_folder)
                    
#                     if converted_doc.text:
#                         st.markdown(converted_doc.text)
#                         st.markdown("---")


#                 # table_docs=st.session_state["table_docs"]
#                 # for idx,doc in enumerate(table_docs):
#                 #     if len(doc.text)>10 and doc.text!="": 
#                 #     #if len(doc.text)>10: 
#                 #         st.markdown("-------------------")
#                 #         st.write(f"表格 **{idx+1}**:")
#                 #         st.write(doc.text)
#                 #         page_number=doc.metadata.get("page_number","unknown")
#                 #         st.write(f"页码:**{page_number}**")   
#             else:
#                 st.warning(f"{output_file_or_folder} 不是一个有效的文件夹")
                
                
#                 #all_files_path = [os.path.join(output_file_or_folder, file) for file in all_files]
#                 #all_files_captions = [f"{file}识别结果" for file in all_files]
#                 #st.image(all_files_path, width=200, caption=all_files_captions)
#             #else:
#                 #st.image(output_file_or_folder, width=200, caption=f"{output_file_or_folder}识别结果")
                
#     except NameError:
#         output_file_or_folder = ""

# def upload_data():
#     clear_dirs()
    
#     if "output_file_or_folder" in st.session_state:
#         del st.session_state["output_file_or_folder"]

#     uploaded_detect_file = st.file_uploader(
#         "请上传您的文档或图像",
#         type=["pdf", "png", "jpg", "jpeg"],
#     )

#     st.session_state["uploaded_detect_file"] = uploaded_detect_file
#     if uploaded_detect_file:
#         with st.spinner("正在识别..."):
#             process_data()


# def detect_demo():
#     st.header("识别工具")
#     with st.sidebar:
#         st.image("/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/374920_tech-logo-png.png", use_column_width=True)

#     upload_data()
#     visulize_img()

#     import gc
#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


import streamlit as st
import os
import json
import pandas as pd
import torch
from .table_detection.inference import detect_image
from .table_detection.inference_pdf import detect_pdf 
# from ..qa_demo.data_preprocessing import parse_pdf,convert_img_to_tables,convert_to_documents
from ..qa_demo.image import convert_img_to_tables

if "uploaded_detect_file" not in st.session_state.keys():
    st.session_state["uploaded_detect_file"] = []

if "output_file_or_folder" not in st.session_state.keys():
    st.session_state["output_file_or_folder"] = []

DATA_DIR = "/home/project/data/jc/Chatbot_Web_Demo/detect_data"
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")




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

        # raw_docs=parse_pdf(filepath,extract_images_in_pdf=True)

        if file.type == "application/pdf":
            os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
            output_dir = os.path.join(DATA_DIR, file_name.split(".")[0])
            detect_pdf(filepath, output_dir)

            # converted_docs=convert_img_to_tables(raw_docs,output_dir)
            # documents,_=convert_to_documents(converted_docs)
            # table_docs=[doc for doc in documents if doc.text.strip().startswith("|")]

            st.session_state["output_file_or_folder"] = output_dir
        else:
            os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
            output_file_path = os.path.join(DATA_DIR, file_name.split(".")[0])
            detect_image(filepath, output_file_path)

            st.session_state["output_file_or_folder"] = output_file_path
        st.success("识别成功！")
        print(f"识别的文件:{filepath}")  
        # st.session_state["table_docs"]=table_docs    


def visulize_img():
    if "output_file_or_folder" in st.session_state.keys():
        output_file_or_folder = st.session_state["output_file_or_folder"]
    try:
        if output_file_or_folder:
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
                for i, file_path in enumerate(detected_files_path):
                    file = detected_files[i]
                    caption = detected_files_captions[i]
                    col1, col2 = st.columns([1, 1])

                     # Display the image
                    with col1:
                        st.image(file_path, width=200, caption=caption)

                    container = st.container()
                    
                    with container:
                     # 添加一个选择器，允许用户选择要处理的文件
                        base_name = file.split('.')[0]  # 获取文件的基本名称，例如：detected_1
                        cropped_files = [
                            f for f in os.listdir(output_file_or_folder)
                            if (
                                (f.startswith(base_name + '_cropped_') or f.startswith(base_name + '_cropped'))
                                and f.endswith('.png')
                            )
                        ]

                         # 如果有可用的裁剪文件，则创建一个选择器
                        if cropped_files:
                            selected_file = st.selectbox(
                                '请选择要处理的裁剪文件',
                                cropped_files,
                                index=0,  # 默认选中第一个文件
                                key=f'select_box_{i}'
                            )

                            # 构建完整的文件路径
                            file_path = os.path.join(output_file_or_folder, selected_file)

                             # 当用户选择了文件后，处理该文件
                            if st.button(f'解析 {caption}', key=f'process_button_{i}'):
                                if os.path.exists(file_path):
                                 # Convert the image to a table when the button is clicked
                                    converted_doc = convert_img_to_tables(file_path)

                                    # Parse the converted document into a dataframe
                                    table_data = [row.split("|")[1:-1] for row in converted_doc.split("\n") if row.strip()]
                                    df = pd.DataFrame(table_data)

                                     # Display the dataframe as a table with a fixed height
                                    st.dataframe(df, width=1000, height=200)
                                else:
                                    st.warning(f"文件 {selected_file} 不存在，请检查文件名是否正确。")
                        else: 
                            st.warning(f"没有找到与 {base_name} 相关的裁剪文件。")
                    st.markdown("<hr style='border: 1px solid black; width: 100%;'>", unsafe_allow_html=True) 
                    # with container:
                    #     # Add a button for processing the image
                    #     if st.button(f'解析 {caption}', key=f'process_button_{i}'):
                    #             file_path = os.path.join(output_file_or_folder, file)
                                
                    #             # Convert the image to a table when the button is clicked
                    #             converted_doc = convert_img_to_tables(file_path)

                    #             # Parse the converted document into a dataframe
                    #             table_data = [row.split("|")[1:-1] for row in converted_doc.split("\n") if row.strip()]
                    #             df = pd.DataFrame(table_data)


                    #             # Display the dataframe as a table with a fixed height
                    #             st.dataframe(df, width=1000, height=200)
                    # st.markdown("<hr style='border: 1px solid black; width: 100%;'>", unsafe_allow_html=True)
                       
                        

                # table_docs=st.session_state["table_docs"]
                # for idx,doc in enumerate(table_docs):
                #     if len(doc.text)>10 and doc.text!="": 
                #     #if len(doc.text)>10: 
                #         st.markdown("-------------------")
                #         st.write(f"表格 **{idx+1}**:")
                #         st.write(doc.text)
                #         page_number=doc.metadata.get("page_number","unknown")
                #         st.write(f"页码:**{page_number}**")   
            else:
                st.warning(f"{output_file_or_folder} 不是一个有效的文件夹")
                
                
                #all_files_path = [os.path.join(output_file_or_folder, file) for file in all_files]
                #all_files_captions = [f"{file}识别结果" for file in all_files]
                #st.image(all_files_path, width=200, caption=all_files_captions)
            #else:
                #st.image(output_file_or_folder, width=200, caption=f"{output_file_or_folder}识别结果")
                
    except NameError:
        output_file_or_folder = ""

def upload_data():
    clear_dirs()
    
    if "output_file_or_folder" in st.session_state:
        del st.session_state["output_file_or_folder"]

    uploaded_detect_file = st.file_uploader(
        "请上传您的文档或图像",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    st.session_state["uploaded_detect_file"] = uploaded_detect_file
    if uploaded_detect_file:
        with st.spinner("正在识别..."):
            process_data()


def detect_demo():
    st.header("识别工具")
    with st.sidebar:
        st.image("/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/374920_tech-logo-png.png", use_column_width=True)

    upload_data()
    visulize_img()

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

'''def visulize_img():
    if "output_file_or_folder" in st.session_state.keys():
        output_file_or_folder = st.session_state["output_file_or_folder"]
    try:
        if output_file_or_folder:
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
                for i, file_path in enumerate(detected_files_path):
                    file = detected_files[i]
                    caption = detected_files_captions[i]
                    col1, col2 = st.columns([1, 1])

                     # Display the image
                    with col1:
                        st.image(file_path, width=200, caption=caption)

                    # Add a button for processing the image
                    with col2:
                        if st.button(f'解析 {caption}', key=f'process_button_{i}'):
                            file_path = os.path.join(output_file_or_folder, file)
                            # Convert the image to a table when the button is clicked
                            converted_doc = convert_img_to_tables(file_path, output_file_or_folder)

                            if converted_doc.text:
                                st.markdown(converted_doc.text)
                                st.markdown("---")

                # table_docs=st.session_state["table_docs"]
                # for idx,doc in enumerate(table_docs):
                #     if len(doc.text)>10 and doc.text!="": 
                #     #if len(doc.text)>10: 
                #         st.markdown("-------------------")
                #         st.write(f"表格 **{idx+1}**:")
                #         st.write(doc.text)
                #         page_number=doc.metadata.get("page_number","unknown")
                #         st.write(f"页码:**{page_number}**")   
            else:
                st.warning(f"{output_file_or_folder} 不是一个有效的文件夹")
                
                
                #all_files_path = [os.path.join(output_file_or_folder, file) for file in all_files]
                #all_files_captions = [f"{file}识别结果" for file in all_files]
                #st.image(all_files_path, width=200, caption=all_files_captions)
            #else:
                #st.image(output_file_or_folder, width=200, caption=f"{output_file_or_folder}识别结果")
                
    except NameError:
        output_file_or_folder = ""
'''

