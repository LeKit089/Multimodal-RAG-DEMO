import cv2
import numpy as np
from .ditod import add_vit_config
import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import os
from PIL import Image
from pdf2image import convert_from_path
import PyPDF2

import warnings

# import streamlit as st
# from tqdm import tqdm
# import uuid

warnings.filterwarnings("ignore", category=UserWarning)


def is_toc_page(page, toc_keywords):
    text = page.extract_text()
    for keyword in toc_keywords:
        if keyword in text:
            return True
    return False


def has_jump_elements(page):
    try:
        if "/Annots" in page and len(page["/Annots"]) > 0:
            return True
    except KeyError:
        pass
    return False


def filter_pdf(input_pdf_path, output_pdf_path, toc_keywords):
    pdf_reader = PyPDF2.PdfReader(input_pdf_path)
    pdf_writer = PyPDF2.PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        if page_num >= 8 or not (
            is_toc_page(page, toc_keywords) or has_jump_elements(page)
        ):
            pdf_writer.add_page(page)

    with open(output_pdf_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)


def detect_pdf(pdf_path, output_folder):

    assert pdf_path.endswith(".pdf"), "Input file must be a PDF file."

    pdf_path = pdf_path
   
    output_folder = output_folder
    # config_file = ".home.project.data.jc.0-TD.icdar19_configs.cascade.cascade_dit_base.yaml"
    # opts = ["MODEL.WEIGHTS", ".home.project.data.jc.0-TD.icdar19_modern.model.pth"]

    #检查路径是否正确
    config_file = "/home/project/data/jc/0-TD/icdar19_configs/cascade/cascade_dit_base.yaml"
    opts = ["MODEL.WEIGHTS", "/home/project/data/jc/0-TD/icdar19_modern/model.pth"]
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在!")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Step 3: set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
   
    

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: convert PDF to images
    print(f"*******正在处理目录下{pdf_path}的文件******")

    # images = convert_from_path(pdf_path)
    toc_keywords = ["内容目录", "图表目录"]
    filtered_pdf_path = "filtered_" + os.path.basename(pdf_path)
    filter_pdf(pdf_path, filtered_pdf_path, toc_keywords)

    # print(f"*******正在处理目录下{filtered_pdf_path}的文件******")
    images = convert_from_path(filtered_pdf_path)

    if os.path.exists(filtered_pdf_path):
        os.remove(filtered_pdf_path)

    # Step 6: run inference for each image and save the results
    for i, image in enumerate(images):
        print(f"===现在对page_{i+1}进行表格检测处理===")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 将原始的图像保存起来
        output_path_o = os.path.join(output_folder, f"detected_{i + 1}_o.png")
        cv2.imwrite(output_path_o, img)

        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if cfg.DATASETS.TEST[0] == "icdar2019_test":
            md.set(thing_classes=["table"])

        else:
            md.set(thing_classes=["text", "title", "list", "table", "figure"])


        output = predictor(img)["instances"]
        check()    
        pred_classes = output.pred_classes
        pred_boxes = output.pred_boxes
        pred_score = output.scores
        # 获取边界框的坐标
        bboxes = pred_boxes.tensor.cpu().numpy().astype(int)
            
        if len(bboxes)>0:    
            # 创建一个图像列表来保存裁剪的结果
            cropped_images = []
            # 遍历每个边界框
            for j, bbox in enumerate(bboxes):
            # 裁剪图像
                x0, y0, x1, y1 = bbox
                cropped_img = img[y0:y1, x0:x1]
                    
            # 可选：保存裁剪的图像
                output_file_name_cropped = os.path.join(
                    output_folder, f"detected_{i + 1}_cropped_{j}.png"
                )
                cv2.imwrite(output_file_name_cropped, cropped_img)

            # 将裁剪的图像添加到列表中
                cropped_images.append(cropped_img)
                
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # 接下来就是对我们裁减后的表格做其他操作

            for cropped_img in cropped_images:
                pass

        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # 这里就是将原图作效果展示的
            v = Visualizer(
                img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
            )
            result = v.draw_instance_predictions(output.to("cpu"))
            result_image = result.get_image()[:, :, ::-1]
        
            # Step 7: save the result image
            output_path = os.path.join(output_folder, f"detected_{i+1}.png")
            cv2.imwrite(output_path, result_image)   
        
       

    print(f"*******表格检测的图片结果，已经保存到{output_folder}文件中，请查看******")

    del predictor
    del cfg

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info=process.memory_info()
    print(f"Current memory usage:{mem_info.rss /1024 /1024:.2f}MB")

# import cv2
# import numpy as np
# from .ditod import add_vit_config
# import torch

# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultPredictor

# import os
# from PIL import Image
# from pdf2image import convert_from_path
# import PyPDF2

# import warnings

# import streamlit as st
# from tqdm import tqdm
# import uuid
# from ...qa_demo.image import convert_img_to_tables

# warnings.filterwarnings("ignore", category=UserWarning)


# def is_toc_page(page, toc_keywords):
#     text = page.extract_text()
#     for keyword in toc_keywords:
#         if keyword in text:
#             return True
#     return False


# def has_jump_elements(page):
#     try:
#         if "/Annots" in page and len(page["/Annots"]) > 0:
#             return True
#     except KeyError:
#         pass
#     return False


# def filter_pdf(input_pdf_path, output_pdf_path, toc_keywords):
#     pdf_reader = PyPDF2.PdfReader(input_pdf_path)
#     pdf_writer = PyPDF2.PdfWriter()

#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         if page_num >= 8 or not (
#             is_toc_page(page, toc_keywords) or has_jump_elements(page)
#         ):
#             pdf_writer.add_page(page)

#     with open(output_pdf_path, "wb") as output_pdf:
#         pdf_writer.write(output_pdf)

# # 图片转为markdown格式
# # def convert_img_to_tables(image_path):
# #     """
# #     Convert the image blocks to tables using the MiniCPM model.
# #     """
# #     # if not os.path.exists(output_dir):
# #     #     os.makedirs(output_dir)
# #     from src.chatbot_web_demo.pages.qa_demo.cpm_convertor import CPMConvertorFactory

# #     factory = CPMConvertorFactory()
# #     cpm_convertor = factory.get_instance()
# #     question = "请你将图像转换为markdown形式的表格"
    

# #     converted_text = cpm_convertor.convert(query=question, img_path=image_path)
# #     print(f'解析后的图片: {converted_text}')

# #     # st.markdown(f'解析后的图片: {converted_text}')

# #     # img_captions_df = pd.DataFrame(img_captions)
# #     # img_captions_df.to_csv(f"{output_dir}/img_captions.csv", index=False)

# #     cpm_convertor.clear_GPU_mem()

# #     return converted_text



# def detect_pdf(pdf_path, output_folder):
#     assert pdf_path.endswith(".pdf"), "Input file must be a PDF file."
    
#     config_file = "/home/project/data/jc/0-TD/icdar19_configs/cascade/cascade_dit_base.yaml"
#     opts = ["MODEL.WEIGHTS", "/home/project/data/jc/0-TD/icdar19_modern/model.pth"]
#     if not os.path.isfile(config_file):
#         raise FileNotFoundError(f"配置文件 '{config_file}' 不存在!")

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     cfg = get_cfg()
#     add_vit_config(cfg)
#     cfg.merge_from_file(config_file)
#     cfg.merge_from_list(opts)

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     cfg.MODEL.DEVICE = device
#     predictor = DefaultPredictor(cfg)

#     toc_keywords = ["内容目录", "图表目录"]
#     filtered_pdf_path = "filtered_" + os.path.basename(pdf_path)
#     filter_pdf(pdf_path, filtered_pdf_path, toc_keywords)
#     images = convert_from_path(filtered_pdf_path)

#     if os.path.exists(filtered_pdf_path):
#         os.remove(filtered_pdf_path)

#     for i, image in enumerate(images):
#         img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         cropped_images = []

#         # Perform detection and crop images
#         output = predictor(img)["instances"]
#         pred_boxes = output.pred_boxes
#         bboxes = pred_boxes.tensor.cpu().numpy().astype(int)
        
#         if len(bboxes) > 0:
#             for j, bbox in enumerate(bboxes):
#                 x0, y0, x1, y1 = bbox
#                 cropped_img = img[y0:y1, x0:x1]
#                 output_file_name_cropped = os.path.join(
#                     output_folder, f"detected_{i + 1}_cropped_{j}.png"
#                 )
#                 cv2.imwrite(output_file_name_cropped, cropped_img)
#                 cropped_images.append(cropped_img)
                
#         # Display and analyze cropped images
#         for cropped_img in cropped_images:
#             col1, col2 = st.columns([1, 1])
#             unique_key = str(uuid.uuid4())

#             with col1:
#                 st.image(cropped_img, use_column_width=True)

#             result_placeholder = st.empty()
            
#             with col2:
#                 if st.button(f'解析', key=unique_key):
#                     temp_img_path = os.path.join(output_folder, f"temp_{unique_key}.png")
#                     cv2.imwrite(temp_img_path, cropped_img)
#                     converted_text = convert_img_to_tables(temp_img_path)
#                     print(f'解析后的图片: {converted_text}')
#                     result_placeholder.write(f'解析后的表格:\n{converted_text}')
#                     print(temp_img_path)
#                     os.remove(temp_img_path)

#     print(f"*******表格检测的图片结果，已经保存到{output_folder}文件中，请查看******")

#     del predictor
#     del cfg

#     import gc
#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

'''
def detect_pdf(pdf_path, output_folder):

    assert pdf_path.endswith(".pdf"), "Input file must be a PDF file."

    pdf_path = pdf_path
   
    output_folder = output_folder
    # config_file = ".home.project.data.jc.0-TD.icdar19_configs.cascade.cascade_dit_base.yaml"
    # opts = ["MODEL.WEIGHTS", ".home.project.data.jc.0-TD.icdar19_modern.model.pth"]

    #检查路径是否正确
    config_file = "/home/project/data/jc/0-TD/icdar19_configs/cascade/cascade_dit_base.yaml"
    opts = ["MODEL.WEIGHTS", "/home/project/data/jc/0-TD/icdar19_modern/model.pth"]
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在!")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Step 3: set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: convert PDF to images
    print(f"*******正在处理目录下{pdf_path}的文件******")

    # images = convert_from_path(pdf_path)
    toc_keywords = ["内容目录", "图表目录"]
    filtered_pdf_path = "filtered_" + os.path.basename(pdf_path)
    filter_pdf(pdf_path, filtered_pdf_path, toc_keywords)

    # print(f"*******正在处理目录下{filtered_pdf_path}的文件******")
    images = convert_from_path(filtered_pdf_path)

    if os.path.exists(filtered_pdf_path):
        os.remove(filtered_pdf_path)

    # Step 6: run inference for each image and save the results
    for i, image in enumerate(images):
        print(f"===现在对page_{i+1}进行表格检测处理===")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 将原始的图像保存起来
        output_path_o = os.path.join(output_folder, f"detected_{i + 1}_o.png")
        cv2.imwrite(output_path_o, img)

        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if cfg.DATASETS.TEST[0] == "icdar2019_test":
            md.set(thing_classes=["table"])

        else:
            md.set(thing_classes=["text", "title", "list", "table", "figure"])


        output = predictor(img)["instances"]
        check()    
        pred_classes = output.pred_classes
        pred_boxes = output.pred_boxes
        pred_score = output.scores
        # 获取边界框的坐标
        bboxes = pred_boxes.tensor.cpu().numpy().astype(int)
            
        if len(bboxes)>0:    
            # 创建一个图像列表来保存裁剪的结果
            cropped_images = []
            # 遍历每个边界框
            for j, bbox in enumerate(bboxes):
            # 裁剪图像
                x0, y0, x1, y1 = bbox
                cropped_img = img[y0:y1, x0:x1]
                    
            # 可选：保存裁剪的图像
                output_file_name_cropped = os.path.join(
                    output_folder, f"detected_{i + 1}_cropped_{j}.png"
                )
                cv2.imwrite(output_file_name_cropped, cropped_img)

            # 将裁剪的图像添加到列表中
                cropped_images.append(cropped_img)
                
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # 接下来就是对我们裁减后的表格做其他操作

            for cropped_img in cropped_images:
                pass

        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # 这里就是将原图作效果展示的
            v = Visualizer(
                img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
            )
            result = v.draw_instance_predictions(output.to("cpu"))
            result_image = result.get_image()[:, :, ::-1]
        
            # Step 7: save the result image
            output_path = os.path.join(output_folder, f"detected_{i+1}.png")
            cv2.imwrite(output_path, result_image)   
        
        # 将图片展示并配有按钮，方便解析
            for cropped_img in cropped_images:
                col1, col2 = st.columns([1, 1])
                unique_key = str(uuid.uuid4())

                with col1:
                    st.image(cropped_img, width=150)
                with col2:
                    if st.button(f'解析', key=unique_key):
                        convert_img_to_tables(cropped_img)

    print(f"*******表格检测的图片结果，已经保存到{output_folder}文件中，请查看******")

    del predictor
    del cfg

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''

def check():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info=process.memory_info()
    print(f"Current memory usage:{mem_info.rss /1024 /1024:.2f}MB")



