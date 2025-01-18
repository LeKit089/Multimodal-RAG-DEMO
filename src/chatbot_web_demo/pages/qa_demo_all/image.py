from unstructured.partition.image import partition_image
from llama_index.core import Document
import pandas as pd
import os
from tqdm import tqdm
from .cpm_convertor import CPMConvertorFactory

def convert_img_to_tables(image_path, output_dir):
    """
    Convert the image blocks to tables using the MiniCPM model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    factory = CPMConvertorFactory()
    cpm_convertor = factory.get_instance()
    question = "请你将图像转换为markdown形式的表格"
    img_captions = []

    # Partition the image to get image blocks
    raw_docs = partition_image(image_path)

    for doc in tqdm(raw_docs, desc="Converting images to tables"):
        if doc.to_dict()["type"] in ["Image", "Table"]:
            file_directory = doc.to_dict()["metadata"]["file_directory"]
            filename = doc.to_dict()["metadata"]["filename"]
            img_path = os.path.join(file_directory, filename)
            converted_text = cpm_convertor.convert(query=question, img_path=img_path)
            img_captions.append(
                {"image_path": img_path, "caption": converted_text}
            )
            converted_doc = doc
            converted_doc.text = converted_text



    img_captions_df = pd.DataFrame(img_captions)
    img_captions_df.to_csv(f"{output_dir}/img_captions.csv", index=False)

    cpm_convertor.clear_GPU_mem()

    return converted_doc