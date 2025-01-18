from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import uvicorn
import base64
import io 
from PIL import Image
from fastapi import Body
from pymilvus import MilvusClient

# 导入函数

from src.chatbot_web_demo.pages.detect_demo.table_detection.inference_pdf import detect_pdf,check
from src.chatbot_web_demo.pages.detect_demo.table_detection.inference import detect_image
from src.chatbot_web_demo.pages.detect_demo.table_detection.tableDetect import table_detect

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_pdf(file):
    return file.content_type=="application/pdf"

output_folder = "output"

@app.post("/detectTables/")
async def detect_tables(file: UploadFile = File(...)):

    check()

    # 读取上传的文件
    file_content = await file.read()

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        if is_pdf(file):
        # 临时保存PDF文件以便处理
            with open("temp.pdf", "wb") as temp_pdf:
                temp_pdf.write(file_content)
                input_pdf_path = "temp.pdf"

            detect_pdf(input_pdf_path, output_folder)
            if os.path.exists(input_pdf_path):
                os.remove(input_pdf_path)
            check()

        # 收集输出文件
            output_files = []
            for filename in os.listdir(output_folder):
                if filename.startswith("detected_") and filename.endswith(".png") and filename.count('_')==1:
                    filepath = os.path.join(output_folder, filename)
                    output_files.append(filepath)



        else:
            with open("temp.png", "wb") as temp_png:
                temp_png.write(file_content)
                input_image_path = "temp.png"

            try:
                output_image_path=os.path.join(output_folder,"output.png")
                detect_image(input_image_path, output_image_path)
                check()
                output_files = []
                output_files.append(output_image_path)

                if os.path.exists(input_image_path):
                    os.remove(input_image_path)                

            except:
                return JSONResponse(status_code=200,content={"message":"该图像不存在表格"})
            

           # 将图片转换为 base64 编码
        base64_images = []
        for image_path in output_files:
            with open(image_path, "rb") as f:
                base64_encoded = base64.b64encode(f.read()).decode('utf-8')
                base64_images.append(base64_encoded)  

    # 返回成功消息和输出的base64编码
        return JSONResponse(status_code=200,content={"message":"successfully","output_files":output_files,"base64_images": base64_images})                

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


# 新增的接口
@app.post("/processBase64/")
async def process_base64(data: dict = Body(...)):
    try:
        # 从 JSON 数据中提取 base64 编码
        base64_images = data.get("base64_images")

        if not base64_images:
            raise ValueError("No base64 images data found in the JSON.")

        # 定义输出文件夹
        output_folder = "TableOutput"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 解码 base64 数据并保存图像
        structured_data = []

        for index, base64_data in enumerate(base64_images):
            # 解码 base64 数据
            decoded_data = base64.b64decode(base64_data)
            # 将解码后的数据转换为 PIL.Image 对象
            image = Image.open(io.BytesIO(decoded_data))
            # 保存图像到文件
            image_path = os.path.join(output_folder, f"image_{index}.png")
            image.save(image_path)

            # 使用 table_detect 函数处理图像
            table_data =table_detect(image_path, output_folder)

            #将处理后的表格数据添加到结构化数据列表
            structured_data.extend(table_data or [])
            # 保存结构化的文本数据
            text_output_path = os.path.join(output_folder, "structured_data.txt")
            # 将数据转换为字符串形式
            data_text = '\n'.join(str(item) for item in structured_data)
            # 保存为文本文件
            with open(text_output_path, 'w') as text_file:
                text_file.write(data_text)
        # 清理临时文件
        if os.path.exists(image_path):
            os.remove(image_path)
        # 返回文本数据路径
            return JSONResponse(status_code=200, content={"message": "Images processed successfully.", "output_folder": output_folder, "text_output_path": text_output_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
   

def get_milvus_collections():
    # Connect to Milvus server
    milvus_client = MilvusClient(
        uri="http://localhost:19530/", db_name="default", token="root:Milvus"
    )
    collections = sorted(milvus_client.list_collections())
    return collections

@app.get("/collections")
async def get_collections():
    collections=get_milvus_collections()
    return {"collections":collections}


if __name__=='__main__':
    uvicorn.run("fasttest:app",port=8080,reload=True)
