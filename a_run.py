

from fastapi import FastAPI, UploadFile, HTTPException, File
from src.chatbot_web_demo.pages.doc_parse_demo.main_dev import process_data, load_model
from src.chatbot_web_demo.pages.qa_demo.summary_utils import read_summary
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os


app = FastAPI()

# 解决跨域问题
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 允许所有来源
#     allow_credentials=True,
#     allow_methods=["*"],  # 允许所有HTTP方法
#     allow_headers=["*"],  # 允许所有HTTP头
# )


DATA_DIR = "/home/project/Chatbot_Web_Demo/doc_parse_data"
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")


@app.post("/recogReport")
async def analyse_Report(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只接受PDF文件")

    # 保存上传的文件
    filepath = os.path.join(INPUT_DIR, file.filename)
    os.makedirs(os.path.join(DATA_DIR, file.filename.split(".")[0]), exist_ok=True)
    output_dir = os.path.join(DATA_DIR, file.filename.split(".")[0])
    unstructured_output_dir = os.path.join(output_dir, "unstructured")
    os.makedirs(INPUT_DIR, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    # 确认文件是否正确保存
    # if not os.path.exists(filepath):
    #     raise HTTPException(status_code=500, detail="文件保存失败")
    # print(f"文件已保存到 {filepath}")
    
    # 处理文件
    output_dir, unstructured_output_dir = process_data(filepath, output_dir, unstructured_output_dir)

     # 加载模型
    load_model()
    
    # 获取摘要和检测结果
    summary = read_summary(unstructured_output_dir)
    
    return JSONResponse(content={
        'success': 'true',
        'msg': "上传成功",
        'result': {
            'summary': summary
        }
    })


if __name__ == '__main__':
    uvicorn.run("a_run:app", port=8080, reload=True)
